from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "bot"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bot.consultant import parse_dialog, search_queries  # noqa: E402
from eval.run_local import DATA, detect_models, read_csv  # noqa: E402
from bot.kb import STOP, Chunk, KnowledgeBase, load_documents, stem, tokenize  # noqa: E402
from bot.settings import CANDIDATES, CONTEXT_CHUNKS  # noqa: E402

JUDGE_SYSTEM = """Ты — строгий и объективный судья соревнования ИИ-консультантов МТС EnergyTool.
Твоя задача — оценить два ответа (Ответ A и Ответ B) на вопрос клиента, опираясь ТОЛЬКО на Эталонный ответ.

Правила оценки:
1. Фактическая точность: все цифры, пороги мощности, сроки, условия и ограничения из Эталона должны быть переданы верно. Выдумывание фактов, цен или советов из общих знаний — грубая ошибка.
2. Поведение консультанта: прямой ответ в начале, краткость (обычно 30–80 слов, связный текст без списков и markdown), в конце — уместный уточняющий вопрос или предложение расчёта специалистом (как в Эталоне).
3. Оцени соответствие Эталону отдельно для A (match_a) и для B (match_b) по шкале:
   - 1.0: все ключевые факты и смысл Эталона переданы верно, нет выдумок;
   - 0.5: ответ частично верный, но упущена важная деталь/оговорка из Эталона или есть лишняя вода;
   - 0.0: ответ неверный, противоречит Эталону или содержит галлюцинации.
4. Выбери победителя (winner): "A", "B" или "tie" (ничья, если оба ответа одинаковы по качеству).

Верни ТОЛЬКО валидный JSON в одну строку без пояснений:
{"match_a": 1.0, "match_b": 0.5, "winner": "A", "reason": "кратко почему"} /no_think"""

JSON_RE = re.compile(r"\{.*?\}", re.S)
WORD_RAW_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
DIGIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

QUESTION_FILLER_STEMS = {
    stem(w)
    for w in (
        "какой", "какая", "какие", "каков", "сколько", "подскажите", "уточните",
        "скажите", "нужен", "нужна", "нужно", "требуется", "планируете", "планируется",
        "рассматриваете", "интересует", "хотите", "можете", "знаете", "имеется",
    )
}


@dataclass
class SemanticScorer:
    """Хранилище BGE-M3 эмбеддингов слов и полных ответов для честного BERTScore + SemSim (с TF-IDF фоллбэком)."""
    word_vecs: dict[str, np.ndarray]
    text_vecs: dict[str, np.ndarray]
    baseline_sim: float
    char_vectorizer: TfidfVectorizer
    has_bge: bool


def _unit_vec(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else v


def content_words(text: str) -> list[str]:
    """Токены без стоп-слов в исходной словоформе (для METEOR и BGE-BERTScore)."""
    words = WORD_RAW_RE.findall(text.lower().replace("ё", "е"))
    return [w for w in words if w not in STOP and len(w) > 1]


def extract_digits(text: str) -> set[str]:
    """Нормализованные числа из текста (1,5 -> 1.5)."""
    return {d.replace(",", ".") for d in DIGIT_RE.findall(text)}


# ---------- Классические NLP/RAG метрики (BLEU, ROUGE, METEOR, BGE-BERTScore) ----------

def compute_bleu4(ref_tokens: list[str], cand_tokens: list[str]) -> float:
    """Sentence BLEU-4 со сглаживанием Lin & Och (method 1) и Brevity Penalty."""
    if not cand_tokens or not ref_tokens:
        return 0.0
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1.0 - len(ref_tokens) / max(len(cand_tokens), 1))
    log_sum = 0.0
    max_n = min(4, len(ref_tokens), len(cand_tokens))
    if max_n == 0:
        return 0.0
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        cand_ngrams = Counter(tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1))
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)
        eps = 0.1 if n > 1 else 0.0
        p_n = (overlap + eps) / (total + eps)
        if p_n <= 0:
            return 0.0
        log_sum += math.log(p_n) / max_n
    return bp * math.exp(log_sum)


def _f1(p: float, r: float) -> float:
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_rouge_n(ref_tokens: list[str], cand_tokens: list[str], n: int = 1) -> float:
    """ROUGE-N (F1-мера по пересечению n-грамм)."""
    if len(ref_tokens) < n or len(cand_tokens) < n:
        return 0.0
    ref_ng = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
    cand_ng = Counter(tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1))
    overlap = sum(min(c, ref_ng[g]) for g, c in cand_ng.items())
    p = overlap / max(sum(cand_ng.values()), 1)
    r = overlap / max(sum(ref_ng.values()), 1)
    return _f1(p, r)


def compute_rouge_l(ref_tokens: list[str], cand_tokens: list[str]) -> float:
    """ROUGE-L (F1-мера на базе длины наибольшей общей подпоследовательности LCS)."""
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    lcs = dp[n]
    return _f1(lcs / n, lcs / m)


def compute_meteor(ref_words: list[str], cand_words: list[str]) -> float:
    """METEOR (Banerjee & Lavie): 3-этапное выравнивание (точное -> стем -> нечёткое >=82) + штраф за фрагментацию."""
    if not ref_words or not cand_words:
        return 0.0
    ref_stems = [stem(w) for w in ref_words]
    cand_stems = [stem(w) for w in cand_words]

    used_ref: set[int] = set()
    alignment: list[tuple[int, int]] = []

    for ci, cw in enumerate(cand_words):
        for ri, rw in enumerate(ref_words):
            if ri not in used_ref and cw == rw:
                used_ref.add(ri)
                alignment.append((ci, ri))
                break

    matched_cand = {ci for ci, _ in alignment}

    for ci, cs in enumerate(cand_stems):
        if ci in matched_cand:
            continue
        for ri, rs in enumerate(ref_stems):
            if ri not in used_ref and cs == rs:
                used_ref.add(ri)
                matched_cand.add(ci)
                alignment.append((ci, ri))
                break

    for ci, cw in enumerate(cand_words):
        if ci in matched_cand or len(cw) < 4:
            continue
        best_ri, best_score = -1, 0.0
        for ri, rw in enumerate(ref_words):
            if ri not in used_ref and len(rw) >= 4:
                sc = fuzz.ratio(cw, rw)
                if sc >= 82.0 and sc > best_score:
                    best_score = sc
                    best_ri = ri
        if best_ri != -1:
            used_ref.add(best_ri)
            matched_cand.add(ci)
            alignment.append((ci, best_ri))

    matches = len(alignment)
    if matches == 0:
        return 0.0

    p = matches / len(cand_words)
    r = matches / len(ref_words)
    f_mean = (10.0 * p * r) / (r + 9.0 * p) if (r + 9.0 * p) > 0 else 0.0

    alignment.sort(key=lambda x: x[0])
    chunks = 1
    for k in range(1, len(alignment)):
        if alignment[k][1] != alignment[k - 1][1] + 1:
            chunks += 1
    frag = chunks / matches
    penalty = 0.5 * (frag ** 3)
    return f_mean * (1.0 - penalty)


def compute_bertscore_and_semsim(
    golden_text: str,
    answer_text: str,
    ref_words: list[str],
    cand_words: list[str],
    scorer: SemanticScorer,
    idf_map: dict[str, float],
    default_idf: float = 2.5,
) -> tuple[float, float]:
    """Возвращает (BERTScore-F1, Whole-Answer Semantic Similarity)."""
    if not ref_words or not cand_words:
        return 0.0, 0.0

    ref_weights = np.array([idf_map.get(stem(w), default_idf) for w in ref_words], dtype=np.float32)
    cand_weights = np.array([idf_map.get(stem(w), default_idf) for w in cand_words], dtype=np.float32)

    if scorer.has_bge and all(w in scorer.word_vecs for w in ref_words) and all(w in scorer.word_vecs for w in cand_words):
        ref_mat = np.vstack([scorer.word_vecs[w] for w in ref_words])
        cand_mat = np.vstack([scorer.word_vecs[w] for w in cand_words])
        raw_sim = ref_mat @ cand_mat.T

        b = scorer.baseline_sim
        sim_matrix = np.clip((raw_sim - b) / max(1.0 - b, 1e-6), 0.0, 1.0)
        for i, rw in enumerate(ref_words):
            rs = stem(rw)
            for j, cw in enumerate(cand_words):
                if rw == cw or rs == stem(cw):
                    sim_matrix[i, j] = 1.0
    else:
        ref_vecs = scorer.char_vectorizer.transform(ref_words)
        cand_vecs = scorer.char_vectorizer.transform(cand_words)
        sim_matrix = cosine_similarity(ref_vecs, cand_vecs)

    r_score = float(np.sum(sim_matrix.max(axis=1) * ref_weights) / max(np.sum(ref_weights), 1e-6))
    p_score = float(np.sum(sim_matrix.max(axis=0) * cand_weights) / max(np.sum(cand_weights), 1e-6))
    bert_f1 = _f1(p_score, r_score)

    g_key, a_key = golden_text.strip()[:2000], answer_text.strip()[:2000]
    if scorer.has_bge and g_key in scorer.text_vecs and a_key in scorer.text_vecs:
        sem_sim = float(np.clip(np.dot(scorer.text_vecs[g_key], scorer.text_vecs[a_key]), 0.0, 1.0))
    else:
        g_tf = scorer.char_vectorizer.transform([golden_text])
        a_tf = scorer.char_vectorizer.transform([answer_text])
        sem_sim = float(cosine_similarity(g_tf, a_tf)[0, 0])

    return bert_f1, sem_sim


# ---------- Послойный анализатор (Слои 1, 2, 3, 4) ----------

def _resolve_top6_chunks(
    debug_row: dict | None,
    chunk_by_id: dict[int, Chunk],
    chunks_by_key: dict[tuple[str, str], list[Chunk]],
    fallback_top16: list[Chunk],
) -> list[Chunk]:
    """Точное восстановление Top-6 чанков по их ID (без склейки длинных разделов в мега-чанк)."""
    if not debug_row or not debug_row.get("sources"):
        return fallback_top16[:CONTEXT_CHUNKS]

    top16_by_key: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for c in fallback_top16:
        top16_by_key[(c.source, c.title)].append(c)

    used_ids: set[int] = set()
    resolved: list[Chunk] = []

    for s in debug_row["sources"]:
        cid = s.get("id")
        if cid is not None and int(cid) in chunk_by_id:
            ch = chunk_by_id[int(cid)]
            resolved.append(ch)
            used_ids.add(ch.id)
            continue

        key = (s.get("source", ""), s.get("title", ""))
        candidates_for_key = top16_by_key.get(key) or chunks_by_key.get(key) or []
        chosen = next((c for c in candidates_for_key if c.id not in used_ids), None)
        if chosen is None and candidates_for_key:
            chosen = candidates_for_key[0]
        if chosen is not None:
            resolved.append(chosen)
            used_ids.add(chosen.id)

    return resolved if resolved else fallback_top16[:CONTEXT_CHUNKS]


def evaluate_all_layers(
    question: str,
    golden: str,
    answer: str,
    debug_row: dict | None,
    kb: KnowledgeBase,
    query_vectors: list | None,
    chunk_by_id: dict[int, Chunk],
    chunks_by_key: dict[tuple[str, str], list[Chunk]],
    scorer: SemanticScorer,
) -> dict:
    idf_map = kb.bm25.idf
    default_idf = 2.5

    def idf_weight(tokens: set[str]) -> float:
        return sum(idf_map.get(t, default_idf) for t in tokens)

    messages = parse_dialog(question)
    queries = (
        debug_row["queries"]
        if (debug_row and debug_row.get("queries"))
        else search_queries(messages)
    )

    last_user_tokens = set(tokenize(messages[-1]["content"]))
    golden_tokens_all = set(tokenize(golden))
    target_facts = golden_tokens_all - last_user_tokens
    if not target_facts:
        target_facts = golden_tokens_all
    target_total_idf = max(idf_weight(target_facts), 1e-6)

    # --- СЛОЙ 1: RETRIEVAL (Топ-16 кандидатов до реранкера) ---
    if debug_row and debug_row.get("candidate_ids"):
        top16_chunks = [chunk_by_id[int(cid)] for cid in debug_row["candidate_ids"] if int(cid) in chunk_by_id]
    else:
        top16_chunks = kb.search(queries, query_vectors=query_vectors, k=CANDIDATES)

    top6_chunks = _resolve_top6_chunks(debug_row, chunk_by_id, chunks_by_key, top16_chunks)

    seen_ids = {c.id for c in top16_chunks}
    all_candidates = list(top16_chunks)
    for c in top6_chunks:
        if c.id not in seen_ids:
            all_candidates.append(c)
            seen_ids.add(c.id)

    top16_tokens: set[str] = set()
    best_rank_16 = 0
    best_overlap_16 = 0.0
    for idx, ch in enumerate(all_candidates, 1):
        ch_toks = set(tokenize(ch.render()))
        top16_tokens |= ch_toks
        ov = idf_weight(target_facts & ch_toks) / target_total_idf
        if ov > best_overlap_16:
            best_overlap_16 = ov
            best_rank_16 = idx
    recall_16 = idf_weight(target_facts & top16_tokens) / target_total_idf

    # --- СЛОЙ 2: RERANKER / FINAL CONTEXT (Топ-6 чанков в промпте) ---
    top6_tokens: set[str] = set()
    top6_digits: set[str] = set()
    best_rank_6 = 0
    best_overlap_6 = 0.0
    useful_chunks_6 = 0
    for idx, ch in enumerate(top6_chunks, 1):
        rendered = ch.render()
        ch_toks = set(tokenize(rendered))
        top6_tokens |= ch_toks
        top6_digits |= extract_digits(rendered)
        ov = idf_weight(target_facts & ch_toks) / target_total_idf
        if ov >= 0.12:
            useful_chunks_6 += 1
        if ov > best_overlap_6:
            best_overlap_6 = ov
            best_rank_6 = idx
    recall_6 = idf_weight(target_facts & top6_tokens) / target_total_idf
    rerank_retention = recall_6 / max(recall_16, 1e-6) if recall_16 > 0 else 1.0

    # --- СЛОЙ 3: КОНТЕКСТ ДИАЛОГА ---
    is_dialog = len(messages) > 1
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    prior_user_tokens = set(tokenize(" ".join(user_turns[:-1]))) if len(user_turns) > 1 else set()
    all_user_tokens = prior_user_tokens | last_user_tokens
    prior_all_tokens = set(tokenize(" ".join(m["content"] for m in messages[:-1]))) if is_dialog else set()

    focused_queries = queries[:-1] if len(queries) > 1 else queries
    focused_q_tokens = set(tokenize(" ".join(focused_queries)))
    if prior_user_tokens:
        focused_history_retention = idf_weight(prior_user_tokens & focused_q_tokens) / max(idf_weight(prior_user_tokens), 1e-6)
    else:
        focused_history_retention = 1.0

    ans_tokens_set = set(tokenize(answer))
    dialog_anchors_in_golden = golden_tokens_all & prior_all_tokens
    if is_dialog and dialog_anchors_in_golden:
        dialog_grounding = idf_weight(dialog_anchors_in_golden & ans_tokens_set) / max(idf_weight(dialog_anchors_in_golden), 1e-6)
    else:
        dialog_grounding = 1.0

    question_novelty = 1.0
    if is_dialog and "?" in answer and prior_user_tokens:
        sents = re.split(r"(?<=[.!?])\s+", answer.strip())
        ask_sent = next((s for s in reversed(sents) if "?" in s), "")
        ask_domain_toks = set(tokenize(ask_sent)) - QUESTION_FILLER_STEMS
        if ask_domain_toks:
            novel_domain_toks = ask_domain_toks - all_user_tokens
            reasked_prior_toks = ask_domain_toks & prior_user_tokens
            novel_idf = idf_weight(novel_domain_toks)
            if len(novel_domain_toks) >= 2 or novel_idf >= 2.0 * default_idf:
                question_novelty = 1.0
            elif len(novel_domain_toks) == 1:
                question_novelty = 0.75 if not reasked_prior_toks else 0.5
            else:
                question_novelty = 0.0 if reasked_prior_toks else 0.5

    # --- СЛОЙ 4: ГЕНЕРАЦИЯ ОТВЕТА (RAG + BLEU, ROUGE, METEOR, BGE-BERTScore, Guardrails) ---
    found_facts_in_top6 = target_facts & top6_tokens
    used_facts_from_top6 = found_facts_in_top6 & ans_tokens_set

    fact_recall = idf_weight(target_facts & ans_tokens_set) / target_total_idf
    context_utilization = (
        idf_weight(used_facts_from_top6) / max(idf_weight(found_facts_in_top6), 1e-6)
        if found_facts_in_top6 else 0.0
    )

    g_stems = tokenize(golden)
    a_stems = tokenize(answer)
    g_cwords = content_words(golden)
    a_cwords = content_words(answer)

    bleu4 = compute_bleu4(g_stems, a_stems)
    rouge1 = compute_rouge_n(g_stems, a_stems, n=1)
    rouge2 = compute_rouge_n(g_stems, a_stems, n=2)
    rougeL = compute_rouge_l(g_stems, a_stems)
    meteor = compute_meteor(g_cwords, a_cwords)
    bertscore_f1, bge_sem_sim = compute_bertscore_and_semsim(
        golden, answer, g_cwords, a_cwords, scorer, idf_map, default_idf
    )

    word_count = len(answer.split())
    length_ok = 25 <= word_count <= 85
    no_markdown_lists = (
        ("**" not in answer)
        and ("__" not in answer)
        and not bool(re.search(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+", answer))
    )
    golden_has_q = "?" in golden
    ans_has_q = "?" in answer
    q_match_ok = (ans_has_q == golden_has_q) or ans_has_q

    g_digits = extract_digits(golden)
    a_digits = extract_digits(answer)
    q_digits = extract_digits(question)
    allowed_digits = g_digits | q_digits | top6_digits

    digits_recall = len(g_digits & a_digits) / len(g_digits) if g_digits else 1.0
    digits_precision = len(a_digits & allowed_digits) / len(a_digits) if a_digits else 1.0
    digits_score = (0.6 * digits_recall + 0.4 * digits_precision) if g_digits else digits_precision

    guardrails_score = (
        (1.0 if length_ok else 0.0)
        + (1.0 if no_markdown_lists else 0.0)
        + (1.0 if q_match_ok else 0.0)
        + digits_score
    ) / 4.0

    return {
        "chunks_titles": [f"[#{c.id}] {c.source} :: {c.title}" for c in top6_chunks],
        "layer1_retrieval": {
            "recall_16": recall_16,
            "best_rank_16": best_rank_16,
            "best_chunk_cov_16": best_overlap_16,
        },
        "layer2_reranker": {
            "recall_6": recall_6,
            "retention": rerank_retention,
            "best_rank_6": best_rank_6,
            "useful_chunks_6": useful_chunks_6,
            "total_chunks_6": len(top6_chunks),
            "reranked_flag": bool(debug_row and debug_row.get("reranked")),
        },
        "layer3_dialog": {
            "is_dialog": is_dialog,
            "turns": len(messages),
            "focused_history_retention": focused_history_retention,
            "dialog_grounding": dialog_grounding,
            "question_novelty": question_novelty,
        },
        "layer4_generation": {
            "fact_recall": fact_recall,
            "context_utilization": context_utilization,
            "bleu4": bleu4,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "meteor": meteor,
            "bertscore_f1": bertscore_f1,
            "bge_sem_sim": bge_sem_sim,
            "guardrails": guardrails_score,
            "word_count": word_count,
            "length_ok": length_ok,
            "no_markdown": no_markdown_lists,
            "has_question": ans_has_q,
            "digits_recall": digits_recall,
            "digits_precision": digits_precision,
        },
    }


# ---------- Инфраструктура запуска и судьи ----------

def resolve_csv(path_str: str) -> Path:
    p = Path(path_str)
    return p / "submission.csv" if p.is_dir() else p


def find_default_runs() -> tuple[Path, Path]:
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        sys.exit("Папка runs/ не найдена. Сначала выполните прогон eval/run_local.py")
    subdirs = sorted([d for d in runs_dir.iterdir() if (d / "submission.csv").is_file()])
    if len(subdirs) < 2:
        sys.exit(f"В папке runs/ найдено прогонов: {len(subdirs)}. Для сравнения нужно минимум 2.")
    return subdirs[0] / "submission.csv", subdirs[-1] / "submission.csv"


def load_debug_rows(csv_path: Path) -> dict[str, dict]:
    dbg = csv_path.parent / "debug.jsonl"
    if not dbg.is_file():
        return {}
    out = {}
    for line in dbg.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


def init_kb_and_embeddings(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    emb_model: str,
    questions: dict[str, str],
    eval_texts: list[str],
) -> tuple[
    KnowledgeBase,
    dict[str, list],
    dict[int, Chunk],
    dict[tuple[str, str], list[Chunk]],
    SemanticScorer,
]:
    kb_path = None
    for name in ("knowledge-base.zip", "knowledge_base.zip", "kb.zip"):
        for folder in (DATA, ROOT / "data", ROOT):
            p = folder / name
            if p.is_file():
                kb_path = p
                break
        if kb_path:
            break
    if not kb_path:
        raise FileNotFoundError("Не найден архив knowledge-base.zip")

    kb = KnowledgeBase(load_documents(kb_path))
    chunk_by_id: dict[int, Chunk] = {}
    chunks_by_key: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    vocab_words: set[str] = set()

    for c in kb.chunks:
        chunk_by_id[c.id] = c
        chunks_by_key[(c.source, c.title)].append(c)
        vocab_words.update(content_words(c.render()))

    for t in eval_texts:
        vocab_words.update(content_words(t))

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    vectorizer.fit(sorted(vocab_words) if vocab_words else ["энергия"])

    query_vectors: dict[str, list] = {}
    word_vecs: dict[str, np.ndarray] = {}
    text_vecs: dict[str, np.ndarray] = {}
    baseline_sim = 0.45
    has_bge = False

    if emb_model:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        def _embed(texts: list[str]) -> list[list[float]]:
            vecs: list[list[float]] = []
            for i in range(0, len(texts), 32):
                batch = texts[i:i + 32]
                r = client.post(f"{base_url}/embeddings", headers=headers, json={"model": emb_model, "input": batch})
                r.raise_for_status()
                items = sorted(r.json()["data"], key=lambda x: x.get("index", 0))
                vecs.extend(it["embedding"] for it in items)
            return vecs

        try:
            kb.enable_dense(_embed)
            dialogs = {qid: parse_dialog(q) for qid, q in questions.items()}
            unique_qs = sorted({q[:2000] for qid in dialogs for q in search_queries(dialogs[qid]) if q.strip()})
            q_emb_map = dict(zip(unique_qs, _embed(unique_qs))) if unique_qs else {}
            query_vectors = {qid: [q_emb_map.get(q[:2000]) for q in search_queries(dialogs[qid])] for qid in dialogs}

            eval_words = sorted({w for t in eval_texts for w in content_words(t)})
            if eval_words:
                raw_w_vecs = _embed(eval_words)
                word_vecs = {w: _unit_vec(np.asarray(v, dtype=np.float32)) for w, v in zip(eval_words, raw_w_vecs)}
                if len(eval_words) >= 8:
                    sample_mat = np.vstack([word_vecs[w] for w in eval_words[:80]])
                    sim_sample = sample_mat @ sample_mat.T
                    off_diag = sim_sample[~np.eye(len(sample_mat), dtype=bool)]
                    baseline_sim = float(np.median(off_diag))

            unique_full_texts = sorted({t.strip()[:2000] for t in eval_texts if t.strip()})
            if unique_full_texts:
                raw_t_vecs = _embed(unique_full_texts)
                text_vecs = {
                    t: _unit_vec(np.asarray(v, dtype=np.float32))
                    for t, v in zip(unique_full_texts, raw_t_vecs)
                }
            has_bge = True
        except Exception as e:
            print(f"[Предупреждение] Эмбеддинги BGE недоступны ({e}), используется BM25 / Char-TFIDF", file=sys.stderr)
            kb.vectors = None
            query_vectors = {}

    scorer = SemanticScorer(
        word_vecs=word_vecs,
        text_vecs=text_vecs,
        baseline_sim=baseline_sim,
        char_vectorizer=vectorizer,
        has_bge=has_bge,
    )
    return kb, query_vectors, chunk_by_id, chunks_by_key, scorer


def call_judge(client: httpx.Client, base_url: str, key: str, model: str,
               question: str, golden: str, ans_a: str, ans_b: str) -> dict:
    user_prompt = (
        f"Вопрос клиента:\n{question}\n\n"
        f"Эталонный ответ:\n{golden}\n\n"
        f"Ответ A:\n{ans_a}\n\n"
        f"Ответ B:\n{ans_b}"
    )
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = client.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    m = JSON_RE.search(content)
    if not m:
        return {"match_a": 0.5, "match_b": 0.5, "winner": "TIE", "reason": "не удалось разобрать JSON"}
    try:
        data = json.loads(m.group(0))
        return {
            "match_a": float(data.get("match_a", 0.5)),
            "match_b": float(data.get("match_b", 0.5)),
            "winner": str(data.get("winner", "tie")).strip().upper(),
            "reason": str(data.get("reason", "")),
        }
    except (ValueError, TypeError):
        return {"match_a": 0.5, "match_b": 0.5, "winner": "TIE", "reason": "ошибка JSON"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Попарный LLM-судья + Послойные метрики RAG")
    parser.add_argument("baseline", nargs="?", help="папка или submission.csv бейзлайна")
    parser.add_argument("candidate", nargs="?", help="папка или submission.csv нового решения")
    args = parser.parse_args()

    if args.baseline and args.candidate:
        base_path, cand_path = resolve_csv(args.baseline), resolve_csv(args.candidate)
    else:
        base_path, cand_path = find_default_runs()

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "lm-studio")
    model = os.getenv("LLM_MODEL", "")
    emb_model = os.getenv("EMB_MODEL", "")
    detected_llm, detected_emb = detect_models(base_url, api_key)
    model = model or detected_llm
    emb_model = emb_model or detected_emb
    if not model:
        sys.exit("В LM Studio не загружена языковая модель")

    questions = read_csv(DATA / "demo_questions.csv")
    golden = read_csv(DATA / "demo_output.csv")
    baseline = read_csv(base_path)
    candidate = read_csv(cand_path)

    qids = [qid for qid in questions if qid in candidate and qid in baseline and qid in golden]
    eval_texts = [golden[qid] for qid in qids] + [baseline[qid] for qid in qids] + [candidate[qid] for qid in qids]

    client = httpx.Client(timeout=120.0, trust_env=False)
    kb, query_vectors, chunk_by_id, chunks_by_key, scorer = init_kb_and_embeddings(
        client, base_url, api_key, emb_model, questions, eval_texts
    )
    base_dbg = load_debug_rows(base_path)
    cand_dbg = load_debug_rows(cand_path)

    bert_label = "BERTScore(BGE)" if scorer.has_bge else "BERTScore(Char)"
    print(f"Судья: {model} | Эмбеддер: {emb_model if scorer.has_bge else 'нет (BM25)'} | Вопросов: {len(qids)}")
    print(f"Baseline : {base_path}")
    print(f"Candidate: {cand_path}\n")

    results = []

    for qid in qids:
        q, g = questions[qid], golden[qid]
        b_ans, c_ans = baseline[qid], candidate[qid]

        b_sources = [(s.get("id"), s.get("title")) for s in base_dbg.get(qid, {}).get("sources", [])]
        c_sources = [(s.get("id"), s.get("title")) for s in cand_dbg.get(qid, {}).get("sources", [])]
        same_chunks = b_sources == c_sources
        tag = "[чанки те же]" if same_chunks else "[ПОИСК ИЗМЕНИЛСЯ]"

        q_vecs = query_vectors.get(qid)
        cand_layers = evaluate_all_layers(
            q, g, c_ans, cand_dbg.get(qid), kb, q_vecs, chunk_by_id, chunks_by_key, scorer
        )
        base_layers = evaluate_all_layers(
            q, g, b_ans, base_dbg.get(qid), kb, q_vecs, chunk_by_id, chunks_by_key, scorer
        )

        l1 = cand_layers["layer1_retrieval"]
        l2 = cand_layers["layer2_reranker"]
        l3 = cand_layers["layer3_dialog"]
        l4 = cand_layers["layer4_generation"]

        r1 = call_judge(client, base_url, api_key, model, q, g, c_ans, b_ans)
        if b_ans.strip() == c_ans.strip():
            cand_match = base_match = r1["match_a"]
            win_rate = 0.5
            reason = f"{tag} Ответы идентичны"
        else:
            r2 = call_judge(client, base_url, api_key, model, q, g, b_ans, c_ans)
            cand_match = (r1["match_a"] + r2["match_b"]) / 2.0
            base_match = (r1["match_b"] + r2["match_a"]) / 2.0
            w1 = 1.0 if r1["winner"] == "A" else (0.0 if r1["winner"] == "B" else 0.5)
            w2 = 1.0 if r2["winner"] == "B" else (0.0 if r2["winner"] == "A" else 0.5)
            win_rate = (w1 + w2) / 2.0
            reason = f"{tag} {r1['reason'] or r2['reason']}"

        score = 0.7 * cand_match + 0.3 * win_rate
        results.append({
            "id": qid,
            "question": q,
            "golden": g,
            "cand_ans": c_ans,
            "base_ans": b_ans,
            "score": score,
            "cand_match": cand_match,
            "base_match": base_match,
            "win_rate": win_rate,
            "reason": reason,
            "layers": cand_layers,
            "base_layers": base_layers,
        })

        print("=" * 105)
        print(f"ВОПРОС #{qid} | Score: {score:.2f} (Match: {cand_match:.2f} vs Base {base_match:.2f} | WinRate: {win_rate:.2f}) {tag}")
        print("-" * 105)
        print(f"Вопрос клиента:\n{q}\n")
        print(f"Эталонный ответ:\n{g}\n")
        print(f"Текущий ответ модели:\n{c_ans}\n")
        print("Чанки в контексте (Top-6):")
        for idx, ct in enumerate(cand_layers["chunks_titles"], 1):
            print(f"  [{idx}] {ct}")
        print("-" * 105)
        print(
            f"  [Слой 1: Поиск Top-{CANDIDATES}]   FactRecall@{CANDIDATES}: {l1['recall_16']:.1%} | "
            f"Ранг лучшего чанка: #{l1['best_rank_16']} (покрытие лучшего: {l1['best_chunk_cov_16']:.1%})"
        )
        print(
            f"  [Слой 2: Реранкер Top-{CONTEXT_CHUNKS}] FactRecall@{CONTEXT_CHUNKS}:  {l2['recall_6']:.1%} | "
            f"Сохранение (Top-6/Top-16): {l2['retention']:.1%} | Полезных чанков: {l2['useful_chunks_6']}/{l2['total_chunks_6']} | "
            f"Ранг лучшего: #{l2['best_rank_6']} | Reranked: {l2['reranked_flag']}"
        )
        if l3["is_dialog"]:
            print(
                f"  [Слой 3: Диалог ({l3['turns']} репл.)] Фокус-запрос: {l3['focused_history_retention']:.1%} | "
                f"Учёт истории в ответе: {l3['dialog_grounding']:.1%} | Новизна вопроса: {l3['question_novelty']:.1%}"
            )
        else:
            print("  [Слой 3: Диалог]           Одиночный вопрос (1 реплика)")
        print(
            f"  [Слой 4: Генерация (RAG)]  FactRecall ответа: {l4['fact_recall']:.1%} | "
            f"Context Utilization: {l4['context_utilization']:.1%} | "
            f"Guardrails: {l4['guardrails']:.1%} ({l4['word_count']} сл., цифры Rec/Prec: {l4['digits_recall']:.0%}/{l4['digits_precision']:.0%})"
        )
        print(
            f"  [Слой 4: NLP-метрики]      BLEU-4: {l4['bleu4']:.3f} | ROUGE-1: {l4['rouge1']:.3f} | "
            f"ROUGE-2: {l4['rouge2']:.3f} | ROUGE-L: {l4['rougeL']:.3f} | "
            f"METEOR: {l4['meteor']:.3f} | {bert_label}: {l4['bertscore_f1']:.3f} | SemSim: {l4['bge_sem_sim']:.3f}"
        )
        print(f"  [Судья] {reason}")

    n = max(len(results), 1)
    avg_match = sum(r["cand_match"] for r in results) / n
    avg_base_match = sum(r["base_match"] for r in results) / n
    avg_win = sum(r["win_rate"] for r in results) / n
    total_score = 0.7 * avg_match + 0.3 * avg_win

    def avg_metric(key1: str, key2: str, use_base: bool = False, only_dialogs: bool = False) -> float:
        key_root = "base_layers" if use_base else "layers"
        rows = [
            r[key_root][key1][key2]
            for r in results
            if (not only_dialogs or r[key_root]["layer3_dialog"]["is_dialog"])
        ]
        return sum(rows) / max(len(rows), 1)

    c_r16, b_r16 = avg_metric("layer1_retrieval", "recall_16"), avg_metric("layer1_retrieval", "recall_16", True)
    c_r6, b_r6 = avg_metric("layer2_reranker", "recall_6"), avg_metric("layer2_reranker", "recall_6", True)
    c_ret, b_ret = avg_metric("layer2_reranker", "retention"), avg_metric("layer2_reranker", "retention", True)

    c_fhist, b_fhist = avg_metric("layer3_dialog", "focused_history_retention", only_dialogs=True), avg_metric("layer3_dialog", "focused_history_retention", True, True)
    c_dgr, b_dgr = avg_metric("layer3_dialog", "dialog_grounding", only_dialogs=True), avg_metric("layer3_dialog", "dialog_grounding", True, True)
    c_qnov, b_qnov = avg_metric("layer3_dialog", "question_novelty", only_dialogs=True), avg_metric("layer3_dialog", "question_novelty", True, True)

    c_frec, b_frec = avg_metric("layer4_generation", "fact_recall"), avg_metric("layer4_generation", "fact_recall", True)
    c_util, b_util = avg_metric("layer4_generation", "context_utilization"), avg_metric("layer4_generation", "context_utilization", True)
    c_guard, b_guard = avg_metric("layer4_generation", "guardrails"), avg_metric("layer4_generation", "guardrails", True)
    c_bleu, b_bleu = avg_metric("layer4_generation", "bleu4"), avg_metric("layer4_generation", "bleu4", True)
    c_r1 = avg_metric("layer4_generation", "rouge1")
    c_r2 = avg_metric("layer4_generation", "rouge2")
    c_rl, b_rl = avg_metric("layer4_generation", "rougeL"), avg_metric("layer4_generation", "rougeL", True)
    c_met, b_met = avg_metric("layer4_generation", "meteor"), avg_metric("layer4_generation", "meteor", True)
    c_bert, b_bert = avg_metric("layer4_generation", "bertscore_f1"), avg_metric("layer4_generation", "bertscore_f1", True)
    c_sem, b_sem = avg_metric("layer4_generation", "bge_sem_sim"), avg_metric("layer4_generation", "bge_sem_sim", True)

    summary = (
        f"\n================ СВОДКА ПО АРХИТЕКТУРНЫМ СЛОЯМ (Cand vs Base) ================\n"
        f"Слой 1 (Retrieval Top-{CANDIDATES}) : FactRecall@{CANDIDATES} = {c_r16:.1%} (Base: {b_r16:.1%}, Δ={c_r16 - b_r16:+.1%})\n"
        f"Слой 2 (Reranker Top-{CONTEXT_CHUNKS})  : FactRecall@{CONTEXT_CHUNKS}  = {c_r6:.1%} (Base: {b_r6:.1%}, Δ={c_r6 - b_r6:+.1%}) | Retention = {c_ret:.1%} (Base: {b_ret:.1%})\n"
        f"Слой 3 (Диалоги)          : Фокус-запрос = {c_fhist:.1%} (Base: {b_fhist:.1%}) | Учёт истории в ответе = {c_dgr:.1%} (Base: {b_dgr:.1%}) | Новизна вопроса = {c_qnov:.1%} (Base: {b_qnov:.1%})\n"
        f"Слой 4 (Генерация RAG)    : FactRecall = {c_frec:.1%} (Base: {b_frec:.1%}, Δ={c_frec - b_frec:+.1%}) | Utilization = {c_util:.1%} (Base: {b_util:.1%}) | Guardrails = {c_guard:.1%} (Base: {b_guard:.1%})\n"
        f"Слой 4 (NLP-метрики)      : BLEU-4={c_bleu:.3f} (Base:{b_bleu:.3f}) | ROUGE-L={c_rl:.3f} (Base:{b_rl:.3f}) | METEOR={c_met:.3f} (Base:{b_met:.3f}) | {bert_label}={c_bert:.3f} (Base:{b_bert:.3f}) | SemSim={c_sem:.3f} (Base:{b_sem:.3f})\n"
        f"------------------------------------------------------------------------------\n"
        f"Candidate MatchRate       : {avg_match:.3f} (Baseline: {avg_base_match:.3f})\n"
        f"Candidate WinRate         : {avg_win:.3f} (>0.500 означает победу над бейзлайном)\n"
        f"ИТОГОВЫЙ SCORE            : {total_score:.3f} (0.7*Match + 0.3*Win)\n"
        f"=============================================================================="
    )
    print(summary)

    if cand_path.parent != ROOT:
        md = [
            f"# Послойный отчёт качества RAG: `{cand_path.parent.name}` против `{base_path.parent.name}`",
            "",
            "## 1. Сводные метрики по слоям (Candidate vs Baseline)",
            f"- **Итоговый Score судьи:** `{total_score:.3f}` (`MatchRate: {avg_match:.3f}` vs `{avg_base_match:.3f}`, `WinRate: {avg_win:.3f}`)",
            f"- **Слой 1 — Retrieval (`Top-{CANDIDATES}`):** `FactRecall@{CANDIDATES} = {c_r16:.1%}` (Baseline: `{b_r16:.1%}`, `Δ = {c_r16 - b_r16:+.1%}`)",
            f"- **Слой 2 — Reranker (`Top-{CONTEXT_CHUNKS}`):** `FactRecall@{CONTEXT_CHUNKS} = {c_r6:.1%}` (Baseline: `{b_r6:.1%}`, `Δ = {c_r6 - b_r6:+.1%}`), `Retention = {c_ret:.1%}` (Baseline: `{b_ret:.1%}`)",
            f"- **Слой 3 — Контекст диалога:** `Focused Query Retention = {c_fhist:.1%}` (Baseline: `{b_fhist:.1%}`), `Dialog Grounding в ответе = {c_dgr:.1%}` (Baseline: `{b_dgr:.1%}`), `Question Novelty = {c_qnov:.1%}` (Baseline: `{b_qnov:.1%}`)",
            f"- **Слой 4 — RAG Генерация:** `Answer FactRecall = {c_frec:.1%}` (Baseline: `{b_frec:.1%}`), `Context Utilization = {c_util:.1%}` (Baseline: `{b_util:.1%}`), `Rule Guardrails = {c_guard:.1%}` (Baseline: `{b_guard:.1%}`)",
            f"- **Слой 4 — Классические и нейросетевые метрики:** `BLEU-4 = {c_bleu:.3f}` (`{b_bleu:.3f}`), `ROUGE-1 = {c_r1:.3f}`, `ROUGE-2 = {c_r2:.3f}`, `ROUGE-L = {c_rl:.3f}` (`{b_rl:.3f}`), `METEOR = {c_met:.3f}` (`{b_met:.3f}`), `{bert_label} = {c_bert:.3f}` (`{b_bert:.3f}`), `BGE-SemSim = {c_sem:.3f}` (`{b_sem:.3f}`)",
            "",
            "## 2. Сводная таблица по всем вопросам",
            "",
            f"| ID | Score | Match (C/B) | Слой 1: Rec@16 | Слой 2: Rec@6 (Ret) | Слой 3: Ground / Nov | Слой 4: FactRec / Util | ROUGE-L / METEOR / {bert_label} | Комментарий судьи |",
            "|---:|---:|:---:|---:|---:|:---:|:---:|:---:|:---|",
        ]
        for r in results:
            ly = r["layers"]
            l1, l2, l3, l4 = ly["layer1_retrieval"], ly["layer2_reranker"], ly["layer3_dialog"], ly["layer4_generation"]
            d_str = f"{l3['dialog_grounding']:.0%} / {l3['question_novelty']:.0%}" if l3["is_dialog"] else "—"
            md.append(
                f"| {r['id']} | **{r['score']:.2f}** | {r['cand_match']:.2f}/{r['base_match']:.2f} | "
                f"{l1['recall_16']:.0%} (#{l1['best_rank_16']}) | {l2['recall_6']:.0%} ({l2['retention']:.0%}) | "
                f"{d_str} | {l4['fact_recall']:.0%} / {l4['context_utilization']:.0%} | "
                f"{l4['rougeL']:.2f} / {l4['meteor']:.2f} / {l4['bertscore_f1']:.2f} | {r['reason']} |"
            )

        md.append("\n## 3. Детальный послойный разбор каждого вопроса\n")
        for r in results:
            ly = r["layers"]
            l1, l2, l3, l4 = ly["layer1_retrieval"], ly["layer2_reranker"], ly["layer3_dialog"], ly["layer4_generation"]
            md.extend([
                f"### Вопрос #{r['id']} (Score: `{r['score']:.2f}` | Match: `{r['cand_match']:.2f}` vs `{r['base_match']:.2f}` | WinRate: `{r['win_rate']:.2f}`)",
                f"**Вопрос клиента:**\n```text\n{r['question']}\n```",
                f"**Эталонный ответ:**\n> {r['golden']}",
                f"**Текущий ответ модели:**\n> {r['cand_ans']}",
                "**Найденные чанки (Top-6):**",
                *[f"{idx}. `{ct}`" for idx, ct in enumerate(ly["chunks_titles"], 1)],
                "",
                "**Анализ по слоям:**",
                f"- **Слой 1 (Retrieval Top-{CANDIDATES}):** `FactRecall@{CANDIDATES} = {l1['recall_16']:.1%}`, ранг лучшего чанка = `#{l1['best_rank_16']}` (покрытие лучшего чанка = `{l1['best_chunk_cov_16']:.1%}`)",
                f"- **Слой 2 (Reranker Top-{CONTEXT_CHUNKS}):** `FactRecall@{CONTEXT_CHUNKS} = {l2['recall_6']:.1%}`, сохранение `Top-6/Top-16 = {l2['retention']:.1%}`, полезных чанков = `{l2['useful_chunks_6']}/{l2['total_chunks_6']}`, ранг лучшего = `#{l2['best_rank_6']}`, `reranked = {l2['reranked_flag']}`",
                (
                    f"- **Слой 3 (Контекст диалога, {l3['turns']} реплик):** фокус-запрос = `{l3['focused_history_retention']:.1%}`, учёт истории диалога в ответе = `{l3['dialog_grounding']:.1%}`, новизна встречного вопроса = `{l3['question_novelty']:.1%}`"
                    if l3["is_dialog"]
                    else "- **Слой 3 (Контекст диалога):** одиночный вопрос"
                ),
                f"- **Слой 4 (RAG Генерация):** `FactRecall ответа = {l4['fact_recall']:.1%}`, `Context Utilization (взято из чанков) = {l4['context_utilization']:.1%}`, `Rule Guardrails = {l4['guardrails']:.1%}` (слов: `{l4['word_count']}`, цифры Rec/Prec: `{l4['digits_recall']:.0%}/{l4['digits_precision']:.0%}`, без списков: `{l4['no_markdown']}`)",
                f"- **Слой 4 (NLP-метрики):** `BLEU-4 = {l4['bleu4']:.3f}`, `ROUGE-1 = {l4['rouge1']:.3f}`, `ROUGE-2 = {l4['rouge2']:.3f}`, `ROUGE-L = {l4['rougeL']:.3f}`, `METEOR = {l4['meteor']:.3f}`, `{bert_label} = {l4['bertscore_f1']:.3f}`, `BGE-SemSim = {l4['bge_sem_sim']:.3f}`",
                f"- **Комментарий LLM-судьи:** {r['reason']}",
                "",
                "---",
            ])

        out_md = cand_path.parent / "judge.md"
        out_md.write_text("\n".join(md), encoding="utf-8")
        print(f"Отчёт сохранён в: {out_md}")


if __name__ == "__main__":
    main()