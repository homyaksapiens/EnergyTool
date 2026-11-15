from __future__ import annotations

import re

import llm_client
from kb import Chunk, KnowledgeBase, tokenize
from prompts import DIALOG_SUMMARY_PROMPT, QUERY_EXPANSION_PROMPT, SYSTEM_PROMPT
from settings import FINAL_K, MAX_CONTEXT_CHARS, MAX_TOKENS, TEMPERATURE, TOP_K

FALLBACK_ANSWER = (
    "По этому вопросу нужна дополнительная проверка параметров вашего объекта. "
    "Подскажите, пожалуйста, основные характеристики объекта и текущую схему учёта, "
    "чтобы мы подобрали точное решение."
)

ROLE_RE = re.compile(
    r"(?:^|\n)\s*(Пользователь|Клиент|Ассистент|Консультант|User|Assistant)\s*:\s*",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_MAX_TOKENS = max(MAX_TOKENS, 2500)


def parse_dialog(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    matches = list(ROLE_RE.finditer(text))
    if not matches:
        return [{"role": "user", "content": text}]

    messages: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        role_label = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if not content:
            continue
        role = "user" if role_label in ("пользователь", "клиент", "user") else "assistant"
        messages.append({"role": role, "content": content})
    return messages or [{"role": "user", "content": text}]


def summarize_dialog(messages: list[dict[str, str]]) -> str | None:
    if len(messages) <= 1:
        return None
    lines = []
    for item in messages:
        role = "Клиент" if item["role"] == "user" else "Консультант"
        lines.append(f"{role}: {item['content']}")
    dialog = "\n".join(lines)[-1800:]
    text, _ = llm_client.chat(
        [
            {"role": "system", "content": DIALOG_SUMMARY_PROMPT},
            {"role": "user", "content": f"/no_think\nДиалог:\n{dialog}\n\nПоисковый запрос:"},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    cleaned = clean_answer(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" «»\"'")
    return cleaned if len(cleaned) >= 5 else None


def expand_single_query(question: str) -> str | None:
    text, _ = llm_client.chat(
        [
            {"role": "system", "content": QUERY_EXPANSION_PROMPT},
            {"role": "user", "content": f"/no_think\nВопрос клиента:\n{question}\n\nКлючевые термины:"},
        ],
        temperature=0.0,
        max_tokens=80,
    )
    cleaned = clean_answer(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" «»\"'")
    return cleaned if len(cleaned) >= 5 else None


def search_queries(messages: list[dict[str, str]]) -> list[str]:
    users = [m["content"] for m in messages if m["role"] == "user"]
    if not users:
        return [""]
    last = users[-1]
    if len(users) == 1:
        expanded = expand_single_query(last)
        if expanded and expanded.lower() != last.lower():
            return [last, f"{last} {expanded}"]
        return [last]
    queries = [last, " ".join(users[-2:]), " ".join(users)]
    summary = summarize_dialog(messages)
    if summary:
        queries.append(summary)
    unique = []
    for q in queries:
        if q and q not in unique:
            unique.append(q)
    return unique


def build_system(chunks: list[Chunk]) -> str:
    blocks = []
    total = 0
    for idx, chunk in enumerate(chunks, 1):
        block = f"[{idx}] {chunk.breadcrumb}\n{chunk.body}"
        if total + len(block) > MAX_CONTEXT_CHARS and blocks:
            break
        blocks.append(block)
        total += len(block)
    context = "\n\n".join(blocks)
    return f"{SYSTEM_PROMPT}\n\nСправочные материалы:\n{context}"


def format_dialog_sandwich(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(messages) <= 1:
        return messages

    history_lines = []
    for m in messages[:-1]:
        role_name = "Клиент" if m["role"] == "user" else "Консультант"
        history_lines.append(f"{role_name}: {m['content']}")
    history_block = "\n".join(history_lines)

    users = [m["content"] for m in messages if m["role"] == "user"]
    known_facts = " ".join(users[:-1]) if len(users) > 1 else ""
    last_question = messages[-1]["content"]

    content = (
        f"Контекст диалога:\n{history_block}\n\n"
        f"Что клиент уже сообщил о своей ситуации: {known_facts}\n\n"
        f"Текущий вопрос клиента: {last_question}\n\n"
        "Ответь на текущий вопрос клиента с учётом всего контекста диалога и справочных материалов: "
        "дай прямой ответ, укажи ключевые условия/ограничения и что нужно проверить или какие данные понадобятся дальше. "
        "В конце при необходимости задай один уточняющий вопрос, но НЕ спрашивай то, что клиент уже сообщил выше."
    )
    return [{"role": "user", "content": content}]


def clean_answer(text: str) -> str:
    cleaned = THINK_RE.sub("", text or "").strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    cleaned = re.sub(r"^(Ответ|Консультант|Ассистент)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    normalized_lines = []
    for line in lines:
        line = re.sub(r"^[\-\•\*]\s+", "", line)
        line = re.sub(r"^\d+[\.\)]\s+", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        normalized_lines.append(line)

    result = " ".join(normalized_lines)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


class Consultant:
    def __init__(self, kb: KnowledgeBase, use_rerank: bool = True) -> None:
        self.kb = kb
        self.use_rerank = use_rerank

    def _retrieve_detailed(
        self,
        messages: list[dict[str, str]],
        query_vectors: list[list[float]] | None = None,
        precomputed_queries: list[str] | None = None,
    ) -> tuple[list[Chunk], list[Chunk], list[str], bool]:
        queries = precomputed_queries if precomputed_queries is not None else search_queries(messages)
        candidates = self.kb.search(queries, query_vectors=query_vectors, limit=TOP_K)
        focus_query = queries[-1]
        if not self.use_rerank or len(candidates) <= FINAL_K:
            return candidates[:FINAL_K], candidates, queries, False

        rerank_query = queries[0] if len(messages) <= 1 else f"Контекст: {focus_query}\nВопрос: {queries[0]}"
        scores = llm_client.rerank(
            rerank_query,
            [f"{c.breadcrumb}\n{c.body}" for c in candidates],
        )
        if not scores or len(scores) != len(candidates):
            return candidates[:FINAL_K], candidates, queries, False

        min_r, max_r = min(scores), max(scores)
        span_r = max(max_r - min_r, 1e-6)
        rerank_norm = [(s - min_r) / span_r for s in scores]
        base_norm = [(len(candidates) - i) / len(candidates) for i in range(len(candidates))]

        focus_terms = [t for t in tokenize(focus_query) if len(t) >= 4]
        doc_sets = [c.tokens | c.title_tokens for c in candidates]
        specificity = []
        for dset in doc_sets:
            spec = 0.0
            for term in focus_terms:
                if term in dset:
                    df = sum(1 for s in doc_sets if term in s)
                    spec += 1.0 / max(1, df)
            specificity.append(spec)
        max_spec = max(specificity) if specificity and max(specificity) > 0 else 1.0

        ranked = []
        for idx, (chunk, r_norm, b_norm) in enumerate(zip(candidates, rerank_norm, base_norm)):
            spec_norm = specificity[idx] / max_spec
            is_direct = 1.0 if chunk.section_kind == "direct" else 0.0
            if len(messages) > 1:
                total = 0.60 * r_norm + 0.24 * b_norm + 0.12 * spec_norm + 0.04 * is_direct
            else:
                total = 0.72 * r_norm + 0.20 * b_norm + 0.05 * spec_norm + 0.03 * is_direct
            ranked.append((total, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        final_chunks = [chunk for _, chunk in ranked[:FINAL_K]]
        return final_chunks, candidates, queries, True

    def retrieve(
        self,
        messages: list[dict[str, str]],
        query_vectors: list[list[float]] | None = None,
        precomputed_queries: list[str] | None = None,
    ) -> tuple[list[Chunk], list[str], bool]:
        final_chunks, _, queries, reranked = self._retrieve_detailed(
            messages, query_vectors=query_vectors, precomputed_queries=precomputed_queries
        )
        return final_chunks, queries, reranked

    def answer(
        self,
        raw_question: str,
        query_vectors: list[list[float]] | None = None,
        precomputed_queries: list[str] | None = None,
    ) -> tuple[str, dict]:
        messages = parse_dialog(raw_question)
        if not messages:
            return FALLBACK_ANSWER, {"fallback": True, "sources": []}

        chunks, queries, reranked = self.retrieve(
            messages, query_vectors=query_vectors, precomputed_queries=precomputed_queries
        )
        convo = [{"role": "system", "content": build_system(chunks)}]
        convo.extend(format_dialog_sandwich(messages))

        reply, err = llm_client.chat(
            convo,
            temperature=TEMPERATURE,
            max_tokens=THINK_MAX_TOKENS,
        )
        cleaned = clean_answer(reply or "")
        if not cleaned:
            return FALLBACK_ANSWER, {
                "fallback": True,
                "error": err,
                "sources": [c.id for c in chunks],
                "reranked": reranked,
            }
        return cleaned, {
            "fallback": False,
            "error": err,
            "sources": [c.id for c in chunks],
            "queries": queries,
            "reranked": reranked,
        }