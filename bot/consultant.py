from __future__ import annotations

import re
import sys
import threading
import time

import llm_client
from kb import Chunk, KnowledgeBase
from prompts import FALLBACK_ANSWER, build_system
from settings import CANDIDATES, CONTEXT_CHUNKS, MAX_TOKENS, TEMPERATURE, USE_RERANK

ROLE_RE = re.compile(r"^\s*(Пользователь|Клиент|User|Ассистент|Консультант|Бот|Assistant)\s*:\s*(.*)$", re.I)
ASSISTANT_ROLES = {"ассистент", "консультант", "бот", "assistant"}
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

THINK_MAX_TOKENS = max(MAX_TOKENS, 1280)

SUMMARY_MAX_TOKENS = 192
SUMMARY_SYSTEM_PROMPT = (
    "Перепиши последнюю реплику диалога в один самодостаточный поисковый запрос для базы знаний (до 25 слов). "
    "Правила:\n"
    "1. Сохрани все ключевые термины и смысл последнего вопроса клиента.\n"
    "2. Замени местоимения и неполные ссылки («это», «таких данных», «четвёртая», «для оценки», «с чего начать») "
    "на конкретные факты из предыдущих реплик: тип объекта, обсуждаемое решение, названные параметры и цифры.\n"
    "3. Добавляй из истории только то, без чего последний вопрос непонятен. Не притягивай второстепенные темы "
    "или вопрос про стоимость из начала диалога, если в конце клиент спрашивает про другое.\n"
    "4. ЗАПРЕЩЕНО использовать водные и мета-слова: «клиент», «пользователь», «консультант», «ассистент», "
    "«спрашивает», «интересуется», «хочет узнать», «просит объяснить», «в диалоге».\n"
    "5. Не отвечай на вопрос и ничего не додумывай. Выведи только одну строку — готовый запрос без кавычек."
)

_SUMMARY_CACHE: dict[tuple[tuple[str, str], ...], str] = {}
_SUMMARY_LOCK = threading.Lock()


def _unpack_chat(res: tuple[str, str | None] | str) -> tuple[str, str | None]:
    if isinstance(res, tuple):
        return (res[0] or ""), (res[1] if len(res) > 1 else None)
    return (res or ""), None


def parse_dialog(query: str) -> list[dict[str, str]]:
    text = (query or "").strip()
    if not text:
        return [{"role": "user", "content": ""}]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    turns: list[dict[str, str]] = []
    for line in lines:
        match = ROLE_RE.match(line)
        if match:
            role_label, content = match.group(1).lower(), match.group(2).strip()
            role = "assistant" if role_label in ASSISTANT_ROLES else "user"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n" + content
            else:
                turns.append({"role": role, "content": content})
        elif turns:
            turns[-1]["content"] += "\n" + line
        else:
            break

    if not turns:
        return [{"role": "user", "content": text}]
    return turns


def clean_answer(text: str | tuple) -> str:
    if isinstance(text, tuple):
        text = text[0] if text else ""
    cleaned = THINK_RE.sub("", text or "")
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    cleaned = re.sub(r"^\s*(Ответ|Консультант|Ассистент)\s*:\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def summarize_dialog(messages: list[dict[str, str]]) -> str:
    user_turns = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    if len(user_turns) <= 1 and len(messages) <= 1:
        return user_turns[-1] if user_turns else ""

    cache_key = tuple((m.get("role", ""), (m.get("content") or "").strip()) for m in messages)
    with _SUMMARY_LOCK:
        cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dialog_lines = []
    for m in messages:
        role_name = "Клиент" if m["role"] == "user" else "Консультант"
        content = m["content"].strip()
        if content:
            dialog_lines.append(f"{role_name}: {content}")
    dialog_text = "\n".join(dialog_lines)

    try:
        raw_res = llm_client.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Диалог:\n{dialog_text}\n\nПоисковый запрос: /no_think"},
            ],
            temperature=0.0,
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        raw, _ = _unpack_chat(raw_res)
        summary = clean_answer(raw)
        summary = re.sub(r"\s+", " ", summary).strip()
        if summary:
            with _SUMMARY_LOCK:
                _SUMMARY_CACHE[cache_key] = summary
            return summary
    except Exception as exc:
        print(f"[consultant] dialog summary error: {exc}", file=sys.stderr)

    fallback = " ".join(user_turns[-2:]) if len(user_turns) >= 2 else (user_turns[-1] if user_turns else "")
    with _SUMMARY_LOCK:
        _SUMMARY_CACHE[cache_key] = fallback
    return fallback


def search_queries(messages: list[dict[str, str]]) -> list[str]:
    user_turns = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    if not user_turns:
        return [""]

    last = user_turns[-1]
    if len(user_turns) == 1:
        return [last]

    global_problem = f"{user_turns[0]} {last}"

    last_exchange_parts = [m["content"].strip() for m in messages[-2:] if m["content"].strip()]
    last_exchange = " ".join(last_exchange_parts)

    full = " ".join(user_turns)
    summary = summarize_dialog(messages)

    result: list[str] = []
    for q in (global_problem, last_exchange, full, summary):
        if q and q not in result:
            result.append(q)
    return result or [last]


def format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for idx, chunk in enumerate(chunks, 1):
        blocks.append(f"[Источник {idx}: {chunk.doc_title} — {chunk.section}]\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def format_dialog_sandwich(messages: list[dict[str, str]]) -> str:
    user_turns = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    if len(messages) <= 1:
        return f"Вопрос клиента:\n{user_turns[-1] if user_turns else ''}"

    global_goal = user_turns[0] if user_turns else ""
    known_facts = "; ".join(user_turns[:-1]) if len(user_turns) > 1 else ""
    last_reply = user_turns[-1] if user_turns else ""

    history_lines = []
    for m in messages:
        role_name = "Клиент" if m["role"] == "user" else "Консультант"
        history_lines.append(f"{role_name}: {m['content'].strip()}")
    history_block = "\n".join(history_lines)

    return (
        f"Контекст диалога:\n"
        f"- Исходная задача клиента: {global_goal}\n"
        f"- Уже известно о клиенте (НЕ переспрашивай эти данные): {known_facts}\n\n"
        f"История диалога:\n{history_block}\n\n"
        f"Актуальная реплика клиента (ответь именно на неё): {last_reply}\n\n"
        f"Напоминание: дай прямой ответ по материалам базы знаний (2–4 предложения, связным текстом без списков), "
        f"учти уже известные факты из истории и при необходимости задай только один новый уточняющий вопрос, "
        f"который ещё не задавался в диалоге."
    )


class Consultant:
    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self.kb = kb or KnowledgeBase()
        self.system_prompt = build_system()

    def _retrieve_detailed(
        self,
        messages: list[dict[str, str]],
        query_vectors: list[list[float] | None] | None = None,
    ) -> tuple[list[Chunk], dict]:
        queries = search_queries(messages)
        candidates = self.kb.search_multi(queries, query_vectors=query_vectors, limit=CANDIDATES)
        reranked = False
        if USE_RERANK and len(candidates) > CONTEXT_CHUNKS:
            order = llm_client.rerank(
                queries[-1],
                [c.text for c in candidates],
                top_n=len(candidates),
            )
            if order:
                reranked = True
                seen: set[int] = set()
                ordered: list[Chunk] = []
                for idx in order:
                    if 0 <= idx < len(candidates) and idx not in seen:
                        ordered.append(candidates[idx])
                        seen.add(idx)
                for idx, chunk in enumerate(candidates):
                    if idx not in seen:
                        ordered.append(chunk)
                candidates = ordered

        selected = candidates[:CONTEXT_CHUNKS]
        meta = {
            "queries": queries,
            "candidate_ids": [c.id for c in candidates],
            "reranked": reranked,
        }
        return selected, meta

    def retrieve(
        self,
        messages: list[dict[str, str]],
        query_vectors: list[list[float] | None] | None = None,
    ) -> list[Chunk]:
        selected, _ = self._retrieve_detailed(messages, query_vectors=query_vectors)
        return selected

    def answer_messages(
        self,
        messages: list[dict[str, str]],
        query_vectors: list[list[float] | None] | None = None,
    ) -> tuple[str, list[Chunk], float]:
        answer, chunks, elapsed, _ = self.answer_detailed(messages, query_vectors=query_vectors)
        return answer, chunks, elapsed

    def answer_detailed(
        self,
        messages: list[dict[str, str]],
        query_vectors: list[list[float] | None] | None = None,
    ) -> tuple[str, list[Chunk], float, dict]:
        t0 = time.perf_counter()
        chunks, retrieval_meta = self._retrieve_detailed(messages, query_vectors=query_vectors)
        context_block = format_context(chunks)
        dialog_block = format_dialog_sandwich(messages)

        convo: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{self.system_prompt}\n\n"
                    f"Материалы базы знаний для ответа:\n\n{context_block}"
                ),
            },
            {
                "role": "user",
                "content": dialog_block,
            },
        ]

        answer = ""
        for attempt in range(2):
            try:
                msgs = convo
                tokens_limit = THINK_MAX_TOKENS
                if attempt == 1 and msgs and msgs[-1]["role"] == "user":
                    msgs = msgs[:-1] + [{**msgs[-1], "content": msgs[-1]["content"] + " /no_think"}]
                    tokens_limit = MAX_TOKENS
                raw_res = llm_client.chat(msgs, temperature=TEMPERATURE, max_tokens=tokens_limit)
                raw, _ = _unpack_chat(raw_res)
                answer = clean_answer(raw)
                if answer:
                    break
            except Exception as exc:
                print(f"[consultant] generation error (attempt {attempt + 1}): {exc}", file=sys.stderr)

        if not answer:
            answer = FALLBACK_ANSWER

        elapsed = time.perf_counter() - t0
        return answer, chunks, elapsed, retrieval_meta

    def answer(
        self,
        query: str,
        query_vectors: list[list[float] | None] | None = None,
    ) -> dict[str, object]:
        messages = parse_dialog(query)
        answer, chunks, elapsed, retrieval_meta = self.answer_detailed(messages, query_vectors=query_vectors)
        return {
            "answer": answer,
            "elapsed_sec": round(elapsed, 2),
            "queries": retrieval_meta["queries"],
            "candidate_ids": retrieval_meta["candidate_ids"],
            "reranked": retrieval_meta["reranked"],
            "sources": [
                {
                    "id": c.id,
                    "path": c.source_path,
                    "title": c.doc_title,
                    "section": c.section,
                    "score": round(c.score, 4),
                    "text": c.text,
                }
                for c in chunks
            ],
        }