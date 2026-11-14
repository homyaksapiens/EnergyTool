"""Ядро консультанта: разбор диалога → быстрая суммаризация контекста → поиск по базе → вызов LLM → чистка ответа.

Используется и в solution.py (автосабмит), и в веб-бэкенде. Tool calling не используем: прокси
платформы его не гарантирует, а лишние вызовы LLM съедают лимит времени.

Берет вопрос клиента, ищет подходящие куски текста через kb.py, пропускает их через реранкер (если он доступен),
подставляет в шаблон промпта и отправляет в языковую модель.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time

import httpx
import llm_client
from kb import Chunk, KnowledgeBase
from prompts import FALLBACK_ANSWER, build_system
from settings import API_KEY, BASE_URL, CANDIDATES, CONTEXT_CHUNKS, LLM_MODEL, MAX_TOKENS, TEMPERATURE, USE_RERANK

ROLE_RE = re.compile(r"^\s*(Пользователь|Клиент|User|Ассистент|Консультант|Бот|Assistant)\s*:\s*(.*)$", re.I)
ASSISTANT_ROLES = {"ассистент", "консультант", "бот", "assistant"}
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
CITE_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Настройки второй (лёгкой) суммаризирующей модели для сжатия контекста сложных диалогов
USE_DIALOG_SUMMARY = os.getenv("ET_USE_SUMMARY", "1").strip() != "0"
SUMMARY_MAX_TOKENS = 160
SUMMARY_TIMEOUT = 15.0
SUMMARY_SYSTEM_PROMPT = (
    "Ты — модуль формирования поискового запроса по истории диалога. Перепиши диалог в один самодостаточный "
    "запрос от первого лица (1–2 предложения, до 35 слов), объединив исходную тему, ВСЕ названные в диалоге "
    "параметры объекта, количественные показатели, сроки, категории, способы работы, высказанные ограничения "
    "и последний вопрос. НЕ используй слова «Клиент», «Консультант», «Пользователь», «начать разговор». "
    "Сохрани все цифры и термины дословно. Не отвечай на вопрос. Пиши сразу готовый запрос без рассуждений. /no_think"
)

_SUMMARY_CACHE: dict[tuple[tuple[str, str], ...], str] = {}
_SUMMARY_LOCK = threading.Lock()
_SUMMARY_FAILURES = 0
_MAX_SUMMARY_FAILURES = 3

# Увеличенный лимит токенов для основного ответа: позволяет модели глубоко рассуждать в <think> без обрыва
ANSWER_MAX_TOKENS = max(MAX_TOKENS, 2048)


def parse_dialog(text: str) -> list[dict]:
    """Строка вопроса → [{"role", "content"}]. Диалог в тестах — реплики «Пользователь: …» / «Ассистент: …»."""
    messages: list[dict] = []
    for line in text.strip().split("\n"):
        m = ROLE_RE.match(line)
        if m:
            role = "assistant" if m.group(1).lower() in ASSISTANT_ROLES else "user"
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n" + m.group(2).strip()
            else:
                messages.append({"role": role, "content": m.group(2).strip()})
        elif messages:
            messages[-1]["content"] += "\n" + line.strip()
        else:
            messages.append({"role": "user", "content": line.strip()})
    return [m for m in messages if m["content"].strip()] or [{"role": "user", "content": text.strip()}]


def clean_answer(text: str) -> str:
    text = text or ""
    if "</think>" in text:  # рассуждение могло начаться ещё в шаблоне промпта — берём то, что после него
        text = text.rsplit("</think>", 1)[1]
    text = THINK_RE.sub("", text)
    if "<think>" in text:  # рассуждение оборвалось по лимиту токенов — ответа нет
        text = text.split("<think>", 1)[0]
    text = MD_LINK_RE.sub(r"\1", text)
    text = CITE_RE.sub("", text)
    text = re.sub(r"\*\*|__|`|^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)
    text = " ".join(text.split())
    return re.sub(
        r"^(?:Консультант|Ассистент|Выжимка|Резюме|Запрос|Контекст|Ответ(?: консультанта)?)\s*:\s*",
        "",
        text,
        flags=re.I,
    )


def _dialog_key(messages: list[dict]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (m.get("role", "user"), (m.get("content") or "").strip())
        for m in messages
        if (m.get("content") or "").strip()
    )


def _deterministic_summary(messages: list[dict]) -> str:
    """Детерминированная сборка контекста: все реплики клиента + последний вопрос ассистента перед ответом."""
    users = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    if not users:
        return " ".join(m["content"].strip() for m in messages if m["content"].strip())
    last = users[-1]
    prev_assistant = ""
    for m in reversed(messages[:-1]):
        if m["role"] == "assistant" and m["content"].strip():
            prev_assistant = m["content"].strip()
            break
    last_exchange = f"{prev_assistant} {last}".strip() if prev_assistant else last
    if len(users) > 1:
        return f"{' '.join(users[:-1])} {last_exchange}".strip()
    return last_exchange


def _fast_summary_call(messages: list[dict]) -> str:
    """Быстрый одиночный запрос к модели с коротким таймаутом и min токенов для суммаризации диалога."""
    if not BASE_URL:
        return ""
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": SUMMARY_MAX_TOKENS,
    }
    response = llm_client._client.post(
        BASE_URL + "/chat/completions",
        headers=headers,
        json=body,
        timeout=httpx.Timeout(SUMMARY_TIMEOUT, connect=5.0),
    )
    if response.status_code >= 400:
        raise llm_client.ApiError(f"HTTP {response.status_code}", response.status_code)
    data = response.json()
    choice = data["choices"][0]
    return (choice.get("message") or {}).get("content") or ""


def summarize_dialog(messages: list[dict]) -> str:
    """Для простых одиночных вопросов возвращает сам вопрос; для многоходовых диалогов вызывает
    быструю конденсацию запроса (Qwen с /no_think и малым лимитом токенов) с кэшированием и фоллбэком.
    """
    global _SUMMARY_FAILURES
    if len(messages) <= 1:
        return messages[0]["content"].strip() if messages else ""

    fallback = _deterministic_summary(messages)
    # Для коротких диалогов без промежуточных реплик достаточно детерминированной склейки
    if len(messages) < 3:
        return fallback

    key = _dialog_key(messages)
    with _SUMMARY_LOCK:
        if key in _SUMMARY_CACHE:
            return _SUMMARY_CACHE[key]
        can_call_llm = USE_DIALOG_SUMMARY and (_SUMMARY_FAILURES < _MAX_SUMMARY_FAILURES)

    if not can_call_llm:
        return fallback

    history_lines = [
        f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content'].strip()}"
        for m in messages
        if m.get("content", "").strip()
    ]
    user_prompt = (
        "Диалог:\n" + "\n".join(history_lines) + "\n\n"
        "Перепиши этот диалог в один самодостаточный вопрос от первого лица со всеми фактами, цифрами и деталями "
        "из всех реплик. /no_think"
    )
    summary_msgs = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    summary = ""
    try:
        raw = _fast_summary_call(summary_msgs)
        summary = clean_answer(raw)
    except Exception as e:  # noqa: BLE001 — при любой ошибке сразу возвращаем детерминированную сборку
        print(f"Суммаризатор диалога: {type(e).__name__}: {e}; используем резервную сборку", file=sys.stderr)

    if not summary or len(summary) < 15:
        with _SUMMARY_LOCK:
            _SUMMARY_FAILURES += 1
        summary = fallback
    else:
        with _SUMMARY_LOCK:
            _SUMMARY_FAILURES = 0

    with _SUMMARY_LOCK:
        _SUMMARY_CACHE[key] = summary
    return summary


def search_queries(messages: list[dict]) -> list[str]:
    """Формирует поисковые запросы без размытия весов токенов:
    - Одиночный вопрос: [last]
    - Диалог:
      1) global_problem (1-я реплика клиента с исходной темой)
      2) last_exchange (последний вопрос бота + ответ клиента)
      3) user_only (все реплики самого клиента без реплик бота)
      4) rerank_target (queries[-1]) — сфокусированный запрос для реранкера (исходная тема + выжимка + последний вопрос).
    """
    if not messages:
        return [""]

    users = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    all_msgs = [m["content"].strip() for m in messages if m["content"].strip()]
    full = " ".join(all_msgs)

    if not users:
        return [all_msgs[-1]] if all_msgs else [""]

    last = users[-1]
    if full == last or len(messages) <= 1:
        return [last]

    global_problem = users[0]
    user_only = " ".join(users)

    prev_assistant = ""
    for m in reversed(messages[:-1]):
        if m["role"] == "assistant" and m["content"].strip():
            prev_assistant = m["content"].strip()
            break
    last_exchange = f"{prev_assistant} {last}".strip() if prev_assistant else last

    summary = summarize_dialog(messages)
    rerank_target = f"{summary} {last}".strip() if last.lower() not in summary.lower() else summary

    queries: list[str] = []
    for q in (global_problem, last_exchange, user_only, rerank_target):
        if q and q not in queries:
            queries.append(q)

    if rerank_target and queries[-1] != rerank_target:
        queries.remove(rerank_target)
        queries.append(rerank_target)

    return queries or [last]


def _diversify_reranked(candidates: list[Chunk], order: list[int], k: int, is_dialog: bool = False) -> list[Chunk]:
    """Защищает Top-k от забивания одноимёнными секциями («Сколько это стоит»)
    и общими вводными секциями («С чего начинается разговор») в многоходовых диалогах.
    """
    selected: list[Chunk] = []
    deferred: list[Chunk] = []
    leaf_counts: dict[str, int] = {}

    for idx in order:
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            leaf = c.title.rsplit("→", 1)[-1].strip().lower()
            # В многоходовом диалоге разговор уже идёт, поэтому общие секции «с чего начинается разговор» откладываем
            if is_dialog and leaf in {"с чего начинается разговор", "с чего начать"}:
                deferred.append(c)
                continue
            max_same_leaf = 1 if leaf in {"сколько это стоит", "следующий шаг", "сроки реализации"} else 2
            if leaf_counts.get(leaf, 0) < max_same_leaf:
                selected.append(c)
                leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
                if len(selected) >= k:
                    return selected
            else:
                deferred.append(c)

    for c in deferred:
        if len(selected) >= k:
            break
        selected.append(c)
    return selected[:k]


class Consultant:
    MAX_RERANK_FAILURES = 3

    def __init__(self, kb: KnowledgeBase, use_rerank: bool = USE_RERANK):
        self.kb = kb
        self.rerank_ok = use_rerank
        self.rerank_failures = 0

    def _retrieve_detailed(
        self, messages: list[dict], query_vectors=None
    ) -> tuple[list[Chunk], list[Chunk], list[str], bool]:
        """Возвращает (top6_chunks, top16_candidates, queries, reranked) для совместимости с eval/judge.py."""
        queries = search_queries(messages)
        if query_vectors is None and self.kb.vectors is not None:
            try:
                query_vectors = llm_client.embed([q[:2000] for q in queries])
            except llm_client.ApiError:
                query_vectors = None
        candidates = self.kb.search(queries, query_vectors, k=CANDIDATES)
        if self.rerank_ok and len(candidates) > CONTEXT_CHUNKS:
            try:
                order = llm_client.rerank(queries[-1][:2000], [c.render()[:2000] for c in candidates], CANDIDATES)
                if order:
                    self.rerank_failures = 0
                    top_chunks = _diversify_reranked(
                        candidates, order, CONTEXT_CHUNKS, is_dialog=(len(messages) > 1)
                    )
                    return top_chunks, candidates, queries, True
            except llm_client.ApiError as e:
                self._rerank_failed(f"Реранкер недоступен: {e}", permanent=e.status in (400, 404, 405, 422, 501))
            except (KeyError, TypeError, ValueError) as e:
                self._rerank_failed(f"Реранкер вернул неожиданный ответ: {e!r}")
        return candidates[:CONTEXT_CHUNKS], candidates, queries, False

    def retrieve(self, messages: list[dict], query_vectors=None) -> tuple[list[Chunk], bool]:
        chunks, _, _, reranked = self._retrieve_detailed(messages, query_vectors)
        return chunks, reranked

    def _rerank_failed(self, message: str, permanent: bool = False) -> None:
        self.rerank_failures += 1
        if permanent or self.rerank_failures >= self.MAX_RERANK_FAILURES:
            if self.rerank_ok:
                print(f"{message} — отключаю реранкер до конца прогона", file=sys.stderr)
            self.rerank_ok = False
        else:
            print(message, file=sys.stderr)

    def answer(self, messages: list[dict], query_vectors=None, deadline: float | None = None) -> dict:
        """Ответ на последнюю реплику клиента: {text, sources, candidate_ids, queries, timings, reranked, fallback}."""
        t0 = time.time()
        chunks, candidates, queries, reranked = self._retrieve_detailed(messages, query_vectors)
        t1 = time.time()

        system_content = build_system(chunks)
        if len(messages) > 1:
            summary = summarize_dialog(messages)
            system_content += (
                f"\n\nСводка контекста текущего диалога (учти все эти детали клиента в ответе и не переспрашивай их):\n"
                f"{summary}"
            )

        # Передаём нативный список реплик messages без искажения ролей
        convo = [{"role": "system", "content": system_content}] + messages
        text = ""
        for attempt in range(2):
            if deadline and time.time() > deadline:
                break
            msgs = convo
            if attempt == 1:  # модель могла уйти в рассуждения и не успеть ответить — просим без них
                msgs = convo[:-1] + [{**convo[-1], "content": convo[-1]["content"] + " /no_think"}]
            try:
                raw, _ = llm_client.chat(msgs, TEMPERATURE, ANSWER_MAX_TOKENS)
            except (llm_client.ApiError, KeyError, IndexError, TypeError, ValueError) as e:
                print(f"LLM: {e!r}", file=sys.stderr)
                break
            text = clean_answer(raw)
            if text:
                break
        return {
            "text": text or FALLBACK_ANSWER,
            "fallback": not text,
            "reranked": reranked,
            "sources": [{"id": c.id, "source": c.source, "title": c.title} for c in chunks],
            "candidate_ids": [c.id for c in candidates],
            "queries": queries,
            "timings": {"retrieval": round(t1 - t0, 2), "llm": round(time.time() - t1, 2)},
        }
