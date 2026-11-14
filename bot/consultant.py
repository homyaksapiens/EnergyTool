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

# Настройки быстрой суммаризации контекста для многоходовых диалогов
USE_DIALOG_SUMMARY = os.getenv("ET_USE_SUMMARY", "1").strip() != "0"
SUMMARY_MAX_TOKENS = 256
SUMMARY_TIMEOUT = 15.0
SUMMARY_SYSTEM_PROMPT = (
    "Ты — модуль конденсации контекста диалога для поиска и ответа по базе знаний. Сожми весь диалог в 1–2 "
    "информативных предложения (до 45 слов), сохранив: исходную задачу и тип объекта, ВСЕ названные клиентом "
    "параметры, количественные показатели, сроки, ценовые категории, способы сбора данных, высказанные ограничения, "
    "предложенные в диалоге варианты (например, предварительную оценку или тип решения) и текущий вопрос клиента. "
    "НЕ используй фразы «начать разговор» или «первичное понимание». Сохрани все цифры и ключевые термины дословно. "
    "Не отвечай на вопрос. Пиши сразу готовую выжимку без рассуждений. /no_think"
)

_SUMMARY_CACHE: dict[tuple[tuple[str, str], ...], str] = {}
_SUMMARY_LOCK = threading.Lock()
_SUMMARY_FAILURES = 0
_MAX_SUMMARY_FAILURES = 3

# Увеличенный лимит токенов для основного ответа: позволяет модели глубоко рассуждать в <think> без обрыва
ANSWER_MAX_TOKENS = max(MAX_TOKENS, 2048)

# Секции, которые в многоходовом диалоге откладываются в резерв, чтобы не вытеснять продуктовые чанки
DIALOG_DEFER_LEAVES = {
    "с чего начинается разговор",
    "с чего начать",
    "для первичного понимания задачи",
}
SINGLE_CAP_LEAVES = {
    "сколько это стоит",
    "сколько стоит расчёт",
    "следующий шаг",
    "сроки реализации",
}


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
    """Детерминированная сборка контекста: все реплики клиента + последний вопрос/предложение ассистента."""
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
    """Быстрый одиночный запрос к модели с коротким таймаутом для конденсации контекста диалога."""
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
    """Для одиночных вопросов возвращает сам вопрос; для многоходовых диалогов вызывает
    быструю конденсацию контекста (Qwen с /no_think) с кэшированием и детерминированным фоллбэком.
    """
    global _SUMMARY_FAILURES
    if len(messages) <= 1:
        return messages[0]["content"].strip() if messages else ""

    fallback = _deterministic_summary(messages)
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
        f"{'Клиент' if m['role'] == 'user' else 'Консультант'}: {m['content'].strip()}"
        for m in messages
        if m.get("content", "").strip()
    ]
    user_prompt = (
        "История диалога:\n" + "\n".join(history_lines) + "\n\n"
        "Сожми этот диалог в 1–2 предложения со всеми фактами, цифрами, условиями клиента, предложенными в диалоге "
        "вариантами и финальным вопросом. /no_think"
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
      2) last_exchange (последняя реплика консультанта + ответ клиента)
      3) user_only (все реплики самого клиента)
      4) rerank_target (queries[-1]) — сфокусированный запрос для реранкера (исходная тема + выжимка + финальный вопрос).
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
    target_parts: list[str] = []
    if global_problem and global_problem.lower() not in summary.lower():
        target_parts.append(global_problem)
    target_parts.append(summary)
    if last and last.lower() not in summary.lower():
        target_parts.append(last)
    rerank_target = " ".join(target_parts).strip()

    queries: list[str] = []
    for q in (global_problem, last_exchange, user_only, rerank_target):
        if q and q not in queries:
            queries.append(q)

    if rerank_target and queries[-1] != rerank_target:
        queries.remove(rerank_target)
        queries.append(rerank_target)

    return queries or [last]


def build_dialog_message(messages: list[dict], summary: str) -> str:
    """Формирует запрос для многоходового диалога по трёхчастной схеме:
    1) {контекст предыдущих сообщений} — сжатая выжимка диалога + предыдущие реплики для сохранения всех деталей;
    2) {текущее сообщение} — последняя реплика клиента, на которую нужно ответить;
    3) {напоминание} — установка следовать правилам системного промпта и дать полноценный развёрнутый ответ.
    """
    if len(messages) <= 1:
        return messages[0]["content"].strip() if messages else ""

    last_user = messages[-1]["content"].strip()
    prev_lines = [
        f"— {'Клиент' if m['role'] == 'user' else 'Консультант'}: {m['content'].strip()}"
        for m in messages[:-1]
        if m.get("content", "").strip()
    ]
    prev_block = "\n".join(prev_lines)

    return (
        f"Контекст предыдущих сообщений диалога:\n"
        f"{summary}\n"
        f"(Краткая справка по предыдущим репликам:\n{prev_block})\n\n"
        f"Текущее сообщение клиента:\n"
        f"{last_user}\n\n"
        f"Напоминание: ответь на текущее сообщение клиента строго по правилам системного промпта "
        f"(связный текст из 2–4 предложений, 35–70 слов). Используй факты, пороги и условия из материалов базы знаний, "
        f"обязательно учти все вводные клиента из контекста диалога и не переспрашивай то, что он уже сообщил."
    )


def _diversify_reranked(candidates: list[Chunk], order: list[int], k: int, is_dialog: bool = False) -> list[Chunk]:
    """Защищает Top-k от:
    1) забивания одноимёнными секциями («Сколько это стоит»);
    2) вытеснения продуктовых чанков общими стартовыми секциями («Для первичного понимания задачи») в диалогах;
    3) перекоса в сторону только одного файла продукта, когда в диалоге затронуты две темы/продукта.
    """
    selected: list[Chunk] = []
    deferred: list[Chunk] = []
    leaf_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    max_per_source = 3 if is_dialog else k

    for idx in order:
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            leaf = c.title.rsplit("→", 1)[-1].strip().lower()
            if is_dialog and leaf in DIALOG_DEFER_LEAVES:
                deferred.append(c)
                continue
            max_same_leaf = 1 if leaf in SINGLE_CAP_LEAVES else 2
            if leaf_counts.get(leaf, 0) >= max_same_leaf or source_counts.get(c.source, 0) >= max_per_source:
                deferred.append(c)
                continue
            selected.append(c)
            leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
            source_counts[c.source] = source_counts.get(c.source, 0) + 1
            if len(selected) >= k:
                return selected

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
            user_content = build_dialog_message(messages, summary)
        else:
            user_content = messages[0]["content"].strip() if messages else ""

        convo = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
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