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
    "Ты — модуль сжатия контекста диалога. Сделай краткую фактическую выжимку диалога клиента с консультантом "
    "в 1–2 предложениях (до 35 слов): кто клиент, какой у него объект, какая исходная задача и какие конкретные "
    "параметры, цифры, сроки или условия он уже сообщил (объединяя вопросы консультанта с короткими ответами "
    "клиента). Сохрани все числа и технические термины дословно. Не отвечай на вопрос клиента и ничего не "
    "додумывай от себя. Пиши сразу выжимку без рассуждений. /no_think"
)

_SUMMARY_CACHE: dict[tuple[tuple[str, str], ...], str] = {}
_SUMMARY_LOCK = threading.Lock()
_SUMMARY_FAILURES = 0
_MAX_SUMMARY_FAILURES = 3

# Безопасный лимит токенов для основного ответа (защита от обрыва <think> при MAX_TOKENS=1024 в settings.py)
ANSWER_MAX_TOKENS = max(MAX_TOKENS, 1536)


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
        r"^(?:Консультант|Ассистент|Выжимка|Резюме|Контекст|Ответ(?: консультанта)?)\s*:\s*",
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
    быструю суммаризацию (Qwen с /no_think и малым лимитом токенов) с кэшированием и фоллбэком.
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
        f"{'Клиент' if m['role'] == 'user' else 'Консультант'}: {m['content'].strip()}"
        for m in messages
        if m.get("content", "").strip()
    ]
    user_prompt = (
        "Диалог:\n" + "\n".join(history_lines) + "\n\n"
        "Сделай краткую выжимку всех фактов о клиенте, его объекте и текущем вопросе в 1–2 предложениях. /no_think"
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
      3) full (весь диалог)
      4) rerank_target (queries[-1]) — сжатая выжимка диалога + последний вопрос клиента
         (используется и в поиске, и как сфокусированный запрос queries[-1] для реранкера).
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

    prev_assistant = ""
    for m in reversed(messages[:-1]):
        if m["role"] == "assistant" and m["content"].strip():
            prev_assistant = m["content"].strip()
            break
    last_exchange = f"{prev_assistant} {last}".strip() if prev_assistant else last

    summary = summarize_dialog(messages)
    rerank_target = f"{summary} {last}".strip() if last not in summary else summary

    queries: list[str] = []
    for q in (global_problem, last_exchange, full, rerank_target):
        if q and q not in queries:
            queries.append(q)

    if rerank_target and queries[-1] != rerank_target:
        queries.remove(rerank_target)
        queries.append(rerank_target)

    return queries or [last]


def format_dialog_sandwich(messages: list[dict]) -> list[dict]:
    """Техника «сэндвича» для многоходовых диалогов:
    фиксирует глобальную задачу клиента, сводку уже известных фактов (включая выжимку суммаризатора)
    и напоминает правила SYSTEM_PROMPT в конце.
    Для одиночных вопросов возвращает messages без изменений.
    """
    if len(messages) <= 1:
        return messages

    users = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    global_goal = users[0] if users else messages[0]["content"].strip()
    raw_facts = " ".join(users)
    summary = summarize_dialog(messages)
    known_facts = f"{summary} ({raw_facts})" if summary and summary != raw_facts else raw_facts
    last_reply = messages[-1]["content"].strip()

    history_lines = [
        f"{'Клиент' if m['role'] == 'user' else 'Консультант'}: {m['content'].strip()}"
        for m in messages[:-1]
    ]
    history_block = "\n".join(history_lines)

    sandwich_prompt = (
        f"Исходная задача клиента (глобальная тема): {global_goal}\n"
        f"Все факты, которые клиент уже сообщил о себе и объекте: {known_facts}\n\n"
        f"Предыдущие реплики диалога:\n{history_block}\n\n"
        f"Последняя реплика клиента (ответь на неё):\n{last_reply}\n\n"
        f"Напоминание правил ответа: ответь на последнюю реплику клиента с учётом его исходной задачи и всех "
        f"сообщённых им фактов строго по материалам EnergyTool. Начни с прямого ответа по сути («Да, …», «Нет, …» "
        f"или сама суть), затем 1–2 предложения: от чего зависит результат для ситуации клиента или важная оговорка. "
        f"Объём — 2–4 предложения, 30–70 слов, обычный связный текст без списков и markdown. В конце задай один "
        f"уточняющий вопрос для подбора решения, но НЕ спрашивай то, что клиент уже сообщил выше."
    )
    return [{"role": "user", "content": sandwich_prompt}]


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
                order = llm_client.rerank(queries[-1][:2000], [c.render()[:2000] for c in candidates], CONTEXT_CHUNKS)
                if order:
                    self.rerank_failures = 0
                    return [candidates[i] for i in order][:CONTEXT_CHUNKS], candidates, queries, True
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
        convo = [{"role": "system", "content": build_system(chunks)}] + format_dialog_sandwich(messages)
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
