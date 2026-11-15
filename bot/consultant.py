"""Ядро консультанта: разбор диалога → поиск по базе знаний → один вызов LLM → чистка ответа.

Используется и в solution.py (автосабмит), и в веб-бэкенде. Tool calling не используем: прокси
платформы его не гарантирует, а лишние вызовы LLM съедают лимит времени.

Берет вопрос клиента, ищет подходящие куски текста через kb.py, пропускает их через реранкер (если он доступен),
подставляет в шаблон промпта и отправляет в языковую модель.
"""
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

# Гарантированный бюджет токенов на 1-й попытке, чтобы глубокое рассуждение <think> не обрывалось
THINK_MAX_TOKENS = max(MAX_TOKENS, 2500)

SUMMARY_MAX_TOKENS = 192
SUMMARY_SYSTEM_PROMPT = (
    "Перепиши последнюю реплику диалога в один самодостаточный поисковый запрос для базы знаний (до 30 слов). "
    "Правила:\n"
    "1. Главный фокус — строго на последнем вопросе и последней реплике клиента.\n"
    "2. Замени местоимения и неполные ссылки («это», «таких данных», «четвёртая», «для оценки») "
    "на конкретные термины, тип объекта, цифры и условия из предыдущих реплик диалога.\n"
    "3. ЗАПРЕЩЕНО использовать водные и мета-слова: «клиент», «пользователь», «консультант», «ассистент», "
    "«спрашивает», «интересуется», «хочет узнать», «просит объяснить», «в диалоге».\n"
    "4. Не отвечай на вопрос и ничего не додумывай. Выведи только одну строку — готовый запрос без кавычек."
)

_SUMMARY_CACHE: dict[tuple[tuple[str, str], ...], str] = {}
_SUMMARY_LOCK = threading.Lock()


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
    text = re.sub(r"\*\*|__|`|^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)
    return " ".join(text.split())


def _dialog_cache_key(messages: list[dict]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (m.get("role", "user"), m.get("content", "").strip())
        for m in messages
        if m.get("content", "").strip()
    )


def summarize_dialog(messages: list[dict]) -> str:
    """Деконтекстуализация последнего вопроса диалога без водных слов (/no_think, с кэшированием)."""
    if len(messages) <= 1:
        return ""

    key = _dialog_cache_key(messages)
    with _SUMMARY_LOCK:
        if key in _SUMMARY_CACHE:
            return _SUMMARY_CACHE[key]

    lines = [
        f"{'Клиент' if m['role'] == 'user' else 'Консультант'}: {m['content'].strip()}"
        for m in messages
        if m["content"].strip()
    ]
    dialog_text = "\n".join(lines)
    convo = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"{dialog_text}\n\nПоисковый запрос по последней реплике: /no_think"},
    ]
    summary = ""
    try:
        raw, _ = llm_client.chat(convo, temperature=0.0, max_tokens=SUMMARY_MAX_TOKENS)
        summary = clean_answer(raw).strip("«»\"'")
    except Exception as e:  # noqa: BLE001 — при любой ошибке работаем по базовым запросам без падения
        print(f"Суммаризатор диалога недоступен: {e!r}", file=sys.stderr)
        summary = ""

    with _SUMMARY_LOCK:
        _SUMMARY_CACHE[key] = summary
    return summary


def search_queries(messages: list[dict]) -> list[str]:
    """Формирует поисковые запросы без размытия весов токенов:
    - Одиночный вопрос: [last]
    - Диалог: [глобальная_проблема, текущий_обмен, весь_диалог, сфокусированный_запрос_без_воды (для реранкера)]
    """
    users = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    last = users[-1] if users else ""
    full = " ".join(m["content"].strip() for m in messages if m["content"].strip())
    if full == last or len(messages) <= 1:
        return [last]

    global_problem = users[0] if users else last
    if len(messages) >= 2 and messages[-2]["role"] == "assistant":
        last_exchange = f"{messages[-2]['content'].strip()} {last}".strip()
    else:
        last_exchange = last

    summary = summarize_dialog(messages)

    queries: list[str] = []
    for q in (global_problem, last_exchange, full, summary):
        if q and q not in queries:
            queries.append(q)
    return queries


def format_dialog_sandwich(messages: list[dict]) -> list[dict]:
    """Техника «сэндвича» для многоходовых диалогов:
    фиксирует глобальную задачу клиента, сводку уже известных фактов, список уже заданных вопросов
    и напоминает правила SYSTEM_PROMPT в конце. Для одиночных вопросов возвращает messages без изменений.
    """
    if len(messages) <= 1:
        return messages

    users = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    assistants = [m["content"].strip() for m in messages[:-1] if m["role"] == "assistant" and m["content"].strip()]
    global_goal = users[0] if users else messages[0]["content"].strip()
    known_facts = " ".join(users)
    already_asked = " | ".join(assistants)
    last_reply = messages[-1]["content"].strip()
    summary = summarize_dialog(messages)

    history_lines = [
        f"{'Клиент' if m['role'] == 'user' else 'Консультант'}: {m['content'].strip()}"
        for m in messages[:-1]
    ]
    history_block = "\n".join(history_lines)
    summary_line = f"Суть текущего вопроса с учётом контекста: {summary}\n" if summary else ""
    asked_line = f"Вопросы, которые консультант УЖЕ задавал (не повторяй их): {already_asked}\n" if already_asked else ""

    sandwich_prompt = (
        f"Исходная задача клиента (глобальная тема): {global_goal}\n"
        f"Все факты, которые клиент уже сообщил о себе и объекте: {known_facts}\n"
        f"{asked_line}"
        f"{summary_line}\n"
        f"Предыдущие реплики диалога:\n{history_block}\n\n"
        f"Последняя реплика клиента (ответь на неё):\n{last_reply}\n\n"
        f"Напоминание правил ответа: ответь на последнюю реплику клиента строго по материалам EnergyTool. "
        f"Начни с прямого ответа по сути («Да, …», «Нет, …» или сама суть) и в первом-втором предложении явно "
        f"привяжи ответ к типу объекта и уже известным условиям клиента из истории диалога (если клиент назвал "
        f"количество словами, например «три склада» или «около сорока», пиши его тоже словами, не заменяя на цифры, "
        f"которых нет в материалах). Затем за 1–2 предложения укажи ключевые условия и оговорки из материалов. "
        f"Объём — 2–4 предложения, 30–70 слов, обычный связный текст без списков и markdown. В конце задай один "
        f"новый уточняющий вопрос для следующего шага, но НЕ спрашивай то, что клиент уже сообщил или что консультант уже спрашивал."
    )
    return [{"role": "user", "content": sandwich_prompt}]


class Consultant:
    def __init__(self, kb: KnowledgeBase, use_rerank: bool = USE_RERANK):
        self.kb = kb
        self.rerank_ok = use_rerank

    def _retrieve_detailed(
        self, messages: list[dict], query_vectors=None
    ) -> tuple[list[Chunk], bool, list[Chunk], list[str]]:
        queries = search_queries(messages)
        candidates = self.kb.search(queries, query_vectors, k=CANDIDATES)
        if self.rerank_ok and len(candidates) > CONTEXT_CHUNKS:
            try:
                order = llm_client.rerank(
                    queries[-1][:2000],
                    [c.render()[:2000] for c in candidates],
                    CONTEXT_CHUNKS,
                )
                if order:
                    return [candidates[i] for i in order][:CONTEXT_CHUNKS], True, candidates, queries
            except llm_client.ApiError as e:
                if e.status in (400, 404, 405, 422, 501):  # эндпоинта нет — больше не пробуем
                    self.rerank_ok = False
                print(f"Реранкер недоступен: {e}", file=sys.stderr)
            except (KeyError, TypeError, ValueError) as e:
                print(f"Реранкер вернул неожиданный ответ: {e!r}", file=sys.stderr)
        return candidates[:CONTEXT_CHUNKS], False, candidates, queries

    def retrieve(self, messages: list[dict], query_vectors=None) -> tuple[list[Chunk], bool]:
        chunks, reranked, _, _ = self._retrieve_detailed(messages, query_vectors)
        return chunks, reranked

    def answer(self, messages: list[dict], query_vectors=None, deadline: float | None = None) -> dict:
        """Ответ на последнюю реплику клиента: {text, sources, timings, reranked, fallback}."""
        t0 = time.time()
        chunks, reranked, candidates, queries = self._retrieve_detailed(messages, query_vectors)
        t1 = time.time()
        convo = [{"role": "system", "content": build_system(chunks)}] + format_dialog_sandwich(messages)
        text = ""
        for attempt in range(2):
            if deadline and time.time() > deadline:
                break
            if attempt == 0:
                msgs = convo[:-1] + [{**convo[-1], "content": convo[-1]["content"] + " /think"}]
                tokens_limit = THINK_MAX_TOKENS
            else:  # модель могла уйти в рассуждения и не успеть ответить — просим без них
                msgs = convo[:-1] + [{**convo[-1], "content": convo[-1]["content"] + " /no_think"}]
                tokens_limit = MAX_TOKENS
            try:
                raw, _ = llm_client.chat(msgs, TEMPERATURE, tokens_limit)
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