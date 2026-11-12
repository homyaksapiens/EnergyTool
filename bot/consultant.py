"""Ядро консультанта: разбор диалога → поиск по базе знаний → один вызов LLM → чистка ответа.

Используется и в solution.py (автосабмит), и в веб-бэкенде. Tool calling не используем: прокси
платформы его не гарантирует, а лишние вызовы LLM съедают лимит времени.
"""
from __future__ import annotations

import re
import sys
import time

import llm_client
from kb import Chunk, KnowledgeBase, diversify_chunks
from prompts import FALLBACK_ANSWER, build_system
from settings import CANDIDATES, CONTEXT_CHUNKS, MAX_TOKENS, TEMPERATURE, USE_RERANK

ROLE_RE = re.compile(r"^\s*(Пользователь|Клиент|User|Ассистент|Консультант|Бот|Assistant)\s*:\s*(.*)$", re.I)
ASSISTANT_ROLES = {"ассистент", "консультант", "бот", "assistant"}
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


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


def search_queries(messages: list[dict]) -> list[str]:
    """Формирует сбалансированные запросы без отрыва темы от вопроса:
    - Для одиночного вопроса: [last]
    - Для диалога: [focused (все реплики клиента + последний вопрос ассистента), full (весь диалог)]
    """
    users = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    last = users[-1] if users else ""
    full = " ".join(m["content"].strip() for m in messages if m["content"].strip())
    if full == last or len(messages) <= 1:
        return [last]

    # Собираем все реплики клиента (чтобы не терять ни исходную задачу U1, ни параметры из середины U2)
    # плюс последний уточняющий вопрос ассистента перед ответом клиента (например, «А какая у вас ценовая категория?»)
    parts = list(users[:-1])
    if len(messages) >= 2 and messages[-2]["role"] == "assistant":
        parts.append(messages[-2]["content"].strip())
    parts.append(last)
    focused = " ".join(p for p in parts if p).strip()

    return [focused, full] if focused != full else [full]


def format_dialog_sandwich(messages: list[dict]) -> list[dict]:
    """Для одиночного вопроса возвращает messages как есть.
    Для многоходового диалога сохраняет историю и добавляет в конец точное напоминание правил SYSTEM_PROMPT,
    чтобы модель не копировала короткий (6-10 слов) стиль промежуточных реплик и связывала всю задачу целиком.
    """
    if len(messages) <= 1:
        return messages

    users = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    initial_task = users[0] if users else messages[0]["content"].strip()
    known_facts = " ".join(users[:-1]) if len(users) > 1 else initial_task

    history_lines = [
        f"{'Клиент' if m['role'] == 'user' else 'Консультант'}: {m['content'].strip()}"
        for m in messages
    ]
    dialog_text = "\n".join(history_lines)

    sandwich_prompt = (
        f"Диалог с клиентом:\n{dialog_text}\n\n"
        f"Сводка контекста: исходная задача клиента — «{initial_task}»; "
        f"уже сообщённые клиентом вводные — «{known_facts}».\n"
        f"Инструкция для ответа на последнюю реплику клиента: дай полный, содержательный ответ (3–4 предложения, "
        f"около 40–55 слов, связным текстом без списков и markdown) строго по материалам EnergyTool. "
        f"Начни сразу с сути ответа на последний вопрос (без лишнего «Да, …», если вопрос открытый или с условием), "
        f"свяжи исходную задачу клиента с деталями из последних реплик (если подходят два наших решения — укажи оба), "
        f"приведи конкретные условия, составляющие и важные оговорки из материалов. "
        f"Если для подбора решения или оценки всё ещё не хватает данных — закончи одним уточняющим вопросом, "
        f"не спрашивая то, что клиент уже сообщил."
    )
    return [{"role": "user", "content": sandwich_prompt}]


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


class Consultant:
    def __init__(self, kb: KnowledgeBase, use_rerank: bool = USE_RERANK):
        self.kb = kb
        self.rerank_ok = use_rerank

    def retrieve(self, messages: list[dict], query_vectors=None) -> tuple[list[Chunk], bool]:
        queries = search_queries(messages)
        candidates = self.kb.search(queries, query_vectors, k=CANDIDATES)
        if self.rerank_ok and len(candidates) > CONTEXT_CHUNKS:
            try:
                # В реранкер передаём сфокусированный запрос queries[0] и ранжируем все кандидаты,
                # после чего отсекаем дубли шаблонных подзаголовков («Следующий шаг», «Сколько это стоит»)
                order = llm_client.rerank(
                    queries[0][:2000],
                    [c.render()[:2000] for c in candidates],
                    len(candidates),
                )
                if order:
                    reranked_chunks = [candidates[i] for i in order]
                    return diversify_chunks(reranked_chunks, CONTEXT_CHUNKS), True
            except llm_client.ApiError as e:
                if e.status in (400, 404, 405, 422, 501):  # эндпоинта нет — больше не пробуем
                    self.rerank_ok = False
                print(f"Реранкер недоступен: {e}", file=sys.stderr)
            except (KeyError, TypeError, ValueError) as e:
                print(f"Реранкер вернул неожиданный ответ: {e!r}", file=sys.stderr)
        return diversify_chunks(candidates, CONTEXT_CHUNKS), False

    def answer(self, messages: list[dict], query_vectors=None, deadline: float | None = None) -> dict:
        """Ответ на последнюю реплику клиента: {text, sources, timings, reranked, fallback}."""
        t0 = time.time()
        chunks, reranked = self.retrieve(messages, query_vectors)
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
                raw, _ = llm_client.chat(msgs, TEMPERATURE, MAX_TOKENS)
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
            "sources": [{"source": c.source, "title": c.title} for c in chunks],
            "timings": {"retrieval": round(t1 - t0, 2), "llm": round(time.time() - t1, 2)},
        }
