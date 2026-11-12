"""Ядро консультанта: разбор диалога → поиск по базе знаний → один вызов LLM → чистка ответа.

Используется и в solution.py (автосабмит), и в веб-бэкенде. Tool calling не используем: прокси
платформы его не гарантирует, а лишние вызовы LLM съедают лимит времени.
"""
from __future__ import annotations

import re
import sys
import time

import llm_client
from kb import Chunk, KnowledgeBase
from prompts import FALLBACK_ANSWER, build_system
from settings import CANDIDATES, CONTEXT_CHUNKS, MAX_TOKENS, TEMPERATURE, USE_RERANK

ROLE_RE = re.compile(
    r"^\s*(Пользователь|Клиент|Покупатель|Заказчик|Вопрос|User|Ассистент|Консультант|Бот|Менеджер|Оператор|Ответ|Assistant)\s*:\s*(.*)$",
    re.I,
)
ASSISTANT_ROLES = {"ассистент", "консультант", "бот", "менеджер", "оператор", "ответ", "assistant"}
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
    """Возвращает [последняя_реплика, контекст_диалога], ограниченные до 2000 символов для безопасного эмбеддинга."""
    users = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    if not users:
        last = (messages[-1]["content"].strip() if messages else "")[:2000]
        return [last]

    last = users[-1][:2000]
    user_history = " ".join(users)

    # Если последняя реплика клиента короткая («Да, давайте», «А какие сроки?»),
    # добавляем предыдущую реплику ассистента, чтобы сохранить тему вопроса, но не тянем старые ответы бота
    if len(last.split()) <= 6 and len(messages) >= 2 and messages[-2]["role"] == "assistant":
        context_query = f"{' '.join(users[:-1])} {messages[-2]['content']} {last}".strip()
    else:
        context_query = user_history.strip()

    # Берём хвост контекста (последние 2000 символов), чтобы не превысить лимит токенов эмбеддера
    context_query = context_query[-2000:]
    return [last] if context_query == last else [last, context_query]


def clean_answer(text: str) -> str:
    text = text or ""
    if "</think>" in text:  # рассуждение могло начаться ещё в шаблоне промпта — берём то, что после него
        text = text.rsplit("</think>", 1)[1]
    text = THINK_RE.sub("", text)
    if "<think>" in text:  # рассуждение оборвалось по лимиту токенов — ответа нет
        text = text.split("<think>", 1)[0]

    # Убираем ссылки на номера фрагментов [1], [2], если модель скопировала их из контекста
    text = re.sub(r"\s*\[\d+(?:[,\s-]+\d+)*\]", "", text)
    text = re.sub(r"\*\*|__|`|^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)

    # При склейке многострочного ответа в один абзац ставим точку, если строка не заканчивалась знаком препинания
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        for idx in range(len(lines) - 1):
            if lines[idx] and lines[idx][-1] not in ".!?:;,»\"":
                lines[idx] += "."
        text = " ".join(lines)

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
                # В реранкер передаём текущий вопрос queries[0] (плюс хвост контекста, если диалог многоходовый)
                if len(queries) == 1:
                    rerank_q = queries[0][:2000]
                else:
                    rerank_q = f"Контекст клиента: {queries[1][-1200:]}\nТекущий вопрос: {queries[0]}"[:2000]

                order = llm_client.rerank(rerank_q, [c.render()[:2000] for c in candidates], CONTEXT_CHUNKS)
                if order:
                    return [candidates[i] for i in order][:CONTEXT_CHUNKS], True
            except llm_client.ApiError as e:
                if e.status in (400, 404, 405, 422, 501):  # эндпоинта нет — больше не пробуем
                    self.rerank_ok = False
                print(f"Реранкер недоступен: {e}", file=sys.stderr)
            except (KeyError, TypeError, ValueError) as e:
                print(f"Реранкер вернул неожиданный ответ: {e!r}", file=sys.stderr)
        return candidates[:CONTEXT_CHUNKS], False

    def answer(self, messages: list[dict], query_vectors=None, deadline: float | None = None) -> dict:
        """Ответ на последнюю реплику клиента: {text, sources, timings, reranked, fallback}."""
        t0 = time.time()
        chunks, reranked = self.retrieve(messages, query_vectors)
        t1 = time.time()
        convo = [{"role": "system", "content": build_system(chunks)}] + messages
        text = ""
        partial_text = ""
        for attempt in range(2):
            if deadline and time.time() > deadline:
                break
            msgs = convo
            if attempt == 1:  # модель могла уйти в рассуждения и не успеть ответить — просим без них
                msgs = convo[:-1] + [{**convo[-1], "content": convo[-1]["content"] + " /no_think"}]
            try:
                raw, finish_reason = llm_client.chat(msgs, TEMPERATURE, MAX_TOKENS)
            except (llm_client.ApiError, KeyError, IndexError, TypeError, ValueError) as e:
                print(f"LLM: {e!r}", file=sys.stderr)
                break
            cleaned = clean_answer(raw)
            if not cleaned:
                continue
            # Если на первой попытке модель потратила токены на <think> и ответ оборвался на полуслове по лимиту
            if (
                attempt == 0
                and finish_reason == "length"
                and ("<think>" in (raw or "") or "</think>" in (raw or ""))
                and cleaned[-1] not in ".!?»\""
            ):
                partial_text = cleaned
                continue
            text = cleaned
            break

        final_text = text or partial_text
        return {
            "text": final_text or FALLBACK_ANSWER,
            "fallback": not final_text,
            "reranked": reranked,
            "sources": [{"source": c.source, "title": c.title} for c in chunks],
            "timings": {"retrieval": round(t1 - t0, 2), "llm": round(time.time() - t1, 2)},
        }
