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
    """Запросы к базе: последняя реплика клиента и (для диалога) весь диалог — реплики ассистента
    часто называют продукт, о котором идёт речь («сервис для выездных проверок»)."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    last = users[-1] if users else ""
    full = " ".join(m["content"] for m in messages)
    return [last] if full == last else [last, full]


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
                order = llm_client.rerank(queries[-1][:2000], [c.render()[:2000] for c in candidates], CONTEXT_CHUNKS)
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
