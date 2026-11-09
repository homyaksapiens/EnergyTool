"""Клиент OpenAI-совместимого прокси: генерация, эмбеддинги, реранкер.

На 429, 5xx и таймауты — ограниченное число повторов с паузой (требование регламента).
"""
from __future__ import annotations

import random
import time
from urllib.parse import urlparse

import httpx

from settings import (API_KEY, BASE_URL, EMB_API_KEY, EMB_MODEL, LLM_MODEL, RERANK_API_KEY, RERANKER_MODEL,
                      REQUEST_TIMEOUT)

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

# httpx.Client потокобезопасен, держим одно пуловое соединение на все потоки.
# К локальному адресу ходим мимо системного прокси (VPN-клиенты в Windows прописывают его для всех запросов)
_client = httpx.Client(
    timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
    trust_env=urlparse(BASE_URL).hostname not in LOCAL_HOSTS,
)


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def post(endpoint: str, key: str, body: dict, attempts: int = 3) -> dict:
    if not BASE_URL:
        raise ApiError("не задан LLM_BASE_URL")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    error = ApiError("запрос не выполнялся")
    for attempt in range(attempts):
        try:
            response = _client.post(BASE_URL + endpoint, headers=headers, json=body)
        except httpx.TransportError as e:  # сюда же попадают таймауты
            error = ApiError(f"{endpoint}: {type(e).__name__}")
        else:
            if response.status_code < 400:
                return response.json()
            error = ApiError(f"{endpoint}: HTTP {response.status_code} {response.text[:200]}", response.status_code)
            if response.status_code not in RETRY_STATUS:
                raise error
        if attempt < attempts - 1:
            time.sleep(1.5 * (attempt + 1) + random.random())
    raise error


def chat(messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, str | None]:
    """Возвращает (текст ответа, finish_reason)."""
    data = post("/chat/completions", API_KEY, {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    choice = data["choices"][0]
    return (choice.get("message") or {}).get("content") or "", choice.get("finish_reason")


def embed(texts: list[str], batch: int = 32) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        data = post("/embeddings", EMB_API_KEY, {"model": EMB_MODEL, "input": part})
        items = sorted(data["data"], key=lambda item: item.get("index", 0))
        if len(items) != len(part):
            raise ApiError("эмбеддингов вернулось меньше, чем текстов")
        vectors.extend(item["embedding"] for item in items)
    return vectors


def rerank(query: str, documents: list[str], top_n: int) -> list[int]:
    """Индексы документов по убыванию релевантности."""
    data = post("/rerank", RERANK_API_KEY, {
        "model": RERANKER_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }, attempts=2)
    items = data.get("results") or data.get("data") or []
    items = sorted(items, key=lambda item: -float(item.get("relevance_score", item.get("score", 0))))
    return [int(item["index"]) for item in items if 0 <= int(item["index"]) < len(documents)]
