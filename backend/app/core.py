"""Подключение ядра консультанта из bot/.

Ядро создаётся один раз при старте сервера. Эмбеддинги базы знаний считаются в фоновом потоке,
чтобы сервер начал отвечать сразу; пока они не готовы или эмбеддер недоступен — поиск идёт по BM25.
"""
import logging
import os
import threading
import time
from urllib.parse import urlparse

import httpx

from . import config

log = logging.getLogger(__name__)

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _detect_models() -> None:
    """Если LLM_MODEL не задан, спрашиваем у сервера список моделей (так же делает eval/run_local.py).

    Вызывается до импорта ядра: оно читает имена моделей из окружения при импорте.
    """
    base_url = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        log.warning("LLM_BASE_URL не задан: модель недоступна, бот будет отдавать запасной ответ")
        return
    if any(os.getenv(name, "").strip() for name in ("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL")):
        return
    key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    try:
        response = httpx.get(
            base_url + "/models",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=5,
            trust_env=urlparse(base_url).hostname not in LOCAL_HOSTS,  # к localhost — мимо системного прокси
        )
        response.raise_for_status()
        ids = [str(m["id"]) for m in response.json()["data"]]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
        log.warning("Не удалось получить список моделей с %s: %s", base_url, type(e).__name__)
        return
    llm = next((i for i in ids if "embed" not in i.lower() and "bge" not in i.lower() and "rerank" not in i.lower()), None)
    emb = next((i for i in ids if "bge-m3" in i.lower()), None)
    if llm:
        os.environ["LLM_MODEL"] = llm
        log.info("LLM_MODEL не задан, выбрана модель сервера: %s", llm)
    if emb and not os.getenv("EMB_MODEL", "").strip():
        os.environ["EMB_MODEL"] = emb


_detect_models()

import llm_client  # noqa: E402  — модули ядра из bot/
import settings as bot_settings  # noqa: E402
from consultant import Consultant, search_queries  # noqa: E402
from kb import KnowledgeBase, load_documents  # noqa: E402

_kb: KnowledgeBase | None = None
_consultant: Consultant | None = None


def start() -> None:
    """Загрузить базу знаний и создать консультанта. Вызывается один раз при старте сервера."""
    global _kb, _consultant
    _kb = KnowledgeBase(load_documents(config.KB_PATH))
    _consultant = Consultant(_kb)
    log.info("База знаний: %d фрагментов из %s", len(_kb.chunks), config.KB_PATH.name)
    if bot_settings.USE_DENSE and bot_settings.BASE_URL:
        threading.Thread(target=_enable_dense, name="kb-embeddings", daemon=True).start()


def _enable_dense() -> None:
    try:
        t0 = time.time()
        _kb.enable_dense(llm_client.embed)
        log.info("Эмбеддинги базы знаний готовы за %.1f с", time.time() - t0)
    except Exception as e:  # эмбеддер необязателен: без него работаем на BM25
        log.warning("Эмбеддер недоступен, поиск только по BM25: %s", e)


def info() -> dict:
    """Состояние ядра для GET /api/health."""
    return {
        "model": bot_settings.LLM_MODEL,
        "kb_chunks": len(_kb.chunks) if _kb else 0,
        "dense": bool(_kb is not None and _kb.vectors is not None),
    }


def answer(messages: list[dict]) -> dict:
    """Ответ консультанта на последнюю реплику: {text, fallback, reranked, sources, timings}.

    Вызов синхронный и долгий — эндпоинты, которые его зовут, объявлены обычным def
    и выполняются в пуле потоков FastAPI.
    """
    t0 = time.time()
    vectors = None
    if _kb.vectors is not None:
        try:  # векторы запросов считает вызывающая сторона, как в bot/solution.py
            vectors = llm_client.embed([q[:2000] for q in search_queries(messages)])
        except Exception as e:
            log.warning("Эмбеддинг запроса не получен, ищем по BM25: %s", e)
    embed_time = time.time() - t0
    result = _consultant.answer(messages, vectors, deadline=time.time() + config.ANSWER_DEADLINE)
    result["timings"]["retrieval"] = round(result["timings"]["retrieval"] + embed_time, 2)
    return result


def complete(messages: list[dict]) -> str:
    """Один вызов модели без поиска по базе — для резюме заявки. Ошибки модели пробрасываются."""
    text, _ = llm_client.chat(messages, bot_settings.TEMPERATURE, bot_settings.MAX_TOKENS)
    return text or ""
