"""Настройки из переменных окружения.

Адрес прокси, ключ и модели платформа подставляет сама (см. техчать.md). Свои параметры поиска
читаем из переменных с префиксом ET_, чтобы не путать с настройками платформы для учебного примера.
"""
import os


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    # платформа может оставить неподставленное значение вида LLM_MODEL=LLM_MODEL
    return default if not value or value == name else value


def first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = env(name)
        if value:
            return value
    return default


def number(name: str, default, lower, upper, kind=int):
    try:
        return max(lower, min(upper, kind(env(name))))
    except ValueError:
        return default


BASE_URL = first_env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/")
API_KEY = first_env("LLM_API_KEY", "OPENAI_API_KEY")
EMB_API_KEY = env("EMBEDDER_API_KEY") or API_KEY
RERANK_API_KEY = env("RERANKER_API_KEY") or API_KEY
LLM_MODEL = first_env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="qwen/qwen3.8-27b")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "qwen/qwen3-reranker-8b")

TEMPERATURE = number("TEMPERATURE", 0.1, 0.0, 1.0, float)
# не меньше 512: если модель начнёт рассуждать, короткий лимит оставит ответ пустым
MAX_TOKENS = max(number("MAX_TOKENS", 1024, 128, 8192), 512)
MAX_WORKERS = number("MAX_WORKERS", 4, 1, 16)
DATA_DIR = env("DATA_DIR", "data")

CANDIDATES = number("ET_CANDIDATES", 28, 4, 50)  # сколько фрагментов отдаём реранкеру
CONTEXT_CHUNKS = number("ET_CONTEXT_CHUNKS", 10, 1, 15)  # сколько фрагментов попадает в промпт
USE_DENSE = env("ET_USE_DENSE", "1") == "1"
USE_RERANK = env("ET_USE_RERANK", "1") == "1"
TIME_BUDGET = number("ET_TIME_BUDGET", 1020, 60, 3600)  # секунд на весь прогон; лимит платформы — 1200
REQUEST_TIMEOUT = number("ET_REQUEST_TIMEOUT", 90.0, 10.0, 600.0, float)
DEBUG = env("ET_DEBUG") == "1"  # писать debug.jsonl с источниками и таймингами