"""Настройки бэкенда из переменных окружения (CONTRACTS.md, раздел 6).

Этот модуль импортируется первым: он читает .env до того, как загрузится ядро из bot/ —
ядро берёт адрес модели и свои настройки из окружения в момент импорта.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Ядро консультанта — плоские модули в bot/ (CONTRACTS.md, раздел 2)
BOT_DIR = ROOT / "bot"
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))


def _path(name: str, default: str) -> Path:
    """Путь из переменной окружения; относительный считается от корня проекта."""
    path = Path(os.getenv(name, "").strip() or default)
    return path if path.is_absolute() else ROOT / path


KB_PATH = _path("KB_PATH", "data/knowledge-base.zip")
DB_PATH = _path("DB_PATH", "data/app.db")
FRONTEND_DIR = _path("FRONTEND_DIR", "frontend")
PORT = int(os.getenv("PORT", "").strip() or 8000)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Панель менеджера: если пароль задан, список сессий и заявки открываются только по логину (HTTP Basic)
MANAGER_USER = os.getenv("MANAGER_USER", "").strip() or "manager"
MANAGER_PASSWORD = os.getenv("MANAGER_PASSWORD", "").strip()

# Показ входа клиента: если задан номер, каждый посетитель считается вошедшим с этим номером
# и видит колонку «Ваши диалоги». Настоящий вход по номеру появится при интеграции с сайтом
DEMO_CLIENT_PHONE = os.getenv("DEMO_CLIENT_PHONE", "").strip()

RETENTION_DAYS = 30        # срок хранения сессий; обещан клиенту на frontend/privacy.html
MAX_MESSAGE_CHARS = 4000   # предел длины одной реплики клиента
HISTORY_MESSAGES = 20      # сколько последних реплик диалога передаём модели
ANSWER_DEADLINE = 75.0     # секунд на ответ: фронт ждёт 90 с, после этого показывает «Повторить»
