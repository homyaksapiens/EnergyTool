import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# "anthropic" — настоящая LLM, "mock" — сценарная заглушка без ключа (для фронта и отладки)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "mock")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("LLM_EFFORT", "low")  # low для быстрого чата; medium/high — вдумчивее, но медленнее

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", ROOT / "knowledge"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", ROOT / "frontend"))
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "leads.db"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

MAX_AGENT_STEPS = 8
