"""Инструменты агента: JSON-схемы (что видит LLM) и их реализация (что делает бэкенд)."""
import re

from . import db
from .config import KNOWLEDGE_DIR
from .notify import notify_manager
from .rag import KnowledgeBase

kb = KnowledgeBase(KNOWLEDGE_DIR)

TOOLS = [
    {
        "name": "search_knowledge",
        "description": (
            "Поиск по базе знаний о продуктах и решениях МТС EnergyTool: что делает решение, для кого, "
            "какие задачи закрывает, как подключается. Вызывай перед тем, как рекомендовать решение "
            "или отвечать на вопрос о продукте. Формулируй запрос через задачу клиента, "
            "например «хищения электроэнергии в сетях» или «почасовой учёт для оптового рынка»."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Поисковый запрос на русском"}},
            "required": ["query"],
        },
    },
    {
        "name": "update_client_profile",
        "description": (
            "Сохраняет факты о клиенте и его задаче, как только они прозвучали в диалоге. "
            "Передавай только новые или изменившиеся поля. Возвращает текущий профиль и список "
            "того, что ещё не выяснено."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "industry": {"type": "string", "description": "Отрасль: УК/ТСЖ, промышленность, сетевая компания, ритейл…"},
                "object_type": {"type": "string", "description": "Тип объектов: МКД, цех, ТП, магазины…"},
                "resources": {"type": "array", "items": {"type": "string"}, "description": "Электроэнергия, вода, тепло, газ"},
                "scale": {"type": "string", "description": "Число объектов или точек учёта"},
                "current_metering": {"type": "string", "description": "Как учёт ведётся сейчас"},
                "pain_points": {"type": "string", "description": "Проблема/цель клиента своими словами"},
                "region": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "handoff_to_manager",
        "description": (
            "Передаёт заявку менеджеру МТС: контакты и краткое резюме диалога. Вызывай только после того, "
            "как клиент дал контакты и согласился на их передачу."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "company": {"type": "string"},
                "phone": {"type": "string"},
                "email": {"type": "string"},
                "consent": {"type": "boolean", "description": "Клиент явно согласился на передачу контактов"},
                "summary": {"type": "string", "description": "3–5 предложений для менеджера: кто клиент, задача, масштаб, боль"},
                "recommended_solutions": {"type": "array", "items": {"type": "string"}},
                "lead_temperature": {
                    "type": "string",
                    "enum": ["hot", "warm", "cold"],
                    "description": "hot — есть конкретная задача и сроки; warm — интерес есть, задача размыта; cold — просто изучает",
                },
                "next_step": {"type": "string", "description": "Что менеджеру сделать: расчёт, демо, пилот, КП…"},
            },
            "required": ["name", "consent", "summary", "recommended_solutions", "lead_temperature"],
        },
    },
]

PROFILE_FIELDS = ["industry", "object_type", "resources", "scale", "current_metering", "pain_points"]
PHONE_RE = re.compile(r"^\+?[\d\s\-()]{10,}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def run_tool(session, name: str, args: dict) -> dict:
    if name == "search_knowledge":
        results = kb.search(args.get("query", ""))
        if not results:
            return {"results": [], "note": "Ничего не найдено. Переформулируй запрос или предложи консультацию менеджера."}
        return {"results": [{"title": r["title"], "text": r["text"]} for r in results]}

    if name == "update_client_profile":
        session.profile.update({k: v for k, v in args.items() if v})
        missing = [f for f in PROFILE_FIELDS if not session.profile.get(f)]
        return {"profile": session.profile, "missing": missing}

    if name == "handoff_to_manager":
        errors = []
        if not args.get("consent"):
            errors.append("нет согласия клиента на передачу контактов")
        phone, email = (args.get("phone") or "").strip(), (args.get("email") or "").strip()
        if not phone and not email:
            errors.append("нужен телефон или e-mail")
        if phone and not PHONE_RE.match(phone):
            errors.append(f"телефон «{phone}» выглядит некорректно")
        if email and not EMAIL_RE.match(email):
            errors.append(f"e-mail «{email}» выглядит некорректно")
        if errors:
            return {"ok": False, "errors": errors, "hint": "Уточни у клиента недостающие данные и вызови снова."}
        if session.lead_id:
            return {"ok": True, "lead_id": session.lead_id, "note": "Заявка по этому диалогу уже передана."}
        lead_id = db.save_lead(session.id, args, session.profile, session.transcript)
        session.lead_id = lead_id
        sent = notify_manager(lead_id, args)
        return {"ok": True, "lead_id": lead_id, "telegram_sent": sent}

    return {"error": f"Неизвестный инструмент {name}"}
