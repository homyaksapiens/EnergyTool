"""Цикл агента: LLM → инструменты → LLM … пока не будет финального ответа.

run_turn — генератор событий, которые сразу уходят на фронт через SSE:
  {"type": "status", "text": "🔍 Подбираю…"}   — агент вызывает инструмент
  {"type": "message", "text": "..."}            — реплика консультанта
  {"type": "profile", "profile": {...}}         — обновился профиль клиента
  {"type": "lead", "lead_id": 1}                — заявка передана менеджеру
  {"type": "error", "text": "..."}
"""
import json
import logging
import uuid
from dataclasses import dataclass, field

from .config import LLM_PROVIDER, MAX_AGENT_STEPS
from .llm import AnthropicLLM, MockLLM
from .prompts import SYSTEM_PROMPT, TOOL_STATUS
from .tools import TOOLS, run_tool

log = logging.getLogger(__name__)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: list = field(default_factory=list)    # история для LLM
    transcript: list = field(default_factory=list)  # человекочитаемая история для менеджера
    profile: dict = field(default_factory=dict)
    lead_id: int | None = None
    llm: object = None


SESSIONS: dict[str, Session] = {}


def get_session(session_id: str | None) -> Session:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    s = Session()
    s.llm = AnthropicLLM() if LLM_PROVIDER == "anthropic" else MockLLM()
    SESSIONS[s.id] = s
    return s


def run_turn(session: Session, user_text: str):
    start = len(session.messages)  # при ошибке или отказе откатываем ход целиком, чтобы история осталась валидной
    session.messages.append({"role": "user", "content": user_text})
    session.transcript.append({"role": "user", "text": user_text})

    for _ in range(MAX_AGENT_STEPS):
        try:
            step = session.llm.step(SYSTEM_PROMPT, session.messages, TOOLS)
        except Exception as e:  # noqa: BLE001 — на демо лучше показать ошибку в чате, чем уронить стрим
            log.exception("LLM error")
            del session.messages[start:]
            yield {"type": "error", "text": f"Ошибка LLM: {e}"}
            return

        if step["stop"] == "refusal":
            del session.messages[start:]
            text = "Извините, с этим я помочь не могу. Давайте вернёмся к учёту энергоресурсов — что у вас за задача?"
            session.transcript.append({"role": "assistant", "text": text})
            yield {"type": "message", "text": text}
            return

        session.messages.append({"role": "assistant", "content": step["assistant_content"]})
        if step["text"].strip():
            session.transcript.append({"role": "assistant", "text": step["text"]})
            yield {"type": "message", "text": step["text"]}

        if step["stop"] != "tool_use":
            return

        results = []
        for call in step["tool_calls"]:
            yield {"type": "status", "text": TOOL_STATUS.get(call["name"], f"⚙️ {call['name']}…")}
            try:
                out = run_tool(session, call["name"], call["input"])
                is_error = False
            except Exception as e:  # noqa: BLE001
                log.exception("Tool error")
                out, is_error = {"error": str(e)}, True
            log.info("tool %s(%s) -> %s", call["name"], call["input"], str(out)[:300])
            results.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": json.dumps(out, ensure_ascii=False),
                "is_error": is_error,
            })
            if call["name"] == "update_client_profile":
                yield {"type": "profile", "profile": session.profile}
            if call["name"] == "handoff_to_manager" and out.get("ok"):
                yield {"type": "lead", "lead_id": out["lead_id"]}
        # все результаты — одним user-сообщением
        session.messages.append({"role": "user", "content": results})

    yield {"type": "error", "text": "Агент сделал слишком много шагов подряд, попробуйте переформулировать."}
