"""Слой LLM. Агент работает с единым форматом шага, поэтому провайдера легко заменить
(например, если организаторы дадут GigaChat/YandexGPT — пишется ещё один класс с методом step).

Шаг возвращает dict:
  stop: "tool_use" | "end" | "refusal"
  text: видимый текст ответа
  tool_calls: [{id, name, input}]
  assistant_content: что положить в историю как сообщение assistant
"""
import re
import uuid

from .config import LLM_EFFORT, LLM_MODEL


class AnthropicLLM:
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    def step(self, system: str, messages: list, tools: list) -> dict:
        resp = self.client.beta.messages.create(
            model=LLM_MODEL,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": LLM_EFFORT},
            # при отказе модели запрос автоматически перезапускается на рекомендованной модели
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            return {"stop": "refusal", "text": "", "tool_calls": [], "assistant_content": None}
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "input": b.input} for b in resp.content if b.type == "tool_use"]
        return {
            "stop": "tool_use" if resp.stop_reason == "tool_use" else "end",
            "text": text,
            "tool_calls": calls,
            "assistant_content": resp.content,  # весь content целиком, включая thinking-блоки
        }


class MockLLM:
    """Сценарная заглушка: позволяет фронту и панели менеджера работать без API-ключа."""

    CONTACT_RE = re.compile(r"(\+?\d[\d\s\-()]{9,}\d)|([^@\s]+@[^@\s]+\.\w+)")

    def step(self, system: str, messages: list, tools: list) -> dict:
        last = messages[-1]["content"]
        if isinstance(last, list):  # пришли результаты инструментов
            name = getattr(self, "_pending", "")
            if name == "search_knowledge":
                return self._text(
                    "[MOCK] Судя по задаче, вам подойдёт **почасовой учёт электроэнергии** от МТС: "
                    "умные счётчики со встроенной связью, данные собираются в облаке, отчётность формируется "
                    "автоматически. Хотите, специалист сделает расчёт под ваши объекты? "
                    "Оставьте имя, компанию и телефон — и подтвердите согласие на передачу контактов."
                )
            if name == "handoff_to_manager":
                return self._text("[MOCK] Готово! Передал заявку менеджеру, он свяжется с вами в ближайшее время.")
            return self._text("[MOCK] Записал. Расскажите, что сейчас больше всего беспокоит в учёте?")

        user_turns = sum(1 for m in messages if m["role"] == "user" and isinstance(m["content"], str))
        m = self.CONTACT_RE.search(last)
        if m:
            phone, email = (m.group(1) or ""), (m.group(2) or "")
            return self._call("handoff_to_manager", {
                "name": last.split(",")[0][:40], "phone": phone, "email": email, "consent": True,
                "summary": "[MOCK] Клиент интересуется автоматизацией учёта электроэнергии.",
                "recommended_solutions": ["Почасовой учёт электроэнергии"],
                "lead_temperature": "warm", "next_step": "Связаться и сделать расчёт",
            })
        if user_turns == 1:
            return self._text(
                "[MOCK] Здравствуйте! Я консультант МТС EnergyTool. Расскажите, пожалуйста, "
                "в какой сфере ваша компания и какие ресурсы хотите учитывать?"
            )
        if user_turns == 2:
            return self._call("update_client_profile", {"pain_points": last[:200]})
        return self._call("search_knowledge", {"query": last})

    def _text(self, text):
        return {"stop": "end", "text": text, "tool_calls": [], "assistant_content": [{"type": "text", "text": text}]}

    def _call(self, name, args):
        self._pending = name
        call = {"id": f"toolu_{uuid.uuid4().hex[:12]}", "name": name, "input": args}
        return {
            "stop": "tool_use", "text": "", "tool_calls": [call],
            "assistant_content": [{"type": "tool_use", **call}],
        }
