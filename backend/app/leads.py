"""Заявка менеджеру: проверка контактов, момент для предложения заявки и резюме диалога."""
import logging
import re

from . import core

log = logging.getLogger(__name__)

# ---------- проверка формы заявки ----------

PHONE_CHARS_RE = re.compile(r"^[\d\s()+\-]+$")
MAX_FIELD_CHARS = 200


def validate_lead(full_name: str, phone: str, company: str, consent: bool) -> str | None:
    """Текст ошибки для клиента или None, если форма заполнена верно."""
    if not consent:
        return "Нужно согласие на обработку персональных данных"
    if len(full_name) < 2 or not re.search(r"[^\W\d_]", full_name):
        return "Укажите ФИО"
    digits = re.sub(r"\D", "", phone)
    if not PHONE_CHARS_RE.match(phone) or not 10 <= len(digits) <= 11:
        return "Проверьте номер телефона: нужно 10–11 цифр"
    if not company:
        return "Укажите название компании"
    if len(full_name) > MAX_FIELD_CHARS or len(company) > MAX_FIELD_CHARS:
        return f"Слишком длинное значение: до {MAX_FIELD_CHARS} символов в поле"
    return None


# ---------- когда предложить заявку ----------

# Клиент сам просит связаться или спрашивает то, что считает только специалист
CLIENT_ASKS_RE = re.compile(
    r"свяж|позвон|перезвон|менеджер|специалист|заявк|оставить (?:контакт|номер|телефон)"
    r"|коммерческ\w* предложен|сколько (?:это )?(?:стоит|будет стоить)|стоимост|расч[её]т|рассчита",
    re.I,
)
# Клиент отказывается от заявки или звонка — не настаиваем (демо-вопрос №18)
CLIENT_REFUSES_RE = re.compile(
    r"\bне (?:хочу|хотим|буду|будем|готов\w*|надо|нужно|стану)\b|\bбез (?:заявк|звонк|контакт|менеджер)|\bпока не\b",
    re.I,
)
# Бот в ответе сам отправляет к человеку
BOT_HANDS_OFF_RE = re.compile(r"специалист|менеджер|заявк|предварительн\w+ оценк|расч[её]т", re.I)


def should_offer_lead(user_text: str, answer: str, user_turns: int) -> bool:
    """Пора ли показать клиенту предложение оставить заявку (поле offer_lead в ответе чата).

    Простое правило по тексту, без лишнего вызова модели: клиент отказывается — не предлагаем;
    просит связаться или посчитать — предлагаем сразу; бот сам отправляет к специалисту —
    предлагаем со второй реплики клиента, чтобы не навязывать форму в ответ на первый же вопрос.
    """
    if CLIENT_REFUSES_RE.search(user_text):
        return False
    if CLIENT_ASKS_RE.search(user_text):
        return True
    return user_turns >= 2 and bool(BOT_HANDS_OFF_RE.search(answer))


# ---------- резюме для менеджера ----------

SECTIONS = ["Задача", "Что выяснили", "Подходящие решения", "Открытые вопросы", "Следующий шаг"]
NOT_DISCUSSED = "не обсуждалось"
MAX_TRANSCRIPT_CHARS = 12000

SUMMARY_PROMPT = """Ты помогаешь менеджеру МТС EnergyTool подготовиться к звонку клиенту. Ниже — диалог клиента \
с ИИ-консультантом на сайте. Составь по нему краткое резюме.

Правила:
1. Используй только то, что прозвучало в диалоге. Ничего не добавляй от себя: ни цен, ни сроков, ни процентов \
экономии, ни характеристик продуктов.
2. Если по разделу в диалоге ничего нет — напиши «не обсуждалось».
3. Каждый раздел — одна строка, 1–2 коротких предложения. Без вступления и заключения.

Формат ответа — ровно пять строк:
**Задача:** что клиент хочет решить, его словами
**Что выяснили:** тип объекта, масштаб, как ведётся учёт, ценовая категория, мощность — только то, что прозвучало
**Подходящие решения:** какие продукты EnergyTool назвал консультант и почему
**Открытые вопросы:** что консультант спрашивал, а клиент не ответил; что осталось неясным
**Следующий шаг:** что менеджеру сделать дальше — расчёт, демо, пилот или уточнить задачу"""

# Заголовок раздела в начале строки: допускаем «**», «-», «1.» перед ним и «**» после двоеточия
SECTION_RE = re.compile(
    r"^[\s*#\-\d.)]*(" + "|".join(re.escape(s) for s in SECTIONS) + r")[\s*]*:[\s*]*(.*)$",
    re.I,
)


def _strip_think(text: str) -> str:
    """Убрать рассуждения модели, как это делает ядро в clean_answer."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.split("<think>", 1)[0]


def parse_sections(text: str) -> dict[str, str]:
    """Ответ модели → {раздел: текст}. Строки без заголовка раздела продолжают предыдущий раздел."""
    found: dict[str, str] = {}
    current = None
    for line in _strip_think(text).splitlines():
        line = line.strip()
        m = SECTION_RE.match(line)
        if m:
            current = next(s for s in SECTIONS if s.lower() == m.group(1).lower())
            found[current] = m.group(2).strip().strip("*").strip()
        elif current and line:
            found[current] = (found[current] + " " + line.strip("*").strip()).strip()
    return {k: v for k, v in found.items() if v}


def _transcript(messages: list[dict]) -> str:
    lines = [("Клиент: " if m["role"] == "user" else "Консультант: ") + m["content"] for m in messages]
    return "\n".join(lines)[-MAX_TRANSCRIPT_CHARS:]


def _template_sections(messages: list[dict]) -> dict[str, str]:
    """Резюме без модели: первая реплика клиента — задача, остальные — что выяснили."""
    asks = [" ".join(m["content"].split()) for m in messages if m["role"] == "user"]
    return {
        "Задача": asks[0] if asks else "клиент сразу оставил контакты, задачу не описал",
        "Что выяснили": "; ".join(asks[1:]) or NOT_DISCUSSED,
        "Подходящие решения": "см. диалог" if asks else NOT_DISCUSSED,
        "Открытые вопросы": "уточнить при звонке",
        "Следующий шаг": "позвонить клиенту и уточнить задачу",
    }


def make_summary(messages: list[dict], full_name: str, phone: str, company: str) -> str:
    """Резюме заявки со строкой о клиенте и пятью разделами.

    messages — история сессии [{"role", "content"}]. Контакты в модель не отправляем:
    строку «Клиент» собираем сами. Если модель не ответила — резюме из реплик клиента.
    """
    sections: dict[str, str] = {}
    if any(m["role"] == "user" for m in messages):
        prompt = [{"role": "system", "content": SUMMARY_PROMPT},
                  {"role": "user", "content": "Диалог:\n" + _transcript(messages)}]
        for attempt in range(2):
            if attempt == 1:  # модель могла уйти в рассуждения и не дойти до ответа — просим без них
                prompt = prompt[:-1] + [{**prompt[-1], "content": prompt[-1]["content"] + " /no_think"}]
            try:
                sections = parse_sections(core.complete(prompt))
            except Exception as e:
                log.warning("Резюме заявки: модель не ответила (%s), собираем по шаблону", type(e).__name__)
                break
            if sections:
                break
    fallback = _template_sections(messages)
    lines = [f"**Клиент:** {full_name}, {company}, {phone}"]
    for name in SECTIONS:
        # модель ответила частично — недостающие разделы помечаем, а не подменяем шаблоном
        lines.append(f"**{name}:** {sections.get(name, NOT_DISCUSSED) if sections else fallback[name]}")
    return "\n".join(lines)
