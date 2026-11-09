import logging

import httpx

from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger(__name__)

TEMP_ICON = {"hot": "🔥", "warm": "🌤", "cold": "❄️"}


def notify_manager(lead_id: int, lead: dict) -> bool:
    """Шлёт карточку лида в Telegram, если настроен бот. Иначе просто пишет в лог."""
    text = (
        f"{TEMP_ICON.get(lead.get('lead_temperature'), '📩')} Новый лид #{lead_id} с сайта EnergyTool\n\n"
        f"👤 {lead.get('name') or '—'} · {lead.get('company') or '—'}\n"
        f"📞 {lead.get('phone') or '—'}   ✉️ {lead.get('email') or '—'}\n\n"
        f"📝 {lead.get('summary') or ''}\n\n"
        f"💡 Решения: {', '.join(lead.get('recommended_solutions') or []) or '—'}\n"
        f"➡️ Следующий шаг: {lead.get('next_step') or '—'}"
    )
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log.info("Telegram не настроен, лид:\n%s", text)
        return False
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except httpx.HTTPError as e:
        log.warning("Не удалось отправить в Telegram: %s", e)
        return False
