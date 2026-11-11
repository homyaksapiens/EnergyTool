"""Уведомление менеджеру о новой заявке в Telegram. Необязательно: без токена ничего не отправляется."""
import logging

import httpx

from . import config

log = logging.getLogger(__name__)


def notify_manager(lead_id: int, summary: str) -> bool:
    """Шлёт резюме заявки в Telegram, если заданы TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID.

    Ошибка отправки не мешает заявке: она уже сохранена и видна в панели менеджера.
    Контакты клиента в лог не пишем.
    """
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return False
    text = f"Новая заявка №{lead_id} с сайта EnergyTool\n\n" + summary.replace("**", "")
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text[:4000]},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except httpx.HTTPError as e:
        # только тип ошибки: в тексте исключения есть адрес запроса с токеном бота
        log.warning("Заявка №%s: не удалось отправить в Telegram (%s)", lead_id, type(e).__name__)
        return False
