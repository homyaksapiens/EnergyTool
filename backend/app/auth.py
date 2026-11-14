"""Быстрый демонстрационный вход. Для интеграции с сайтом заменить выдачу сессии.

Выбор аккаунта открыт намеренно: это имитация входа, а не проверка личности.
Роли и принадлежность диалогов проверяет сервер.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, Response

from . import config, db

COOKIE = "et_account"
ACCOUNTS = [
    {"id": "user-1", "role": "user", "full_name": "Петров Пётр Петрович", "phone": "+7 900 000-00-01", "company": "ООО «Ромашка»"},
    {"id": "user-2", "role": "user", "full_name": "Иванова Анна Сергеевна", "phone": "+7 900 000-00-02", "company": "СНТ «Пример»"},
    {"id": "admin-1", "role": "admin", "full_name": "Смирнова Мария Андреевна", "phone": None, "company": "EnergyTool"},
    {"id": "admin-2", "role": "admin", "full_name": "Орлов Иван Алексеевич", "phone": None, "company": "EnergyTool"},
]


def init():
    with db.connect() as c:
        c.execute("CREATE TABLE IF NOT EXISTS auth_sessions (token TEXT PRIMARY KEY, account_id TEXT NOT NULL, expires_at TEXT NOT NULL)")
        c.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (db.now(),))


def account(request: Request):
    if not config.DEMO_AUTH:
        return None
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    with db.connect() as c:
        row = c.execute("SELECT account_id FROM auth_sessions WHERE token = ? AND expires_at > ?",
                        (hashlib.sha256(token.encode()).hexdigest(), db.now())).fetchone()
    return next((a for a in ACCOUNTS if row and a["id"] == row["account_id"]), None)


def require_account(request: Request):
    user = account(request)
    if config.DEMO_AUTH and not user:
        raise HTTPException(401, "Войдите в аккаунт")
    return user


def require_client(request: Request):
    user = require_account(request)
    if user and user["role"] != "user":
        raise HTTPException(403, "Для консультации выберите аккаунт пользователя")
    return user


def check_session(request: Request, session_id: str):
    user = require_account(request)
    if not user or user["role"] == "admin":
        return
    with db.connect() as c:
        row = c.execute("SELECT account_id FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row or row["account_id"] != user["id"]:
        raise HTTPException(404, "Сессия не найдена")


def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE)
    if token:
        with db.connect() as c:
            c.execute("DELETE FROM auth_sessions WHERE token = ?", (hashlib.sha256(token.encode()).hexdigest(),))
    response.delete_cookie(COOKIE)


def login(request: Request, response: Response, account_id: str):
    if not config.DEMO_AUTH:
        raise HTTPException(404, "Демонстрационный вход отключён")
    user = next((a for a in ACCOUNTS if a["id"] == account_id), None)
    if not user:
        raise HTTPException(422, "Выберите тестовый аккаунт")
    logout(request, response)
    token = secrets.token_urlsafe(32)
    expiry = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(timespec="milliseconds")
    with db.connect() as c:
        c.execute("INSERT INTO auth_sessions VALUES (?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), user["id"], expiry))
    response.set_cookie(COOKIE, token, max_age=86400, httponly=True, samesite="strict", secure=request.url.scheme == "https")
    return {"authenticated": True, **user}
