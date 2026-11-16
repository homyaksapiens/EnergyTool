"""Хранение сессий, сообщений и заявок в SQLite.

Каждая операция открывает своё соединение: эндпоинты работают в разных потоках,
а соединение sqlite3 между потоками передавать нельзя.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from . import config

LEAD_STATUSES = ("new", "in_progress", "done", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    client_phone TEXT,             -- номер вошедшего клиента; у гостя пусто; наружу не отдаётся
    visitor_id   TEXT              -- анонимная метка браузера из cookie; наружу не отдаётся
);
CREATE TABLE IF NOT EXISTS messages (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role              TEXT NOT NULL,   -- user | assistant
    text              TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    debug             TEXT,            -- JSON, только у ответов бота
    client_message_id TEXT             -- метка реплики клиента от фронта, может быть пустой
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS leads (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,  -- одна заявка на диалог
    created_at TEXT NOT NULL,
    full_name  TEXT NOT NULL,
    phone      TEXT NOT NULL,
    company    TEXT NOT NULL,
    summary    TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'new'
);
CREATE INDEX IF NOT EXISTS leads_by_status ON leads(status, id);
CREATE INDEX IF NOT EXISTS sessions_by_updated ON sessions(updated_at, id);
"""


def now() -> str:
    """Текущее время в UTC, ISO 8601. Одинаковый формат позволяет сравнивать даты как строки."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@contextmanager
def connect():
    """Соединение на одну операцию: commit при успехе, rollback при ошибке, в конце закрывается."""
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # без этого ON DELETE CASCADE не работает
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init() -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.execute("PRAGMA journal_mode = WAL")  # читатели не ждут пишущего
        c.executescript(SCHEMA)
        # База, созданная до появления метки браузера: добавляем колонку, старые диалоги остаются без метки
        if "visitor_id" not in {row["name"] for row in c.execute("PRAGMA table_info(sessions)")}:
            c.execute("ALTER TABLE sessions ADD COLUMN visitor_id TEXT")
        c.execute("CREATE INDEX IF NOT EXISTS sessions_by_visitor ON sessions(visitor_id)")
        if "account_id" not in {row["name"] for row in c.execute("PRAGMA table_info(sessions)")}:
            c.execute("ALTER TABLE sessions ADD COLUMN account_id TEXT")
        c.execute("CREATE INDEX IF NOT EXISTS sessions_by_account ON sessions(account_id)")


# ---------- сессии ----------

def create_session(client_phone: str | None = None, visitor_id: str | None = None, account_id: str | None = None) -> str:
    session_id = uuid.uuid4().hex
    t = now()
    with connect() as c:
        c.execute("INSERT INTO sessions (id, created_at, updated_at, client_phone, visitor_id, account_id) VALUES (?,?,?,?,?,?)",
                  (session_id, t, t, client_phone or None, visitor_id or None, account_id))
    return session_id


def session_exists(session_id: str) -> bool:
    with connect() as c:
        return c.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is not None


def get_session(session_id: str) -> dict | None:
    """Сессия целиком, как в GET /api/sessions/{id}; None — такой сессии нет."""
    with connect() as c:
        s = c.execute("SELECT id, created_at, updated_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if s is None:
            return None
        lead = c.execute("SELECT id FROM leads WHERE session_id = ?", (session_id,)).fetchone()
        rows = c.execute("SELECT role, text, created_at, debug, client_message_id FROM messages "
                         "WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
    return {
        "session_id": s["id"],
        "created_at": s["created_at"],
        "updated_at": s["updated_at"],
        "lead_id": lead["id"] if lead else None,
        "messages": [_message(r) for r in rows],
    }


def _message(row: sqlite3.Row) -> dict:
    m = {"role": row["role"], "text": row["text"], "created_at": row["created_at"]}
    if row["debug"]:
        m["debug"] = json.loads(row["debug"])
    if row["client_message_id"]:
        m["client_message_id"] = row["client_message_id"]
    return m


def list_sessions() -> list[dict]:
    """Все сессии с хотя бы одним сообщением, новые сверху — для панели менеджера."""
    return _list_sessions("", ())


def list_client_sessions(client_phone: str | None, visitor_id: str | None, account_id: str | None = None) -> list[dict]:
    """Диалоги одного клиента: по номеру вошедшего и по метке его браузера. Нет ни того, ни другого — пусто.

    Отдельная функция, а не параметры list_sessions: пустой фильтр не должен случайно отдать чужие диалоги.
    """
    if account_id:
        return _list_sessions("WHERE s.account_id = ?", (account_id,))
    conditions, params = [], []
    if client_phone:
        conditions.append("s.client_phone = ?")
        params.append(client_phone)
    if visitor_id:
        conditions.append("s.visitor_id = ?")
        params.append(visitor_id)
    if not conditions:
        return []
    return _list_sessions("WHERE " + " OR ".join(conditions), tuple(params))


def _list_sessions(where: str, params: tuple) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            f"""SELECT s.id, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS messages,
                       (SELECT l.id FROM leads l WHERE l.session_id = s.id) AS lead_id,
                       (SELECT m.text FROM messages m WHERE m.session_id = s.id AND m.role = 'user'
                         ORDER BY m.id LIMIT 1) AS preview
                FROM sessions s {where}
                ORDER BY s.updated_at DESC, s.rowid DESC""",
            params,
        ).fetchall()
    return [
        {"session_id": r["id"], "created_at": r["created_at"], "updated_at": r["updated_at"],
         "messages": r["messages"], "lead_id": r["lead_id"], "preview": (r["preview"] or "")[:200]}
        for r in rows if r["messages"]  # пустые «Новый диалог» без реплик не показываем
    ]


# ---------- сообщения ----------

def add_message(session_id: str, role: str, text: str, debug: dict | None = None,
                client_message_id: str | None = None) -> None:
    t = now()
    with connect() as c:
        c.execute(
            "INSERT INTO messages (session_id, role, text, created_at, debug, client_message_id) VALUES (?,?,?,?,?,?)",
            (session_id, role, text, t, json.dumps(debug, ensure_ascii=False) if debug is not None else None,
             client_message_id or None),
        )
        c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (t, session_id))


# ---------- заявки ----------

def create_lead(session_id: str, full_name: str, phone: str, company: str, summary: str) -> int | None:
    """Номер новой заявки; None — по этой сессии заявка уже есть (или сессию успели удалить)."""
    t = now()
    try:
        with connect() as c:
            cur = c.execute(
                "INSERT INTO leads (session_id, created_at, full_name, phone, company, summary) VALUES (?,?,?,?,?,?)",
                (session_id, t, full_name, phone, company, summary),
            )
            c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (t, session_id))
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def _lead(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["lead_id"] = d.pop("id")
    return d


def get_lead_by_session(session_id: str) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM leads WHERE session_id = ?", (session_id,)).fetchone()
    return _lead(row) if row else None


def list_leads() -> list[dict]:
    with connect() as c:
        return [_lead(r) for r in c.execute("SELECT * FROM leads ORDER BY id DESC").fetchall()]


def set_lead_status(lead_id: int, status: str) -> bool:
    """False — заявки с таким номером нет."""
    with connect() as c:
        return c.execute("UPDATE leads SET status = ? WHERE id = ?", (status, lead_id)).rowcount > 0


# ---------- срок хранения ----------

def purge_old(days: int | None = None) -> int:
    """Удалить сессии, в которых не было сообщений дольше срока хранения, вместе с их сообщениями и заявками."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days or config.RETENTION_DAYS)).isoformat(timespec="milliseconds")
    with connect() as c:
        return c.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,)).rowcount
