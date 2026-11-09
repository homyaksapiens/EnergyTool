import json
import sqlite3
from datetime import datetime, timezone

from .config import DB_PATH

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_id TEXT NOT NULL,
                name TEXT, company TEXT, phone TEXT, email TEXT,
                temperature TEXT,
                summary TEXT,
                solutions TEXT,      -- JSON-список
                next_step TEXT,
                profile TEXT,        -- JSON
                transcript TEXT,     -- JSON-список {role, text}
                status TEXT DEFAULT 'new'
            )"""
        )


def save_lead(session_id: str, lead: dict, profile: dict, transcript: list[dict]) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO leads (created_at, session_id, name, company, phone, email, temperature,
                                  summary, solutions, next_step, profile, transcript)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                session_id,
                lead.get("name"), lead.get("company"), lead.get("phone"), lead.get("email"),
                lead.get("lead_temperature"),
                lead.get("summary"),
                json.dumps(lead.get("recommended_solutions", []), ensure_ascii=False),
                lead.get("next_step"),
                json.dumps(profile, ensure_ascii=False),
                json.dumps(transcript, ensure_ascii=False),
            ),
        )
        return cur.lastrowid


def list_leads() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM leads ORDER BY id DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("solutions", "profile", "transcript"):
            d[k] = json.loads(d[k] or "null")
        out.append(d)
    return out


def set_status(lead_id: int, status: str):
    with _conn() as c:
        c.execute("UPDATE leads SET status=? WHERE id=?", (status, lead_id))
