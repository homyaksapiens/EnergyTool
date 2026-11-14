"""Выборки панели менеджера. Поиск, фильтры и страницы выполняются в SQLite."""
import re
from datetime import datetime, timedelta, timezone

from . import db


def _conditions(q: str, days: int, sessions: bool) -> tuple[list[str], list]:
    clauses, values = [], []
    if sessions:
        clauses.append("EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)")
    if days:
        clauses.append(f"{'s.updated_at' if sessions else 'l.created_at'} >= ?")
        values.append((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="milliseconds"))
    q = q.strip().casefold()
    if q:
        # instr ищет буквально: % и _ не становятся шаблонами. casefold поддерживает кириллицу.
        fields = ["l.full_name", "l.company", "l.phone", "l.summary", "CAST(l.id AS TEXT)"]
        if sessions:
            fields.append("s.id")
        search = [f"instr(casefold(COALESCE({field}, '')), ?) > 0" for field in fields]
        values.extend([q] * len(fields))
        digits = re.sub(r"\D", "", q)
        if len(digits) >= 4 and re.fullmatch(r"[+\d\s()\-]+", q):
            search.append("instr(phone_digits(l.phone), ?) > 0")
            values.append(digits)
        search.append("EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id AND instr(casefold(m.text), ?) > 0)")
        values.append(q)
        clauses.append("(" + " OR ".join(search) + ")")
    return clauses, values


def _prepare(c):
    c.create_function("casefold", 1, lambda text: (text or "").casefold(), deterministic=True)
    c.create_function("phone_digits", 1, lambda text: re.sub(r"\D", "", text or ""), deterministic=True)
    c.execute("BEGIN")  # счётчики и строки одной страницы читаем из одного снимка


def _task(summary: str, preview: str) -> str:
    match = re.search(r"(?:\*\*)?Задача:(?:\*\*)?\s*([^\n]+)", summary, re.IGNORECASE)
    return (match.group(1).replace("**", "").strip() if match else preview or "Задача не указана")[:200]


def leads_page(q="", status="all", days=0, sort="newest", limit=50, offset=0):
    clauses, values = _conditions(q, days, False)
    base = " FROM leads l JOIN sessions s ON s.id = l.session_id"
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with db.connect() as c:
        _prepare(c)
        counts = {s: 0 for s in db.LEAD_STATUSES}
        counts.update({r["status"]: r["n"] for r in c.execute("SELECT l.status, COUNT(*) AS n" + base + where + " GROUP BY l.status", values)})
        counts["all"] = sum(counts.values())
        counts["active"] = counts["new"] + counts["in_progress"]
        if status == "active":
            clauses.append("l.status IN ('new', 'in_progress')")
        elif status != "all":
            clauses.append("l.status = ?")
            values.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        order = "l.id ASC" if sort == "oldest" else "s.updated_at DESC, l.id DESC" if sort == "activity" else "l.id DESC"
        total = counts[status]
        rows = c.execute("""SELECT l.id AS lead_id, l.session_id, l.created_at, l.full_name, l.company,
                            l.status, s.updated_at, substr(l.summary, 1, 1600) AS summary,
                            (SELECT m.text FROM messages m WHERE m.session_id = s.id AND m.role = 'user' ORDER BY m.id LIMIT 1) AS preview"""
                         + base + where + " ORDER BY " + order + " LIMIT ? OFFSET ?", [*values, limit, offset]).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["task"] = _task(item.pop("summary"), item.pop("preview"))
        items.append(item)
    return dict(items=items, total=total, counts=counts, limit=limit, offset=offset)


def sessions_page(q="", has_lead="all", days=0, sort="newest", limit=50, offset=0):
    clauses, values = _conditions(q, days, True)
    if has_lead != "all":
        clauses.append("l.id IS NOT NULL" if has_lead == "yes" else "l.id IS NULL")
    base = " FROM sessions s LEFT JOIN leads l ON l.session_id = s.id WHERE " + " AND ".join(clauses)
    order = "s.updated_at ASC, s.rowid ASC" if sort == "oldest" else "s.updated_at DESC, s.rowid DESC"
    with db.connect() as c:
        _prepare(c)
        total = c.execute("SELECT COUNT(*)" + base, values).fetchone()[0]
        rows = c.execute("""SELECT s.id AS session_id, s.created_at, s.updated_at, l.id AS lead_id,
                            l.full_name, l.company,
                            (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS messages,
                            (SELECT substr(m.text, 1, 200) FROM messages m WHERE m.session_id = s.id AND m.role = 'user' ORDER BY m.id LIMIT 1) AS preview"""
                         + base + " ORDER BY " + order + " LIMIT ? OFFSET ?", [*values, limit, offset]).fetchall()
    return dict(items=[dict(r) for r in rows], total=total, limit=limit, offset=offset)


def lead_detail(lead_id: int):
    with db.connect() as c:
        row = c.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
    return db._lead(row) if row else None
