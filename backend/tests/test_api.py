"""Проверка HTTP API по CONTRACTS.md, раздел 4. Модель и база знаний подменены, сеть не нужна.

Запуск из папки backend/:  ..\\.venv\\Scripts\\python -m unittest discover tests -v
Все персональные данные в тестах выдуманы.
"""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Окружение задаём до импорта приложения: настройки и ядро читают его при импорте.
# Пустые значения нужны, чтобы .env разработчика не подмешал свои (load_dotenv не трогает заданные переменные)
TMP = tempfile.TemporaryDirectory()
ROOT = Path(TMP.name)
(ROOT / "kb").mkdir()
(ROOT / "kb" / "test.md").write_text("# Тестовая база\n\n## Раздел\n\nВыдуманный текст для проверки поиска.\n", encoding="utf-8")
(ROOT / "frontend" / "js").mkdir(parents=True)
(ROOT / "frontend" / "index.html").write_text("<!doctype html><title>сайт</title>", encoding="utf-8")
(ROOT / "frontend" / "manager.html").write_text("<!doctype html><title>менеджер</title>", encoding="utf-8")
(ROOT / "frontend" / "js" / "mock.js").write_text("const MockServer = {};", encoding="utf-8")
os.environ.update({
    "KB_PATH": str(ROOT / "kb"), "DB_PATH": str(ROOT / "test.db"), "FRONTEND_DIR": str(ROOT / "frontend"),
    "LLM_BASE_URL": "", "OPENAI_BASE_URL": "", "ET_USE_DENSE": "0",
    "MANAGER_PASSWORD": "", "DEMO_CLIENT_PHONE": "", "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "",
})

from fastapi.testclient import TestClient  # noqa: E402

from app import config, core, db, leads, main  # noqa: E402

SOURCES = [{"source": "test.md", "title": "Тестовая база → Раздел"}]
SUMMARY = ("**Задача:** снизить счёт за электроэнергию\n**Что выяснили:** три магазина\n"
           "**Подходящие решения:** платформа учёта\n**Открытые вопросы:** модели счётчиков\n"
           "**Следующий шаг:** демо")
LEAD = {"full_name": "Иванов Иван Иванович", "phone": "+7 900 000-00-00", "company": "ООО «Пример»", "consent": True}


def fake_answer(text="Да, поможем. Сколько у вас объектов?", fallback=False):
    """Подмена core.answer: запоминает историю, которую получила, и возвращает заданный ответ."""
    def answer(messages):
        answer.calls.append([dict(m) for m in messages])
        return {"text": text, "fallback": fallback, "reranked": False, "sources": SOURCES,
                "timings": {"retrieval": 0.01, "llm": 0.02}}
    answer.calls = []
    return answer


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.client.__enter__()  # запускает старт сервера: база, ядро

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        TMP.cleanup()

    def setUp(self):
        self.answer = fake_answer()
        for target, replacement in ((core, {"answer": self.answer}), (core, {"complete": lambda messages: SUMMARY})):
            patcher = mock.patch.multiple(target, **replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def ask(self, message="Счёт за свет вырос, что делать?", **extra):
        return self.client.post("/api/chat", json={"message": message, **extra})

    # ---------- диалог ----------

    def test_health(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["ok"], True)
        self.assertEqual(body["kb_chunks"], 1)
        self.assertIs(body["dense"], False)
        self.assertIn("model", body)

    def test_chat_creates_session_and_stores_history(self):
        r = self.ask(client_message_id="m1")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["answer"], "Да, поможем. Сколько у вас объектов?")
        self.assertIs(body["offer_lead"], False)
        self.assertEqual(body["debug"]["sources"], SOURCES)
        self.assertEqual(body["debug"]["timings"], {"retrieval": 0.01, "llm": 0.02})
        self.assertIs(body["debug"]["fallback"], False)
        self.assertIsInstance(body["debug"]["elapsed_ms"], int)

        session = self.client.get(f"/api/sessions/{body['session_id']}").json()
        self.assertEqual([m["role"] for m in session["messages"]], ["user", "assistant"])
        self.assertIsNone(session["lead_id"])
        self.assertNotIn("debug", session["messages"][0])
        self.assertEqual(session["messages"][1]["debug"]["sources"], SOURCES)
        self.assertNotIn("client_message_id", session["messages"][0])
        self.assertNotIn("client_phone", session)

    def test_core_gets_whole_dialog(self):
        sid = self.ask("Первый вопрос").json()["session_id"]
        self.ask("Три магазина", session_id=sid)
        self.assertEqual(self.answer.calls[-1], [
            {"role": "user", "content": "Первый вопрос"},
            {"role": "assistant", "content": "Да, поможем. Сколько у вас объектов?"},
            {"role": "user", "content": "Три магазина"},
        ])

    def test_new_dialog_has_no_old_context(self):
        self.ask("Старый вопрос")
        r = self.client.post("/api/sessions")
        self.assertEqual(r.status_code, 201)
        self.ask("Новый вопрос", session_id=r.json()["session_id"])
        self.assertEqual(self.answer.calls[-1], [{"role": "user", "content": "Новый вопрос"}])

    def test_fallback_answer_is_not_passed_to_model(self):
        with mock.patch.object(core, "answer", fake_answer("Запасная фраза", fallback=True)):
            body = self.ask("Первый вопрос").json()
        self.assertIs(body["debug"]["fallback"], True)
        self.assertIs(body["offer_lead"], False)
        self.ask("Второй вопрос", session_id=body["session_id"])
        self.assertEqual(self.answer.calls[-1], [{"role": "user", "content": "Первый вопрос\nВторой вопрос"}])

    def test_bad_requests(self):
        for payload, status in (({"message": "   "}, 422), ({"message": "я" * 4001}, 422),
                                ({"message": "Вопрос", "session_id": "net-takoy"}, 404), ({"message": 5}, 422)):
            r = self.client.post("/api/chat", json=payload)
            self.assertEqual(r.status_code, status, payload)
            self.assertIsInstance(r.json()["detail"], str)
        self.assertEqual(self.client.get("/api/sessions/net-takoy").status_code, 404)

    def test_repeat_after_error_does_not_duplicate_question(self):
        sid = self.client.post("/api/sessions").json()["session_id"]
        with mock.patch.object(core, "answer", side_effect=RuntimeError("модель упала")):
            r = self.ask(session_id=sid, client_message_id="m1")
        self.assertEqual(r.status_code, 502)
        self.assertIsInstance(r.json()["detail"], str)

        r = self.ask(session_id=sid, client_message_id="m1")  # кнопка «Повторить вопрос»
        self.assertEqual(r.status_code, 200)
        roles = [m["role"] for m in self.client.get(f"/api/sessions/{sid}").json()["messages"]]
        self.assertEqual(roles, ["user", "assistant"])

        # ответ уже записан, а клиент повторил ещё раз — отдаём тот же ответ, модель не зовём
        calls = len(self.answer.calls)
        again = self.ask(session_id=sid, client_message_id="m1")
        self.assertEqual(again.json()["answer"], r.json()["answer"])
        self.assertEqual(len(self.answer.calls), calls)
        self.assertEqual(len(self.client.get(f"/api/sessions/{sid}").json()["messages"]), 2)

    def test_same_text_without_label_is_a_new_question(self):
        sid = self.ask("Да").json()["session_id"]
        self.ask("Да", session_id=sid)
        self.assertEqual(len(self.client.get(f"/api/sessions/{sid}").json()["messages"]), 4)

    # ---------- предложение заявки ----------

    def test_offer_lead_rule(self):
        self.assertTrue(leads.should_offer_lead("Пусть мне перезвонит менеджер", "Хорошо.", 1))
        self.assertTrue(leads.should_offer_lead("Сколько это стоит?", "Зависит от объекта.", 1))
        self.assertFalse(leads.should_offer_lead("Счёт вырос", "Расчёт подготовит специалист.", 1))
        self.assertTrue(leads.should_offer_lead("Пять магазинов", "Расчёт подготовит специалист.", 2))
        self.assertFalse(leads.should_offer_lead("Пять магазинов", "Какие у вас счётчики?", 3))
        # клиент отказался — не предлагаем, даже если бот упомянул специалиста
        self.assertFalse(leads.should_offer_lead("Не хочу оставлять заявку, просто расскажите", "Расчёт подготовит специалист.", 3))
        self.assertFalse(leads.should_offer_lead("Давайте пока без звонков и менеджеров", "Хорошо.", 2))
        self.assertFalse(leads.should_offer_lead("Телефон оставлять не буду", "Точный расчёт сделает менеджер.", 4))

    def test_offer_lead_stops_after_lead(self):
        body = self.ask("Свяжите меня с менеджером").json()
        self.assertIs(body["offer_lead"], True)
        self.client.post("/api/leads", json={**LEAD, "session_id": body["session_id"]})
        again = self.ask("Свяжите меня с менеджером", session_id=body["session_id"]).json()
        self.assertIs(again["offer_lead"], False)

    # ---------- заявки ----------

    def test_lead_main_path(self):
        sid = self.ask().json()["session_id"]
        r = self.client.post("/api/leads", json={**LEAD, "session_id": sid})
        self.assertEqual(r.status_code, 201)
        lead_id, summary = r.json()["lead_id"], r.json()["summary"]
        self.assertTrue(summary.startswith("**Клиент:** Иванов Иван Иванович, ООО «Пример», +7 900 000-00-00\n"))
        self.assertIn("**Задача:** снизить счёт за электроэнергию", summary)
        self.assertIn("**Следующий шаг:** демо", summary)

        # заявку видит менеджер, сессия помнит её номер
        lead = next(x for x in self.client.get("/api/leads").json() if x["lead_id"] == lead_id)
        self.assertEqual((lead["session_id"], lead["status"], lead["company"]), (sid, "new", "ООО «Пример»"))
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").json()["lead_id"], lead_id)
        listed = next(s for s in self.client.get("/api/sessions").json() if s["session_id"] == sid)
        self.assertEqual((listed["lead_id"], listed["messages"]), (lead_id, 2))
        self.assertEqual(listed["preview"], "Счёт за свет вырос, что делать?")

        # повторная отправка формы не создаёт вторую заявку
        again = self.client.post("/api/leads", json={**LEAD, "session_id": sid})
        self.assertEqual(again.json()["lead_id"], lead_id)

        # статус
        self.assertEqual(self.client.patch(f"/api/leads/{lead_id}", json={"status": "in_progress"}).json(), {"ok": True})
        lead = next(x for x in self.client.get("/api/leads").json() if x["lead_id"] == lead_id)
        self.assertEqual(lead["status"], "in_progress")
        self.assertEqual(self.client.patch(f"/api/leads/{lead_id}", json={"status": "unknown"}).status_code, 422)
        self.assertEqual(self.client.patch("/api/leads/999999", json={"status": "done"}).status_code, 404)

    def test_lead_validation(self):
        sid = self.client.post("/api/sessions").json()["session_id"]
        cases = [({"consent": False}, "согласие"), ({"phone": "12345"}, "телефон"), ({"phone": "звоните вечером"}, "телефон"),
                 ({"full_name": " "}, "ФИО"), ({"company": ""}, "компани")]
        for change, word in cases:
            r = self.client.post("/api/leads", json={**LEAD, "session_id": sid, **change})
            self.assertEqual(r.status_code, 422, change)
            self.assertIn(word, r.json()["detail"])
        self.assertEqual(self.client.post("/api/leads", json={**LEAD, "session_id": "net-takoy"}).status_code, 404)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").json()["lead_id"], None)

    def test_lead_summary_without_model(self):
        sid = self.ask("Нужен учёт по пяти магазинам").json()["session_id"]
        with mock.patch.object(core, "complete", side_effect=RuntimeError("модель упала")):
            r = self.client.post("/api/leads", json={**LEAD, "session_id": sid})
        self.assertEqual(r.status_code, 201)
        self.assertIn("**Задача:** Нужен учёт по пяти магазинам", r.json()["summary"])
        self.assertIn("**Следующий шаг:**", r.json()["summary"])

    def test_lead_without_dialog(self):
        sid = self.client.post("/api/sessions").json()["session_id"]
        with mock.patch.object(core, "complete", side_effect=AssertionError("модель звать не нужно")):
            r = self.client.post("/api/leads", json={**LEAD, "session_id": sid})
        self.assertEqual(r.status_code, 201)
        self.assertIn("задачу не описал", r.json()["summary"])

    def test_summary_parsing(self):
        raw = "<think>думаю</think>Вот резюме:\n1. **Задача**: учёт\n- Что выяснили: офис\nи склад\n**Следующий шаг:** «демо»"
        self.assertEqual(leads.parse_sections(raw), {"Задача": "учёт", "Что выяснили": "офис и склад", "Следующий шаг": "«демо»"})
        with mock.patch.object(core, "complete", lambda messages: raw):
            summary = leads.make_summary([{"role": "user", "content": "вопрос"}], "А Б", "+7 900 000-00-00", "ООО")
        self.assertIn("**Подходящие решения:** не обсуждалось", summary)
        self.assertIn("**Задача:** учёт", summary)

    def test_summary_prompt_has_no_contacts(self):
        seen = []
        with mock.patch.object(core, "complete", lambda messages: seen.append(messages) or SUMMARY):
            leads.make_summary([{"role": "user", "content": "вопрос"}], "Иванов Иван", "+7 900 000-00-00", "ООО «Пример»")
        sent = " ".join(m["content"] for m in seen[0])
        self.assertNotIn("Иванов", sent)
        self.assertNotIn("900", sent)

    # ---------- история и вход клиента ----------

    def test_empty_sessions_are_hidden(self):
        sid = self.client.post("/api/sessions").json()["session_id"]
        self.assertNotIn(sid, [s["session_id"] for s in self.client.get("/api/sessions").json()])

    def test_sessions_newest_first(self):
        first = self.ask("Первый").json()["session_id"]
        second = self.ask("Второй").json()["session_id"]
        ids = [s["session_id"] for s in self.client.get("/api/sessions").json()]
        self.assertLess(ids.index(second), ids.index(first))
        self.ask("Ещё вопрос", session_id=first)  # новое сообщение поднимает диалог наверх
        ids = [s["session_id"] for s in self.client.get("/api/sessions").json()]
        self.assertLess(ids.index(first), ids.index(second))

    def test_guest_sees_dialogs_of_own_browser_only(self):
        # Два «браузера»: у каждого свой набор cookie
        anna, boris = TestClient(main.app), TestClient(main.app)
        self.assertEqual(anna.get("/api/me").json(), {"authenticated": False, "phone": None})
        self.assertEqual(anna.get("/api/me/sessions").json(), [])  # метки ещё нет — список пуст
        self.assertNotIn("et_visitor", anna.cookies)               # и без диалога метка не выдаётся

        r = anna.post("/api/chat", json={"message": "Вопрос Анны"})
        cookie = r.headers["set-cookie"]
        self.assertIn("et_visitor=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=lax", cookie)
        first = r.json()["session_id"]
        second = anna.post("/api/sessions").json()["session_id"]  # «Новый диалог» — та же метка
        anna.post("/api/chat", json={"message": "Второй вопрос Анны", "session_id": second})
        other = boris.post("/api/chat", json={"message": "Вопрос Бориса"}).json()["session_id"]

        mine = anna.get("/api/me/sessions").json()
        self.assertEqual([s["session_id"] for s in mine], [second, first])  # новые сверху
        self.assertEqual(mine[1]["preview"], "Вопрос Анны")
        self.assertNotIn("visitor_id", mine[0])
        self.assertEqual([s["session_id"] for s in boris.get("/api/me/sessions").json()], [other])
        self.assertNotIn("visitor_id", anna.get(f"/api/sessions/{first}").json())

    def test_broken_visitor_cookie_is_ignored(self):
        self.ask("Чей-то диалог")
        stranger = TestClient(main.app, cookies={"et_visitor": "' OR 1=1 --"})
        self.assertEqual(stranger.get("/api/me/sessions").json(), [])

    def test_signed_in_client(self):
        guest, client, other = TestClient(main.app), TestClient(main.app), TestClient(main.app)
        guest_sid = guest.post("/api/chat", json={"message": "Вопрос гостя"}).json()["session_id"]
        with mock.patch.object(config, "DEMO_CLIENT_PHONE", "+7 900 000-00-01"):
            self.assertEqual(client.get("/api/me").json(), {"authenticated": True, "phone": "+7 900 000-00-01"})
            own_sid = client.post("/api/chat", json={"message": "Вопрос клиента"}).json()["session_id"]
            mine = [s["session_id"] for s in client.get("/api/me/sessions").json()]
            # тот же номер с другого устройства — диалоги по номеру видны и там
            elsewhere = [s["session_id"] for s in other.get("/api/me/sessions").json()]
        self.assertIn(own_sid, mine)
        self.assertNotIn(guest_sid, mine)
        self.assertIn(own_sid, elsewhere)

    def test_old_database_gets_visitor_column(self):
        old = Path(TMP.name) / "old.db"
        with sqlite3.connect(old) as c:
            c.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, client_phone TEXT)")
            c.execute("INSERT INTO sessions VALUES ('staraya', '2026-11-11T10:00:00.000+00:00', '2099-01-01T00:00:00.000+00:00', NULL)")
        c.close()
        with mock.patch.object(config, "DB_PATH", old):
            db.init()
            db.init()  # повторный старт ничего не ломает
            sid = db.create_session(visitor_id="metka-brauzera-000000000")
            db.add_message(sid, "user", "Вопрос")
            self.assertEqual([s["session_id"] for s in db.list_client_sessions(None, "metka-brauzera-000000000")], [sid])
            self.assertTrue(db.session_exists("staraya"))

    def test_manager_password(self):
        sid = self.ask().json()["session_id"]
        with mock.patch.object(config, "MANAGER_PASSWORD", "sekret"):
            for path in ("/api/leads", "/api/sessions", "/manager.html", "/manager"):
                r = self.client.get(path)
                self.assertEqual(r.status_code, 401, path)
                self.assertIn("Basic", r.headers["www-authenticate"])
                self.assertEqual(self.client.get(path, auth=("manager", "ne-tot")).status_code, 401, path)
                self.assertEqual(self.client.get(path, auth=("manager", "sekret")).status_code, 200, path)
            self.assertEqual(self.client.patch("/api/leads/1", json={"status": "done"}).status_code, 401)
            # чат клиента паролем не закрыт
            self.assertEqual(self.ask(session_id=sid).status_code, 200)
            self.assertEqual(self.client.get(f"/api/sessions/{sid}").status_code, 200)
            self.assertEqual(self.client.get("/api/me/sessions").status_code, 200)
        self.assertEqual(self.client.get("/manager.html").status_code, 200)

    # ---------- срок хранения ----------

    def test_old_sessions_are_purged(self):
        old = self.ask("Старый диалог").json()["session_id"]
        self.client.post("/api/leads", json={**LEAD, "session_id": old})
        fresh = self.ask("Свежий диалог").json()["session_id"]
        with sqlite3.connect(config.DB_PATH) as c:
            c.execute("UPDATE sessions SET updated_at = '2020-01-01T00:00:00.000+00:00' WHERE id = ?", (old,))
        c.close()
        self.assertEqual(db.purge_old(), 1)
        self.assertEqual(self.client.get(f"/api/sessions/{old}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/sessions/{fresh}").status_code, 200)
        self.assertNotIn(old, [x["session_id"] for x in self.client.get("/api/leads").json()])
        with sqlite3.connect(config.DB_PATH) as c:
            left = c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (old,)).fetchone()[0]
        c.close()
        self.assertEqual(left, 0)

    # ---------- фронтенд ----------

    def test_frontend_is_served_without_mock(self):
        self.assertIn("сайт", self.client.get("/").text)
        mock_js = self.client.get("/js/mock.js")
        self.assertEqual(mock_js.status_code, 200)
        self.assertNotIn("MockServer", mock_js.text)


if __name__ == "__main__":
    unittest.main()
