"""HTTP API бэкенда и раздача фронтенда. Пути и поля — CONTRACTS.md, раздел 4.

Запуск из папки backend/:  uvicorn app.main:app --port 8000
"""
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager, contextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Журнал настраиваем до импорта ядра: оно пишет в него уже при загрузке (выбор модели)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # иначе в лог попадает адрес Telegram с токеном бота

from . import config, core, db, leads  # noqa: E402
from .notify import notify_manager  # noqa: E402

log = logging.getLogger(__name__)

MODEL_DOWN = "Модель сейчас не отвечает. Попробуйте ещё раз через минуту."


# ---------- старт сервера ----------

def _purge_daily() -> None:
    """Раз в сутки удаляет сессии старше срока хранения (CONTRACTS.md, раздел 5)."""
    while True:
        time.sleep(24 * 3600)
        try:
            removed = db.purge_old()
            if removed:
                log.info("Срок хранения: удалено сессий — %d", removed)
        except Exception:
            log.exception("Не удалось удалить старые сессии")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    db.purge_old()
    core.start()
    threading.Thread(target=_purge_daily, name="retention", daemon=True).start()
    yield


app = FastAPI(title="EnergyTool — ИИ-консультант", lifespan=lifespan)


# ---------- ошибки: всегда {"detail": "текст для человека"} ----------

@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    fields = ", ".join(sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")}))
    return JSONResponse({"detail": f"Проверьте поля запроса: {fields or 'тело запроса'}"}, status_code=422)


@app.exception_handler(Exception)
async def server_error(request: Request, exc: Exception):
    log.exception("Ошибка при обработке %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "Внутренняя ошибка сервера. Попробуйте ещё раз."}, status_code=500)


# ---------- кто делает запрос ----------

def current_phone(request: Request) -> str | None:
    """Номер вошедшего клиента или None для гостя.

    Единственное место, которое знает, как бэкенд узнаёт клиента. Сейчас входа нет (жюри в Docker — гость);
    DEMO_CLIENT_PHONE включает показ «Ваших диалогов». При интеграции с сайтом здесь появится
    чтение cookie или токена (CONTRACTS.md, раздел 8).
    """
    return config.DEMO_CLIENT_PHONE or None


basic = HTTPBasic(auto_error=False)


def require_manager(credentials: HTTPBasicCredentials | None = Depends(basic)) -> None:
    """Доступ к панели менеджера. Пароль не задан — открыто (локальная разработка и демо)."""
    if not config.MANAGER_PASSWORD:
        return
    ok = credentials is not None and secrets.compare_digest(
        credentials.username.encode(), config.MANAGER_USER.encode()
    ) and secrets.compare_digest(credentials.password.encode(), config.MANAGER_PASSWORD.encode())
    if not ok:
        # по этому заголовку браузер сам покажет окно входа и запомнит логин для следующих запросов
        raise HTTPException(401, "Нужен вход менеджера", {"WWW-Authenticate": 'Basic realm="EnergyTool manager"'})


# ---------- диалог ----------

class ChatIn(BaseModel):
    message: str = ""
    session_id: str | None = None
    client_message_id: str | None = None


class SessionLocks:
    """Замок на каждую сессию: два запроса в один диалог обрабатываются по очереди.

    Без него повтор вопроса, пришедший, пока первый ещё считается, породил бы второй ответ.
    Замок живёт, только пока его кто-то держит или ждёт, — словарь не растёт.
    """

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[str, list] = {}  # id сессии → [замок, сколько запросов его держат или ждут]

    @contextmanager
    def hold(self, session_id: str):
        with self._guard:
            entry = self._locks.setdefault(session_id, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._guard:
                entry[1] -= 1
                if entry[1] == 0:
                    self._locks.pop(session_id, None)


session_locks = SessionLocks()


def model_history(stored: list[dict]) -> list[dict]:
    """Сообщения сессии из базы → история для ядра: [{"role", "content"}], последняя реплика — клиента."""
    history: list[dict] = []
    for m in stored:
        if m["role"] == "assistant" and (m.get("debug") or {}).get("fallback"):
            continue  # запасная фраза — не ответ модели, в контекст её не берём
        if history and history[-1]["role"] == m["role"]:
            history[-1]["content"] += "\n" + m["text"]  # две реплики клиента подряд — после сбоя ответа
        else:
            history.append({"role": m["role"], "content": m["text"]})
    history = history[-config.HISTORY_MESSAGES:]
    while history and history[0]["role"] != "user":
        history.pop(0)
    return history


@app.post("/api/sessions", status_code=201)
def new_session(request: Request):
    """Кнопка «Новый диалог»: пустая сессия без истории. Старая остаётся в базе."""
    return {"session_id": db.create_session(current_phone(request))}


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    """Реплика клиента → ответ консультанта. Обычный def: вызов модели долгий и идёт в пуле потоков."""
    text = body.message.strip()
    if not text:
        raise HTTPException(422, "Пустое сообщение")
    if len(text) > config.MAX_MESSAGE_CHARS:
        raise HTTPException(422, f"Сообщение слишком длинное: до {config.MAX_MESSAGE_CHARS} символов")
    client_id = (body.client_message_id or "").strip()[:100] or None

    session_id = body.session_id
    if session_id:
        if not db.session_exists(session_id):
            raise HTTPException(404, "Сессия не найдена")
    else:
        session_id = db.create_session(current_phone(request))

    with session_locks.hold(session_id):
        session = db.get_session(session_id)
        if session is None:  # сессию удалили, пока ждали очереди
            raise HTTPException(404, "Сессия не найдена")
        stored = session["messages"]
        has_lead = session["lead_id"] is not None

        # «Повторить вопрос» после ошибки приходит с той же меткой — второй раз реплику не записываем
        last_user = max((i for i, m in enumerate(stored) if m["role"] == "user"), default=None)
        is_repeat = bool(client_id) and last_user is not None and stored[last_user].get("client_message_id") == client_id
        if is_repeat:
            answered = next((m for m in stored[last_user + 1:] if m["role"] == "assistant"), None)
            if answered:  # ответ успели записать, но клиент его не дождался — отдаём тот же
                user_turns = sum(m["role"] == "user" for m in stored)
                return _chat_response(session_id, answered["text"], answered.get("debug") or {},
                                      stored[last_user]["text"], user_turns, has_lead)
        else:
            db.add_message(session_id, "user", text, client_message_id=client_id)
            stored = stored + [{"role": "user", "text": text}]

        t0 = time.time()
        try:
            result = core.answer(model_history(stored))
        except Exception:
            log.exception("Ядро не вернуло ответ")
            raise HTTPException(502, MODEL_DOWN) from None
        debug = {
            "elapsed_ms": round((time.time() - t0) * 1000),
            "timings": result["timings"],
            "fallback": result["fallback"],
            "reranked": result["reranked"],
            "sources": result["sources"],
        }
        db.add_message(session_id, "assistant", result["text"], debug=debug)
        user_turns = sum(m["role"] == "user" for m in stored)
        return _chat_response(session_id, result["text"], debug, text, user_turns, has_lead)


def _chat_response(session_id: str, answer: str, debug: dict, user_text: str, user_turns: int, has_lead: bool) -> dict:
    offer = not has_lead and not debug.get("fallback") and leads.should_offer_lead(user_text, answer, user_turns)
    return {"session_id": session_id, "answer": answer, "offer_lead": offer, "debug": debug}


# ---------- история сессий ----------

@app.get("/api/sessions", dependencies=[Depends(require_manager)])
def all_sessions():
    """Все сессии, в том числе без заявки — для панели менеджера и проверки жюри."""
    return db.list_sessions()


@app.get("/api/me")
def me(request: Request):
    phone = current_phone(request)
    return {"authenticated": bool(phone), "phone": phone}


@app.get("/api/me/sessions")
def my_sessions(request: Request):
    """Диалоги вошедшего клиента для колонки «Ваши диалоги». Гостю — пустой список."""
    phone = current_phone(request)
    return db.list_sessions(client_phone=phone) if phone else []


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    """Сессия целиком. Открыта по id: клиент возвращается к своему диалогу, а id нельзя подобрать."""
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Сессия не найдена")
    for m in session["messages"]:
        m.pop("client_message_id", None)  # служебная метка, фронту не нужна
    return session


# ---------- заявки ----------

class LeadIn(BaseModel):
    session_id: str = ""
    full_name: str = ""
    phone: str = ""
    company: str = ""
    consent: bool = False


class StatusIn(BaseModel):
    status: str = ""


@app.post("/api/leads", status_code=201)
def create_lead(body: LeadIn, background: BackgroundTasks):
    """Клиент оставил контакты: проверяем форму, пишем резюме диалога и сохраняем заявку."""
    full_name, phone, company = (" ".join(v.split()) for v in (body.full_name, body.phone, body.company))
    error = leads.validate_lead(full_name, phone, company, body.consent)
    if error:
        raise HTTPException(422, error)

    with session_locks.hold(body.session_id):
        session = db.get_session(body.session_id)
        if session is None:
            raise HTTPException(404, "Сессия не найдена")
        existing = db.get_lead_by_session(body.session_id)
        if existing:  # повторная отправка формы — второй заявки по диалогу не создаём
            return JSONResponse({"lead_id": existing["lead_id"], "summary": existing["summary"]})

        history = [{"role": m["role"], "content": m["text"]} for m in session["messages"]
                   if not (m.get("debug") or {}).get("fallback")]
        summary = leads.make_summary(history, full_name, phone, company)
        lead_id = db.create_lead(body.session_id, full_name, phone, company, summary)
        if lead_id is None:
            raise HTTPException(404, "Сессия не найдена")

    background.add_task(notify_manager, lead_id, summary)  # после ответа клиенту: Telegram может отвечать долго
    return {"lead_id": lead_id, "summary": summary}


@app.get("/api/leads", dependencies=[Depends(require_manager)])
def all_leads():
    return db.list_leads()


@app.patch("/api/leads/{lead_id}", dependencies=[Depends(require_manager)])
def update_lead(lead_id: int, body: StatusIn):
    if body.status not in db.LEAD_STATUSES:
        raise HTTPException(422, "Неизвестный статус заявки")
    if not db.set_lead_status(lead_id, body.status):
        raise HTTPException(404, "Заявка не найдена")
    return {"ok": True}


# ---------- служебное ----------

@app.get("/api/health")
def health():
    return {"ok": True, **core.info()}


# ---------- фронтенд ----------

@app.get("/js/mock.js", include_in_schema=False)
def no_mock():
    """Вместо заглушки API отдаём пустой скрипт: api.js не находит MockServer и ходит в настоящий бэкенд.

    Сам файл frontend/js/mock.js остаётся для работы над интерфейсом без бэкенда.
    """
    return Response("/* заглушка отключена: страницу раздаёт бэкенд */\n",
                    media_type="application/javascript", headers={"Cache-Control": "no-store"})


@app.get("/manager", include_in_schema=False, dependencies=[Depends(require_manager)])
@app.get("/manager.html", include_in_schema=False, dependencies=[Depends(require_manager)])
def manager_page():
    page = config.FRONTEND_DIR / "manager.html"
    if not page.is_file():
        raise HTTPException(404, "Панель менеджера не найдена")
    return FileResponse(page)


if config.FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
else:
    log.warning("Папка фронтенда не найдена: %s — работает только API", config.FRONTEND_DIR)


if __name__ == "__main__":  # python -m app.main — порт из PORT, адрес из HOST
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "").strip() or "127.0.0.1", port=config.PORT)
