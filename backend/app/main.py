import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .agent import SESSIONS, get_session, run_turn
from .config import FRONTEND_DIR, LLM_MODEL, LLM_PROVIDER
from .tools import kb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="EnergyTool AI Consultant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
db.init_db()


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


class StatusIn(BaseModel):
    status: str


@app.get("/api/health")
def health():
    return {"ok": True, "provider": LLM_PROVIDER, "model": LLM_MODEL, "kb_chunks": len(kb.chunks)}


@app.post("/api/chat")
def chat(body: ChatIn):
    if not body.message.strip():
        raise HTTPException(400, "Пустое сообщение")
    session = get_session(body.session_id)

    def sse():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session.id})}\n\n"
        for event in run_turn(session, body.message.strip()[:4000]):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/sessions/{session_id}")
def session_info(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404)
    return {"profile": s.profile, "transcript": s.transcript, "lead_id": s.lead_id}


@app.get("/api/search")
def search(q: str):
    """Отладка RAG: что находит база знаний по запросу."""
    return kb.search(q)


@app.get("/api/leads")
def leads():
    return db.list_leads()


@app.patch("/api/leads/{lead_id}")
def update_lead(lead_id: int, body: StatusIn):
    db.set_status(lead_id, body.status)
    return {"ok": True}


@app.get("/manager")
def manager_page():
    return FileResponse(FRONTEND_DIR / "manager.html")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
