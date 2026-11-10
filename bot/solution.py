"""Точка входа автосабмита: $DATA_DIR/questions.csv + knowledge-base.zip → submission.csv.

Файл с запасными ответами пишется сразу и обновляется по ходу работы, чтобы сбой или тайм-аут
не оставили платформу без submission.csv.

Главный файл для отправки на платформу хакатона. Он читает тестовые вопросы, прогоняет их через Consultant
и сохраняет ответы в таблицу submission.csv.
"""

from __future__ import annotations

import time

START = time.time()

import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from concurrent.futures import TimeoutError as FuturesTimeout  # noqa: E402
from pathlib import Path  # noqa: E402

import llm_client  # noqa: E402
from consultant import Consultant, parse_dialog, search_queries  # noqa: E402
from kb import KnowledgeBase, load_documents  # noqa: E402
from prompts import FALLBACK_ANSWER  # noqa: E402
from settings import DATA_DIR, DEBUG, EMB_MODEL, LLM_MODEL, MAX_WORKERS, TIME_BUDGET, USE_DENSE  # noqa: E402

OUT = Path("submission.csv")
MAX_ANSWER_CHARS = 9500  # лимит платформы — 10 000


def log(message: str) -> None:
    print(f"[{time.time() - START:6.1f}s] {message}", file=sys.stderr, flush=True)


def find_file(*names: str) -> Path:
    here = Path(__file__).resolve().parent
    for root in (Path(DATA_DIR), here / DATA_DIR, Path.cwd(), here):
        for name in names:
            path = root / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"не найдены файлы {names} (DATA_DIR={DATA_DIR})")


def write_submission(ids: list[str], answers: dict[str, str]) -> None:
    tmp = OUT.with_name(OUT.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for qid in ids:
            writer.writerow([qid, (answers.get(qid) or FALLBACK_ANSWER)[:MAX_ANSWER_CHARS]])
    os.replace(tmp, OUT)


def embed_queries(kb: KnowledgeBase, ids: list[str], dialogs: dict) -> dict:
    """Эмбеддинги фрагментов базы и всех запросов — пачками заранее. При ошибке поиск идёт только по BM25."""
    if not USE_DENSE:
        return {}
    try:
        kb.enable_dense(llm_client.embed)
        texts = sorted({q for qid in ids for q in search_queries(dialogs[qid]) if q.strip()})
        vectors = dict(zip(texts, llm_client.embed(texts)))
        return {qid: [vectors.get(q) for q in search_queries(dialogs[qid])] for qid in ids}
    except Exception as e:  # noqa: BLE001 — без эмбеддингов работаем дальше на BM25
        kb.vectors = None
        log(f"Эмбеддинги недоступны ({type(e).__name__}: {e}); поиск только BM25")
        return {}


def main() -> None:
    sys.stderr.reconfigure(errors="replace")
    deadline = START + TIME_BUDGET
    questions_path = find_file("questions.csv", "train.csv")
    with questions_path.open(encoding="utf-8-sig", newline="") as f:
        rows = [(str(r["id"]).strip(), str(r.get("question") or "")) for r in csv.DictReader(f)]
    ids = list(dict.fromkeys(qid for qid, _ in rows))
    questions = dict(rows)
    answers: dict[str, str] = {}
    write_submission(ids, answers)  # страховка: файл есть, даже если дальше что-то упадёт

    kb = KnowledgeBase(load_documents(find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")))
    dialogs = {qid: parse_dialog(questions[qid]) for qid in ids}
    log(f"Вопросов: {len(ids)}, фрагментов базы: {len(kb.chunks)}, модель: {LLM_MODEL}")
    query_vectors = embed_queries(kb, ids, dialogs)
    log(f"Эмбеддинги ({EMB_MODEL}): {'да' if kb.vectors is not None else 'нет'}")

    consultant = Consultant(kb)
    debug: list[dict] = []

    def work(qid: str) -> tuple[str, dict | None]:
        if time.time() > deadline:
            return qid, None
        t = time.time()
        try:
            result = consultant.answer(dialogs[qid], query_vectors.get(qid), deadline)
        except Exception as e:  # noqa: BLE001 — один упавший вопрос не должен ронять прогон
            log(f"Вопрос {qid}: {type(e).__name__}: {e}")
            return qid, None
        result["seconds"] = round(time.time() - t, 2)
        return qid, result

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = [pool.submit(work, qid) for qid in ids]
    done = 0
    try:
        for future in as_completed(futures, timeout=max(1.0, deadline + 60 - time.time())):
            qid, result = future.result()
            done += 1
            if result:
                if not result["fallback"]:
                    answers[qid] = result["text"]
                debug.append({"id": qid, "question": questions[qid], **result})
            if done % 20 == 0:
                write_submission(ids, answers)
                log(f"Готово {done}/{len(ids)}")
    except FuturesTimeout:
        log("Время вышло: оставшиеся вопросы получат запасной ответ")

    write_submission(ids, answers)
    fallbacks = len(ids) - len(answers)
    reranked = sum(1 for d in debug if d.get("reranked"))
    log(f"Готов {OUT}: {len(ids)} ответов, запасных: {fallbacks}, с реранкером: {reranked}")
    if DEBUG:
        with open("debug.jsonl", "w", encoding="utf-8") as f:
            for item in debug:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if any(not f.done() for f in futures):
        os._exit(0)  # не ждём зависшие запросы: файл уже записан
    pool.shutdown()


if __name__ == "__main__":
    main()
