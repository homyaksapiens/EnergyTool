"""Локальный прогон bot/solution.py на демо-вопросах и отчёт рядом с эталонными ответами.

По умолчанию — LM Studio на localhost:1234 (модель определяется сама). Если заданы LLM_BASE_URL,
LLM_API_KEY, LLM_MODEL, EMB_MODEL — используются они.

    .venv\\Scripts\\python eval\\run_local.py [--limit 5] [--workers 2]

Результат — папка runs/<время>/: submission.csv, debug.jsonl, run.log, report.md.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def detect_models(base_url: str, key: str) -> tuple[str | None, str | None]:
    request = urllib.request.Request(base_url + "/models", headers={"Authorization": f"Bearer {key}"})
    ids = [m["id"] for m in json.load(urllib.request.urlopen(request, timeout=5))["data"]]
    llm = next((i for i in ids if "embed" not in i.lower() and "bge" not in i.lower()), None)
    emb = next((i for i in ids if "bge-m3" in i.lower()), None)
    return llm, emb


def read_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return {row[0]: row[1] for row in reader}


def words(text: str) -> int:
    return len(text.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="сколько вопросов прогнать (0 — все)")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("LLM_BASE_URL", "http://localhost:1234/v1")
    env.setdefault("LLM_API_KEY", "lm-studio")
    if not env.get("LLM_MODEL"):
        llm, emb = detect_models(env["LLM_BASE_URL"], env["LLM_API_KEY"])
        if not llm:
            sys.exit("В LM Studio не загружена языковая модель")
        env["LLM_MODEL"] = llm
        if emb and not env.get("EMB_MODEL"):
            env["EMB_MODEL"] = emb
    if not env.get("EMB_MODEL"):
        env.setdefault("ET_USE_DENSE", "0")  # bge-m3 локально нет — поиск только BM25
    env.update({"DATA_DIR": "data", "ET_DEBUG": "1", "MAX_WORKERS": str(args.workers), "PYTHONIOENCODING": "utf-8"})

    run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    (run_dir / "data").mkdir(parents=True)
    questions = read_csv(DATA / "demo_questions.csv")
    if args.limit:
        questions = dict(list(questions.items())[:args.limit])
    with (run_dir / "data" / "questions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "question"])
        writer.writerows(questions.items())
    shutil.copy(DATA / "knowledge-base.zip", run_dir / "data" / "knowledge-base.zip")

    print(f"Модель: {env['LLM_MODEL']}, эмбеддинги: {env.get('EMB_MODEL', 'нет')}, вопросов: {len(questions)}")
    t0 = time.time()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(ROOT / "bot" / "solution.py")], cwd=run_dir, env=env,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        for line in proc.stderr:
            print(line, end="")
            log.write(line)
        proc.wait()
    total = time.time() - t0

    golden = read_csv(DATA / "demo_output.csv")
    ours = read_csv(run_dir / "submission.csv")
    debug = {}
    if (run_dir / "debug.jsonl").exists():
        for line in (run_dir / "debug.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            debug[item["id"]] = item

    seconds = [d["seconds"] for d in debug.values()]
    fallbacks = sum(1 for qid in questions if debug.get(qid, {}).get("fallback", True))
    summary = (
        f"Модель: {env['LLM_MODEL']} · эмбеддинги: {env.get('EMB_MODEL', 'нет')} · вопросов: {len(questions)}\n"
        f"Общее время: {total:.0f} с · на вопрос: {sum(seconds) / max(len(seconds), 1):.1f} с "
        f"(макс. {max(seconds, default=0):.1f} с) · запасных ответов: {fallbacks}\n"
        f"Средняя длина: наша {sum(words(a) for a in ours.values()) / max(len(ours), 1):.0f} слов, "
        f"эталон {sum(words(golden[q]) for q in questions if q in golden) / max(len(questions), 1):.0f} слов"
    )
    lines = [f"# Прогон {run_dir.name}", "", summary.replace("\n", "  \n"), ""]
    for qid, question in questions.items():
        d = debug.get(qid, {})
        answer = ours.get(qid, "")
        lines += [
            f"## {qid}",
            "**Вопрос:**  ",
            question.replace("\n", "  \n"),
            "",
            f"**Эталон ({words(golden.get(qid, ''))} сл.):** {golden.get(qid, '—')}",
            "",
            f"**Ответ ({words(answer)} сл., {d.get('seconds', '?')} с):** {answer}",
            "",
            "**Источники:** " + "; ".join(s["title"] for s in d.get("sources", [])),
            "",
        ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + summary)
    print(f"Отчёт: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
