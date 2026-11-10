"""Локальный LLM-судья: попарное сравнение нового прогона с бейзлайном относительно эталона (2 раунда).

Формула платформы: Score = 0.7 * MatchRate + 0.3 * WinRate.

Запуск (сравнит два последних прогона в папке runs/ или явно указанные папки/csv):
    .venv\\Scripts\\python eval\\judge.py [путь_к_baseline] [путь_к_candidate]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))

from eval.run_local import DATA, detect_models, read_csv  # noqa: E402
import httpx  # noqa: E402

JUDGE_SYSTEM = """Ты — строгий и объективный судья соревнования ИИ-консультантов МТС EnergyTool.
Твоя задача — оценить два ответа (Ответ A и Ответ B) на вопрос клиента, опираясь ТОЛЬКО на Эталонный ответ.

Правила оценки:
1. Фактическая точность: все цифры, пороги мощности, сроки, условия и ограничения из Эталона должны быть переданы верно. Выдумывание фактов, цен или советов из общих знаний — грубая ошибка.
2. Поведение консультанта: прямой ответ в начале, краткость (обычно 30–80 слов, связный текст без списков и markdown), в конце — уместный уточняющий вопрос или предложение расчёта специалистом (как в Эталоне).
3. Оцени соответствие Эталону отдельно для A (match_a) и для B (match_b) по шкале:
   - 1.0: все ключевые факты и смысл Эталона переданы верно, нет выдумок;
   - 0.5: ответ частично верный, но упущена важная деталь/оговорка из Эталона или есть лишняя вода;
   - 0.0: ответ неверный, противоречит Эталону или содержит галлюцинации.
4. Выбери победителя (winner): "A", "B" или "tie" (ничья, если оба ответа одинаковы по качеству).

Верни ТОЛЬКО валидный JSON в одну строку без пояснений:
{"match_a": 1.0, "match_b": 0.5, "winner": "A", "reason": "кратко почему"} /no_think"""

JSON_RE = re.compile(r"\{.*?\}", re.S)


def resolve_csv(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        return p / "submission.csv"
    return p


def find_default_runs() -> tuple[Path, Path]:
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        sys.exit("Папка runs/ не найдена. Сначала выполните прогон eval/run_local.py")
    subdirs = sorted([d for d in runs_dir.iterdir() if (d / "submission.csv").is_file()])
    if len(subdirs) < 2:
        sys.exit(f"В папке runs/ найдено прогонов: {len(subdirs)}. Для сравнения нужно минимум 2 (или укажите пути явно).")
    return subdirs[0] / "submission.csv", subdirs[-1] / "submission.csv"


def call_judge(client: httpx.Client, base_url: str, key: str, model: str,
               question: str, golden: str, ans_a: str, ans_b: str) -> dict:
    user_prompt = (
        f"Вопрос клиента:\n{question}\n\n"
        f"Эталонный ответ:\n{golden}\n\n"
        f"Ответ A:\n{ans_a}\n\n"
        f"Ответ B:\n{ans_b}"
    )
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = client.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    m = JSON_RE.search(content)
    if not m:
        return {"match_a": 0.5, "match_b": 0.5, "winner": "tie", "reason": "не удалось разобрать JSON"}
    try:
        data = json.loads(m.group(0))
        return {
            "match_a": float(data.get("match_a", 0.5)),
            "match_b": float(data.get("match_b", 0.5)),
            "winner": str(data.get("winner", "tie")).strip().upper(),
            "reason": str(data.get("reason", "")),
        }
    except (ValueError, TypeError):
        return {"match_a": 0.5, "match_b": 0.5, "winner": "tie", "reason": "ошибка JSON"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Попарный LLM-судья (2 раунда)")
    parser.add_argument("baseline", nargs="?", help="папка или submission.csv бейзлайна")
    parser.add_argument("candidate", nargs="?", help="папка или submission.csv нового решения")
    args = parser.parse_args()

    if args.baseline and args.candidate:
        base_path, cand_path = resolve_csv(args.baseline), resolve_csv(args.candidate)
    else:
        base_path, cand_path = find_default_runs()

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "lm-studio")
    model = os.getenv("LLM_MODEL", "")
    if not model:
        model, _ = detect_models(base_url, api_key)
        if not model:
            sys.exit("В LM Studio не загружена языковая модель")

    questions = read_csv(DATA / "demo_questions.csv")
    golden = read_csv(DATA / "demo_output.csv")
    baseline = read_csv(base_path)
    candidate = read_csv(cand_path)

    qids = [qid for qid in questions if qid in candidate and qid in baseline and qid in golden]
    print(f"Судья: {model} | Вопросов: {len(qids)}")
    print(f"Baseline : {base_path}")
    print(f"Candidate: {cand_path}\n")

    client = httpx.Client(timeout=120.0, trust_env=False)
    results = []

    for qid in qids:
        q, g = questions[qid], golden[qid]
        b_ans, c_ans = baseline[qid], candidate[qid]

        # Раунд 1: A = Candidate, B = Baseline
        r1 = call_judge(client, base_url, api_key, model, q, g, c_ans, b_ans)
        # Раунд 2: A = Baseline, B = Candidate (меняем местами)
        r2 = call_judge(client, base_url, api_key, model, q, g, b_ans, c_ans)

        # MatchRate кандидата (среднее за 2 раунда)
        cand_match = (r1["match_a"] + r2["match_b"]) / 2.0
        base_match = (r1["match_b"] + r2["match_a"]) / 2.0

        # WinRate кандидата за 2 раунда
        w1 = 1.0 if r1["winner"] == "A" else (0.0 if r1["winner"] == "B" else 0.5)
        w2 = 1.0 if r2["winner"] == "B" else (0.0 if r2["winner"] == "A" else 0.5)
        win_rate = (w1 + w2) / 2.0

        score = 0.7 * cand_match + 0.3 * win_rate
        results.append({
            "id": qid,
            "score": score,
            "cand_match": cand_match,
            "base_match": base_match,
            "win_rate": win_rate,
            "reason": r1["reason"] or r2["reason"],
        })
        print(f"ID {qid:>2}: Score={score:.2f} (Match: {cand_match:.2f} vs {base_match:.2f}, WinRate: {win_rate:.2f}) | {results[-1]['reason']}")

    avg_match = sum(r["cand_match"] for r in results) / max(len(results), 1)
    avg_base_match = sum(r["base_match"] for r in results) / max(len(results), 1)
    avg_win = sum(r["win_rate"] for r in results) / max(len(results), 1)
    total_score = 0.7 * avg_match + 0.3 * avg_win

    summary = (
        f"\n================ ИТОГ СУДЬИ ================\n"
        f"Candidate MatchRate : {avg_match:.3f} (Baseline: {avg_base_match:.3f})\n"
        f"Candidate WinRate   : {avg_win:.3f} (>0.500 означает победу над бейзлайном)\n"
        f"ИТОГОВЫЙ SCORE      : {total_score:.3f} (0.7*Match + 0.3*Win)\n"
        f"============================================"
    )
    print(summary)

    # Если кандидат лежит в папке прогона runs/<время>/ — сохраняем туда же judge.md
    if cand_path.parent != ROOT:
        report_lines = [
            f"# Оценка судьи: {cand_path.parent.name} против {base_path.parent.name}",
            "",
            f"- **Итоговый Score:** `{total_score:.3f}`",
            f"- **MatchRate:** `{avg_match:.3f}` (у бейзлайна `{avg_base_match:.3f}`)",
            f"- **WinRate:** `{avg_win:.3f}`",
            "",
            "| ID | Score | Match (Cand / Base) | WinRate | Комментарий судьи |",
            "|---:|---:|---:|---:|:---|",
        ]
        for r in results:
            report_lines.append(
                f"| {r['id']} | **{r['score']:.2f}** | {r['cand_match']:.2f} / {r['base_match']:.2f} | "
                f"{r['win_rate']:.2f} | {r['reason']} |"
            )
        out_md = cand_path.parent / "judge.md"
        out_md.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Отчёт сохранён в: {out_md}")


if __name__ == "__main__":
    main()