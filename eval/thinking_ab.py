"""Reproduce the controlled thinking comparison on EnergyTool commit 3b607ea."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import subprocess
import sys
import time
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "3b607ea64e4db4eed48bba50dae518772b12f0d1"
MODES = ("off", "on")
MODEL = "mts-qwen3-4b"
EMBEDDER = "text-embedding-bge-m3"
JUDGE = "huihui-qwen3-vl-8b-instruct-abliterated"
REASONING_BUDGET = 128
MAX_TOKENS = 3072
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0}

sys.path.insert(0, str(ROOT / "bot"))
import consultant
import llm_client
from kb import KnowledgeBase, load_documents


def post(client, url, body):
    response = client.post(url, json=body)
    response.raise_for_status()
    return response.json()


def read_csv(path, column):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["id"]: row[column] for row in csv.DictReader(stream)}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def chat(client, generator, messages, mode, limit=MAX_TOKENS, temperature=0.6):
    started = time.perf_counter()
    data = post(client, f"{generator}/v1/chat/completions", {
        "model": MODEL,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": mode == "on"},
        **SAMPLING,
        "temperature": temperature,
        "max_tokens": limit,
        "seed": 42,
        "cache_prompt": False,
    })
    choice = data["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = data["usage"]
    tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    if reasoning and not tokens:
        tokenized = post(client, f"{generator}/tokenize", {"content": reasoning})
        tokens = len(tokenized["tokens"])
    return {
        "answer": consultant.clean_answer(message.get("content") or ""),
        "seconds": round(time.perf_counter() - started, 3),
        "reasoning_chars": len(reasoning),
        "stats": {"total_output_tokens": usage["completion_tokens"], "reasoning_output_tokens": tokens},
        "hit_output_limit": choice["finish_reason"] == "length",
        "model_instance_id": data.get("model"),
    }


def verify_mode(mode, result):
    tokens = result["stats"]["reasoning_output_tokens"]
    if mode == "off" and (tokens or result["reasoning_chars"]):
        raise RuntimeError("The server emitted reasoning in the off variant.")
    if mode == "on" and not 0 < tokens <= REASONING_BUDGET:
        raise RuntimeError(f"Expected 1–{REASONING_BUDGET} reasoning tokens, got {tokens}.")


def prepare(client, args, questions, golden):
    def summarize(messages, temperature, max_tokens):
        result = chat(client, args.generator, messages, "off", max_tokens, temperature)
        return result["answer"], None

    def embed_request(endpoint, key, body, attempts=3):
        return post(client, args.embeddings + "/v1" + endpoint, {**body, "model": EMBEDDER})

    index = KnowledgeBase(load_documents(args.data / "knowledge-base.zip"))
    bot = consultant.Consultant(index, use_rerank=False)
    prepared = []
    with patch.object(llm_client, "chat", summarize), patch.object(llm_client, "post", embed_request):
        index.enable_dense(llm_client.embed)
        for qid, question in questions.items():
            messages = consultant.parse_dialog(question)
            queries = consultant.search_queries(messages)
            vectors = llm_client.embed([query[:2000] for query in queries])
            chunks, _, _, _ = bot._retrieve_detailed(messages, vectors)
            convo = [{"role": "system", "content": consultant.build_system(chunks)}]
            convo.extend(consultant.format_dialog_sandwich(messages))
            prepared.append({
                "id": qid,
                "question": question,
                "golden": golden[qid],
                "messages": convo,
                "context_sha256": hashlib.sha256(json.dumps(convo, ensure_ascii=False).encode()).hexdigest(),
                "sources": [chunk.id for chunk in chunks],
            })
    return prepared


def generate(args):
    if subprocess.run(["git", "diff", "--quiet", BASE, "--", "bot"], cwd=ROOT).returncode:
        raise SystemExit(f"bot/ must match the original snapshot {BASE}.")
    questions = read_csv(args.data / "demo_questions.csv", "question")
    golden = read_csv(args.data / "demo_output.csv", "answer")
    if not questions or set(questions) != set(golden):
        raise SystemExit("Questions and reference answers must contain the same nonempty set of IDs.")
    questions = dict(list(questions.items())[:args.limit]) if args.limit else questions
    metadata = {
        "commit": BASE,
        "questions_sha256": hashlib.sha256((args.data / "demo_questions.csv").read_bytes()).hexdigest(),
        "golden_sha256": hashlib.sha256((args.data / "demo_output.csv").read_bytes()).hexdigest(),
        "kb_sha256": hashlib.sha256((args.data / "knowledge-base.zip").read_bytes()).hexdigest(),
        "ids": list(questions),
        "reasoning_budget": REASONING_BUDGET,
        "max_tokens": MAX_TOKENS,
    }
    meta_path = args.output / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise SystemExit("The inputs differ from this output directory. Select a fresh --output.")
    write_json(meta_path, metadata)

    with httpx.Client(timeout=180, trust_env=False) as client:
        probe = [
            {"role": "system", "content": "Ответь по-русски."},
            {"role": "user", "content": "Чему равно 17 * 19 + 5?"},
        ]
        for mode in MODES:
            result = chat(client, args.generator, probe, mode, limit=1024)
            verify_mode(mode, result)
            if not result["answer"]:
                raise RuntimeError(f"The {mode} probe returned no final answer.")
        context_path = args.output / "prepared-contexts.json"
        if context_path.exists():
            prepared = json.loads(context_path.read_text(encoding="utf-8"))
        else:
            prepared = prepare(client, args, questions, golden)
            write_json(context_path, prepared)
        answer_paths = {mode: args.output / mode / "debug.jsonl" for mode in MODES}
        done = {}
        for mode, path in answer_paths.items():
            path.parent.mkdir(exist_ok=True)
            done[mode] = {row["id"] for row in read_jsonl(path)}
        for index, case in enumerate(prepared):
            order = MODES if index % 2 == 0 else MODES[::-1]
            for mode in order:
                if case["id"] in done[mode]:
                    continue
                result = chat(client, args.generator, case["messages"], mode)
                verify_mode(mode, result)
                append(answer_paths[mode], {
                    "id": case["id"], "mode": mode, **result,
                    "fallback": not result["answer"], "context_sha256": case["context_sha256"],
                })
                print(f"{case['id']} {mode}: {result['seconds']:.2f}s", flush=True)


def judge_system():
    tree = ast.parse((ROOT / "eval/judge.py").read_text(encoding="utf-8"))
    return next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "JUDGE_SYSTEM" for target in node.targets)
    )


def judge(args):
    cases = json.loads((args.output / "prepared-contexts.json").read_text(encoding="utf-8"))
    answers = {mode: {row["id"]: row for row in read_jsonl(args.output / mode / "debug.jsonl")}
               for mode in MODES}
    expected = {case["id"] for case in cases}
    if any(set(group) != expected for group in answers.values()):
        raise SystemExit("Finish generation before running the judge.")
    path = args.output / "judgments.jsonl"
    done = {(row["id"], row["round"]) for row in read_jsonl(path)}
    system = judge_system()
    with httpx.Client(timeout=180, trust_env=False) as client:
        for index, case in enumerate(cases):
            qid = case["id"]
            for round_no in (1, 2):
                if (qid, round_no) in done:
                    continue
                a, b = MODES if (index + round_no) % 2 else MODES[::-1]
                prompt = (
                    f"Вопрос клиента:\n{case['question']}\n\nЭталонный ответ:\n{case['golden']}\n\n"
                    f"Ответ A:\n{answers[a][qid]['answer'] or consultant.FALLBACK_ANSWER}\n\n"
                    f"Ответ B:\n{answers[b][qid]['answer'] or consultant.FALLBACK_ANSWER}"
                )
                for _ in range(2):
                    result = post(client, f"{args.judge_url}/v1/chat/completions", {
                        "model": args.judge_model,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        "temperature": 0, "max_tokens": 700, "seed": 42 + index,
                    })
                    content = result["choices"][0]["message"].get("content") or ""
                    match = re.search(r"\{.*\}", content, re.S)
                    try:
                        parsed = json.loads(match.group()) if match else {}
                        if parsed["winner"].lower() not in ("a", "b", "tie"):
                            raise ValueError("Unsupported winner")
                        if not all(0 <= float(parsed[f"match_{c}"]) <= 1 for c in ("a", "b")):
                            raise ValueError("Match score outside [0, 1]")
                        break
                    except (ValueError, TypeError, KeyError, AttributeError):
                        continue
                else:
                    raise RuntimeError(f"Invalid judge JSON for case {qid}, round {round_no}.")
                append(path, {"id": qid, "round": round_no, "a_mode": a, "b_mode": b, "judgment": parsed})
    report(args)


def report(args):
    outcomes = read_jsonl(args.output / "judgments.jsonl")
    expected = {(row["id"], order) for row in read_jsonl(args.output / "on" / "debug.jsonl") for order in (1, 2)}
    judged = {(row["id"], row["round"]) for row in outcomes}
    if not expected or expected != judged or len(judged) != len(outcomes):
        raise SystemExit("The report requires two judgments per case.")
    summary = {}
    for mode in MODES:
        rows = read_jsonl(args.output / mode / "debug.jsonl")
        if {(row["id"], order) for row in rows for order in (1, 2)} != expected:
            raise SystemExit(f"Missing answers in the {mode} variant.")
        matches, wins = [], []
        for row in outcomes:
            letter = "a" if row["a_mode"] == mode else "b"
            verdict = row["judgment"]
            matches.append(float(verdict[f"match_{letter}"]))
            winner = verdict["winner"].lower()
            wins.append(0.5 if winner == "tie" else float(winner == letter))
        match_rate, win_rate = mean(matches), mean(wins)
        summary[mode] = {
            "answers": len(rows),
            "empty": sum(row["fallback"] for row in rows),
            "limit_hits": sum(row["hit_output_limit"] for row in rows),
            "total_seconds": sum(row["seconds"] for row in rows),
            "reasoning_output_tokens": sum(row["stats"]["reasoning_output_tokens"] for row in rows),
            "match_rate": match_rate,
            "win_rate": win_rate,
            "score": 0.7 * match_rate + 0.3 * win_rate,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "judge", "report"))
    parser.add_argument("--data", type=Path, default=ROOT / "данные")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "thinking-ab-3b607ea")
    parser.add_argument("--generator", default="http://127.0.0.1:12347")
    parser.add_argument("--embeddings", default="http://127.0.0.1:1234")
    parser.add_argument("--judge-url", default="http://127.0.0.1:1234")
    parser.add_argument("--judge-model", default=JUDGE)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    args.output.mkdir(parents=True, exist_ok=True)
    {"generate": generate, "judge": judge, "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
