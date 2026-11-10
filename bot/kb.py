"""База знаний: загрузка из zip, разбор pandoc-таблиц, нарезка по заголовкам, гибридный поиск.

Поиск — BM25 (всегда) + эмбеддинги bge-m3 (если эмбеддер доступен), списки объединяются через RRF.

Распаковывает архив knowledge-base.zip, превращает таблицы в удобный текст, режет файлы по заголовкам ## на кусочки (чанки)
и ищет по ним двумя способами: по ключевым словам (BM25) и по смыслу через эмбеддинги (bge-m3).
"""

from __future__ import annotations

import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_CHUNK_CHARS = 1800


@dataclass
class Chunk:
    id: int
    source: str  # путь файла внутри архива
    title: str  # «Документ → Раздел → Подраздел»
    text: str

    def render(self) -> str:
        return f"{self.title}\n{self.text}"


# ---------- загрузка и нормализация ----------

def load_documents(path) -> list[tuple[str, str]]:
    """[(имя файла, текст)] из zip-архива или папки с .md/.txt."""
    path = Path(path)
    docs = []
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in (".md", ".txt"):
                docs.append((p.relative_to(path).as_posix(), p.read_text(encoding="utf-8-sig", errors="replace")))
    else:
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.startswith("__MACOSX/") or Path(name).suffix.lower() not in (".md", ".txt"):
                    continue
                docs.append((name, archive.read(name).decode("utf-8-sig", errors="replace")))
    return [(name, text.replace("\r\n", "\n")) for name, text in docs if text.strip()]


SPEC_RE = re.compile(r"^\s*-{3,}(?:\s+-{3,})+\s*$")  # разметка колонок: несколько групп дефисов
RULE_RE = re.compile(r"^\s*-{3,}\s*$")  # сплошная линия: рамка таблицы или горизонтальная черта


def _spans(spec_line: str) -> list[tuple[int, int | None]]:
    starts = [m.start() for m in re.finditer(r"-+", spec_line)]
    return [(s, starts[i + 1] if i + 1 < len(starts) else None) for i, s in enumerate(starts)]


def _cells(lines: list[str], spans) -> list[str]:
    cells = []
    for start, end in spans:
        parts = [(line[start:end] if end else line[start:]).strip() for line in lines]
        cells.append(" ".join(p for p in parts if p))
    return cells


def normalize_tables(text: str) -> str:
    """Pandoc multiline-таблицы (ячейки перенесены по строкам) → строки «- Шапка: значение; …»."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not SPEC_RE.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        spans = _spans(lines[i])
        # шапка — непустые строки прямо над разметкой колонок, до верхней рамки
        header_lines: list[str] = []
        while out and out[-1].strip() and not RULE_RE.match(out[-1]):
            header_lines.insert(0, out.pop())
        if out and RULE_RE.match(out[-1]):
            out.pop()
        headers = _cells(header_lines, spans) if header_lines else [""] * len(spans)
        i += 1
        rows, row = [], []
        # строки таблицы идут с отступом; конец — рамка или текст без отступа
        while i < len(lines) and not (RULE_RE.match(lines[i]) or SPEC_RE.match(lines[i])):
            line = lines[i]
            if line.strip() and not line.startswith(" "):
                break
            if line.strip():
                row.append(line)
            elif row:
                rows.append(row)
                row = []
            i += 1
        if row:
            rows.append(row)
        if i < len(lines) and (RULE_RE.match(lines[i]) or SPEC_RE.match(lines[i])):
            i += 1  # нижняя рамка
        for r in rows:
            cells = _cells(r, spans)
            parts = [f"{h}: {c}" if h else c for h, c in zip(headers, cells) if c]
            if parts:
                out.append("- " + "; ".join(parts))
        out.append("")
    return "\n".join(line for line in out if not RULE_RE.match(line))


def normalize(text: str) -> str:
    text = normalize_tables(text)
    text = text.replace("---", "—").replace("--", "–").replace("**", "")
    return re.sub(r"\n{3,}", "\n\n", text)


HEAD_RE = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")


def _sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
    doc, stack = fallback_title, []
    buf: list[str] = []
    out = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            out.append((" → ".join([doc] + stack), body))
        buf.clear()

    for line in text.split("\n"):
        m = HEAD_RE.match(line)
        if not m:
            buf.append(line)
            continue
        flush()
        level, heading = len(m.group(1)), m.group(2)
        if level == 1:
            doc, stack = heading, []
        else:
            stack = stack[:level - 2] + [heading]
    flush()
    return out


def _split_long(body: str, limit: int = MAX_CHUNK_CHARS, sep: str = "\n\n") -> list[str]:
    if len(body) <= limit:
        return [body]
    parts, cur = [], ""
    for piece in body.split(sep):
        if len(piece) > limit and sep == "\n\n":
            if cur:
                parts.append(cur)
                cur = ""
            parts.extend(_split_long(piece, limit, "\n"))
            continue
        if cur and len(cur) + len(piece) + len(sep) > limit:
            parts.append(cur)
            cur = piece
        else:
            cur = f"{cur}{sep}{piece}" if cur else piece
    if cur:
        parts.append(cur)
    return parts


# ---------- BM25 ----------

WORD_RE = re.compile(r"[a-zа-я0-9]+")
STOP = set("""и в во на с со по для от до из к ко о об а но или не что это как у же ли бы мы вы они он она оно
их его ее мне нам вам вас нас там тут так то при за под над про через если ну да нет есть быть был была были
будет можно нужно ваш наш ли уже еще""".split())
SUFFIXES = sorted(set("""ами ями ого его ому ему ыми ими ых их ой ей ий ый ая яя ое ее ые ие ую юю ам ям ах ях ом ем
ов ев ия ии ию ья ье ьи ью ать ять ить еть ет ит ут ют ат ят ешь ишь ете ите ла ли ло а я о е ы и у ю ь й""".split()),
                  key=len, reverse=True)


def stem(word: str) -> str:
    """Грубый стемминг для русского: срезаем окончание (а для разговорных -ка/-ки — и суффикс к)."""
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix in SUFFIXES:
        min_root = 3 if suffix in ("а", "и", "у", "е", "ой", "ами", "ах") else 4
        if word.endswith(suffix) and len(word) - len(suffix) >= min_root:
            base = word[:-len(suffix)][:7]
            # Разговорный суффикс -к- в коротких словах (сим-к-и -> сим, фот-к-а -> фот)
            if len(base) == 4 and base.endswith("к"):
                return base[:-1]
            return base
    return word[:7]


def tokenize(text: str) -> list[str]:
    words = WORD_RE.findall(text.lower().replace("ё", "е"))
    return [stem(w) for w in words if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.n = len(docs)
        lengths = [len(d) for d in docs]
        avg = sum(lengths) / max(self.n, 1)
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            norm = k1 * (1 - b + b * lengths[i] / avg)
            for word, tf in Counter(doc).items():
                self.postings[word].append((i, tf * (k1 + 1) / (tf + norm)))
        self.idf = {w: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        for word in set(query):
            for i, weight in self.postings.get(word, ()):
                out[i] += self.idf[word] * weight
        return out


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-9)


# ---------- база знаний ----------

class KnowledgeBase:
    def __init__(self, documents: list[tuple[str, str]]):
        self.chunks: list[Chunk] = []
        for name, text in documents:
            for title, body in _sections(normalize(text), Path(name).stem):
                for part in _split_long(body):
                    self.chunks.append(Chunk(len(self.chunks), name, title, part))
        if not self.chunks:
            raise ValueError("в базе знаний нет текстов .md/.txt")
        self.bm25 = BM25([tokenize(c.render()) for c in self.chunks])
        self.vectors: np.ndarray | None = None  # нормированные эмбеддинги фрагментов

    def enable_dense(self, embed_fn) -> None:
        vectors = np.asarray(embed_fn([c.render()[:2000] for c in self.chunks]), dtype=np.float32)
        self.vectors = _unit(vectors)

    def search(self, queries: list[str], query_vectors=None, k: int = 16) -> list[Chunk]:
        """Гибридный поиск: по каждому запросу свой список BM25 (+ векторный), затем RRF."""
        ranked: list[list[int]] = []
        for query in queries:
            scores = self.bm25.scores(tokenize(query))
            ranked.append([int(i) for i in np.argsort(-scores)[:50] if scores[i] > 0])
        if self.vectors is not None and query_vectors:
            for vector in query_vectors:
                if vector is None:
                    continue
                sims = self.vectors @ _unit(np.asarray(vector, dtype=np.float32))
                ranked.append([int(i) for i in np.argsort(-sims)[:50]])
        fused: Counter = Counter()
        for order in ranked:
            for rank, i in enumerate(order):
                fused[i] += 1.0 / (60 + rank)
        return [self.chunks[i] for i, _ in fused.most_common(k)]
