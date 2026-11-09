"""Простой RAG без внешних зависимостей: markdown → чанки по заголовкам → BM25.

Для хакатона этого хватает: база маленькая, поиск мгновенный, ничего не надо поднимать.
Если захотите эмбеддинги — замените KnowledgeBase.search на Chroma/pgvector, интерфейс тот же.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOP = set("и в во на с со по для от до из к ко о об а но или не что это как у же ли бы мы вы они он она оно их его её мне нам вам".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))
    # грубый «стемминг» для русского: обрезаем окончания до 6 символов
    return [w[:6] for w in words if w not in STOP and len(w) > 1]


@dataclass
class Chunk:
    source: str
    title: str
    text: str


class KnowledgeBase:
    def __init__(self, directory: Path):
        self.chunks: list[Chunk] = []
        for path in sorted(directory.glob("*.md")):
            self._add_file(path)
        self._index()

    def _add_file(self, path: Path):
        content = path.read_text(encoding="utf-8-sig")  # -sig: файлы из Windows-редакторов бывают с BOM
        doc_title = next((l[2:].strip() for l in content.splitlines() if l.startswith("# ")), path.stem)
        for part in re.split(r"\n(?=## )", content):
            part = part.strip()
            if not part:
                continue
            heading = part.splitlines()[0].lstrip("# ").strip()
            title = doc_title if heading == doc_title else f"{doc_title} → {heading}"
            self.chunks.append(Chunk(path.name, title, part))

    def _index(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.title + " " + c.text)) for c in self.chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        q = tokenize(query)
        scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [
            {"source": self.chunks[i].source, "title": self.chunks[i].title, "text": self.chunks[i].text, "score": round(s, 2)}
            for s, i in scored[:top_k]
        ]
