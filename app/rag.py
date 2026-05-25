"""Lightweight RAG over the character's static documents.

Design choices:
- Core identity files (`person_setup.md`, `world.md`) are always included in
  the system prompt — they're load-bearing and small enough that retrieval
  risks dropping the wrong slice.
- Other files are chunked by markdown sections and retrieved via BM25.
- Tokenization uses character bigrams, which works for Chinese without
  pulling in a segmenter (jieba) as a dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CORE_FILES = ("person_setup.md", "world.md", "sample_conversations.md")
RAG_FILES = ("personality.md", "hobbies.md", "others.md")


@dataclass
class Chunk:
    text: str
    source: str
    heading: str

    def render(self) -> str:
        return f"【{self.source} · {self.heading}】\n{self.text}"


def _split_sections(content: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs using level-1/2/3 headers."""
    lines = content.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "（开头）"
    current_body: list[str] = []
    header_re = re.compile(r"^#{1,3}\s+(.+)")
    for line in lines:
        m = header_re.match(line)
        if m:
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, current_body))
    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def _chunk_body(body: str, max_chars: int = 600) -> list[str]:
    """Further split long sections by numbered/bulleted items, paragraph-aware."""
    if len(body) <= max_chars:
        return [body]
    # Split on numbered list items at start of line ("1.", "2.")
    parts = re.split(r"\n(?=\s*\d+\.\s)", body)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = part
        else:
            buf = f"{buf}\n{part}".strip() if buf else part
    if buf:
        chunks.append(buf.strip())
    return chunks


def chunk_markdown(content: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _split_sections(content):
        for piece in _chunk_body(body):
            chunks.append(Chunk(text=piece, source=source, heading=heading))
    return chunks


def tokenize(text: str) -> list[str]:
    """Character n-gram tokens + ASCII word tokens.

    Emits both unigrams and bigrams over CJK + alphanumeric runs, plus
    ASCII word-level tokens. Unigrams give partial credit when a key
    character (e.g. 猫) appears in different bigram contexts; BM25's IDF
    naturally downweights ubiquitous characters like 的/了/我.
    """
    ascii_words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
    compact = re.sub(r"[^\w一-鿿]+", "", text)
    compact = re.sub(r"\s+", "", compact)
    if not compact:
        return ascii_words
    unigrams = [f"1:{c}" for c in compact]
    bigrams = [f"2:{compact[i:i+2]}" for i in range(len(compact) - 1)]
    return unigrams + bigrams + ascii_words


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.tf: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def fit(self, tokenized_docs: list[list[str]]) -> None:
        self.docs = tokenized_docs
        self.tf = [Counter(d) for d in tokenized_docs]
        n = max(1, len(tokenized_docs))
        self.avgdl = sum(len(d) for d in tokenized_docs) / n
        df: Counter[str] = Counter()
        for d in tokenized_docs:
            for term in set(d):
                df[term] += 1
        self.idf = {
            term: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for term, f in df.items()
        }

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        if not self.docs:
            return 0.0
        doc = self.docs[doc_idx]
        if not doc:
            return 0.0
        dl = len(doc)
        tf = self.tf[doc_idx]
        s = 0.0
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * num / denom
        return s

    def top_k(self, query_tokens: list[str], k: int = 4) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(len(self.docs))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]


class CharacterRAG:
    """Loads static character files, exposes core text + retrieval over the rest.

    `file_overrides` lets callers substitute the content of one or more
    static files at runtime without touching disk (used by the in-app
    prompt editor — overrides are kept off-git).
    """

    def __init__(
        self,
        static_dir: str | Path,
        file_overrides: dict[str, str] | None = None,
    ):
        self.static_dir = Path(static_dir)
        if not self.static_dir.exists():
            raise FileNotFoundError(f"Static directory not found: {self.static_dir}")
        self.file_overrides = dict(file_overrides or {})
        self._core_text: str = ""
        self._chunks: list[Chunk] = []
        self._bm25 = BM25()
        self._load()

    def _read(self, fname: str) -> str:
        if fname in self.file_overrides:
            return self.file_overrides[fname]
        fpath = self.static_dir / fname
        if fpath.exists():
            return fpath.read_text(encoding="utf-8")
        return ""

    def _load(self) -> None:
        # Core files concatenated, with file labels for traceability.
        core_parts: list[str] = []
        for fname in CORE_FILES:
            content = self._read(fname).strip()
            if content:
                core_parts.append(f"### 来源文件：{fname}\n{content}")
        self._core_text = "\n\n".join(core_parts)

        for fname in RAG_FILES:
            content = self._read(fname)
            if content:
                self._chunks.extend(chunk_markdown(content, fname))

        tokenized = [tokenize(c.text + " " + c.heading) for c in self._chunks]
        self._bm25.fit(tokenized)

    @property
    def core_text(self) -> str:
        return self._core_text

    @property
    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        if not query.strip() or not self._chunks:
            return []
        tokens = tokenize(query)
        hits = self._bm25.top_k(tokens, k=k)
        return [self._chunks[i] for i, _ in hits]

    def retrieve_with_scores(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not query.strip() or not self._chunks:
            return []
        tokens = tokenize(query)
        hits = self._bm25.top_k(tokens, k=k)
        return [(self._chunks[i], s) for i, s in hits]


def retrieve_history_chunks(
    messages,
    query: str,
    k: int = 3,
    exclude_recent_count: int = 30,
) -> list[Chunk]:
    """Build a per-session BM25 index over older (user, assistant) pairs and
    return the top-k most relevant ones for `query`.

    `messages` is the FULL list of past Message objects (or .role/.content
    dataclass instances). The most recent `exclude_recent_count` messages are
    skipped — those are already in the API's working window and don't need
    retrieval.
    """
    if not query.strip() or not messages:
        return []

    older_count = len(messages) - exclude_recent_count
    if older_count <= 0:
        return []
    older = messages[:older_count]

    # Pair user turns with the immediately-following assistant turn.
    pairs: list[tuple] = []
    i = 0
    while i < len(older):
        m = older[i]
        if m.role == "user":
            if i + 1 < len(older) and older[i + 1].role == "assistant":
                pairs.append((m, older[i + 1]))
                i += 2
            else:
                pairs.append((m, None))
                i += 1
        else:
            i += 1

    if not pairs:
        return []

    docs: list[list[str]] = []
    chunks: list[Chunk] = []
    for idx, (u, a) in enumerate(pairs):
        u_text = (u.content or "").strip()
        a_text = (a.content if a else "").strip()
        docs.append(tokenize(u_text + " " + a_text))
        body = f"用户：{u_text}"
        if a_text:
            body += f"\n西比莉娜：{a_text}"
        chunks.append(Chunk(text=body, source="对话历史", heading=f"第 {idx + 1} 轮"))

    bm = BM25()
    bm.fit(docs)
    hits = bm.top_k(tokenize(query), k=k)
    return [chunks[i] for i, _ in hits]
