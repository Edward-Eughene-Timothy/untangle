"""Phase 1: one-time web warm-up.

Fetch counseling-framework pages for the topics the user picked, strip the HTML, chunk the text
(~500 chars / 50 overlap), embed locally with FastEmbed and persist to on-disk Qdrant. Once at
least one page is indexed, `initialized` is set in SQLite and the app never needs the network
again (apart from optional re-warm-ups).
"""
import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.config import Settings
from app.core.memory_db import MemoryDB
from app.schemas.setup import TopicInfo, WarmupRequest, WarmupState, WarmupStatus
from app.services.rag_engine import Chunk, RagEngine

logger = logging.getLogger(__name__)

WIKI = "https://en.wikipedia.org/wiki/"


def _wiki(*titles: str) -> list[str]:
    return [WIKI + t for t in titles]


# Default, openly licensed sources. Users can add more pages via `extra_urls`.
TOPIC_CATALOG: dict[str, TopicInfo] = {
    t.id: t
    for t in [
        TopicInfo(
            id="cbt",
            label="Cognitive Behavioral Therapy",
            sources=_wiki(
                "Cognitive_behavioral_therapy",
                "Cognitive_distortion",
                "Cognitive_restructuring",
                "Socratic_questioning",
            ),
        ),
        TopicInfo(id="ifs", label="Internal Family Systems", sources=_wiki("Internal_Family_Systems_Model")),
        TopicInfo(
            id="nvc",
            label="Nonviolent Communication",
            sources=_wiki("Nonviolent_communication", "Active_listening"),
        ),
        TopicInfo(
            id="decision_paralysis",
            label="Decision Paralysis",
            sources=_wiki("Analysis_paralysis", "Decision_fatigue"),
        ),
        TopicInfo(id="imposter_syndrome", label="Imposter Syndrome", sources=_wiki("Impostor_syndrome")),
        TopicInfo(id="conflict_resolution", label="Conflict Resolution", sources=_wiki("Conflict_resolution")),
        TopicInfo(
            id="act", label="Acceptance and Commitment Therapy", sources=_wiki("Acceptance_and_commitment_therapy")
        ),
        TopicInfo(id="dbt", label="Dialectical Behavior Therapy", sources=_wiki("Dialectical_behavior_therapy")),
        TopicInfo(
            id="inner_conflict",
            label="Inner Conflict & Ambivalence",
            sources=_wiki("Cognitive_dissonance", "Motivational_interviewing"),
        ),
        TopicInfo(id="attachment", label="Attachment", sources=_wiki("Attachment_theory")),
    ]
}

# --------------------------------------------------------------------------- HTML → text

_DROP_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe", "button"]
_DROP_CLASSES = {
    "navbox", "reflist", "references", "reference", "mw-editsection", "toc", "infobox", "hatnote",
    "metadata", "sidebar", "mw-references-wrap", "noprint", "thumb", "navbar", "ambox",
}
_STOP_HEADINGS = {
    "references", "external links", "see also", "further reading", "notes", "bibliography",
    "sources", "footnotes",
}


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, clean_text). Keeps headings, paragraphs and list items; drops chrome,
    citations and everything from the 'References' section onwards."""
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")

    for tag in soup(_DROP_TAGS):
        if not getattr(tag, "decomposed", False):
            tag.decompose()
    for el in soup.find_all(class_=lambda c: bool(c) and c in _DROP_CLASSES):
        if not getattr(el, "decomposed", False):
            el.decompose()

    root = soup.find(id="mw-content-text") or soup.find("main") or soup.find("article") or soup.body or soup
    parts: list[str] = []
    for el in root.find_all(["h2", "h3", "h4", "p", "li"]):
        if el.name in ("h2", "h3", "h4"):
            heading = re.sub(r"\[edit\]", "", el.get_text(" ", strip=True)).strip()
            if heading.lower() in _STOP_HEADINGS:
                break
            if heading:
                parts.append(heading + ".")
            continue
        if el.name == "li" and el.find(["p", "li", "ul", "ol"]):
            continue  # avoid duplicating text that is captured by its children
        text = re.sub(r"\[\s*\d+\s*\]", "", el.get_text(" ", strip=True))
        if len(text) >= 25:
            parts.append(text)
    return title, "\n".join(parts)


# --------------------------------------------------------------------------- chunking


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split into ~`size`-char chunks with `overlap` chars of shared context, preferring
    sentence (then word) boundaries so chunks stay readable."""
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut < size * 0.5:
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt)  # start the next chunk on a word boundary
        start = space + 1 if 0 <= space < end else nxt
    # Drop tiny trailing fragments that carry no meaning of their own.
    if len(chunks) > 1 and len(chunks[-1]) < 40:
        chunks.pop()
    return chunks


# --------------------------------------------------------------------------- fetching


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str


PageFetcher = Callable[[str], Awaitable[FetchedPage]]


class HttpPageFetcher:
    """httpx + BeautifulSoup fetcher that honours robots.txt."""

    def __init__(self, client: httpx.AsyncClient, user_agent: str):
        self._client = client
        self._ua = user_agent
        self._robots: dict[str, robotparser.RobotFileParser] = {}

    async def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                resp = await self._client.get(origin + "/robots.txt")
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                    rp.modified()
                elif resp.status_code in (401, 403):
                    rp.disallow_all = True
                else:
                    rp.allow_all = True
            except httpx.HTTPError:
                rp.allow_all = True
            self._robots[origin] = rp
        return self._robots[origin].can_fetch(self._ua, url)

    async def __call__(self, url: str) -> FetchedPage:
        if not await self._allowed(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        resp = await self._client.get(url)
        resp.raise_for_status()
        if "html" not in resp.headers.get("content-type", ""):
            raise ValueError(f"{url} is not an HTML page")
        title, text = await asyncio.to_thread(html_to_text, resp.text)
        if len(text) < 200:
            raise ValueError(f"No extractable content at {url}")
        return FetchedPage(url=url, title=title, text=text)


# --------------------------------------------------------------------------- service


class WarmupBusy(RuntimeError):
    pass


class UnknownTopic(ValueError):
    pass


@dataclass
class WarmupJob:
    sources: list[tuple[str, str]]  # (topic_id, url)
    status: WarmupStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WarmupService:
    def __init__(self, db: MemoryDB, rag: RagEngine, settings: Settings, fetcher: PageFetcher | None = None):
        self.db = db
        self.rag = rag
        self.settings = settings
        self._fetcher = fetcher
        self._running = False

    def catalog(self) -> list[TopicInfo]:
        return list(TOPIC_CATALOG.values())

    async def is_initialized(self) -> bool:
        return (await self.db.get_state("initialized")) == "true"

    async def get_status(self) -> WarmupStatus:
        raw = await self.db.get_state("warmup_status")
        status = WarmupStatus.model_validate_json(raw) if raw else WarmupStatus()
        if status.state == WarmupState.running and not self._running:
            status.state = WarmupState.failed  # the process died mid-run
            status.error = status.error or "Warm-up was interrupted. Start it again."
        status.initialized = await self.is_initialized()
        return status

    async def _save(self, status: WarmupStatus) -> None:
        await self.db.set_state("warmup_status", status.model_dump_json())

    async def start(self, req: WarmupRequest) -> WarmupJob:
        unknown = [t for t in req.topics if t not in TOPIC_CATALOG]
        if unknown:
            raise UnknownTopic(f"Unknown topic(s): {', '.join(unknown)}")
        if self._running:
            raise WarmupBusy("A warm-up is already running.")

        seen: set[str] = set()
        sources: list[tuple[str, str]] = []
        for topic in req.topics:
            for url in TOPIC_CATALOG[topic].sources:
                if url not in seen:
                    seen.add(url)
                    sources.append((topic, url))
        for url in req.extra_urls:
            if str(url) not in seen:
                seen.add(str(url))
                sources.append(("custom", str(url)))

        self._running = True
        status = WarmupStatus(
            state=WarmupState.running,
            topics=list(req.topics),
            pages_total=len(sources),
            started_at=_now(),
        )
        await self._save(status)
        status.initialized = await self.is_initialized()
        return WarmupJob(sources=sources, status=status)

    async def run(self, job: WarmupJob) -> None:
        status = job.status
        sem = asyncio.Semaphore(self.settings.CRAWL_CONCURRENCY)
        last_error: str | None = None
        try:
            async with AsyncExitStack() as stack:
                if self._fetcher is not None:
                    fetch = self._fetcher
                else:
                    client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers={"User-Agent": self.settings.CRAWL_USER_AGENT},
                            follow_redirects=True,
                            timeout=httpx.Timeout(self.settings.CRAWL_TIMEOUT_S),
                        )
                    )
                    fetch = HttpPageFetcher(client, self.settings.CRAWL_USER_AGENT)

                async def one(topic: str, url: str) -> None:
                    nonlocal last_error
                    async with sem:
                        try:
                            page = await fetch(url)
                            texts = chunk_text(page.text, self.settings.CHUNK_SIZE, self.settings.CHUNK_OVERLAP)
                            chunks = [
                                Chunk(text=t, source_url=url, title=page.title, topic=topic, index=i)
                                for i, t in enumerate(texts)
                            ]
                            stored = await self.rag.replace_source(url, chunks)
                            status.chunks_indexed += stored  # not `+= await ...`: that races
                            status.pages_done += 1
                        except Exception as exc:  # one bad page must not sink the run
                            logger.warning("Warm-up failed for %s: %s", url, exc)
                            last_error = f"{type(exc).__name__}: {exc}"
                            status.pages_failed += 1
                            status.failed_urls.append(url)
                        await self._save(status)

                await asyncio.gather(*(one(t, u) for t, u in job.sources))

            if status.chunks_indexed > 0:
                status.state = WarmupState.completed
                await self.db.set_state("initialized", "true")
            else:
                status.state = WarmupState.failed
                status.error = f"No pages could be indexed. Last error: {last_error}"
        except Exception as exc:
            logger.exception("Warm-up crashed")
            status.state = WarmupState.failed
            status.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._running = False
            status.finished_at = _now()
            await self._save(status)
