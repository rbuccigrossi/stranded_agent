"""Planning, goal, and public-web tools for the STRANDed Agent.

Every tool here is a plain Python function. Strands derives the JSON schema the
model sees from the type hints and docstring, and plan and goal state lives in
``agent.state`` so it is saved and restored with the session.
"""

import ipaddress
import json
import socket
import time
from html.parser import HTMLParser
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel
from strands import ToolContext, tool


MAX_BYTES = 2_000_000
MAX_OUTPUT = 12_000
WEB_TIMEOUT = 15
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "Chrome/131.0 Safari/537.36 STRANDedAgent/1.0")


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _trim(value: str, limit: int = MAX_OUTPUT) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "\n[output truncated]"


class PlanStep(BaseModel):
    """One step of the visible plan."""

    step: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


def _write_plan(steps: List[PlanStep], tool_context: ToolContext, explanation: str) -> str:
    """Validate a plan and store it on the agent state.

    Strands hands tool arguments over as plain JSON, so the steps are validated
    against ``PlanStep`` here rather than arriving already parsed.
    """
    plan = [PlanStep.model_validate(step).model_dump() for step in steps]
    if not plan:
        raise ValueError("steps must be a non-empty list")
    if len(plan) > 100:
        raise ValueError("a plan may contain at most 100 steps")
    if sum(step["status"] == "in_progress" for step in plan) > 1:
        raise ValueError("only one plan step may be in_progress")
    tool_context.agent.state.set("plan", plan)
    return _compact({"status": "updated", "plan": plan, "explanation": explanation})


@tool(context=True)
def create_plan(steps: List[PlanStep], tool_context: ToolContext, explanation: str = "") -> str:
    """Create a visible multi-step plan for the current task.

    Args:
        steps: The ordered plan steps, at most one of them in_progress.
        explanation: Why the plan looks like this, in a short sentence.
    """
    return _write_plan(steps, tool_context, explanation)


@tool(context=True)
def update_plan(steps: List[PlanStep], tool_context: ToolContext, explanation: str = "") -> str:
    """Update the visible plan as work progresses.

    Args:
        steps: The full plan again, with each step's current status.
        explanation: What changed since the last update.
    """
    return _write_plan(steps, tool_context, explanation)


@tool(context=True)
def create_goal(objective: str, tool_context: ToolContext, token_budget: Optional[int] = None,
                max_steps: Optional[int] = None) -> str:
    """Create an explicit bounded objective for continuing work.

    Args:
        objective: What the agent should keep working toward.
        token_budget: Optional token ceiling for the objective.
        max_steps: Optional ceiling on unprompted continuation steps.
    """
    if not objective.strip():
        raise ValueError("objective is required")
    goal: Dict[str, Any] = {"objective": objective.strip(), "status": "active",
                            "created": int(time.time())}
    if token_budget is not None:
        goal["token_budget"] = max(1, int(token_budget))
    if max_steps is not None:
        goal["max_steps"] = max(1, int(max_steps))
    tool_context.agent.state.set("goal", goal)
    return _compact({"status": "created", "goal": goal})


@tool(context=True)
def get_goal(tool_context: ToolContext) -> str:
    """Read the current explicit goal and its status."""
    return _compact({"goal": tool_context.agent.state.get("goal")})


@tool(context=True)
def update_goal(status: Literal["active", "complete", "blocked"], tool_context: ToolContext,
                explanation: str = "") -> str:
    """Mark the current goal active, complete, or blocked.

    Args:
        status: The new goal status.
        explanation: Why the goal reached this status.
    """
    goal = tool_context.agent.state.get("goal")
    if not isinstance(goal, dict):
        raise ValueError("no active goal exists")
    goal = dict(goal, status=status, updated=int(time.time()))
    if explanation:
        goal["explanation"] = explanation
    tool_context.agent.state.set("goal", goal)
    return _compact({"status": "updated", "goal": goal})


class DocumentParser(HTMLParser):
    """Extract readable document text without executing page content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.lines: List[str] = []
        self.links: List[Dict[str, str]] = []
        self._skip = 0
        self._title = False
        self._link: Optional[Dict[str, str]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1
        elif tag == "title" and not self._skip:
            self._title = True
        elif tag == "a" and not self._skip and attributes.get("href"):
            self._link = {"url": attributes["href"] or "", "text": ""}

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._title = False
        elif tag == "a" and self._link:
            if self._link["text"].strip():
                self.links.append({key: value.strip() for key, value in self._link.items()})
            self._link = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._title:
            self.title += (" " if self.title else "") + text
        if self._link is not None:
            self._link["text"] += (" " if self._link["text"] else "") + text
        self.lines.append(text)


class DuckDuckGoParser(HTMLParser):
    """Pull the title, url, and snippet out of a DuckDuckGo HTML results page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._current: Optional[Dict[str, str]] = None
        self._mode: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self._current = {"title": "", "url": "", "snippet": ""}
            self._current["url"] = _unwrap_search_url(attributes.get("href") or "")
            self._mode = "title"
        elif self._current and tag in {"a", "div", "span"} and "result__snippet" in classes:
            self._mode = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if self._current and tag == "a" and self._mode == "title":
            self._mode = None
        elif self._current and self._mode == "snippet" and tag in {"div", "span"}:
            self._mode = None
            if self._current["title"] or self._current["snippet"]:
                self.results.append(self._current)
                self._current = None

    def handle_data(self, data: str) -> None:
        if not self._current or not self._mode:
            return
        text = " ".join(data.split())
        if text:
            self._current[self._mode] += (" " if self._current[self._mode] else "") + text


def _unwrap_search_url(value: str) -> str:
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(value)
    if parsed.path.endswith("/l/") or parsed.path == "/l/":
        return unquote(parse_qs(parsed.query).get("uddg", [value])[0])
    return value


def _safe_url(url: str) -> str:
    """Reject anything that would let the web tools reach the local network."""
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must use http or https")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("local hosts are not allowed")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port, type=socket.SOCK_STREAM)}
    except socket.gaierror as error:
        raise ValueError(f"could not resolve host: {host}") from error
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError("private or local network addresses are not allowed")
    return parsed.geturl()


def _fetch_urllib(url: str) -> Tuple[bytes, str, str]:
    request = Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1",
    })
    with urlopen(request, timeout=WEB_TIMEOUT) as response:
        _safe_url(response.geturl())
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise RuntimeError(f"response exceeds {MAX_BYTES} bytes")
        charset = response.headers.get_content_charset() or "utf-8"
        return data, response.headers.get_content_type(), charset


def _fetch_playwright(url: str) -> Tuple[bytes, str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError("request was blocked and optional Playwright is not installed") from error
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=USER_AGENT)
                page.goto(url, wait_until="domcontentloaded", timeout=WEB_TIMEOUT * 1000)
                _safe_url(page.url)
                content = page.content().encode("utf-8")
            finally:
                browser.close()
            if len(content) > MAX_BYTES:
                raise RuntimeError(f"response exceeds {MAX_BYTES} bytes")
            return content, "text/html", "utf-8"
    except Exception as error:
        raise RuntimeError(f"Playwright fetch failed: {error}") from error


def _fetch(url: str) -> Tuple[bytes, str, str]:
    safe = _safe_url(url)
    try:
        return _fetch_urllib(safe)
    except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
        if isinstance(error, RuntimeError) and "exceeds" in str(error):
            raise
        return _fetch_playwright(safe)


def _decode(data: bytes, charset: str) -> str:
    try:
        return data.decode(charset, "replace")
    except (LookupError, UnicodeError):
        return data.decode("utf-8", "replace")


@tool
def web_fetch(url: str) -> str:
    """Fetch a public URL and extract readable content and links.

    Args:
        url: The http or https address to read.
    """
    data, content_type, charset = _fetch(url)
    text = _decode(data, charset)
    if content_type in {"text/html", "application/xhtml+xml"} or "<html" in text[:1000].lower():
        parser = DocumentParser()
        parser.feed(text)
        body = "\n".join(dict.fromkeys(parser.lines))
        output: Dict[str, Any] = {"url": url, "title": parser.title, "text": _trim(body),
                                  "links": parser.links[:30]}
        sources = [{"title": parser.title or url, "url": url}]
    else:
        output = {"url": url, "content_type": content_type, "text": _trim(text)}
        sources = [{"title": url, "url": url}]
    return _compact({**output, "sources": sources})


@tool
def web_search(query: str, domains: Optional[List[str]] = None, max_results: int = 5) -> str:
    """Search the public web and return source-linked results.

    Args:
        query: What to search for.
        domains: Optional list of domains to restrict the search to.
        max_results: How many results to return, between 1 and 10.
    """
    if not query.strip():
        raise ValueError("query is required")
    max_results = max(1, min(int(max_results), 10))
    terms = [query.strip()] + [f"site:{domain}" for domain in (domains or []) if str(domain).strip()]
    data, _, charset = _fetch("https://html.duckduckgo.com/html/?q=" + quote_plus(" ".join(terms)))
    parser = DuckDuckGoParser()
    parser.feed(_decode(data, charset))
    results = parser.results[:max_results]
    if not results:
        raise RuntimeError("search returned no parseable results")
    sources = [{"title": item["title"], "url": item["url"]} for item in results if item.get("url")]
    return _compact({"query": query, "results": results, "sources": sources})


TOOLS = [create_plan, update_plan, create_goal, get_goal, update_goal, web_search, web_fetch]
