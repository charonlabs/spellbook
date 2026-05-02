"""Web search, read, and answer tools powered by Exa.

Three tools:
  WebSearch — semantic search with highlights or full text
  WebRead  — clean text extraction from known URLs
  WebAnswer — Q&A with grounded citations from web search
"""

from __future__ import annotations

import os
from typing import Any, Optional

from exa_py import Exa
from exa_py.api import (
    AnswerResponse,
    Result,
    ResultWithText,
    SearchResponse,
)
from pydantic import BaseModel, Field

from spellbook.ir_types import IRToolTextBlock
from spellbook.tools.common import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)

# --- Lazy Exa client ---

_client: Exa | None = None


def _get_exa_client() -> Exa:
    global _client
    if _client is None:
        key = os.environ.get("EXA_API_KEY")
        if not key:
            raise ToolError(
                "EXA_API_KEY is not set. Web search is unavailable. "
                "Set the EXA_API_KEY environment variable to enable web tools."
            )
        _client = Exa(api_key=key)
    return _client


# --- Input schemas ---


class WebSearchInput(BaseModel):
    """Search the web with semantic understanding."""

    query: str = Field(
        description="The search query. Be specific and descriptive.",
    )
    num_results: int = Field(
        default=5,
        description="Number of results to return (1-10).",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category filter: 'research paper', 'news', 'company', 'people'. "
            "Omit for general search."
        ),
    )
    mode: str = Field(
        default="highlights",
        description=(
            "Content mode: 'highlights' (key excerpts, default) or 'text' (full content). "
            "Use highlights for scanning, text for deep reading."
        ),
    )
    start_date: Optional[str] = Field(
        default=None,
        description=(
            "Filter results published after this date (ISO format, e.g. '2026-01-01'). "
            "Use this instead of putting years in the query."
        ),
    )
    end_date: Optional[str] = Field(
        default=None,
        description=(
            "Filter results published before this date (ISO format, e.g. '2026-04-01')."
        ),
    )


class WebReadInput(BaseModel):
    """Read clean text content from one or more URLs."""

    urls: list[str] = Field(
        description="URLs to extract content from.",
    )
    max_characters: int = Field(
        default=10000,
        description="Maximum characters per URL (default: 10000).",
    )


class WebAnswerInput(BaseModel):
    """Get a grounded answer with citations from web search."""

    query: str = Field(
        description=(
            "The question to answer. Best for factual lookups: "
            "'What is the architecture of Gemma 4?', "
            "'When was Python 3.12 released?'"
        ),
    )


# --- Implementations ---


async def exec_web_search(
    meta: ToolMetadata, input: WebSearchInput
) -> ToolExecutionResult:
    """Search the web with semantic understanding."""
    exa = _get_exa_client()

    num_results = max(1, min(10, input.num_results))

    kwargs: dict[str, Any] = {
        "query": input.query,
        "type": "auto",
        "num_results": num_results,
    }

    if input.category:
        kwargs["category"] = input.category

    if input.start_date:
        kwargs["start_published_date"] = input.start_date
    if input.end_date:
        kwargs["end_published_date"] = input.end_date

    if input.mode == "text":
        kwargs["contents"] = {"text": {"max_characters": 10000}}
    else:
        kwargs["contents"] = {"highlights": {"max_characters": 3000}}

    try:
        results = exa.search(**kwargs)
    except Exception as e:
        raise ToolError(f"Web search failed: {e}") from e

    output = _format_search_results(results, mode=input.mode)
    items = results.results
    titles = [r.title or "(untitled)" for r in items[:3]]
    return ToolExecutionResult(
        content=[IRToolTextBlock(text=output)],
        display={
            "kind": "web_search",
            "query": input.query,
            "num_results": len(items),
            "result_titles": titles,
        },
    )


async def exec_web_read(meta: ToolMetadata, input: WebReadInput) -> ToolExecutionResult:
    """Read clean text content from known URLs."""
    exa = _get_exa_client()

    if not input.urls:
        raise ToolError("No URLs provided.")

    max_chars = max(1000, min(50000, input.max_characters))

    try:
        results = exa.get_contents(
            input.urls,
            text={"max_characters": max_chars},
        )
    except Exception as e:
        raise ToolError(f"Web read failed: {e}") from e

    output = _format_read_results(results)
    items = results.results
    first = items[0] if items else None
    title = first.title or "(untitled)" if first else ""
    url = input.urls[0] if input.urls else ""
    text = first.text if first else ""
    preview = text[:150].strip().replace("\n", " ") if text else ""

    return ToolExecutionResult(
        content=[IRToolTextBlock(text=output)],
        display={
            "kind": "web_read",
            "url": url,
            "title": title,
            "preview": preview,
        },
    )


async def exec_web_answer(
    meta: ToolMetadata, input: WebAnswerInput
) -> ToolExecutionResult:
    """Get a grounded answer with citations from web search."""
    exa = _get_exa_client()

    try:
        result = exa.answer(
            input.query,
            text=True,
        )
    except Exception as e:
        raise ToolError(f"Web answer failed: {e}") from e
    assert isinstance(result, AnswerResponse)
    output = _format_answer_result(result)
    answer = str(result.answer)
    citations = result.citations
    preview = answer[:150].strip().replace("\n", " ") if answer else ""

    return ToolExecutionResult(
        content=[IRToolTextBlock(text=output)],
        display={
            "kind": "web_answer",
            "query": input.query,
            "answer_preview": preview,
            "citation_count": len(citations),
        },
    )


# --- Formatters ---


def _format_search_results(results: SearchResponse[Result], *, mode: str) -> str:
    """Format Exa search results into readable text."""
    items = results.results
    if not items:
        return "No results found."

    parts = [f"{len(items)} results:\n"]

    for i, result in enumerate(items, 1):
        title = result.title or "(untitled)"
        url = result.url

        parts.append(f"## {i}. {title}")
        parts.append(f"URL: {url}")

        if mode == "text":
            text = result.text
            if text:
                parts.append(text.strip())
        else:
            highlights = result.highlights
            if highlights:
                for highlight in highlights:
                    parts.append(f"> {highlight}")
            else:
                # Fallback to text snippet if highlights empty
                text = result.text
                if text:
                    parts.append(text[:500].strip())

        parts.append("")

    return "\n".join(parts)


def _format_read_results(results: SearchResponse[ResultWithText]) -> str:
    """Format Exa content extraction results."""
    items = results.results
    if not items:
        return "No content extracted."

    parts = []

    for result in items:
        title = result.title or "(untitled)"
        url = result.url
        text = result.text

        parts.append(f"## {title}")
        parts.append(f"URL: {url}")
        parts.append(text.strip())
        parts.append("")

    return "\n".join(parts)


def _format_answer_result(result: AnswerResponse) -> str:
    """Format Exa answer with citations."""
    answer = str(result.answer)
    citations = result.citations

    parts = [answer]

    if citations:
        parts.append("\n### Sources")
        for i, citation in enumerate(citations, 1):
            title = citation.title or "(untitled)"
            url = citation.url
            parts.append(f"{i}. [{title}]({url})")

    return "\n".join(parts)


WEB_SEARCH_TOOL: Tool[WebSearchInput] = Tool(
    name="WebSearch",
    input_model=WebSearchInput,
    exec=exec_web_search,
    category="web",
)

WEB_READ_TOOL: Tool[WebReadInput] = Tool(
    name="WebRead",
    input_model=WebReadInput,
    exec=exec_web_read,
    category="web",
)

WEB_ANSWER_TOOL: Tool[WebAnswerInput] = Tool(
    name="WebAnswer",
    input_model=WebAnswerInput,
    exec=exec_web_answer,
    category="web",
)
