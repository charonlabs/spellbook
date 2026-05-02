from pathlib import Path

import pytest
from exa_py.api import Result, SearchResponse

from spellbook.ir_types import IRToolTextBlock
from spellbook.tools import web as web_tools
from spellbook.tools.common import ToolMetadata

pytestmark = pytest.mark.asyncio


class _FakeExaClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []

    def search(self, **kwargs: object) -> SearchResponse[Result]:
        self.search_calls.append(kwargs)
        contents = kwargs.get("contents")
        if contents == {"highlights": {"max_characters": 3000}}:
            return SearchResponse(
                [
                    Result(
                        url="https://example.com/highlight",
                        id="highlight",
                        title="Highlighted Result",
                        highlights=["A useful excerpt."],
                    )
                ],
                None,
                None,
            )
        if contents == {"text": {"max_characters": 10000}}:
            return SearchResponse(
                [
                    Result(
                        url="https://example.com/text",
                        id="text",
                        title="Text Result",
                        text="Full page text.",
                    )
                ],
                None,
                None,
            )
        raise AssertionError(f"Unexpected contents argument: {contents!r}")


async def test_web_search_uses_exa_search_contents_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeExaClient()
    monkeypatch.setattr(web_tools, "_get_exa_client", lambda: fake)
    meta = ToolMetadata(cwd=tmp_path, transcript_path=Path())

    highlights = await web_tools.exec_web_search(
        meta,
        web_tools.WebSearchInput(
            query="semantic memory systems",
            num_results=99,
            category="research paper",
            start_date="2026-01-01",
            end_date="2026-04-01",
        ),
    )
    text = await web_tools.exec_web_search(
        meta,
        web_tools.WebSearchInput(query="spellbook docs", mode="text"),
    )

    assert fake.search_calls == [
        {
            "query": "semantic memory systems",
            "type": "auto",
            "num_results": 10,
            "category": "research paper",
            "start_published_date": "2026-01-01",
            "end_published_date": "2026-04-01",
            "contents": {"highlights": {"max_characters": 3000}},
        },
        {
            "query": "spellbook docs",
            "type": "auto",
            "num_results": 5,
            "contents": {"text": {"max_characters": 10000}},
        },
    ]
    assert highlights.display == {
        "kind": "web_search",
        "query": "semantic memory systems",
        "num_results": 1,
        "result_titles": ["Highlighted Result"],
    }
    assert isinstance(highlights.content[0], IRToolTextBlock)
    assert "> A useful excerpt." in highlights.content[0].text
    assert text.display == {
        "kind": "web_search",
        "query": "spellbook docs",
        "num_results": 1,
        "result_titles": ["Text Result"],
    }
    assert isinstance(text.content[0], IRToolTextBlock)
    assert "Full page text." in text.content[0].text
