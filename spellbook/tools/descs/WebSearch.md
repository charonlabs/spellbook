Search the web with semantic understanding powered by Exa.

Returns relevant results with titles, URLs, and content excerpts. Uses neural search that understands meaning, not just keywords.

Usage:
- `WebSearch(query="transformer architecture improvements")` — general semantic search
- `WebSearch(query="attention mechanism papers", category="research paper")` — search academic papers
- `WebSearch(query="OpenAI announcements", category="news")` — search news
- `WebSearch(query="React hooks best practices", mode="text")` — full text content instead of highlights
- `WebSearch(query="LLM reasoning", start_date="2026-01-01")` — results from 2026 onward

Parameters:
- `query`: Be specific and descriptive. "transformer architecture improvements" works better than "transformers".
- `num_results`: 1-10 results (default: 5). Use fewer for focused queries, more for broad research.
- `category`: Optional filter — "research paper", "news", "company", "people". Omit for general search.
- `mode`: "highlights" (default, key excerpts) or "text" (full content). Use highlights for scanning results, text when you need to read in depth.
- `start_date`: ISO date (e.g. "2026-01-01"). Filter to results published after this date.
- `end_date`: ISO date (e.g. "2026-04-01"). Filter to results published before this date.

Important:
- **Don't put years in your search query** — use `start_date`/`end_date` filters instead. "LLM reasoning" with `start_date="2026-01-01"` works much better than "LLM reasoning 2026".
- Start with highlights mode to scan results, then use WebRead on specific URLs for full content.
- For academic research, use `category="research paper"` to search arxiv, paperswithcode, etc.
