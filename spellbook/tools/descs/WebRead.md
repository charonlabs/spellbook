Read clean text content from one or more URLs.

Extracts readable text from web pages, stripping navigation, ads, and boilerplate. Use this when you have a specific URL and want to read its content.

Usage:
- `WebRead(urls=["https://arxiv.org/abs/2401.12345"])` — read a paper
- `WebRead(urls=["https://docs.python.org/3/library/asyncio.html"], max_characters=20000)` — read documentation with more content

Parameters:
- `urls`: One or more URLs to extract content from.
- `max_characters`: Maximum characters per URL (default: 10000, max: 50000). Increase for long documents.

Tips:
- Use WebSearch first to find relevant URLs, then WebRead to get full content.
- For long documents, increase max_characters. For quick checks, the default is usually sufficient.
- Works well with documentation pages, blog posts, papers, and articles.
