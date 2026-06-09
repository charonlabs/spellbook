See or change your runtime settings — the policies that shape how your context is managed.

Call with no arguments to see current settings. Call with `key` and `value` to change one.

Current settings:

- `ttl_enabled` — whether large tool results automatically register for TTL collapse.
- `ttl_turns` — how many turns a registered result stays full before collapsing.
- `ttl_char_threshold` — how large a result must be (in characters) to trigger auto-registration.

Changes are persisted and take effect immediately for future tool calls. Existing TTL records aren't rewritten — use ForgetToolResult to collapse something that's already tracked.
