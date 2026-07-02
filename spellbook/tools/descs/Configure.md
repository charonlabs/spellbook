See or change your runtime settings — the policies that shape how your context is managed.

Call with no arguments to see current settings. Call with `key` and `value` to change one.

Current settings:

- `ttl_enabled` — whether large tool results automatically register for TTL collapse.
- `ttl_turns` — how many turns a registered result stays full before collapsing.
- `ttl_char_threshold` — how large a result must be (in characters) to trigger auto-registration.
- `hearth_enabled` — whether the ambient hearth scheduler may start idle crackle turns.
- `hearth_interval_minutes` — how many idle minutes pass before a crackle turn; minimum 5.
- `hearth_quiet_hours` — optional local quiet window as `HH:MM-HH:MM`; empty string means none.

Changes are persisted and take effect immediately. Existing TTL records aren't rewritten — use ForgetToolResult to collapse something that's already tracked.
