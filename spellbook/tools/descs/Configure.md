Inspect or update runtime configuration.

Call `Configure()` with no arguments to see current runtime settings. To update a setting, call `Configure(key=..., value=...)`.

The current configurable settings are for tool result TTL behavior:

- `ttl_enabled` controls future automatic TTL registration for large tool results.
- `ttl_turns` controls how many completed turns newly auto-registered tool results remain full before collapsing.
- `ttl_char_threshold` controls how large textual tool output must be before it is auto-registered for TTL.

Configuration changes are persisted to the transcript and affect future behavior. Existing TTL records are not rewritten; use `ForgetToolResult` to collapse a specific existing result immediately.
