Collapse a specific tool result immediately.

Use this when a tool output is no longer useful enough to keep in active context. Pass the `call_id` shown by `ReflectToolResults`; unique prefixes are accepted.

The full textual output is preserved in the session's `tool-outputs/` directory when needed, and the rendered context will show a compact pointer instead of the original result.
