Minecraft as a worn surface through the configured Golem harness.

Use one action string and an `args` object. Core actions:

- `boot`: connect to the game and turn chat routing on.
- `shutdown`: stop the surface and turn chat routing off.
- `tick`: read the heartbeat.
- `goto`: move to coordinates with `x`, `y`, `z`, optional `range`.
- `scene`: glance at the current surroundings.
- `mine`: mine visible blocks by `name`, optional `count` and `radius`.
- `craft`: craft an `item`, optional `count`.
- `chat`: speak with `msg`, send paced `lines`, or listen with no args.
- `config`: set `chat_routing` or `tool_call_echo` true/false.

When booted, normal assistant text is also spoken in Minecraft chat, and Minecraft tool calls are echoed as short `[Tool] ...` chat lines unless `tool_call_echo` is off. Full verb reference lives in the Minecraft skill/field guide.
