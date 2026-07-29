"""Tool access to a configured Golem Minecraft surface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import httpx
from pydantic import BaseModel, Field

from spellbook.ir_types import IRToolTextBlock
from spellbook.minecraft_surface import MinecraftFocusMode, MinecraftSurface
from spellbook.tools.common import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)

MINECRAFT_TIMEOUT_SECONDS = 300.0
NO_GAME_MESSAGE = "the game isn't running."
NO_SURFACE_MESSAGE = (
    "Minecraft is unavailable because this entity has no Minecraft URL configured."
)

MinecraftAction = str

LOCAL_ACTIONS = frozenset({"boot", "shutdown", "config"})
GOLEM_ACTIONS = frozenset(
    {
        "activate",
        "batch",
        "blockat",
        "brace",
        "bridging",
        "build",
        "chat",
        "chatlog",
        "chest",
        "climb",
        "collect",
        "come",
        "craft",
        "deposit",
        "dig_stair",
        "dig_tunnel",
        "digat",
        "eat",
        "enchant",
        "entities",
        "events",
        "find",
        "fish",
        "follow",
        "gather",
        "gaze",
        "gesture",
        "goto",
        "goto_wp",
        "guard",
        "inventory",
        "job",
        "job_stop",
        "jobs",
        "journal",
        "listen",
        "lookat",
        "map",
        "mine",
        "outline",
        "passages",
        "pillar",
        "place",
        "placeitem",
        "pov",
        "pulse",
        "record",
        "reflexes",
        "rename",
        "roof",
        "safe_goto",
        "scene",
        "section",
        "shoot",
        "sign",
        "smelt",
        "snapshot",
        "stance",
        "state",
        "stop",
        "strike",
        "threatdebug",
        "tidy",
        "toss",
        "unfollow",
        "use",
        "useon",
        "waypoint",
        "waypoints",
        "walls",
        "where",
        "withdraw",
    }
)
ALL_ACTIONS = LOCAL_ACTIONS | GOLEM_ACTIONS | {"tick"}

REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "activate": ("x", "y", "z"),
    "blockat": ("x", "y", "z"),
    "come": ("name",),
    "craft": ("item",),
    "deposit": ("name",),
    "digat": ("x", "y", "z"),
    "enchant": ("item",),
    "find": ("name",),
    "follow": ("name",),
    "gather": ("resource",),
    "goto": ("x", "y", "z"),
    "goto_wp": ("name",),
    "lookat": ("x", "y", "z"),
    "mine": ("name",),
    "place": ("name",),
    "placeitem": ("name", "x", "y", "z"),
    "shoot": ("name",),
    "sign": ("x", "y", "z", "text"),
    "smelt": ("item",),
    "strike": ("name",),
    "toss": ("name",),
    "use": ("x", "y", "z"),
    "useon": ("name", "item"),
    "waypoint": ("name",),
    "withdraw": ("name",),
}

ECHO_PRIMARY_ARGS = ("name", "item", "resource", "template")
ECHO_ARG_ORDER = (
    "x",
    "y",
    "z",
    "count",
    "amount",
    "radius",
    "range",
    "dir",
    "up",
    "steps",
    "length",
    "face",
    "dest",
    "on",
    "slot",
    "set",
    "sec",
)
ECHO_SKIP_ARGS = frozenset({"brief", "verbose", "then", "fresh", "wait", "hud", "pov"})
MAX_ECHO_ARGS = 6
MAX_ECHO_CHARS = 220


class MinecraftInput(BaseModel):
    """Use the configured Minecraft surface."""

    action: MinecraftAction = Field(
        description="Minecraft action to perform. Core actions: boot, shutdown, tick, goto, scene, mine, craft, chat, config."
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific arguments, validated at runtime.",
    )


async def exec_minecraft(
    meta: ToolMetadata, input: MinecraftInput
) -> ToolExecutionResult:
    minecraft_url = _normalize_minecraft_url(meta.minecraft_url)
    surface = _ensure_surface(meta, minecraft_url)
    action = _normalize_action(input.action)
    args = _normalize_args(input.args)

    if action not in ALL_ACTIONS:
        raise ToolError(_unknown_action_message(action))

    if action != "boot" and not surface.booted:
        return _text_result(NO_GAME_MESSAGE, action=action, running=False)

    if action == "config":
        return _exec_config(surface=surface, args=args)

    async with httpx.AsyncClient(timeout=MINECRAFT_TIMEOUT_SECONDS) as client:
        if action == "boot":
            return await _exec_boot(
                client, minecraft_url=minecraft_url, surface=surface
            )
        if action == "shutdown":
            return await _exec_shutdown(
                client, minecraft_url=minecraft_url, surface=surface
            )

        _validate_required_args(action, args)
        data = await _get_json(
            client,
            minecraft_url=minecraft_url,
            path=f"/{action}",
            params=args,
        )
        await _echo_tool_call(
            client,
            minecraft_url=minecraft_url,
            surface=surface,
            action=action,
            args=args,
        )
        _require_ok(data)
        return _formatted_result(action, data, surface=surface)


async def _exec_boot(
    client: httpx.AsyncClient, *, minecraft_url: str, surface: MinecraftSurface
) -> ToolExecutionResult:
    data = await _get_json(client, minecraft_url=minecraft_url, path="/boot", params={})
    _require_ok(data)
    surface.mark_booted(
        chat_cursor=data.get("chatSeq") or data.get("chat_cursor"),
        event_cursor=data.get("eventSeq") or data.get("event_cursor"),
    )
    await _echo_tool_call(
        client, minecraft_url=minecraft_url, surface=surface, action="boot", args={}
    )
    result = _formatted_result("boot", data, surface=surface)
    text = _first_text(result)
    if text:
        text += "\n\nChat routing is on. Assistant text will be spoken in Minecraft."
        return _text_result(
            text, action="boot", data=result.display.get("data"), running=True
        )
    return result


async def _exec_shutdown(
    client: httpx.AsyncClient, *, minecraft_url: str, surface: MinecraftSurface
) -> ToolExecutionResult:
    try:
        await _get_json(client, minecraft_url=minecraft_url, path="/stop", params={})
        await _echo_tool_call(
            client,
            minecraft_url=minecraft_url,
            surface=surface,
            action="shutdown",
            args={},
        )
    except ToolError:
        pass
    surface.mark_shutdown()
    return _text_result(
        "Minecraft surface shut down. Chat routing is off.",
        action="shutdown",
        running=False,
    )


def _exec_config(
    *, surface: MinecraftSurface, args: dict[str, Any]
) -> ToolExecutionResult:
    allowed = {"chat_routing", "tool_call_echo", "focus_mode", "focus"}
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ToolError(f"config does not understand: {', '.join(unknown)}.")
    chat_routing = (
        _bool_arg(args["chat_routing"], field_name="chat_routing")
        if "chat_routing" in args
        else None
    )
    tool_call_echo = (
        _bool_arg(args["tool_call_echo"], field_name="tool_call_echo")
        if "tool_call_echo" in args
        else None
    )
    focus_value = args.get("focus_mode", args.get("focus"))
    focus_mode = _focus_mode(focus_value) if focus_value is not None else None
    surface.configure(
        chat_routing=chat_routing,
        tool_call_echo=tool_call_echo,
        focus_mode=focus_mode,
    )
    return _text_result(
        "Minecraft config: "
        f"chat routing {'on' if surface.chat_routing else 'off'}, "
        f"tool echo {'on' if surface.tool_call_echo else 'off'}, "
        f"focus {surface.focus_mode}.",
        action="config",
        running=surface.booted,
        chat_routing=surface.chat_routing,
        tool_call_echo=surface.tool_call_echo,
        focus_mode=surface.focus_mode,
    )


async def _get_json(
    client: httpx.AsyncClient,
    *,
    minecraft_url: str,
    path: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    query = {key: _query_value(value) for key, value in params.items()}
    query.setdefault("brief", "1")
    try:
        response = await client.get(f"{minecraft_url}{path}", params=query)
    except httpx.TimeoutException as exc:
        raise ToolError(f"Minecraft did not answer {path} before the timeout.") from exc
    except httpx.HTTPError as exc:
        raise ToolError(
            f"the game isn't running; I could not reach Golem at {minecraft_url}."
        ) from exc
    return _response_json(response)


def _response_json(response: httpx.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise ToolError(
            f"Minecraft returned HTTP {response.status_code}: "
            f"{_compact_response_text(response.text)}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ToolError("Minecraft returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise ToolError("Minecraft returned an unexpected response.")
    return cast(dict[str, Any], data)


def _require_ok(data: dict[str, Any]) -> None:
    ok = data.get("ok")
    if ok is False:
        raise ToolError(_natural_error(data.get("error")))
    if ok is not True:
        raise ToolError("Minecraft returned an unexpected response.")


async def _echo_tool_call(
    client: httpx.AsyncClient,
    *,
    minecraft_url: str,
    surface: MinecraftSurface,
    action: str,
    args: dict[str, Any],
) -> None:
    if action == "chat" or not surface.tool_call_echo:
        return
    message = _tool_echo_message(action, args)
    try:
        data = await _get_json(
            client,
            minecraft_url=minecraft_url,
            path="/chat",
            params={"msg": message},
        )
        _require_ok(data)
    except ToolError:
        return


def _tool_echo_message(action: str, args: dict[str, Any]) -> str:
    consumed: set[str] = set()
    parts = ["[Tool]", action]
    for key in ECHO_PRIMARY_ARGS:
        if key in args:
            parts.append(_echo_value(args[key]))
            consumed.add(key)
            break

    for key in ECHO_ARG_ORDER:
        if key in args and key not in consumed and key not in ECHO_SKIP_ARGS:
            parts.append(f"{key}={_echo_value(args[key])}")
            consumed.add(key)
        if len(parts) >= MAX_ECHO_ARGS + 2:
            break

    if len(parts) < MAX_ECHO_ARGS + 2:
        for key in sorted(args):
            if key in consumed or key in ECHO_SKIP_ARGS:
                continue
            parts.append(f"{key}={_echo_value(args[key])}")
            if len(parts) >= MAX_ECHO_ARGS + 2:
                break

    text = " ".join(part for part in parts if part)
    if len(text) <= MAX_ECHO_CHARS:
        return text
    return f"{text[: MAX_ECHO_CHARS - 3].rstrip()}..."


def _echo_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = " ".join(str(value).split())
    return text.encode("ascii", "ignore").decode("ascii")


def _formatted_result(
    action: str, data: dict[str, Any], *, surface: MinecraftSurface
) -> ToolExecutionResult:
    clean = cast(dict[str, Any], _strip_nulls(data))
    text = _format_text(action, clean)
    return ToolExecutionResult(
        content=[IRToolTextBlock(text=text)],
        display={
            "kind": "minecraft",
            "action": action,
            "body": text,
            "data": clean,
            "running": surface.booted,
            "chat_routing": surface.chat_routing,
            "tool_call_echo": surface.tool_call_echo,
            "focus_mode": surface.focus_mode,
        },
    )


def _format_text(action: str, data: dict[str, Any]) -> str:
    summary = data.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines = [summary.strip()]
    else:
        lines = [_headline(action, data)]

    spatial = _spatial_context(data)
    if spatial:
        lines.append("")
        lines.append("Spatial context:")
        lines.extend(f"- {line}" for line in spatial[:8])

    details = _compact_json(_detail_payload(data))
    if details:
        lines.append("")
        lines.append(f"Result: {details}")
    return "\n".join(lines)


def _headline(action: str, data: dict[str, Any]) -> str:
    match action:
        case "tick":
            pos = _render_pos(data.get("pos"))
            hp = data.get("hp")
            food = data.get("food")
            vitals = ", ".join(
                part
                for part in (
                    f"at {pos}" if pos else None,
                    f"HP {hp}/20" if isinstance(hp, int | float) else None,
                    f"food {food}/20" if isinstance(food, int | float) else None,
                )
                if part
            )
            return f"Minecraft tick: {vitals or 'no major changes'}."
        case "chat":
            if "said" in data:
                return f"You said in Minecraft chat: {data['said']}"
            recent = data.get("recent")
            if isinstance(recent, list):
                return f"Minecraft chat returned {len(recent)} recent line(s)."
        case "shutdown":
            return "Minecraft surface shut down."
    if "job" in data:
        return f"Minecraft started job {data['job']}."
    return f"Minecraft {action} complete."


def _detail_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if key not in {"ok", "summary"} and value not in ({}, [])
    }


def _spatial_context(value: object, *, label: str | None = None) -> list[str]:
    lines: list[str] = []
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        pos = mapping.get("pos") or mapping.get("at")
        rendered = _render_pos(pos)
        if rendered:
            name = mapping.get("name") or mapping.get("kind") or label
            dist = mapping.get("dist")
            direction = mapping.get("dir") or mapping.get("bearing")
            suffix = _render_distance(direction, dist)
            lines.append(
                f"{name + ' ' if isinstance(name, str) and name else ''}{rendered}{suffix}"
            )
        for key, child in mapping.items():
            if key in {"pos", "at"}:
                continue
            lines.extend(_spatial_context(child, label=str(key)))
    elif isinstance(value, list):
        for child in value:
            lines.extend(_spatial_context(child, label=label))
    return _dedupe(lines)


def _render_pos(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    mapping = cast(Mapping[str, object], value)
    coords: list[str] = []
    for axis in ("x", "y", "z"):
        item = mapping.get(axis)
        if not isinstance(item, int | float):
            return None
        coords.append(f"{item:g}")
    return f"({', '.join(coords)})"


def _render_distance(direction: object, dist: object) -> str:
    parts: list[str] = []
    if isinstance(direction, str) and direction:
        parts.append(direction)
    if isinstance(dist, int | float):
        parts.append(f"{dist:g} blocks")
    return f" - {'; '.join(parts)}" if parts else ""


def _dedupe(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return deduped


def _normalize_minecraft_url(value: str | None) -> str:
    if value is None or not value.strip():
        raise ToolError(NO_SURFACE_MESSAGE)
    return value.strip().rstrip("/")


def _ensure_surface(meta: ToolMetadata, minecraft_url: str) -> MinecraftSurface:
    surface = meta.minecraft_surface
    if isinstance(surface, MinecraftSurface):
        return surface
    surface = MinecraftSurface(minecraft_url=minecraft_url)
    meta.minecraft_surface = surface
    return surface


def _normalize_action(value: str) -> str:
    action = value.strip().lstrip("/").lower()
    if not action:
        raise ToolError("Minecraft action is required.")
    return action


def _normalize_args(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolError("Minecraft args must be an object.")
    args: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key or raw_value is None:
            continue
        if isinstance(raw_value, str | int | float | bool):
            args[key] = raw_value
        else:
            raise ToolError(f"args.{key} must be a string, number, or boolean.")
    return args


def _validate_required_args(action: str, args: dict[str, Any]) -> None:
    missing = [
        name
        for name in REQUIRED_ARGS.get(action, ())
        if name not in args or (isinstance(args[name], str) and not args[name].strip())
    ]
    if missing:
        raise ToolError(f"{action} requires {', '.join(missing)}.")


def _query_value(value: object) -> str | int | float:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str | int | float):
        return value
    raise ToolError("Minecraft args must be strings, numbers, or booleans.")


def _bool_arg(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ToolError(f"{field_name} must be true or false.")


def _focus_mode(value: object) -> MinecraftFocusMode:
    if not isinstance(value, str):
        raise ToolError("focus_mode must be explore, combat, build, or idle.")
    normalized = value.strip().lower()
    if normalized not in {"explore", "combat", "build", "idle"}:
        raise ToolError("focus_mode must be explore, combat, build, or idle.")
    return cast(MinecraftFocusMode, normalized)


def _strip_nulls(value: object) -> object:
    if isinstance(value, list):
        return [_strip_nulls(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _strip_nulls(item)
            for key, item in value.items()
            if item is not None
        }
    return value


def _compact_json(value: object) -> str:
    if value in ({}, []):
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _natural_error(value: object) -> str:
    text = _compact_response_text(str(value or "unknown Minecraft error"))
    if text.startswith("not ready") or text.startswith("bot not spawned"):
        return "the game isn't running; Golem has not spawned into the world yet."
    if text.startswith("no "):
        return f"you do not have or cannot see {text[3:]}."
    if "not in my line of sight" in text:
        return "you can't see that from here; walk closer or clear the view first."
    return f"Minecraft says: {text}"


def _unknown_action_message(action: str) -> str:
    core = "boot, shutdown, tick, goto, scene, mine, craft, chat, config"
    return f"unknown Minecraft action '{action}'. Core actions are: {core}."


def _compact_response_text(text: str) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "(empty response body)"
    if len(compact) > 500:
        return f"{compact[:497]}..."
    return compact


def _text_result(text: str, *, action: str, **display: Any) -> ToolExecutionResult:
    return ToolExecutionResult(
        content=[IRToolTextBlock(text=text)],
        display={"kind": "minecraft", "action": action, "body": text, **display},
    )


def _first_text(result: ToolExecutionResult) -> str:
    for block in result.content:
        if isinstance(block, IRToolTextBlock):
            return block.text
    return ""


MINECRAFT_TOOL: Tool[MinecraftInput] = Tool(
    name="Minecraft",
    input_model=MinecraftInput,
    exec=exec_minecraft,
    category="minecraft",
)
