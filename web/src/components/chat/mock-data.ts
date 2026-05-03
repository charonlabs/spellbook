/**
 * Mock conversation data for the Meta Chat surface.
 *
 * Models the Spellbook app-server protocol shapes:
 * - UIUserMessageEvent, UIAssistantTextEvent, UIThinkingEvent
 * - UIToolStartEvent, UIToolResultEvent
 * - AwarenessView (budget, segments, proposals)
 */

// ─── Content block types ────────────────────────────────────────

export type MockToolCall = {
  kind: 'tool_call'
  callId: string
  tool: string
  summary: string
}

export type ToolDisplay = Record<string, unknown> & { kind: string }

export type MockToolResult = {
  kind: 'tool_result'
  callId: string
  tool: string
  summary: string
  result: string
  isError?: boolean
  display?: ToolDisplay
}

export type MockContentBlock =
  | { kind: 'text'; text: string }
  | { kind: 'thinking'; text: string }
  | MockToolCall
  | MockToolResult

// ─── Turn types ─────────────────────────────────────────────────

type BaseMockTurn = {
  id: string
  role: 'user' | 'assistant'
  timestamp: Date
  /** Source of user messages. 'user'/'human'/'tui'/'web' = direct human, anything else = conduit */
  source?: string
  /** Duration of assistant turns in milliseconds (from first to last event) */
  durationMs?: number
}

export type ToolPair = {
  call: MockToolCall
  result?: MockToolResult
}

export interface ToolGroup {
  tool: string
  pairs: ToolPair[]
}

export type PreparedTurnSegment =
  | { type: 'text'; text: string }
  | { type: 'tools'; groups: ToolGroup[]; pairs: ToolPair[] }

export type ConduitType = 'notification' | 'context' | 'forwarded'

export type RawMockTurn = BaseMockTurn & {
  blocks: MockContentBlock[]
}

export type MockTurn = BaseMockTurn & {
  render:
    | { kind: 'user'; text: string }
    | {
        kind: 'conduit'
        source: string
        conduitType: ConduitType
        content: string
      }
    | {
        kind: 'assistant'
        thinkingText: string | null
        segments: PreparedTurnSegment[]
      }
}

// ─── Awareness types ────────────────────────────────────────────

export type MockBlock = {
  id: string
  title: string
  mode: 'full' | 'summary' | 'index' | 'headline'
  tokens: number
  turnRange: [number, number]
  pinned: boolean
  pinnedFacetCount: number
}

export type MockAwareness = {
  maxTokens: number
  usedTokens: number
  regime: 'calm' | 'warning' | 'forced'
  blocks: MockBlock[]
  model: string
  gauge: string
}

// ─── Turn preparation helpers ───────────────────────────────────

export const HUMAN_SOURCES = new Set(['user', 'queued', 'human', 'tui', 'web'])

export function containsConduitMarkup(text: string): boolean {
  return /(?:<|&lt;)\s*(?:chorus-)?(?:conduit|notification)\b/i.test(text)
}

export function stripConduitXml(text: string): string {
  return text
    .replace(/<(?:chorus-)?(?:conduit|notification)[^>]*>\n?/g, '')
    .replace(/\n?<\/(?:chorus-)?(?:conduit|notification)>/g, '')
    .replace(/&lt;(?:chorus-)?(?:conduit|notification)[^&]*&gt;\n?/g, '')
    .replace(/\n?&lt;\/(?:chorus-)?(?:conduit|notification)&gt;/g, '')
    .trim()
}

export function conduitTypeFromSource(source: string): ConduitType {
  if (source === 'telegram' || source.startsWith('telegram.')) return 'forwarded'
  if (source.includes('notification')) return 'notification'
  return 'context'
}

function conduitTypeFromTurn(source: string | undefined, text: string): ConduitType {
  if (source && !HUMAN_SOURCES.has(source)) {
    return conduitTypeFromSource(source)
  }

  if (/(?:<|&lt;)\s*(?:chorus-)?notification\b/i.test(text)) {
    return 'notification'
  }

  return 'context'
}

export function buildToolPairs(blocks: MockContentBlock[]): ToolPair[] {
  const pairs: ToolPair[] = []
  const pendingCalls = new Map<string, number>()

  for (const block of blocks) {
    if (block.kind === 'tool_call') {
      pendingCalls.set(block.callId, pairs.length)
      pairs.push({ call: block })
      continue
    }

    if (block.kind === 'tool_result') {
      const pairIndex = pendingCalls.get(block.callId)
      if (pairIndex !== undefined) {
        pairs[pairIndex].result = block
      }
    }
  }

  return pairs
}

export function groupToolBlocks(blocks: MockContentBlock[]): ToolGroup[] {
  const groups: ToolGroup[] = []

  for (const block of blocks) {
    if (block.kind === 'tool_call') {
      const lastGroup = groups[groups.length - 1]
      if (lastGroup && lastGroup.tool === block.tool) {
        lastGroup.pairs.push({ call: block })
      } else {
        groups.push({ tool: block.tool, pairs: [{ call: block }] })
      }
      continue
    }

    if (block.kind === 'tool_result') {
      for (let groupIndex = groups.length - 1; groupIndex >= 0; groupIndex -= 1) {
        const group = groups[groupIndex]

        for (const pair of group.pairs) {
          if (pair.call.callId === block.callId && !pair.result) {
            pair.result = block
            break
          }
        }

        if (group.pairs.some((pair) => pair.call.callId === block.callId && pair.result === block)) {
          break
        }
      }
    }
  }

  return groups
}

function prepareAssistantSegments(blocks: MockContentBlock[]): PreparedTurnSegment[] {
  const segments: PreparedTurnSegment[] = []
  let toolBuffer: MockContentBlock[] = []

  const flushTools = () => {
    if (toolBuffer.length === 0) return

    const groups = groupToolBlocks(toolBuffer)
    const pairs = buildToolPairs(toolBuffer)
    if (groups.length > 0 || pairs.length > 0) {
      segments.push({ type: 'tools', groups, pairs })
    }
    toolBuffer = []
  }

  for (const block of blocks) {
    if (block.kind === 'thinking') continue

    if (block.kind === 'text') {
      flushTools()
      segments.push({ type: 'text', text: block.text })
      continue
    }

    toolBuffer.push(block)
  }

  flushTools()
  return segments
}

export function prepareTurn(turn: RawMockTurn): MockTurn {
  if (turn.role === 'user') {
    const text = turn.blocks
      .filter((block): block is Extract<MockContentBlock, { kind: 'text' }> => block.kind === 'text')
      .map((block) => block.text)
      .join('\n\n')

    if (
      (turn.source && !HUMAN_SOURCES.has(turn.source)) ||
      containsConduitMarkup(text)
    ) {
      return {
        ...turn,
        render: {
          kind: 'conduit',
          source: turn.source || 'conduit',
          conduitType: conduitTypeFromTurn(turn.source, text),
          content: stripConduitXml(text),
        },
      }
    }

    return {
      ...turn,
      render: {
        kind: 'user',
        text,
      },
    }
  }

  const thinkingText = turn.blocks
    .filter((block): block is Extract<MockContentBlock, { kind: 'thinking' }> => block.kind === 'thinking')
    .map((block) => block.text)
    .join('\n\n')

  return {
    ...turn,
    render: {
      kind: 'assistant',
      thinkingText: thinkingText || null,
      segments: prepareAssistantSegments(turn.blocks),
    },
  }
}

// ─── Mock conversation ──────────────────────────────────────────

const now = new Date()
const minutesAgo = (m: number) => new Date(now.getTime() - m * 60_000)

const RAW_MOCK_TURNS: RawMockTurn[] = [
  // Turn 1: User asks to explore architecture
  {
    id: 'turn-1-user',
    role: 'user',
    timestamp: minutesAgo(47),
    blocks: [
      {
        kind: 'text',
        text: "I want to build a new page in the interface for the Meta Chat surface. Can you read the existing code and understand the architecture first?",
      },
    ],
  },
  // Turn 1: Assistant explores codebase
  {
    id: 'turn-1-assistant',
    role: 'assistant',
    timestamp: minutesAgo(46),
    blocks: [
      {
        kind: 'thinking',
        text: "The user wants me to understand the interface architecture before building. Let me read the key files — the page layout, the component structure, the design system, and how the existing data flow works. I should look at the SSE relay, the daemon hook, the stream components, and the globals.css to understand the visual language.\n\nThis is a substantial request — they want a whole new surface for the IIDE. Let me be thorough in my reading.",
      },
      {
        kind: 'text',
        text: "I'll start by reading the existing interface code to understand the architecture.",
      },
      { kind: 'tool_call', callId: 'tc-1a', tool: 'Read', summary: 'interface/src/app/page.tsx' },
      { kind: 'tool_result', callId: 'tc-1a', tool: 'Read', summary: 'interface/src/app/page.tsx', result: '194 lines' },
      { kind: 'tool_call', callId: 'tc-1b', tool: 'Read', summary: 'interface/src/app/layout.tsx' },
      { kind: 'tool_result', callId: 'tc-1b', tool: 'Read', summary: 'interface/src/app/layout.tsx', result: '48 lines' },
      { kind: 'tool_call', callId: 'tc-1c', tool: 'Read', summary: 'interface/src/app/globals.css' },
      { kind: 'tool_result', callId: 'tc-1c', tool: 'Read', summary: 'interface/src/app/globals.css', result: '256 lines' },
      { kind: 'tool_call', callId: 'tc-1d', tool: 'Grep', summary: '"useDaemon|use-daemon" in interface/' },
      { kind: 'tool_result', callId: 'tc-1d', tool: 'Grep', summary: 'Grep for useDaemon', result: '3 files matched' },
      { kind: 'tool_call', callId: 'tc-1e', tool: 'Read', summary: 'interface/src/lib/use-daemon.ts' },
      { kind: 'tool_result', callId: 'tc-1e', tool: 'Read', summary: 'interface/src/lib/use-daemon.ts', result: '445 lines' },
      {
        kind: 'text',
        text: "The interface is a Next.js 16 app with a clean architecture:\n\n**Stack:** React 19, Tailwind v4, shadcn/ui, Motion for animations, d3-force for the World Tree constellation.\n\n**Current surfaces:**\n- `/` — The Yard: entity cards, interleaved stream (chat + entity output + system events), meta status bar\n- `/tree` — World Tree constellation (canvas-based d3-force simulation)\n\n**Data flow:** An SSE relay at `/api/events` connects server-side to the daemon WebSocket at port 7700, relaying transcript IR and lifecycle events. The `useDaemon` hook consumes these on the client, maintaining entity state and building stream items in real-time.\n\n**Design system:** Dark-only with warm neutrals — ground `#0a0a0c`, surface `#111114`, accent gold `#d4a050`. Entity identity colors (builder=gold, reader=teal, reviewer=purple). Custom `.entity-prose` for monospace entity output.\n\nThe key thing I see: the current Yard mixes meta-Claude conversation with entity events in one stream. For the Meta Chat surface, we need a dedicated conversational view that connects directly to the Spellbook app server at port 8100, completely separate from the entity timeline.",
      },
    ],
  },
  // Turn 2: User discusses design
  {
    id: 'turn-2-user',
    role: 'user',
    timestamp: minutesAgo(38),
    blocks: [
      {
        kind: 'text',
        text: "Good analysis. Now let's think about the design. The feeling should be: \"I'm sitting across from a mind that's thinking with me.\" Dark, amber, constellation — same family as the World Tree. Two voices that look genuinely different.",
      },
    ],
  },
  // Turn 2: Assistant responds with design ideas
  {
    id: 'turn-2-assistant',
    role: 'assistant',
    timestamp: minutesAgo(37),
    blocks: [
      {
        kind: 'thinking',
        text: "Ryan wants the Meta Chat to feel intimate and conversational — a workspace where a mind lives. The design direction he shared earlier is clear:\n\n- Warmth without coziness. A well-lit study, not a living room.\n- Depth on demand. Clean surface with detail one click away.\n- Presence, not performance. Honest about state.\n- The conversation as landscape. Old messages age visually.\n- Two kinds of voice. Different visual treatments for human vs entity.\n- Tool calls as craftsmanship.\n- Context gauge as ambient awareness.\n\nLet me think about how to translate this into React components...",
      },
      {
        kind: 'text',
        text: "Here's how I'd translate that design direction into concrete UI decisions:\n\n**Two voices:**\n- Human messages: right-aligned, compact, warm amber tint — your words are seeds, not essays\n- Entity messages: full-width, monospace `.entity-prose`, the working surface — this is where the mind unfolds\n\n**Depth on demand:**\n- Thinking blocks collapsed by default → \"*thought for a moment*\" with expand toggle\n- Tool calls as compact grouped badges → expand to see call/result detail\n- Code blocks with syntax highlighting, diffs as real diffs\n\n**The conversation as landscape:**\n- Recent messages vivid, older messages gradually fade (opacity shift)\n- Block boundaries subtly visible — like geological strata\n\n**Context gauge as ambient awareness:**\n- Not a progress bar. A thin 2px line below the header that shifts color with regime:\n  - Calm: warm gold\n  - Warning: amber pulse\n  - Forced: red\n- The gauge chip in the header shows the number, but the *feeling* comes from the line\n\n**Presence:**\n- A subtle breathing dot when the entity is thinking\n- The input area reflects state — active when idle, muted when working\n- The header honestly shows: model, working state, context usage",
      },
    ],
  },
  // Turn 3: User says build it
  {
    id: 'turn-3-user',
    role: 'user',
    timestamp: minutesAgo(22),
    blocks: [
      {
        kind: 'text',
        text: "Love it. Build the mockup. Start with just a page at /chat with dummy data — no wiring yet. I want to see it and iterate on the visual feel.",
      },
    ],
  },
  // Turn 3: Assistant builds it
  {
    id: 'turn-3-assistant',
    role: 'assistant',
    timestamp: minutesAgo(21),
    blocks: [
      {
        kind: 'text',
        text: "On it. I'll create the mockup with realistic dummy data so we can iterate on the visual feel before wiring anything.",
      },
      { kind: 'tool_call', callId: 'tc-3a', tool: 'Write', summary: 'components/chat/mock-data.ts' },
      { kind: 'tool_result', callId: 'tc-3a', tool: 'Write', summary: 'Created mock-data.ts', result: 'ok' },
      { kind: 'tool_call', callId: 'tc-3b', tool: 'Write', summary: 'components/chat/context-gauge.tsx' },
      { kind: 'tool_result', callId: 'tc-3b', tool: 'Write', summary: 'Created context-gauge.tsx', result: 'ok' },
      { kind: 'tool_call', callId: 'tc-3c', tool: 'Write', summary: 'components/chat/tool-cards.tsx' },
      { kind: 'tool_result', callId: 'tc-3c', tool: 'Write', summary: 'Created tool-cards.tsx', result: 'ok' },
      { kind: 'tool_call', callId: 'tc-3d', tool: 'Write', summary: 'components/chat/turn.tsx' },
      { kind: 'tool_result', callId: 'tc-3d', tool: 'Write', summary: 'Created turn.tsx', result: 'ok' },
      { kind: 'tool_call', callId: 'tc-3e', tool: 'Write', summary: 'app/chat/page.tsx' },
      { kind: 'tool_result', callId: 'tc-3e', tool: 'Write', summary: 'Created chat page', result: 'ok' },
      {
        kind: 'text',
        text: "The mockup is live at `/chat`. Head over and take a look — I'm curious what you think of the two-voice treatment and the ambient gauge.\n\nKey things to notice:\n- User messages are compact and warm, right-aligned\n- Entity text uses monospace and takes full width\n- Tool calls group by type (Read×3 collapses nicely)\n- The context gauge line below the header shifts from gold → red\n- Older turns fade slightly — the conversation ages like a landscape",
      },
    ],
  },
  // Turn 4: User feedback
  {
    id: 'turn-4-user',
    role: 'user',
    timestamp: minutesAgo(3),
    blocks: [
      {
        kind: 'text',
        text: "This looks great. Can you make the thinking blocks a bit more subtle? And I think the user messages could use a slightly different font — not monospace, something warmer.",
      },
    ],
  },
  // Turn 4: Assistant responds
  {
    id: 'turn-4-assistant',
    role: 'assistant',
    timestamp: minutesAgo(2),
    blocks: [
      {
        kind: 'text',
        text: "Good calls. I'll make thinking blocks use a dimmer, italic treatment and switch user messages to the Outfit font (your `--font-sans` variable) — it's already loaded and has that warm, readable quality that contrasts nicely with the monospace entity voice.\n\nThe two-font split reinforces the *two kinds of voice* principle: proportional for the human (conversational, flowing), monospace for the entity (precise, working).",
      },
    ],
  },
]

export const MOCK_TURNS: MockTurn[] = RAW_MOCK_TURNS.map((turn) => prepareTurn(turn))

// ─── Mock awareness state ───────────────────────────────────────

export const MOCK_AWARENESS: MockAwareness = {
  maxTokens: 200_000,
  usedTokens: 127_400,
  regime: 'calm',
  model: 'claude-opus-4-6',
  gauge: '█████████░░░░░ 64%',
  blocks: [
    {
      id: 'block-orientation',
      title: 'Orientation & repo exploration',
      mode: 'summary',
      tokens: 2_400,
      turnRange: [1, 4],
      pinned: false,
      pinnedFacetCount: 0,
    },
    {
      id: 'block-architecture',
      title: 'Interface architecture deep-dive',
      mode: 'summary',
      tokens: 3_100,
      turnRange: [5, 8],
      pinned: true,
      pinnedFacetCount: 1,
    },
    {
      id: 'block-spellbook',
      title: 'Spellbook protocol analysis',
      mode: 'full',
      tokens: 18_200,
      turnRange: [9, 14],
      pinned: false,
      pinnedFacetCount: 0,
    },
    {
      id: 'block-design',
      title: 'Design direction & mockup',
      mode: 'full',
      tokens: 24_600,
      turnRange: [15, 22],
      pinned: false,
      pinnedFacetCount: 0,
    },
  ],
}
