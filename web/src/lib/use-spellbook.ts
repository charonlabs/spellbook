'use client'

import { startTransition, useCallback, useEffect, useRef, useState } from 'react'
import {
  prepareTurn,
  type MockAwareness,
  type MockBlock,
  type MockContentBlock,
  type MockTurn,
  type RawMockTurn,
  type ToolDisplay,
} from '@/components/chat/mock-data'

// ─── Spellbook server event types ───────────────────────────────

interface SpellbookEvent {
  type: string
  [key: string]: unknown
}

interface CatchupEvent {
  type: 'catchup'
  main_pane: CatchupReplayEvent[]
  awareness: CatchupAwareness
  gauge: string
  working: boolean
  model: string
  cwd: string
  surface?: string
  surface_time?: string
  tool_cards_expanded_by_default?: boolean
}

interface CatchupReplayEvent {
  _type: string
  text?: string
  source?: string
  time?: string | null
  call_id?: string
  name?: string
  summary?: string
  result?: string
  is_error?: boolean
  display?: unknown
  block_id?: string
  mode?: string
  title?: string
  turn_range?: [number, number]
  body?: string
}

interface CatchupAwareness {
  budget?: {
    max_input_tokens?: number
    current_input_tokens?: number | null
    estimated_input_tokens?: number
    regime?: string
  }
  segments?: CatchupSegment[]
}

interface CatchupSegment {
  id: string
  kind: string
  title: string
  mode: string
  estimated_tokens: number
  block_id?: string | null
  pinned?: boolean
  pinned_facet_count?: number
  turn_range?: [number, number] | null
}

// ─── Core server event types ───────────────────────────────────

interface CoreServerEvent {
  kind: string
  time?: string | null
  [key: string]: unknown
}

interface CoreCatchupEvent extends CoreServerEvent {
  kind: 'catchup'
  rehydrated: CoreRehydration
  surface?: string | null
  surface_time?: string | null
}

interface CoreAwarenessResponse {
  kind: 'awareness'
  snapshot: CoreAwarenessSnapshot
}

interface CoreHealthResponse {
  kind: 'health'
  model: string
  state: string
}

interface CoreAwarenessSnapshot {
  homunculus?: {
    budget?: {
      max_tokens?: number
      current_input_tokens?: number | null
      regime?: string
    }
    semantic_blocks?: CoreSemanticBlock[]
  }
}

interface CoreRehydration {
  records?: CoreRecord[]
  blocks?: CoreBlock[]
  config?: {
    model?: string
    hom_config?: {
      max_tokens?: number
    }
  }
  semantic_blocks?: CoreSemanticBlock[]
}

interface CoreRecord {
  ir: string
  time?: string | null
  turn?: number
  seq?: number
  event?: CoreBlock
}

interface CoreSemanticBlock {
  id: string
  idx?: number
  title?: string
  mode?: string
  toks?: CoreTokenCount | null
  full_toks?: CoreTokenCount | null
  range?: {
    start_block?: number
    end_block?: number
  }
  pin?: unknown
  facet_pins?: unknown[]
}

interface CoreTokenCount {
  tokens?: number
}

interface CoreBlock {
  type: string
  time?: string | null
  origin?: string
  text?: string
  call_id?: string
  tool?: string
  input?: Record<string, unknown>
  content?: CoreToolResultContent[]
  display?: unknown
  is_error?: boolean
  source?: CoreImageSource
}

interface CoreImageSource {
  type?: string
  url?: string
  media_type?: string
}

interface CoreToolResultContent {
  type: string
  text?: string
  source?: CoreImageSource
}

interface CoreInboundMessage {
  blocks?: CoreBlock[]
  source_metadata?: Record<string, unknown>
}

interface CoreStreamEvent {
  kind: string
  time?: string | null
  text?: string
}

// ─── Connection state ───────────────────────────────────────────

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error'

const DEFAULT_AWARENESS: MockAwareness = {
  maxTokens: 200_000,
  usedTokens: 0,
  regime: 'calm',
  blocks: [],
  model: 'unknown',
  gauge: '',
}

// ─── Turn builder (accumulates streaming events into turns) ─────

class TurnBuilder {
  private turns: MockTurn[] = []
  private currentTurn: MockContentBlock[] | null = null
  private currentThinking = ''
  private currentText = ''
  private currentTurnStartTime: Date | null = null
  private currentTurnEndTime: Date | null = null
  private turnCounter = 0

  private pushTurn(turn: RawMockTurn): void {
    this.turns.push(prepareTurn(turn))
  }

  private reset(): void {
    this.turns = []
    this.currentTurn = null
    this.currentThinking = ''
    this.currentText = ''
    this.currentTurnStartTime = null
    this.currentTurnEndTime = null
    this.turnCounter = 0
  }

  private ensureTurn(): void {
    if (!this.currentTurn) {
      this.startTurn()
    }
  }

  /** Reset and load from already-projected raw turns */
  loadRawTurns(turns: RawMockTurn[]): MockTurn[] {
    this.reset()

    for (const turn of turns) {
      this.pushTurn(turn)
    }

    this.turnCounter = turns.length
    return [...this.turns]
  }

  /** Reset and load from catchup replay events */
  loadCatchup(events: CatchupReplayEvent[]): MockTurn[] {
    this.reset()

    let currentUserBlocks: MockContentBlock[] = []
    let currentUserSource = 'user'
    let currentUserTime: Date | null = null
    let currentAssistantBlocks: MockContentBlock[] = []
    let currentAssistantStartTime: Date | null = null
    let currentAssistantEndTime: Date | null = null
    let inAssistant = false

    const parseTime = (time?: string | null): Date | null => {
      if (!time) return null
      const d = new Date(time)
      return isNaN(d.getTime()) ? null : d
    }

    const flushUser = () => {
      if (currentUserBlocks.length === 0) return

      this.pushTurn({
        id: `catchup-user-${this.turnCounter++}`,
        role: 'user',
        blocks: currentUserBlocks,
        timestamp: currentUserTime || new Date(),
        source: currentUserSource,
      })
      currentUserBlocks = []
      currentUserSource = 'user'
      currentUserTime = null
    }

    const flushAssistant = () => {
      if (currentAssistantBlocks.length === 0) return

      const start = currentAssistantStartTime
      const end = currentAssistantEndTime || currentAssistantStartTime
      let durationMs: number | undefined
      if (start && end) {
        durationMs = end.getTime() - start.getTime()
        if (durationMs < 0) durationMs = undefined
      }

      this.pushTurn({
        id: `catchup-assistant-${this.turnCounter++}`,
        role: 'assistant',
        blocks: currentAssistantBlocks,
        timestamp: end || new Date(),
        durationMs,
      })
      currentAssistantBlocks = []
      currentAssistantStartTime = null
      currentAssistantEndTime = null
    }

    for (const event of events) {
      switch (event._type) {
        case 'UIUserMessageEvent': {
          if (inAssistant) {
            flushAssistant()
            inAssistant = false
          }
          flushUser()
          currentUserSource = event.source || 'user'
          currentUserTime = parseTime(event.time)
          currentUserBlocks.push({ kind: 'text', text: event.text || '' })
          flushUser()
          break
        }
        case 'UIThinkingEvent': {
          if (!inAssistant) {
            flushUser()
            inAssistant = true
          }
          const t0 = parseTime(event.time)
          if (!currentAssistantStartTime) currentAssistantStartTime = t0
          if (t0) currentAssistantEndTime = t0
          currentAssistantBlocks.push({ kind: 'thinking', text: event.text || '' })
          break
        }
        case 'UIAssistantTextEvent': {
          if (!inAssistant) {
            flushUser()
            inAssistant = true
          }
          const t1 = parseTime(event.time)
          if (!currentAssistantStartTime) currentAssistantStartTime = t1
          if (t1) currentAssistantEndTime = t1
          currentAssistantBlocks.push({ kind: 'text', text: event.text || '' })
          break
        }
        case 'UIToolStartEvent': {
          if (!inAssistant) {
            flushUser()
            inAssistant = true
          }
          const t2 = parseTime(event.time)
          if (!currentAssistantStartTime) currentAssistantStartTime = t2
          if (t2) currentAssistantEndTime = t2
          currentAssistantBlocks.push({
            kind: 'tool_call',
            callId: event.call_id || `tc-${this.turnCounter}`,
            tool: event.name || 'Tool',
            summary: event.summary || '',
          })
          break
        }
        case 'UIToolResultEvent': {
          if (!inAssistant) {
            flushUser()
            inAssistant = true
          }
          const t3 = parseTime(event.time)
          if (!currentAssistantStartTime) currentAssistantStartTime = t3
          if (t3) currentAssistantEndTime = t3
          currentAssistantBlocks.push({
            kind: 'tool_result',
            callId: event.call_id || `tr-${this.turnCounter}`,
            tool: event.name || 'Tool',
            summary: event.summary || '',
            result: event.result || '',
            isError: event.is_error || false,
            display: (event.display as ToolDisplay) || undefined,
          })
          break
        }
        case 'UIMemoryCardEvent': {
          break
        }
      }
    }

    flushUser()
    flushAssistant()

    return [...this.turns]
  }

  /** Begin a new streaming turn */
  startTurn(time?: string | null): void {
    this.currentTurn = []
    this.currentThinking = ''
    this.currentText = ''
    this.currentTurnStartTime = time ? new Date(time) : new Date()
    this.currentTurnEndTime = null
  }

  /** Update the end time from a streaming event */
  noteEventTime(time?: string | null): void {
    if (time) {
      const d = new Date(time)
      if (!isNaN(d.getTime())) this.currentTurnEndTime = d
    }
  }

  /** Accumulate a thinking delta */
  addThinkingDelta(text: string): void {
    this.ensureTurn()
    this.currentThinking += text
  }

  /** End thinking — flush as a block */
  endThinking(): void {
    this.ensureTurn()
    if (this.currentThinking && this.currentTurn) {
      this.currentTurn.push({ kind: 'thinking', text: this.currentThinking })
      this.currentThinking = ''
    }
  }

  /** Accumulate a text delta */
  addTextDelta(text: string): void {
    this.ensureTurn()
    this.currentText += text
  }

  /** End text — flush as a block */
  endText(): void {
    this.ensureTurn()
    if (this.currentText && this.currentTurn) {
      this.currentTurn.push({ kind: 'text', text: this.currentText })
      this.currentText = ''
    }
  }

  /** Add a tool start */
  addToolStart(callId: string, tool: string, summary: string): void {
    this.ensureTurn()
    if (this.currentTurn) {
      this.currentTurn.push({ kind: 'tool_call', callId, tool, summary })
    }
  }

  /** Add a tool result */
  addToolResult(
    callId: string,
    tool: string,
    summary: string,
    result: string,
    isError: boolean,
    display?: ToolDisplay
  ): void {
    this.ensureTurn()
    if (this.currentTurn) {
      this.currentTurn.push({ kind: 'tool_result', callId, tool, summary, result, isError, display })
    }
  }

  private buildInProgressBlocks(): MockContentBlock[] {
    if (!this.currentTurn) return []

    const inProgressBlocks: MockContentBlock[] = [...this.currentTurn]

    if (this.currentThinking) {
      inProgressBlocks.push({ kind: 'thinking', text: this.currentThinking })
    }
    if (this.currentText) {
      inProgressBlocks.push({ kind: 'text', text: this.currentText })
    }

    return inProgressBlocks
  }

  /** End the current turn, commit to turn list */
  endTurn(time?: string | null): MockTurn[] {
    if (this.currentText) this.endText()
    if (this.currentThinking) this.endThinking()

    if (time) this.noteEventTime(time)
    const endTime = this.currentTurnEndTime || this.currentTurnStartTime || new Date()
    const startTime = this.currentTurnStartTime
    let durationMs: number | undefined
    if (startTime && endTime) {
      durationMs = endTime.getTime() - startTime.getTime()
      if (durationMs < 0) durationMs = undefined
    }

    if (this.currentTurn && this.currentTurn.length > 0) {
      this.pushTurn({
        id: `live-assistant-${this.turnCounter++}`,
        role: 'assistant',
        blocks: this.currentTurn,
        timestamp: endTime,
        durationMs,
      })
    }

    this.currentTurn = null
    this.currentTurnStartTime = null
    this.currentTurnEndTime = null
    return [...this.turns]
  }

  /** Add a user message (from queued_message_delivered or direct send) */
  addUserMessage(text: string, source: string = 'user', time?: string | null): MockTurn[] {
    const timestamp = parseOptionalDate(time) || new Date()

    this.pushTurn({
      id: `live-user-${this.turnCounter++}`,
      role: 'user',
      blocks: [{ kind: 'text', text }],
      timestamp,
      source,
    })
    return [...this.turns]
  }

  /** Get current snapshot including in-progress turn */
  snapshot(): MockTurn[] {
    const result = [...this.turns]
    const inProgressBlocks = this.buildInProgressBlocks()

    if (this.currentTurn && inProgressBlocks.length > 0) {
      result.push(
        prepareTurn({
          id: `live-streaming-${this.turnCounter}`,
          role: 'assistant',
          blocks: inProgressBlocks,
          timestamp: this.currentTurnEndTime || this.currentTurnStartTime || new Date(),
        })
      )
    }

    return result
  }
}

// ─── Awareness mapper ───────────────────────────────────────────

function mapAwareness(raw: CatchupAwareness | undefined, model: string, gauge: string): MockAwareness {
  const budget = raw?.budget
  const maxTokens = budget?.max_input_tokens || 200_000
  const usedTokens = budget?.current_input_tokens ?? budget?.estimated_input_tokens ?? 0
  const regime = (budget?.regime === 'warning' || budget?.regime === 'forced')
    ? budget.regime as 'warning' | 'forced'
    : 'calm'

  const blocks: MockBlock[] = (raw?.segments || [])
    .filter((seg) => seg.block_id)
    .map((seg) => ({
      id: seg.block_id || seg.id,
      title: seg.title,
      mode: (seg.mode as MockBlock['mode']) || 'full',
      tokens: seg.estimated_tokens,
      turnRange: (seg.turn_range || [0, 0]) as [number, number],
      pinned: seg.pinned || false,
      pinnedFacetCount: seg.pinned_facet_count || 0,
    }))

  return { maxTokens, usedTokens, regime, blocks, model, gauge }
}

function parseOptionalDate(time?: string | null): Date | null {
  if (!time) return null
  const parsed = new Date(time)
  return isNaN(parsed.getTime()) ? null : parsed
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isCoreServerEvent(event: SpellbookEvent | CoreServerEvent): event is CoreServerEvent {
  return typeof (event as CoreServerEvent).kind === 'string'
}

function isCoreCatchup(event: CoreServerEvent): event is CoreCatchupEvent {
  return event.kind === 'catchup' && isRecord(event.rehydrated)
}

function httpBaseFromServerUrl(url: string): string {
  const parsed = new URL(url)
  if (parsed.protocol === 'ws:') {
    parsed.protocol = 'http:'
  } else if (parsed.protocol === 'wss:') {
    parsed.protocol = 'https:'
  }
  parsed.pathname = ''
  parsed.search = ''
  parsed.hash = ''
  return parsed.toString().replace(/\/$/, '')
}

function modelFromCoreRehydration(rehydrated: CoreRehydration | undefined): string {
  return rehydrated?.config?.model || 'unknown'
}

function mapCoreRehydrationAwareness(rehydrated: CoreRehydration | undefined): MockAwareness {
  const model = modelFromCoreRehydration(rehydrated)
  const maxTokens = rehydrated?.config?.hom_config?.max_tokens || DEFAULT_AWARENESS.maxTokens
  const blocks = mapCoreSemanticBlocks(rehydrated?.semantic_blocks)
  return {
    maxTokens,
    usedTokens: 0,
    regime: 'calm',
    blocks,
    model,
    gauge: '',
  }
}

function mapCoreAwarenessResponse(
  raw: CoreAwarenessResponse,
  model: string,
  gauge: string,
): MockAwareness {
  const homunculus = raw.snapshot.homunculus
  const budget = homunculus?.budget
  const maxTokens = budget?.max_tokens || DEFAULT_AWARENESS.maxTokens
  const usedTokens = budget?.current_input_tokens ?? 0
  const regime = mapCoreRegime(budget?.regime)
  const blocks = mapCoreSemanticBlocks(homunculus?.semantic_blocks)

  return {
    maxTokens,
    usedTokens,
    regime,
    blocks,
    model,
    gauge,
  }
}

function mapCoreRegime(regime: string | undefined): MockAwareness['regime'] {
  if (regime === 'warning' || regime === 'forced') return regime
  if (regime === 'critical') return 'forced'
  return 'calm'
}

function mapCoreSemanticBlocks(blocks: CoreSemanticBlock[] | undefined): MockBlock[] {
  return (blocks || []).map((block) => {
    const start = block.range?.start_block ?? 0
    const end = block.range?.end_block ?? start
    const pinnedFacetCount = block.facet_pins?.length || 0

    return {
      id: block.id,
      title: block.title || `Block ${block.idx ?? '?'}`,
      mode: mapCoreBlockMode(block.mode),
      tokens: block.toks?.tokens ?? block.full_toks?.tokens ?? 0,
      turnRange: [start, end] as [number, number],
      pinned: Boolean(block.pin) || pinnedFacetCount > 0,
      pinnedFacetCount,
    }
  })
}

function mapCoreBlockMode(mode: string | undefined): MockBlock['mode'] {
  if (mode === 'summary' || mode === 'index' || mode === 'headline') return mode
  return 'full'
}

function coreRehydrationToTurns(rehydrated: CoreRehydration): RawMockTurn[] {
  const turns: RawMockTurn[] = []
  let assistantBlocks: MockContentBlock[] = []
  let assistantStartTime: Date | null = null
  let assistantEndTime: Date | null = null

  const flushAssistant = () => {
    if (assistantBlocks.length === 0) return

    const start = assistantStartTime
    const end = assistantEndTime || assistantStartTime || new Date()
    let durationMs: number | undefined
    if (start && end) {
      const duration = end.getTime() - start.getTime()
      if (duration >= 0) durationMs = duration
    }

    turns.push({
      id: `core-catchup-assistant-${turns.length}`,
      role: 'assistant',
      blocks: assistantBlocks,
      timestamp: end,
      durationMs,
    })

    assistantBlocks = []
    assistantStartTime = null
    assistantEndTime = null
  }

  const noteAssistantTime = (time?: string | null) => {
    const parsed = parseOptionalDate(time)
    if (!parsed) return
    if (!assistantStartTime) assistantStartTime = parsed
    assistantEndTime = parsed
  }

  for (const record of rehydrated.records || []) {
    if (record.ir === 'turn_end') {
      flushAssistant()
      continue
    }

    if (record.ir !== 'event' || !record.event) continue

    const block = record.event
    const time = parseOptionalDate(block.time || record.time)

    if (isCoreVisibleUserBlock(block)) {
      flushAssistant()
      turns.push({
        id: `core-catchup-user-${turns.length}`,
        role: 'user',
        blocks: [coreUserBlockToText(block)],
        timestamp: time || new Date(),
        source: coreUserSource(block),
      })
      continue
    }

    const contentBlock = coreAssistantBlockToContent(block)
    if (!contentBlock) continue

    noteAssistantTime(block.time || record.time)
    assistantBlocks.push(contentBlock)
  }

  flushAssistant()
  return turns
}

function isCoreVisibleUserBlock(block: CoreBlock): boolean {
  if (block.type === 'user_text') {
    return block.origin === 'human' || block.origin === 'conduit'
  }

  if (block.type === 'image') {
    return block.origin === 'human' || block.origin === 'conduit'
  }

  return false
}

function coreUserBlockToText(block: CoreBlock): MockContentBlock {
  if (block.type === 'image') {
    return { kind: 'text', text: coreImagePlaceholder(block.source) }
  }

  return { kind: 'text', text: block.text || '' }
}

function coreUserSource(block: CoreBlock): string {
  if (block.origin === 'human') return 'user'
  if (block.origin === 'conduit') return 'conduit'
  return block.origin || 'user'
}

function coreAssistantBlockToContent(block: CoreBlock): MockContentBlock | null {
  switch (block.type) {
    case 'assistant_text':
      return { kind: 'text', text: block.text || '' }
    case 'thinking':
      return { kind: 'thinking', text: block.text || '' }
    case 'tool_call':
      return {
        kind: 'tool_call',
        callId: block.call_id || 'unknown-call',
        tool: block.tool || 'Tool',
        summary: summarizeToolInput(block.input),
      }
    case 'tool_result': {
      const result = coreToolResultText(block)
      return {
        kind: 'tool_result',
        callId: block.call_id || 'unknown-call',
        tool: block.tool || 'Tool',
        summary: summarizeToolResult(block, result),
        result,
        isError: block.is_error || false,
        display: coreToolDisplay(block.display),
      }
    }
    default:
      return null
  }
}

function extractCoreInboundText(message: CoreInboundMessage | undefined): string {
  return (message?.blocks || [])
    .map((block) => {
      if (block.type === 'user_text') return block.text || ''
      if (block.type === 'image') return coreImagePlaceholder(block.source)
      return ''
    })
    .filter(Boolean)
    .join('\n\n')
}

function extractCoreInboundSource(message: CoreInboundMessage | undefined): string {
  const metadata = message?.source_metadata
  const source = metadata?.source
  if (typeof source === 'string' && source) return source

  const origin = metadata?.origin
  if (typeof origin === 'string' && origin) return origin

  const firstOrigin = message?.blocks?.find((block) => typeof block.origin === 'string')?.origin
  if (firstOrigin === 'human') return 'user'
  return firstOrigin || 'user'
}

function coreToolResultText(block: CoreBlock): string {
  return (block.content || [])
    .map((item) => {
      if (item.type === 'tool_text') return item.text || ''
      if (item.type === 'image') return coreImagePlaceholder(item.source)
      return ''
    })
    .filter(Boolean)
    .join('\n\n')
}

function coreToolDisplay(display: unknown): ToolDisplay | undefined {
  if (!isRecord(display) || typeof display.kind !== 'string') return undefined
  return display as ToolDisplay
}

function summarizeToolInput(input: Record<string, unknown> | undefined): string {
  if (!input) return ''

  const summaryKeys = [
    'description',
    'command',
    'file_path',
    'path',
    'query',
    'pattern',
    'block_id',
    'facet_id',
    'skill',
  ]

  for (const key of summaryKeys) {
    const value = input[key]
    if (typeof value === 'string' && value.trim()) return truncateOneLine(value)
  }

  return truncateOneLine(JSON.stringify(input))
}

function summarizeToolResult(block: CoreBlock, result: string): string {
  if (isRecord(block.display)) {
    const summary = block.display.summary
    if (typeof summary === 'string' && summary.trim()) return truncateOneLine(summary)
  }

  const firstLine = result.split('\n').find((line) => line.trim())
  return truncateOneLine(firstLine || (block.is_error ? 'error' : 'done'))
}

function coreImagePlaceholder(source: CoreImageSource | undefined): string {
  if (source?.type === 'url' && source.url) return `[Image: ${source.url}]`
  if (source?.type === 'base64' && source.media_type) return `[Image: ${source.media_type}]`
  return '[Image]'
}

function truncateOneLine(value: string, maxLength = 160): string {
  const normalized = value.replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength - 3)}...`
}

// ─── Hook ───────────────────────────────────────────────────────

const DEFAULT_URL = 'ws://localhost:8100/ws'
const STORAGE_KEY = 'spellbook-ws-url'
const WS_CLIENT_LABEL = 'spellbook-web-ui'

function withWsClientLabel(rawUrl: string, client: string): string {
  const base = typeof window === 'undefined' ? undefined : window.location.href
  const url = new URL(rawUrl, base)
  url.searchParams.set('client', client)
  return url.toString()
}

export function useSpellbook() {
  const [serverUrl, setServerUrl] = useState<string>(() => {
    if (typeof window === 'undefined') return DEFAULT_URL
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_URL
  })
  const [turns, setTurns] = useState<MockTurn[]>([])
  const [awareness, setAwareness] = useState<MockAwareness>(DEFAULT_AWARENESS)
  const [working, setWorking] = useState(false)
  const [queuedCount, setQueuedCount] = useState(0)
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected')
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const builderRef = useRef(new TurnBuilder())
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const turnsAnimationFrameRef = useRef<number | null>(null)
  const pendingTurnsRef = useRef<MockTurn[] | null>(null)
  const serverUrlRef = useRef(serverUrl)
  const connectRef = useRef<() => void>(() => {})

  const flushPendingTurns = useCallback(() => {
    turnsAnimationFrameRef.current = null

    const nextTurns = pendingTurnsRef.current
    if (nextTurns === null) return

    pendingTurnsRef.current = null
    startTransition(() => {
      setTurns(nextTurns)
    })
  }, [])

  const scheduleTurnsUpdate = useCallback((nextTurns: MockTurn[]) => {
    pendingTurnsRef.current = nextTurns

    if (turnsAnimationFrameRef.current !== null) return

    turnsAnimationFrameRef.current = requestAnimationFrame(() => {
      flushPendingTurns()
    })
  }, [flushPendingTurns])

  const applyTurnsNow = useCallback((nextTurns: MockTurn[]) => {
    pendingTurnsRef.current = null

    if (turnsAnimationFrameRef.current !== null) {
      cancelAnimationFrame(turnsAnimationFrameRef.current)
      turnsAnimationFrameRef.current = null
    }

    startTransition(() => {
      setTurns(nextTurns)
    })
  }, [])

  const updateServerUrl = useCallback((url: string) => {
    setServerUrl(url)
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY, url)
    }
  }, [])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (turnsAnimationFrameRef.current !== null) {
      cancelAnimationFrame(turnsAnimationFrameRef.current)
      turnsAnimationFrameRef.current = null
    }

    pendingTurnsRef.current = null

    if (wsRef.current) {
      wsRef.current.onopen = null
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.onmessage = null
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  const refreshAwareness = useCallback(async () => {
    try {
      const response = await fetch(`${httpBaseFromServerUrl(serverUrlRef.current)}/awareness`)
      if (!response.ok) return

      const awarenessResponse = await response.json() as CoreAwarenessResponse
      if (awarenessResponse.kind !== 'awareness') return

      setAwareness((current) =>
        mapCoreAwarenessResponse(awarenessResponse, current.model, current.gauge)
      )
    } catch {
      // Awareness is a convenience projection; live chat can continue without it.
    }
  }, [])

  const refreshRuntimeState = useCallback(async () => {
    try {
      const response = await fetch(`${httpBaseFromServerUrl(serverUrlRef.current)}/health`)
      if (!response.ok) return

      const health = await response.json() as CoreHealthResponse
      if (health.kind !== 'health') return

      setWorking(health.state === 'running' || health.state === 'dreaming')
      setAwareness((current) => ({ ...current, model: health.model || current.model }))
    } catch {
      // Health refresh is best-effort; websocket state remains authoritative enough for UI presence.
    }
  }, [])

  const handleCoreEvent = useCallback((event: CoreServerEvent) => {
    const builder = builderRef.current

    switch (event.kind) {
      case 'catchup': {
        if (!isCoreCatchup(event)) break

        applyTurnsNow(builder.loadRawTurns(coreRehydrationToTurns(event.rehydrated)))
        setAwareness(mapCoreRehydrationAwareness(event.rehydrated))
        setQueuedCount(0)
        void refreshAwareness()
        void refreshRuntimeState()
        break
      }

      case 'turn_started': {
        const message = isRecord(event.message)
          ? event.message as CoreInboundMessage
          : undefined
        const text = extractCoreInboundText(message)
        const source = extractCoreInboundSource(message)

        if (text) {
          builder.addUserMessage(text, source, event.time)
        }

        builder.startTurn(event.time)
        setWorking(true)
        setQueuedCount((count) => Math.max(0, count - 1))
        applyTurnsNow(builder.snapshot())
        break
      }

      case 'stream': {
        const streamEvent = isRecord(event.event)
          ? event.event as unknown as CoreStreamEvent
          : undefined
        if (!streamEvent) break

        switch (streamEvent.kind) {
          case 'thinking_delta':
            builder.addThinkingDelta(streamEvent.text || '')
            scheduleTurnsUpdate(builder.snapshot())
            break
          case 'thinking_end':
            builder.noteEventTime(streamEvent.time)
            builder.endThinking()
            scheduleTurnsUpdate(builder.snapshot())
            break
          case 'text_delta':
            builder.addTextDelta(streamEvent.text || '')
            scheduleTurnsUpdate(builder.snapshot())
            break
          case 'text_end':
            builder.noteEventTime(streamEvent.time)
            builder.endText()
            scheduleTurnsUpdate(builder.snapshot())
            break
        }
        break
      }

      case 'context_block_added': {
        const block = isRecord(event.block)
          ? event.block as unknown as CoreBlock
          : undefined
        if (!block) break

        if (block.type === 'tool_call') {
          builder.noteEventTime(block.time)
          builder.addToolStart(
            block.call_id || '',
            block.tool || 'Tool',
            summarizeToolInput(block.input),
          )
          scheduleTurnsUpdate(builder.snapshot())
          break
        }

        if (block.type === 'tool_result') {
          const result = coreToolResultText(block)
          builder.noteEventTime(block.time)
          builder.addToolResult(
            block.call_id || '',
            block.tool || 'Tool',
            summarizeToolResult(block, result),
            result,
            block.is_error || false,
            coreToolDisplay(block.display),
          )
          scheduleTurnsUpdate(builder.snapshot())
        }
        break
      }

      case 'turn_ended': {
        applyTurnsNow(builder.endTurn(event.time))
        setWorking(false)
        void refreshAwareness()
        break
      }

      case 'runtime_state': {
        const state = event.state
        if (state === 'running' || state === 'dreaming') {
          setWorking(true)
        } else if (state === 'idle' || state === 'suspended') {
          setWorking(false)
        }
        break
      }

      case 'message_queued': {
        setQueuedCount((c) => c + 1)
        break
      }
    }
  }, [applyTurnsNow, refreshAwareness, refreshRuntimeState, scheduleTurnsUpdate])

  const handleEvent = useCallback((event: SpellbookEvent | CoreServerEvent) => {
    if (isCoreServerEvent(event)) {
      handleCoreEvent(event)
      return
    }

    const builder = builderRef.current

    switch (event.type) {
      case 'catchup': {
        const catchup = event as unknown as CatchupEvent
        applyTurnsNow(builder.loadCatchup(catchup.main_pane || []))
        setAwareness(mapAwareness(catchup.awareness, catchup.model || 'unknown', catchup.gauge || ''))
        setWorking(catchup.working || false)
        setQueuedCount(0)
        break
      }

      case 'turn_started': {
        const text = (event.text as string) || ''
        const source = (event.source as string) || 'user'
        const time = (event.time as string) || null

        if (text) {
          builder.addUserMessage(text, source)
        }
        builder.startTurn(time)
        setWorking(true)
        setQueuedCount(0)
        if (text) {
          applyTurnsNow(builder.snapshot())
        }
        break
      }

      case 'thinking_delta': {
        builder.addThinkingDelta((event.text as string) || '')
        scheduleTurnsUpdate(builder.snapshot())
        break
      }

      case 'thinking_end': {
        builder.endThinking()
        scheduleTurnsUpdate(builder.snapshot())
        break
      }

      case 'text_delta': {
        builder.addTextDelta((event.text as string) || '')
        scheduleTurnsUpdate(builder.snapshot())
        break
      }

      case 'text_end': {
        builder.noteEventTime((event.time as string) || null)
        builder.endText()
        scheduleTurnsUpdate(builder.snapshot())
        break
      }

      case 'tool_start': {
        builder.noteEventTime((event.time as string) || null)
        builder.addToolStart(
          (event.call_id as string) || '',
          (event.tool as string) || 'Tool',
          (event.summary as string) || '',
        )
        scheduleTurnsUpdate(builder.snapshot())
        break
      }

      case 'tool_result': {
        builder.noteEventTime((event.time as string) || null)
        builder.addToolResult(
          (event.call_id as string) || '',
          (event.tool as string) || 'Tool',
          (event.summary as string) || '',
          (event.result as string) || '',
          (event.is_error as boolean) || false,
          (event.display as ToolDisplay) || undefined,
        )
        scheduleTurnsUpdate(builder.snapshot())
        break
      }

      case 'turn_end': {
        const awarenessData = event.awareness as CatchupAwareness | undefined
        const gauge = (event.gauge as string) || ''
        const time = (event.time as string) || null

        applyTurnsNow(builder.endTurn(time))
        setWorking(false)
        setAwareness((current) => {
          if (awarenessData) {
            return mapAwareness(awarenessData, current.model, gauge)
          }

          if (gauge) {
            return { ...current, gauge }
          }

          return current
        })
        break
      }

      case 'entity_idle': {
        setWorking(false)
        break
      }

      case 'awareness_update': {
        const awarenessData = event.awareness as CatchupAwareness | undefined
        const gauge = (event.gauge as string) || ''

        if (awarenessData) {
          setAwareness((current) =>
            mapAwareness(awarenessData, current.model, gauge || current.gauge)
          )
        } else if (gauge) {
          setAwareness((current) => ({ ...current, gauge }))
        }
        break
      }

      case 'message_queued': {
        setQueuedCount((c) => c + 1)
        break
      }

      case 'queued_message_delivered': {
        const text = (event.text as string) || ''
        const source = (event.source as string) || 'user'

        setQueuedCount((c) => Math.max(0, c - 1))

        if (text) {
          builder.addUserMessage(text, source)
          builder.startTurn()
          applyTurnsNow(builder.snapshot())
        }
        break
      }

      case 'error': {
        setWorking(false)
        setError((event.message as string) || 'An error occurred')
        break
      }
    }
  }, [applyTurnsNow, handleCoreEvent, scheduleTurnsUpdate])

  const connect = useCallback(() => {
    disconnect()
    const url = serverUrlRef.current

    setConnectionState('connecting')
    setError(null)

    try {
      const wsUrl = withWsClientLabel(url, WS_CLIENT_LABEL)
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
          reconnectTimeoutRef.current = null
        }

        setConnectionState('connected')
        setError(null)
      }

      ws.onclose = () => {
        setConnectionState((current) => (current === 'error' ? current : 'disconnected'))

        reconnectTimeoutRef.current = setTimeout(() => {
          if (serverUrlRef.current === url) {
            connectRef.current()
          }
        }, 3000)
      }

      ws.onerror = () => {
        setConnectionState('error')
        setError(`Failed to connect to ${url}`)
      }

      ws.onmessage = (message) => {
        try {
          const data: SpellbookEvent = JSON.parse(message.data)
          handleEvent(data)
        } catch {
          // Malformed message
        }
      }
    } catch {
      setConnectionState('error')
      setError(`Invalid URL: ${url}`)
    }
  }, [disconnect, handleEvent])

  const sendMessage = useCallback(async (text: string) => {
    try {
      const response = await fetch(`${httpBaseFromServerUrl(serverUrlRef.current)}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) {
        throw new Error(`Message request failed with ${response.status}`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message')
      setConnectionState('error')
    }
  }, [])

  const sendCommand = useCallback((command: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    wsRef.current.send(JSON.stringify({
      type: 'command',
      command,
    }))
  }, [])

  const interrupt = useCallback(async () => {
    try {
      const response = await fetch(`${httpBaseFromServerUrl(serverUrlRef.current)}/interrupt`, {
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error(`Interrupt request failed with ${response.status}`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to interrupt')
      setConnectionState('error')
    }
  }, [])

  useEffect(() => {
    serverUrlRef.current = serverUrl
  }, [serverUrl])

  useEffect(() => {
    connectRef.current = connect
  }, [connect])

  useEffect(() => {
    const connectionTimer = setTimeout(() => {
      connect()
    }, 0)

    return () => {
      clearTimeout(connectionTimer)
      disconnect()
    }
  }, [serverUrl, connect, disconnect])

  return {
    turns,
    awareness,
    working,
    queuedCount,
    connectionState,
    error,
    serverUrl,
    updateServerUrl,
    sendMessage,
    sendCommand,
    interrupt,
    reconnect: connect,
  }
}
