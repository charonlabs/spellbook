'use client'

import { memo, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import type { ConnectionState } from '@/lib/use-spellbook'
import type { MockTurn } from './mock-data'
import { MetaChatInput } from './chat-input'
import { Turn, type AgeBucket, type ToolDisplayMode } from './turn'

const CONVERSATION_WIDTH = 'max-w-4xl'
const INITIAL_TURN_WINDOW = 60
const TURN_WINDOW_CHUNK = 40
const LOAD_OLDER_THRESHOLD_PX = 160
const TURN_ROW_STYLE: CSSProperties = {
  contentVisibility: 'auto',
  containIntrinsicSize: '240px',
  paddingInline: '4px',
}

interface ConversationPaneProps {
  turns: MockTurn[]
  toolDisplayMode: ToolDisplayMode
  connectionState: ConnectionState
  error: string | null
  working: boolean
  queuedCount?: number
  onSend: (message: string) => void
  onOpenSettings: () => void
}

function getAgeBucket(age: number): AgeBucket {
  if (age <= 3) return 0
  if (age <= 6) return 1
  if (age <= 12) return 2
  return 3
}

function getDefaultWindowStart(turnCount: number): number {
  return Math.max(0, turnCount - INITIAL_TURN_WINDOW)
}

export const ConversationPane = memo(
  function ConversationPane({
    turns,
    toolDisplayMode,
    connectionState,
    error,
    working,
    queuedCount,
    onSend,
    onOpenSettings,
  }: ConversationPaneProps) {
    const scrollRef = useRef<HTMLDivElement>(null)
    const scrollRafRef = useRef<number | null>(null)
    const prependRestoreRef = useRef<{ scrollHeight: number; scrollTop: number } | null>(null)
    const isLoadingOlderRef = useRef(false)
    const [userScrolled, setUserScrolled] = useState(false)
    const [windowStart, setWindowStart] = useState<number | null>(null)
    const totalTurns = turns.length
    const defaultWindowStart = getDefaultWindowStart(totalTurns)
    const effectiveWindowStart =
      windowStart === null
        ? defaultWindowStart
        : Math.min(windowStart, defaultWindowStart)
    const visibleTurns = turns.slice(effectiveWindowStart)
    const hiddenTurnCount = effectiveWindowStart

    // Auto-scroll: immediate scroll + short polling to catch async content.
    const scrollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
    const userScrolledRef = useRef(userScrolled)
    userScrolledRef.current = userScrolled

    useEffect(() => {
      if (userScrolledRef.current || !scrollRef.current) return

      const el = scrollRef.current

      // Immediate scroll
      el.scrollTop = el.scrollHeight

      // Clear any previous polling
      if (scrollIntervalRef.current) {
        clearInterval(scrollIntervalRef.current)
      }

      // Then poll for 500ms to catch animations / async renders
      let lastHeight = el.scrollHeight
      let stableCount = 0

      scrollIntervalRef.current = setInterval(() => {
        if (!el || userScrolledRef.current) {
          if (scrollIntervalRef.current) clearInterval(scrollIntervalRef.current)
          return
        }

        const h = el.scrollHeight
        if (h !== lastHeight) {
          el.scrollTop = h
          lastHeight = h
          stableCount = 0
        } else {
          stableCount++
          if (stableCount >= 5) {
            if (scrollIntervalRef.current) clearInterval(scrollIntervalRef.current)
          }
        }
      }, 50)

      const timeout = setTimeout(() => {
        if (scrollIntervalRef.current) clearInterval(scrollIntervalRef.current)
      }, 1500)

      return () => {
        if (scrollIntervalRef.current) clearInterval(scrollIntervalRef.current)
        clearTimeout(timeout)
      }
    }, [turns, effectiveWindowStart])

    useLayoutEffect(() => {
      if (!scrollRef.current || !prependRestoreRef.current) return

      const { scrollHeight, scrollTop } = prependRestoreRef.current
      prependRestoreRef.current = null

      const nextScrollHeight = scrollRef.current.scrollHeight
      scrollRef.current.scrollTop = nextScrollHeight - scrollHeight + scrollTop
      isLoadingOlderRef.current = false
    }, [visibleTurns])

    useEffect(() => {
      return () => {
        if (scrollRafRef.current !== null) {
          cancelAnimationFrame(scrollRafRef.current)
        }
      }
    }, [])

    const handleScroll = () => {
      if (!scrollRef.current) return

      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
      if (
        scrollTop <= LOAD_OLDER_THRESHOLD_PX &&
        effectiveWindowStart > 0 &&
        !isLoadingOlderRef.current
      ) {
        isLoadingOlderRef.current = true
        prependRestoreRef.current = {
          scrollHeight,
          scrollTop,
        }
        setWindowStart(Math.max(0, effectiveWindowStart - TURN_WINDOW_CHUNK))
      }

      const isNearBottom = scrollHeight - scrollTop - clientHeight < 80
      if (isNearBottom && windowStart !== null) {
        setWindowStart(null)
      }
      setUserScrolled((current) => (current === !isNearBottom ? current : !isNearBottom))
    }

    return (
      <div className="flex-1 flex flex-col min-h-0 min-w-0 relative">
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto"
        >
          <div className="h-6" />

          <div className={cn(CONVERSATION_WIDTH, 'mx-auto px-6 space-y-6 pb-4 transition-all duration-300')}>
            {hiddenTurnCount > 0 && (
              <div className="flex justify-center py-1">
                <span className="text-[10px] font-mono text-ch-hint">
                  Showing latest {visibleTurns.length} of {totalTurns} turns. Scroll up to load older messages.
                </span>
              </div>
            )}

            {turns.length === 0 && connectionState === 'connected' && (
              <div className="text-center py-20">
                <p className="text-ch-hint text-sm">No conversation yet.</p>
                <p className="text-ch-hint/60 text-xs mt-1">Send a message to begin.</p>
              </div>
            )}

            {turns.length === 0 && connectionState !== 'connected' && (
              <div className="text-center py-20">
                <p className="text-ch-hint text-sm">
                  {connectionState === 'connecting'
                    ? 'Connecting to Spellbook server...'
                    : connectionState === 'error'
                      ? 'Unable to connect'
                      : 'Disconnected'}
                </p>
                {error && (
                  <p className="text-attention/70 text-xs mt-1.5">{error}</p>
                )}
                <button
                  onClick={onOpenSettings}
                  className="mt-3 text-xs text-accent hover:text-accent-bright transition-colors"
                >
                  configure connection
                </button>
              </div>
            )}

            {visibleTurns.map((turn, index) => (
              <div key={turn.id} style={TURN_ROW_STYLE}>
                <Turn
                  turn={turn}
                  ageBucket={getAgeBucket(totalTurns - 1 - (effectiveWindowStart + index))}
                  toolDisplayMode={toolDisplayMode}
                />
              </div>
            ))}
          </div>
        </div>

        {userScrolled && (
          <motion.button
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            onClick={() => {
              if (!scrollRef.current) return

              if (windowStart !== null) {
                setWindowStart(null)
              }
              scrollRef.current.scrollTo({
                top: scrollRef.current.scrollHeight,
                behavior: 'smooth',
              })
              setUserScrolled(false)
            }}
            className="absolute bottom-20 left-1/2 -translate-x-1/2 px-3.5 py-1.5 rounded-full bg-surface border border-edge text-xs text-ch-subtext hover:text-ch-text hover:border-edge-bright transition-colors shadow-lg shadow-ground/50 z-10"
          >
            Scroll to latest
          </motion.button>
        )}

        <MetaChatInput
          onSend={onSend}
          working={working}
          queuedCount={queuedCount}
          maxWidthClass={CONVERSATION_WIDTH}
        />
      </div>
    )
  },
  (previous, next) =>
    previous.turns === next.turns &&
    previous.toolDisplayMode === next.toolDisplayMode &&
    previous.connectionState === next.connectionState &&
    previous.error === next.error &&
    previous.working === next.working &&
    previous.queuedCount === next.queuedCount &&
    previous.onSend === next.onSend &&
    previous.onOpenSettings === next.onOpenSettings
)
