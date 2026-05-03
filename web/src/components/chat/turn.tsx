'use client'

import { memo, useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn, formatRelativeTime, formatDurationCompact } from '@/lib/utils'
import type { ConduitType, MockTurn, PreparedTurnSegment } from './mock-data'
import { ToolCards } from './tool-cards'
import { ToolCardVerbose } from './tool-card-verbose'

export type ToolDisplayMode = 'badges' | 'cards'
export type AgeBucket = 0 | 1 | 2 | 3

interface TurnProps {
  turn: MockTurn
  toolDisplayMode?: ToolDisplayMode
  ageBucket: AgeBucket
}

export const Turn = memo(
  function Turn({ turn, ageBucket, toolDisplayMode = 'badges' }: TurnProps) {
    switch (turn.render.kind) {
      case 'user':
        return <UserTurn turn={turn} ageBucket={ageBucket} />
      case 'conduit':
        return (
          <ConduitTurn
            turn={turn}
            ageBucket={ageBucket}
            displayMode={toolDisplayMode}
          />
        )
      case 'assistant':
        return (
          <AssistantTurn
            turn={turn}
            ageBucket={ageBucket}
            toolDisplayMode={toolDisplayMode}
          />
        )
      default:
        return null
    }
  },
  (previous, next) =>
    previous.turn === next.turn &&
    previous.ageBucket === next.ageBucket &&
    previous.toolDisplayMode === next.toolDisplayMode
)

function opacityForAge(ageBucket: AgeBucket): number {
  switch (ageBucket) {
    case 0:
      return 1
    case 1:
      return 0.85
    case 2:
      return 0.7
    case 3:
      return 0.55
  }
}

// ─── User Turn ──────────────────────────────────────────────────

function UserTurn({ turn, ageBucket }: { turn: MockTurn; ageBucket: AgeBucket }) {
  if (turn.render.kind !== 'user' || !turn.render.text) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: opacityForAge(ageBucket), y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex justify-end"
    >
      <div className="max-w-[80%] group">
        <div className="bg-accent/[0.12] border border-accent/[0.18] rounded-2xl rounded-br-md px-4 py-3 user-prose">
          <Markdown remarkPlugins={[remarkGfm]}>{turn.render.text}</Markdown>
        </div>
        <div className="flex justify-end mt-1 pr-1">
          <span className="text-[10px] text-ch-hint opacity-0 group-hover:opacity-100 transition-opacity">
            {formatRelativeTime(turn.timestamp)}
          </span>
        </div>
      </div>
    </motion.div>
  )
}

// ─── Assistant Turn ─────────────────────────────────────────────

function AssistantTurn({
  turn,
  ageBucket,
  toolDisplayMode,
}: {
  turn: MockTurn
  ageBucket: AgeBucket
  toolDisplayMode: ToolDisplayMode
}) {
  if (turn.render.kind !== 'assistant') return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: opacityForAge(ageBucket), y: 0 }}
      transition={{ duration: 0.3 }}
      className="group"
    >
      {turn.render.thinkingText && (
        <ThinkingIndicator text={turn.render.thinkingText} />
      )}

      <div className="space-y-1">
        {turn.render.segments.map((segment, index) => {
          if (segment.type === 'text') {
            return (
              <div key={index} className="entity-prose">
                <Markdown remarkPlugins={[remarkGfm]}>{segment.text}</Markdown>
              </div>
            )
          }

          if (toolDisplayMode === 'cards') {
            return (
              <div key={index} className="space-y-1 py-1">
                {segment.pairs.map((pair, pairIndex) => (
                  <ToolCardVerbose
                    key={pair.call.callId || pairIndex}
                    call={pair.call}
                    result={pair.result}
                  />
                ))}
              </div>
            )
          }

          return <ToolCards key={index} groups={segment.groups} />
        })}
      </div>

      <div className="mt-1.5 flex items-center gap-2">
        <CopyLastBlock segments={turn.render.segments} />
        <span className="text-[10px] text-ch-hint opacity-0 group-hover:opacity-100 transition-opacity">
          {turn.id.startsWith('live-streaming-') ? (
            'thinking...'
          ) : (
            <>
              Finished {formatRelativeTime(turn.timestamp)}
              {turn.durationMs != null && turn.durationMs > 0 && (
                <span className="ml-1 text-ch-hint/60">
                  · took {formatDurationCompact(turn.durationMs)}
                </span>
              )}
            </>
          )}
        </span>
      </div>
    </motion.div>
  )
}

// ─── Thinking Indicator ─────────────────────────────────────────

// ─── Copy last text block ───────────────────────────────────────

function CopyLastBlock({ segments }: { segments: PreparedTurnSegment[] }) {
  const [copied, setCopied] = useState(false)

  // Find the last text segment
  const lastText = [...segments].reverse().find((s) => s.type === 'text')
  if (!lastText || lastText.type !== 'text') return null

  const handleCopy = () => {
    navigator.clipboard.writeText(lastText.text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <button
      onClick={handleCopy}
      className="text-ch-hint/50 hover:text-ch-hint opacity-0 group-hover:opacity-100 transition-all p-0.5"
      title="Copy last text block"
    >
      {copied ? (
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-working">
          <path d="M2.5 6.5L5 9l4.5-6" />
        </svg>
      ) : (
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="4" y="4" width="6.5" height="6.5" rx="1" />
          <path d="M8 4V2.5A1 1 0 0 0 7 1.5H2.5A1 1 0 0 0 1.5 2.5V7A1 1 0 0 0 2.5 8H4" />
        </svg>
      )}
    </button>
  )
}

// ─── Thinking Indicator ─────────────────────────────────────────

function ThinkingIndicator({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] transition-colors',
          expanded
            ? 'bg-inset text-ch-subtext'
            : 'text-ch-hint hover:text-ch-subtext hover:bg-elevated/50'
        )}
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          className="text-ch-hint"
        >
          <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2" />
          <circle cx="6" cy="6" r="1.5" fill="currentColor" opacity="0.5" />
        </svg>
        <span className="italic">thought for a moment</span>
        <svg
          width="8"
          height="8"
          viewBox="0 0 8 8"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          className={cn(
            'transition-transform duration-150',
            expanded && 'rotate-180'
          )}
        >
          <path d="M2 3L4 5L6 3" />
        </svg>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="mt-1.5 pl-3 border-l border-edge/50">
              <pre className="text-xs text-ch-hint/70 leading-relaxed whitespace-pre-wrap font-mono italic">
                {text}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Conduit Turn ───────────────────────────────────────────────

const CONDUIT_STYLES: Record<
  ConduitType,
  { border: string; label: string; icon: string }
> = {
  notification: { border: 'border-idle/30', label: 'text-idle', icon: '🔔' },
  context: { border: 'border-familiar/30', label: 'text-familiar', icon: '↻' },
  forwarded: { border: 'border-accent/30', label: 'text-accent', icon: '↗' },
}

function ConduitTurn({
  turn,
  ageBucket,
  displayMode,
}: {
  turn: MockTurn
  ageBucket: AgeBucket
  displayMode: ToolDisplayMode
}) {
  if (turn.render.kind !== 'conduit') return null

  const style = CONDUIT_STYLES[turn.render.conduitType]

  if (displayMode === 'badges') {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: opacityForAge(ageBucket) }}
        transition={{ duration: 0.25 }}
        className="flex items-center gap-3 py-1"
      >
        <div className="flex-1 h-px bg-edge/50" />
        <span className={cn('text-[11px] whitespace-nowrap flex items-center gap-1.5', style.label)}>
          <span>{style.icon}</span>
          <span>{turn.render.source}</span>
        </span>
        <div className="flex-1 h-px bg-edge/50" />
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: opacityForAge(ageBucket), y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <div className={cn('rounded-lg border overflow-hidden', style.border)}>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-elevated/40 border-b border-edge/30">
          <span className="text-[10px]">{style.icon}</span>
          <span className={cn('text-xs font-mono font-semibold', style.label)}>
            {turn.render.source}
          </span>
          <span className="text-[10px] text-ch-hint">{turn.render.conduitType}</span>
        </div>
        {turn.render.content && (
          <div className="px-3 py-2 entity-prose">
            <Markdown remarkPlugins={[remarkGfm]}>{turn.render.content}</Markdown>
          </div>
        )}
      </div>
    </motion.div>
  )
}
