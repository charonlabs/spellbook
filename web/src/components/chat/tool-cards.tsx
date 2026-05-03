'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { cn } from '@/lib/utils'
import type { MockToolCall, MockToolResult, MockContentBlock } from './mock-data'

// Icons for common tools
const TOOL_ICONS: Record<string, string> = {
  Read: '📄',
  Write: '✏️',
  Edit: '✏️',
  Bash: '▶',
  Grep: '🔍',
  Glob: '📁',
  Reflect: '🪞',
  Recall: '↩',
  Pin: '📌',
  Forget: '💨',
  Amend: '✎',
  Skill: '⚡',
}

export interface ToolGroup {
  tool: string
  pairs: { call: MockToolCall; result?: MockToolResult }[]
}

/** Group consecutive tool call/result blocks into display groups */
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
    } else if (block.kind === 'tool_result') {
      // Walk backwards to find matching call and attach result
      for (let g = groups.length - 1; g >= 0; g--) {
        let found = false
        for (const pair of groups[g].pairs) {
          if (pair.call.callId === block.callId && !pair.result) {
            pair.result = block
            found = true
            break
          }
        }
        if (found) break
      }
    }
  }

  return groups
}

interface ToolCardsProps {
  groups: ToolGroup[]
}

export function ToolCards({ groups }: ToolCardsProps) {
  const [expandedGroup, setExpandedGroup] = useState<number | null>(null)

  if (groups.length === 0) return null

  return (
    <div className="py-2">
      <div className="flex flex-wrap gap-1.5">
        {groups.map((group, i) => {
          const hasError = group.pairs.some((p) => p.result?.isError)
          const icon = TOOL_ICONS[group.tool] || '⚙'
          const isExpanded = expandedGroup === i

          return (
            <button
              key={i}
              onClick={() => setExpandedGroup(isExpanded ? null : i)}
              className={cn(
                'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono transition-all duration-150',
                isExpanded
                  ? 'bg-accent/15 text-accent ring-1 ring-accent/20'
                  : hasError
                    ? 'bg-attention/10 text-attention/80 hover:text-attention hover:bg-attention/15'
                    : 'bg-elevated/80 text-ch-hint hover:text-ch-subtext hover:bg-inset'
              )}
            >
              <span className="text-[10px]">{icon}</span>
              {group.pairs.length > 1 && (
                <span className="text-[10px] opacity-50">
                  {group.pairs.length}×
                </span>
              )}
              <span>{group.tool}</span>
              <svg
                width="8"
                height="8"
                viewBox="0 0 8 8"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className={cn(
                  'transition-transform duration-150',
                  isExpanded && 'rotate-180'
                )}
              >
                <path d="M2 3L4 5L6 3" />
              </svg>
            </button>
          )
        })}
      </div>

      <AnimatePresence>
        {expandedGroup !== null && groups[expandedGroup] && (
          <motion.div
            key={expandedGroup}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="mt-2 rounded-md bg-elevated/50 border border-edge/50 p-2.5 space-y-1">
              {groups[expandedGroup].pairs.map((pair, j) => (
                <div
                  key={j}
                  className="flex items-start gap-2 text-xs font-mono"
                >
                  <span className="text-ch-hint shrink-0 select-none w-3 text-right">
                    {j === groups[expandedGroup].pairs.length - 1 ? '└' : '├'}
                  </span>
                  <span className="text-ch-subtext">{pair.call.summary}</span>
                  {pair.result && (
                    <>
                      <span className="text-ch-hint">→</span>
                      <span
                        className={cn(
                          pair.result.isError ? 'text-attention' : 'text-ch-hint'
                        )}
                      >
                        {pair.result.summary}
                      </span>
                    </>
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
