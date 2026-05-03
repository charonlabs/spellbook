'use client'

import { cn } from '@/lib/utils'
import type { MockAwareness } from './mock-data'

interface AwarenessSidebarProps {
  awareness: MockAwareness
}

const MODE_COLORS: Record<string, string> = {
  full: 'bg-working/60',
  summary: 'bg-accent/60',
  index: 'bg-idle/60',
  headline: 'bg-ch-hint/40',
}

export function AwarenessSidebar({ awareness }: AwarenessSidebarProps) {
  const pct = Math.round((awareness.usedTokens / awareness.maxTokens) * 100)

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-edge">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono text-ch-hint uppercase tracking-wider">
            Context
          </span>
          <span
            className={cn(
              'text-xs font-mono',
              awareness.regime === 'calm'
                ? 'text-ch-subtext'
                : awareness.regime === 'warning'
                  ? 'text-idle'
                  : 'text-attention'
            )}
          >
            {pct}%
          </span>
        </div>

        {/* Mini gauge */}
        <div className="mt-2 h-1 bg-edge/50 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all duration-700',
              awareness.regime === 'calm'
                ? 'bg-accent/60'
                : awareness.regime === 'warning'
                  ? 'bg-idle/80'
                  : 'bg-attention/80'
            )}
            style={{ width: `${pct}%` }}
          />
        </div>

        <div className="flex justify-between mt-1.5">
          <span className="text-[10px] text-ch-hint">
            {(awareness.usedTokens / 1000).toFixed(0)}k
          </span>
          <span className="text-[10px] text-ch-hint">
            {(awareness.maxTokens / 1000).toFixed(0)}k
          </span>
        </div>
      </div>

      {/* Block list */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-1.5">
        <span className="text-[10px] text-ch-hint uppercase tracking-wider px-1 mb-1 block">
          Memory Blocks
        </span>

        {awareness.blocks.map((block) => (
          <button
            key={block.id}
            className="w-full text-left px-2.5 py-2 rounded-md hover:bg-elevated/60 transition-colors group"
          >
            <div className="flex items-center gap-2">
              <div
                className={cn(
                  'w-1.5 h-1.5 rounded-full shrink-0',
                  MODE_COLORS[block.mode] || 'bg-ch-hint/40'
                )}
              />
              <span className="text-xs text-ch-text/80 truncate flex-1">
                {block.title}
              </span>
              {block.pinned && (
                <span className="text-[9px] text-accent/60">📌</span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-0.5 pl-3.5">
              <span className="text-[10px] text-ch-hint">{block.mode}</span>
              <span className="text-[10px] text-ch-hint">·</span>
              <span className="text-[10px] text-ch-hint">
                {(block.tokens / 1000).toFixed(1)}k
              </span>
            </div>
          </button>
        ))}
      </div>

      {/* Footer legend */}
      <div className="px-4 py-3 border-t border-edge">
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {Object.entries(MODE_COLORS).map(([mode, color]) => (
            <div key={mode} className="flex items-center gap-1.5">
              <div className={cn('w-1.5 h-1.5 rounded-full', color)} />
              <span className="text-[10px] text-ch-hint">{mode}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
