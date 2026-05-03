'use client'

import { cn } from '@/lib/utils'
import type { MockAwareness } from './mock-data'

interface ContextGaugeProps {
  awareness: MockAwareness
}

export function ContextGauge({ awareness }: ContextGaugeProps) {
  const pct = (awareness.usedTokens / awareness.maxTokens) * 100
  const regimeColor =
    awareness.regime === 'forced'
      ? 'bg-attention'
      : awareness.regime === 'warning'
        ? 'bg-idle'
        : 'bg-accent'

  return (
    <div className="w-full h-[2px] bg-edge/30 relative overflow-hidden">
      <div
        className={cn('h-full transition-all duration-1000 ease-out', regimeColor)}
        style={{ width: `${pct}%`, opacity: awareness.regime === 'calm' ? 0.5 : 0.8 }}
      />
    </div>
  )
}
