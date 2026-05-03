'use client'

import { useState, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSend: (message: string) => void
  working?: boolean
  maxWidthClass?: string
  queuedCount?: number
}

export function MetaChatInput({ onSend, working, maxWidthClass = 'max-w-3xl', queuedCount = 0 }: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [value])

  const handleSubmit = () => {
    const trimmed = value.trim()
    if (!trimmed) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div
      className={cn(
        'border-t backdrop-blur-sm transition-colors duration-300',
        working
          ? 'bg-accent/[0.08] border-accent/30'
          : 'bg-surface/60 border-edge'
      )}
    >
      <div className={cn(maxWidthClass, 'mx-auto px-6 transition-all duration-300')}>
        <div className="flex items-stretch">
          {/* Queued badge — left of input */}
          <div className="flex items-center pr-3">
            {queuedCount > 0 ? (
              <span className="text-[10px] font-mono text-accent/70 whitespace-nowrap">
                {queuedCount} queued
              </span>
            ) : (
              /* Spacer to keep layout stable */
              <span className="text-[10px] invisible">0 queued</span>
            )}
          </div>

          {/* Text input — flat, bordered sides, fills full height */}
          <div
            className={cn(
              'flex-1 border-r px-4 py-3 flex items-center transition-colors duration-300',
              working
                ? 'border-accent/40'
                : 'border-edge focus-within:border-edge-bright'
            )}
          >
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                working ? 'Type to queue a message...' : 'Message...'
              }
              rows={1}
              className={cn(
                'w-full bg-transparent text-sm resize-none focus:outline-none min-h-[24px] max-h-[200px] leading-relaxed',
                working
                  ? 'text-accent/70 placeholder:text-accent/30'
                  : 'text-ch-text placeholder:text-ch-hint/60'
              )}
            />
          </div>

          {/* Right column: thinking indicator above send button */}
          <div className="flex flex-col items-center justify-end gap-1.5 pl-3 py-2">
            {/* Thinking indicator — inverted when working */}
            {working && (
              <div className="flex items-center gap-1.5 px-1">
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-status" />
                <span className="text-[10px] text-accent/70 whitespace-nowrap">
                  thinking...
                </span>
              </div>
            )}

            {/* Send button — wider to match thinking text, inverted when working */}
            <button
              onClick={handleSubmit}
              disabled={!value.trim()}
              className={cn(
                'shrink-0 min-w-[5rem] h-8 rounded-lg flex items-center justify-center transition-all duration-300',
                !value.trim()
                  ? 'bg-elevated text-ch-hint/30'
                  : working
                    ? 'bg-accent/30 text-accent hover:bg-accent/40'
                    : 'bg-accent text-ground hover:bg-accent-bright'
              )}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M3 8h10M9 4l4 4-4 4" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
