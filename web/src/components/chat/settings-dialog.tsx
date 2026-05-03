'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { cn } from '@/lib/utils'
import type { ConnectionState } from '@/lib/use-spellbook'
import type { ToolDisplayMode } from './turn'

interface SettingsDialogProps {
  open: boolean
  onClose: () => void
  serverUrl: string
  onUpdateUrl: (url: string) => void
  connectionState: ConnectionState
  error: string | null
  onReconnect: () => void
  toolDisplayMode: ToolDisplayMode
  onToolDisplayModeChange: (mode: ToolDisplayMode) => void
}

const CONNECTION_STATUS: Record<ConnectionState, { label: string; dotClass: string }> = {
  disconnected: { label: 'disconnected', dotClass: 'bg-ch-hint' },
  connecting: { label: 'connecting...', dotClass: 'bg-idle animate-pulse-status' },
  connected: { label: 'connected', dotClass: 'bg-working' },
  error: { label: 'error', dotClass: 'bg-attention' },
}

export function SettingsDialog({
  open,
  onClose,
  serverUrl,
  onUpdateUrl,
  connectionState,
  error,
  onReconnect,
  toolDisplayMode,
  onToolDisplayModeChange,
}: SettingsDialogProps) {
  const [urlDraft, setUrlDraft] = useState(serverUrl)

  // Sync draft when dialog opens
  useEffect(() => {
    if (open) setUrlDraft(serverUrl)
  }, [open, serverUrl])

  // Escape to close
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  const isDirty = urlDraft.trim() !== serverUrl
  const status = CONNECTION_STATUS[connectionState]

  const handleApply = () => {
    const trimmed = urlDraft.trim()
    if (trimmed && trimmed !== serverUrl) {
      onUpdateUrl(trimmed)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleApply()
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-50 bg-ground/70 backdrop-blur-sm flex items-center justify-center p-6"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="w-full max-w-md bg-surface rounded-xl border border-edge overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-edge">
              <h2 className="text-sm font-semibold text-ch-text">Connection Settings</h2>
              <button
                onClick={onClose}
                className="text-ch-hint hover:text-ch-text transition-colors p-1"
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 14 14"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                >
                  <path d="M3 3L11 11M11 3L3 11" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="px-5 py-4 space-y-4">
              {/* Connection status */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-ch-hint">Status</span>
                <div className="flex items-center gap-2">
                  <div className={cn('w-1.5 h-1.5 rounded-full', status.dotClass)} />
                  <span className={cn(
                    'text-xs',
                    connectionState === 'connected' ? 'text-working' :
                    connectionState === 'error' ? 'text-attention' :
                    'text-ch-subtext'
                  )}>
                    {status.label}
                  </span>
                </div>
              </div>

              {/* Server URL */}
              <div className="space-y-1.5">
                <label className="text-xs text-ch-hint block">
                  WebSocket URL
                </label>
                <input
                  type="text"
                  value={urlDraft}
                  onChange={(e) => setUrlDraft(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="ws://localhost:8100/ws"
                  className="w-full bg-ground/50 border border-edge rounded-lg px-3 py-2 text-sm font-mono text-ch-text placeholder:text-ch-hint/50 focus:outline-none focus:border-edge-bright transition-colors"
                />
              </div>

              {/* Error display */}
              {error && (
                <div className="px-3 py-2 rounded-md bg-attention/10 border border-attention/20">
                  <p className="text-xs text-attention">{error}</p>
                </div>
              )}

              {/* Tool display mode */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-ch-hint">Tool Display</span>
                <div className="flex items-center bg-elevated/60 rounded-md p-0.5">
                  {(['badges', 'cards'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => onToolDisplayModeChange(mode)}
                      className={cn(
                        'px-2.5 py-1 rounded text-[11px] font-mono transition-colors',
                        toolDisplayMode === mode
                          ? 'bg-accent/15 text-accent'
                          : 'text-ch-hint hover:text-ch-subtext'
                      )}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center justify-between pt-1">
                <button
                  onClick={onReconnect}
                  className="text-xs text-ch-hint hover:text-ch-subtext transition-colors px-2 py-1 rounded-md hover:bg-elevated/50"
                >
                  reconnect
                </button>

                <div className="flex items-center gap-2">
                  <button
                    onClick={onClose}
                    className="text-xs text-ch-hint hover:text-ch-subtext transition-colors px-3 py-1.5 rounded-md hover:bg-elevated/50"
                  >
                    cancel
                  </button>
                  <button
                    onClick={handleApply}
                    disabled={!isDirty}
                    className={cn(
                      'text-xs px-3 py-1.5 rounded-md transition-all',
                      isDirty
                        ? 'bg-accent/15 text-accent hover:bg-accent/25'
                        : 'bg-elevated text-ch-hint/30 cursor-not-allowed'
                    )}
                  >
                    apply & reconnect
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
