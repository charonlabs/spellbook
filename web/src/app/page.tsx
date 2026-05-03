'use client'

import { useCallback, useState } from 'react'
import Link from 'next/link'
import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { ConversationPane } from '@/components/chat/conversation-pane'
import type { ToolDisplayMode } from '@/components/chat/turn'
import { ContextGauge } from '@/components/chat/context-gauge'
import { AwarenessSidebar } from '@/components/chat/awareness-sidebar'
import { SettingsDialog } from '@/components/chat/settings-dialog'
import { useSpellbook } from '@/lib/use-spellbook'

const SIDEBAR_PUSH_OFFSET = 110
const TOOL_DISPLAY_MODE_STORAGE_KEY = 'spellbook-tool-display-mode'

function getStoredToolDisplayMode(): ToolDisplayMode {
  if (typeof window === 'undefined') return 'badges'

  const storedMode = localStorage.getItem(TOOL_DISPLAY_MODE_STORAGE_KEY)
  return storedMode === 'cards' || storedMode === 'badges'
    ? storedMode
    : 'badges'
}

export default function SpellbookPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [toolDisplayMode, setToolDisplayMode] = useState<ToolDisplayMode>(() => getStoredToolDisplayMode())

  const {
    turns,
    awareness,
    working,
    queuedCount,
    connectionState,
    error,
    serverUrl,
    updateServerUrl,
    sendMessage,
    interrupt,
    reconnect,
  } = useSpellbook()

  const openSettings = useCallback(() => {
    setSettingsOpen(true)
  }, [])

  const updateToolDisplayMode = useCallback((mode: ToolDisplayMode) => {
    setToolDisplayMode(mode)
    if (typeof window !== 'undefined') {
      localStorage.setItem(TOOL_DISPLAY_MODE_STORAGE_KEY, mode)
    }
  }, [])

  const pct = awareness.maxTokens > 0
    ? Math.round((awareness.usedTokens / awareness.maxTokens) * 100)
    : 0
  const conversationOffset = sidebarOpen ? -SIDEBAR_PUSH_OFFSET : 0

  // Connection state for the presence dot
  const presenceDot = connectionState === 'connected'
    ? working
      ? 'bg-working animate-pulse-status'
      : 'bg-working'
    : connectionState === 'connecting'
      ? 'bg-idle animate-pulse-status'
      : connectionState === 'error'
        ? 'bg-attention'
        : 'bg-ch-hint'

  return (
    <div className="h-screen flex flex-col bg-ground">
      {/* Header */}
      <header className="flex items-center justify-between px-5 py-2.5 border-b border-edge bg-surface/60 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <div className={cn('w-2 h-2 rounded-full', presenceDot)} />
          <h1 className="font-display text-sm font-semibold text-ch-text tracking-tight">
            spellbook
          </h1>
          <span className="text-[10px] font-mono text-ch-hint">
            {awareness.model}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Context gauge chip */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className={cn(
              'flex items-center gap-2 px-2.5 py-1 rounded-md text-xs font-mono transition-colors',
              sidebarOpen
                ? 'bg-accent/15 text-accent'
                : 'text-ch-hint hover:text-ch-subtext hover:bg-elevated/50'
            )}
          >
            {pct > 0 && (
              <>
                <span className="text-[10px]">{pct}%</span>
                <div className="w-12 h-1 bg-edge/50 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-accent/50 rounded-full"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </>
            )}
            {pct === 0 && (
              <span className="text-[10px]">context</span>
            )}
          </button>

          {/* Playground link */}
          <Link
            href="/playground"
            className="text-[11px] font-mono text-ch-hint hover:text-accent transition-colors px-1.5 py-0.5 rounded hover:bg-elevated/50"
            title="Context playground"
          >
            playground
          </Link>

          {/* Settings gear */}
          <button
            onClick={openSettings}
            className="text-ch-hint hover:text-ch-subtext transition-colors p-1 rounded-md hover:bg-elevated/50"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="7" cy="7" r="2" />
              <path d="M5.7 1.3l-.4 1.2a4.5 4.5 0 0 0-1 .6L3.1 2.7l-1.4 1.4.4 1.2a4.5 4.5 0 0 0-.6 1L.3 5.7v2l1.2.4c.1.4.3.7.6 1l-.4 1.2 1.4 1.4 1.2-.4c.3.3.6.5 1 .6l.4 1.2h2l.4-1.2a4.5 4.5 0 0 0 1-.6l1.2.4 1.4-1.4-.4-1.2c.3-.3.5-.6.6-1l1.2-.4v-2l-1.2-.4a4.5 4.5 0 0 0-.6-1l.4-1.2-1.4-1.4-1.2.4a4.5 4.5 0 0 0-1-.6L8.3 1.3z" />
            </svg>
          </button>

          {/* Interrupt button */}
          <button
            onClick={interrupt}
            disabled={!working}
            className={cn(
              'text-[11px] px-2 py-1 rounded-md transition-colors',
              working
                ? 'text-attention hover:bg-attention/10'
                : 'text-ch-hint/30 cursor-not-allowed'
            )}
          >
            interrupt
          </button>
        </div>
      </header>

      {/* Context gauge line */}
      <ContextGauge awareness={awareness} />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden relative">
        <motion.div
          initial={false}
          animate={{ x: conversationOffset }}
          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-1 min-h-0 min-w-0 transform-gpu will-change-transform"
        >
          <ConversationPane
            turns={turns}
            toolDisplayMode={toolDisplayMode}
            connectionState={connectionState}
            error={error}
            working={working}
            queuedCount={queuedCount}
            onSend={sendMessage}
            onOpenSettings={openSettings}
          />
        </motion.div>

        {/* Awareness Sidebar */}
        <motion.div
          initial={false}
          animate={{
            x: sidebarOpen ? 0 : 280,
            opacity: sidebarOpen ? 1 : 0,
          }}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className={cn(
            'absolute inset-y-0 right-0 z-20 w-[260px] border-l border-edge bg-surface/85 backdrop-blur-md overflow-hidden shadow-xl shadow-ground/40 transform-gpu',
            sidebarOpen ? 'pointer-events-auto' : 'pointer-events-none'
          )}
        >
          <AwarenessSidebar awareness={awareness} />
        </motion.div>
      </div>

      {/* Settings Dialog */}
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        serverUrl={serverUrl}
        onUpdateUrl={updateServerUrl}
        connectionState={connectionState}
        error={error}
        onReconnect={reconnect}
        toolDisplayMode={toolDisplayMode}
        onToolDisplayModeChange={updateToolDisplayMode}
      />
    </div>
  )
}
