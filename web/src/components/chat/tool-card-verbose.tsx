'use client'

import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { cn } from '@/lib/utils'
import type { MockToolCall, MockToolResult, ToolDisplay } from './mock-data'

// ─── Types ──────────────────────────────────────────────────────

interface ToolCardVerboseProps {
  call: MockToolCall
  result?: MockToolResult
}

// ─── Tool name colors ───────────────────────────────────────────

const TOOL_COLORS: Record<string, { border: string; title: string }> = {
  Read:    { border: 'border-ch-hint/30', title: 'text-ch-subtext' },
  Write:   { border: 'border-accent/30', title: 'text-accent' },
  Edit:    { border: 'border-accent/30', title: 'text-accent' },
  Bash:    { border: 'border-working/30', title: 'text-working' },
  Grep:    { border: 'border-reader/30', title: 'text-reader' },
  Glob:    { border: 'border-ch-hint/30', title: 'text-ch-subtext' },
  Reflect: { border: 'border-reviewer/30', title: 'text-reviewer' },
  Recall:  { border: 'border-reviewer/30', title: 'text-reviewer' },
  Pin:     { border: 'border-reviewer/30', title: 'text-reviewer' },
  Forget:  { border: 'border-reviewer/30', title: 'text-reviewer' },
  Amend:   { border: 'border-reviewer/30', title: 'text-reviewer' },
  Skill:   { border: 'border-familiar/30', title: 'text-familiar' },
}

const DEFAULT_COLORS = { border: 'border-edge', title: 'text-ch-subtext' }
const DIFF_THEME = 'github-dark'
const MAX_HIGHLIGHTABLE_DIFF_LINES = 400
const MAX_HIGHLIGHTABLE_DIFF_CHARS = 50_000
const MAX_DIFF_CACHE_ENTRIES = 100

type DiffHighlighter = {
  codeToHtml: (code: string, options: { lang: string; theme: string }) => string
}

let shikiHighlighterPromise: Promise<DiffHighlighter> | null = null
const highlightedDiffCache = new Map<string, string | null>()
const pendingDiffHighlights = new Map<string, Promise<string | null>>()

// ─── Main Component ─────────────────────────────────────────────

export function ToolCardVerbose({ call, result }: ToolCardVerboseProps) {
  const [manualCollapsed, setManualCollapsed] = useState<boolean | null>(null)
  const colors = TOOL_COLORS[call.tool] || DEFAULT_COLORS
  const isRunning = !result
  const isError = result?.isError

  const titleColor = isError ? 'text-attention' : isRunning ? 'text-idle' : colors.title
  const borderColor = isError ? 'border-attention/30' : isRunning ? 'border-idle/30' : colors.border

  const hasExpandableContent = result?.display && isExpandable(result.display)
  const collapsed = manualCollapsed ?? !hasExpandableContent

  return (
    <div className={cn('rounded-lg border overflow-hidden my-1.5', borderColor)}>
      {/* Header */}
      <button
        onClick={() => hasExpandableContent && setManualCollapsed(!collapsed)}
        className={cn(
          'w-full flex items-center justify-between px-3 py-1.5 text-left',
          'bg-elevated/40 border-b border-edge/30',
          hasExpandableContent && 'cursor-pointer hover:bg-elevated/60'
        )}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className={cn('text-xs font-mono font-semibold', titleColor)}>
            {call.tool}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0 ml-2">
          {isRunning && (
            <span className="text-[10px] text-idle animate-pulse-status">⟳</span>
          )}
          {isError && (
            <span className="text-[10px] text-attention">✗</span>
          )}
          {!isRunning && !isError && (
            <span className="text-[10px] text-working">✓</span>
          )}
          {hasExpandableContent && (
            <svg
              width="10"
              height="10"
              viewBox="0 0 10 10"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className={cn(
                'text-ch-hint transition-transform duration-150',
                !collapsed && 'rotate-180'
              )}
            >
              <path d="M2.5 4L5 6.5L7.5 4" />
            </svg>
          )}
        </div>
      </button>

      {/* Body — display-driven content */}
      <AnimatePresence initial={false}>
        {!collapsed && result?.display && (
          <motion.div
            key="card-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="px-3 py-2">
              <DisplayBody display={result.display} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Compact body for non-expandable results with display */}
      {result && !hasExpandableContent && !isRunning && result.display && (
        <CompactResultBody result={result} />
      )}

      {/* Fallback: no display data, show summary */}
      {result && !result.display && !isRunning && !isError && (
        <div className="px-3 py-1.5">
          <span className="text-xs font-mono text-ch-hint">{call.summary}</span>
        </div>
      )}

      {/* Running state with no display yet */}
      {isRunning && (
        <div className="px-3 py-1.5">
          <span className="text-xs font-mono text-ch-hint">{call.summary}</span>
        </div>
      )}

      {/* Error output */}
      {isError && result?.result && (
        <div className="px-3 py-2 border-t border-attention/20">
          <pre className="text-xs font-mono text-attention/80 whitespace-pre-wrap">
            {result.result}
          </pre>
        </div>
      )}
    </div>
  )
}

// ─── Display Body Router ────────────────────────────────────────

function DisplayBody({ display }: { display: ToolDisplay }) {
  switch (display.kind) {
    case 'diff':
      return <DiffBody display={display} />
    case 'command':
      return <CommandBody display={display} />
    case 'read':
      return <ReadBody display={display} />
    case 'grep':
      return <GrepBody display={display} />
    case 'glob':
      return <GlobBody display={display} />
    case 'text':
      return <TextBody display={display} />
    default:
      return <GenericBody display={display} />
  }
}

// ─── Diff Display ───────────────────────────────────────────────

function DiffBody({ display }: { display: ToolDisplay }) {
  const diff = (display.diff as string) || ''
  const stats = display.stats as { added?: number; removed?: number } | undefined
  const language = display.language as string | undefined
  const path = display.path as string | undefined
  const changeType = display.change_type as string | undefined

  return (
    <div className="space-y-1.5">
      {path && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-semibold text-ch-text">{path}</span>
          {changeType && changeType !== 'edit' && (
            <span className="text-[10px] font-mono text-ch-hint">({changeType})</span>
          )}
        </div>
      )}
      {stats && (
        <div className="flex items-center gap-2 text-[11px] font-mono">
          {stats.added !== undefined && (
            <span className="text-working">+{stats.added}</span>
          )}
          {stats.removed !== undefined && (
            <span className="text-attention">-{stats.removed}</span>
          )}
        </div>
      )}
      <DiffRenderer diff={diff} language={language} />
    </div>
  )
}

// ─── Diff Renderer with Shiki ───────────────────────────────────

function DiffRenderer({ diff, language }: { diff: string; language?: string }) {
  const cacheKey = buildDiffCacheKey(diff, language)
  const cachedHtml = highlightedDiffCache.get(cacheKey)
  const [highlightState, setHighlightState] = useState<{ key: string; html: string | null }>(() => {
    return { key: cacheKey, html: cachedHtml ?? null }
  })

  useEffect(() => {
    let cancelled = false

    if (cachedHtml !== undefined) {
      return () => {
        cancelled = true
      }
    }

    void highlightDiffHtml(diff, language).then((html) => {
      if (!cancelled) {
        setHighlightState({ key: cacheKey, html })
      }
    })

    return () => {
      cancelled = true
    }
  }, [cacheKey, cachedHtml, diff, language])

  const highlightedHtml = highlightState.key === cacheKey
    ? highlightState.html
    : cachedHtml ?? null

  if (highlightedHtml === null) {
    return <PlainDiff diff={diff} />
  }

  return (
    <div
      className="diff-container rounded-md overflow-x-auto text-xs font-mono"
      dangerouslySetInnerHTML={{ __html: highlightedHtml }}
    />
  )
}

function PlainDiff({ diff }: { diff: string }) {
  const lines = diff.split('\n').filter(
    (l) => !l.startsWith('---') && !l.startsWith('+++') && !l.startsWith('@@')
  )

  return (
    <div className="diff-container rounded-md overflow-x-auto text-xs font-mono">
      {lines.map((line, i) => {
        const isAdd = line.startsWith('+')
        const isRemove = line.startsWith('-')
        return (
          <div
            key={i}
            className={cn(
              'diff-line',
              isAdd && 'diff-added',
              isRemove && 'diff-removed'
            )}
          >
            <span className="diff-prefix">
              {isAdd ? '+' : isRemove ? '-' : ' '}
            </span>
            <span className="diff-code">
              {line.startsWith('+') || line.startsWith('-') || line.startsWith(' ')
                ? line.slice(1)
                : line}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ─── Command Display ────────────────────────────────────────────

function CommandBody({ display }: { display: ToolDisplay }) {
  const command = (display.command as string) || ''
  const exitCode = (display.exit_code as number) ?? 0
  const durationMs = (display.duration_ms as number) || 0
  const cwd = display.cwd as string | undefined
  const stdout = stripAnsi((display.stdout as string) || '')
  const stderr = stripAnsi((display.stderr as string) || '')

  return (
    <div className="space-y-1.5">
      {/* Command line */}
      <pre className="text-xs font-mono whitespace-pre-wrap">
        <span className="text-working">$ </span>
        <span className="text-ch-text">{command}</span>
      </pre>

      {/* Stats */}
      <div className="flex items-center gap-3 text-[10px] font-mono">
        <span className={exitCode === 0 ? 'text-working' : 'text-attention'}>
          exit {exitCode}
        </span>
        {durationMs > 0 && (
          <span className="text-ch-hint">{durationMs}ms</span>
        )}
        {cwd && (
          <span className="text-ch-hint">{cwd}</span>
        )}
      </div>

      {/* Output */}
      {stdout && (
        <pre className="text-xs font-mono text-ch-hint whitespace-pre-wrap max-h-[300px] overflow-y-auto">
          {stdout}
        </pre>
      )}
      {stderr && (
        <pre className="text-xs font-mono text-attention/60 whitespace-pre-wrap max-h-[200px] overflow-y-auto">
          {stderr}
        </pre>
      )}
    </div>
  )
}

// ─── Read Display ───────────────────────────────────────────────

function ReadBody({ display }: { display: ToolDisplay }) {
  const path = (display.path as string) || ''
  const startLine = (display.start_line as number) || 1
  const endLine = (display.end_line as number) || 0
  const totalLines = (display.total_lines as number) || 0
  const lineCount = (display.line_count as number) || 0

  return (
    <div className="text-xs font-mono">
      <span className="text-ch-text font-semibold">{path}</span>
      {startLine === 1 && endLine === totalLines ? (
        <span className="text-ch-hint ml-2">{totalLines} lines</span>
      ) : (
        <span className="text-ch-hint ml-2">
          Lines {startLine}-{endLine} ({lineCount} of {totalLines})
        </span>
      )}
    </div>
  )
}

// ─── Grep Display ───────────────────────────────────────────────

function GrepBody({ display }: { display: ToolDisplay }) {
  const pattern = (display.pattern as string) || ''
  const searchPath = (display.search_path as string) || ''
  const matchCount = (display.match_count as number) ?? 0
  const fileCount = (display.file_count as number) ?? 0

  return (
    <div className="text-xs font-mono">
      <span className="text-ch-text font-semibold">&quot;{pattern}&quot;</span>
      {searchPath && <span className="text-ch-hint ml-1">in {searchPath}</span>}
      <div className="text-ch-hint mt-0.5">
        {matchCount === 0 && fileCount === 0
          ? 'No matches'
          : matchCount === -1
            ? `${fileCount} file${fileCount !== 1 ? 's' : ''}`
            : `${matchCount} matches across ${fileCount} files`}
      </div>
    </div>
  )
}

// ─── Glob Display ───────────────────────────────────────────────

function GlobBody({ display }: { display: ToolDisplay }) {
  const pattern = (display.pattern as string) || ''
  const matchCount = (display.match_count as number) ?? 0

  return (
    <div className="text-xs font-mono">
      <span className="text-ch-text font-semibold">{pattern}</span>
      <span className="text-ch-hint ml-2">
        {matchCount === 0 ? 'no matches' : `${matchCount} file${matchCount !== 1 ? 's' : ''} matched`}
      </span>
    </div>
  )
}

// ─── Text Display (fallback) ────────────────────────────────────

function TextBody({ display }: { display: ToolDisplay }) {
  return (
    <div className="text-xs font-mono">
      <p className="text-ch-text font-semibold">{(display.title as string) || ''}</p>
      <p className="text-ch-hint mt-0.5 whitespace-pre-wrap">{(display.body as string) || ''}</p>
    </div>
  )
}

// ─── Generic fallback ───────────────────────────────────────────

function GenericBody({ display }: { display: ToolDisplay }) {
  return (
    <div className="text-xs font-mono text-ch-hint">
      <span className="text-ch-subtext">{display.kind}</span>
    </div>
  )
}

// ─── Compact result for non-expandable tools ────────────────────

function CompactResultBody({ result }: { result: MockToolResult }) {
  if (!result.display) return null
  const display = result.display

  // Render inline info for simple tools
  const kind = display.kind
  if (kind === 'read' || kind === 'grep' || kind === 'glob') {
    return (
      <div className="px-3 py-1.5">
        <DisplayBody display={display} />
      </div>
    )
  }

  return null
}

// ─── Helpers ────────────────────────────────────────────────────

function isExpandable(display: ToolDisplay): boolean {
  const kind = display.kind
  return kind === 'diff' || kind === 'command' || kind === 'text'
}

async function highlightDiffHtml(diff: string, language?: string): Promise<string | null> {
  const cacheKey = buildDiffCacheKey(diff, language)

  if (highlightedDiffCache.has(cacheKey)) {
    return highlightedDiffCache.get(cacheKey) ?? null
  }

  const pending = pendingDiffHighlights.get(cacheKey)
  if (pending) {
    return pending
  }

  const highlightPromise = (async () => {
    try {
      const lines = parseDiffLines(diff)
      if (
        lines.length === 0 ||
        lines.length > MAX_HIGHLIGHTABLE_DIFF_LINES ||
        diff.length > MAX_HIGHLIGHTABLE_DIFF_CHARS
      ) {
        cacheHighlightedDiff(cacheKey, null)
        return null
      }

      const lang = mapLanguage(language)
      const highlighter = await getDiffHighlighter(lang)
      const combinedCode = lines.map((line) => line.code).join('\n')
      const highlighted = highlighter.codeToHtml(combinedCode, {
        lang,
        theme: DIFF_THEME,
      })
      const highlightedLines = extractHighlightedLines(highlighted)

      const html = lines.map((line, index) => {
        const prefixHtml = line.prefix
          ? `<span class="diff-prefix">${escapeHtml(line.prefix)}</span>`
          : ''
        const codeHtml = normalizeHighlightedLine(highlightedLines[index], line.code)

        return `<div class="diff-line ${line.bgClass}">${prefixHtml}<span class="diff-code">${codeHtml}</span></div>`
      }).join('\n')

      cacheHighlightedDiff(cacheKey, html)
      return html
    } catch {
      cacheHighlightedDiff(cacheKey, null)
      return null
    } finally {
      pendingDiffHighlights.delete(cacheKey)
    }
  })()

  pendingDiffHighlights.set(cacheKey, highlightPromise)
  return highlightPromise
}

async function getDiffHighlighter(lang: string): Promise<DiffHighlighter> {
  if (!shikiHighlighterPromise) {
    shikiHighlighterPromise = import('shiki').then(async (shiki) =>
      (shiki.getSingletonHighlighter as (...args: unknown[]) => Promise<DiffHighlighter>)({
        themes: [DIFF_THEME],
        langs: [lang],
      })
    )
    return shikiHighlighterPromise
  }

  const shiki = await import('shiki')
  return (shiki.getSingletonHighlighter as (...args: unknown[]) => Promise<DiffHighlighter>)({
    themes: [DIFF_THEME],
    langs: [lang],
  })
}

function buildDiffCacheKey(diff: string, language?: string): string {
  return `${mapLanguage(language)}\u0000${diff}`
}

function cacheHighlightedDiff(cacheKey: string, html: string | null): void {
  if (highlightedDiffCache.has(cacheKey)) {
    highlightedDiffCache.delete(cacheKey)
  }

  highlightedDiffCache.set(cacheKey, html)

  if (highlightedDiffCache.size > MAX_DIFF_CACHE_ENTRIES) {
    const oldestKey = highlightedDiffCache.keys().next().value
    if (oldestKey) {
      highlightedDiffCache.delete(oldestKey)
    }
  }
}

function parseDiffLines(diff: string): Array<{ prefix: string; code: string; bgClass: string }> {
  return diff
    .split('\n')
    .filter((line) => !line.startsWith('---') && !line.startsWith('+++') && !line.startsWith('@@'))
    .map((line) => {
      if (line.startsWith('+')) {
        return { prefix: '+', code: line.slice(1), bgClass: 'diff-added' }
      }
      if (line.startsWith('-')) {
        return { prefix: '-', code: line.slice(1), bgClass: 'diff-removed' }
      }
      if (line.startsWith(' ')) {
        return { prefix: ' ', code: line.slice(1), bgClass: '' }
      }
      return { prefix: '', code: line, bgClass: '' }
    })
}

function extractHighlightedLines(highlightedHtml: string): string[] {
  const innerMatch = highlightedHtml.match(/<code[^>]*>([\s\S]*?)<\/code>/)
  if (!innerMatch) return []
  return innerMatch[1].split('\n')
}

function normalizeHighlightedLine(highlightedLine: string | undefined, code: string): string {
  if (!highlightedLine) {
    return escapeHtml(code)
  }

  return highlightedLine
    .replace(/^<span class="line">/, '')
    .replace(/<\/span>$/, '') || escapeHtml(code)
}

function mapLanguage(lang?: string): string {
  if (!lang) return 'text'
  const map: Record<string, string> = {
    'py': 'python',
    'ts': 'typescript',
    'tsx': 'tsx',
    'js': 'javascript',
    'jsx': 'jsx',
    'rs': 'rust',
    'go': 'go',
    'css': 'css',
    'html': 'html',
    'json': 'json',
    'yaml': 'yaml',
    'yml': 'yaml',
    'md': 'markdown',
    'sh': 'bash',
    'bash': 'bash',
    'toml': 'toml',
    'sql': 'sql',
  }
  return map[lang.toLowerCase()] || lang.toLowerCase()
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** Strip ANSI escape sequences from terminal output */
function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
}
