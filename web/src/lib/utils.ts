import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatRelativeTime(date: Date): string {
  const diff = Date.now() - date.getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  const rem = minutes % 60
  if (hours < 24) return rem > 0 ? `${hours}h ${rem}m ago` : `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

/** Strip daemon log format markers from entity output text */
export function sanitizeEntityOutput(text: string): string {
  return text
    .replace(/<!--\s*(user|assistant)\s*-->/g, '')
    .replace(/\*\*Claude:\*\*/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/** Format a duration in ms as compact "Xm Ys" or "Xs" */
export function formatDurationCompact(ms: number): string {
  const seconds = Math.round(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const rem = seconds % 60
  return rem > 0 ? `${minutes}m ${rem}s` : `${minutes}m`
}

export function formatDuration(date: Date): string {
  const diff = Date.now() - date.getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const rem = minutes % 60
  return rem > 0 ? `${hours}h ${rem}m` : `${hours}h`
}
