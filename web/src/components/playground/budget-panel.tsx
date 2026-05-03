"use client";

import { cn } from "@/lib/utils";
import { formatTokens, formatTokensLong } from "@/lib/playground/fidelity";
import type {
  BootstrapResponse,
  PlaygroundOverrides,
  SnapshotResponse,
} from "@/lib/playground/types";

interface BudgetPanelProps {
  bootstrap: BootstrapResponse;
  snapshot: SnapshotResponse;
  loading: boolean;
  overrides: PlaygroundOverrides;
  overridesAreDefault: boolean;
  onUpdateOverrides: (patch: Partial<PlaygroundOverrides>) => void;
  onResetOverrides: () => void;
}

interface Segment {
  key: string;
  label: string;
  tokens: number;
  color: string;
  hint?: string;
}

export function BudgetPanel({
  bootstrap,
  snapshot,
  loading,
  overrides,
  overridesAreDefault,
  onUpdateOverrides,
  onResetOverrides,
}: BudgetPanelProps) {
  const b = snapshot.budget_breakdown;
  const total = b.estimated_input_tokens;

  const segments: Segment[] = [
    {
      key: "frame",
      label: "Frame",
      tokens: b.frame_tokens,
      color: "#5890e8",
      hint: "system prompt + tools + identity",
    },
    {
      key: "compacted",
      label: "Compacted blocks",
      tokens: b.compacted_block_tokens,
      color: "#a78bfa",
      hint: "summary / index / headline",
    },
    {
      key: "full",
      label: "Full blocks",
      tokens: b.full_block_tokens,
      color: "#3dbfa0",
      hint: "blocks at full fidelity",
    },
    {
      key: "tail",
      label: "Tail",
      tokens: b.tail_tokens,
      color: "#e8b040",
      hint: "recent unblocked turns",
    },
    {
      key: "overlay",
      label: "Overlays",
      tokens: b.overlay_tokens,
      color: "#e87060",
      hint: "tool results, planner overlays",
    },
  ];

  const target = bootstrap.target_message_tokens;
  const warning = bootstrap.warning_message_tokens;
  const forced = bootstrap.forced_message_tokens;

  // Use the largest threshold as the gauge ceiling so we can see overruns clearly.
  const gaugeMax = Math.max(forced, total) * 1.02;

  const regimeLabel = (() => {
    if (b.regime !== "unknown") return b.regime;
    if (total >= forced) return "forced";
    if (total >= warning) return "warning";
    return "calm";
  })();

  const regimeColor = {
    calm: "text-working bg-working/10 border-working/30",
    warning: "text-idle bg-idle/10 border-idle/30",
    forced: "text-attention bg-attention/10 border-attention/30",
    unknown: "text-ch-subtext bg-ch-subtext/10 border-ch-subtext/20",
  }[regimeLabel] ?? "text-ch-subtext bg-ch-subtext/10 border-ch-subtext/20";

  return (
    <div className={cn("flex flex-col h-full", loading && "opacity-80")}>
      <PanelHeader title="Token budget" subtitle={`turn ${snapshot.turn}`} loading={loading} />

      <div className="px-4 pt-3 pb-1">
        <div className="flex items-baseline justify-between">
          <div>
            <div className="text-2xl font-mono font-medium text-ch-text leading-none">
              {formatTokens(total)}
            </div>
            <div className="text-[10px] text-ch-hint mt-1 font-mono tabular-nums">
              {formatTokensLong(total)} estimated
            </div>
          </div>
          <span
            className={cn(
              "px-2 py-0.5 rounded text-[10px] uppercase tracking-wider font-mono border",
              regimeColor,
            )}
          >
            {regimeLabel}
          </span>
        </div>

        {/* Stacked bar against thresholds */}
        <div className="mt-4">
          <div className="relative h-2.5 rounded-full bg-edge/40 overflow-hidden">
            {(() => {
              let acc = 0;
              return segments.map((s) => {
                const pct = (s.tokens / gaugeMax) * 100;
                const left = (acc / gaugeMax) * 100;
                acc += s.tokens;
                if (s.tokens === 0) return null;
                return (
                  <div
                    key={s.key}
                    className="absolute top-0 bottom-0"
                    style={{
                      left: `${left}%`,
                      width: `${pct}%`,
                      background: s.color,
                      opacity: 0.85,
                    }}
                  />
                );
              });
            })()}
          </div>
          {/* Threshold ticks */}
          <div className="relative h-3 mt-1">
            <ThresholdMark value={target} max={gaugeMax} label="target" color="#5c5a52" />
            <ThresholdMark value={warning} max={gaugeMax} label="warn" color="#e8b040" />
            <ThresholdMark value={forced} max={gaugeMax} label="forced" color="#e86060" />
          </div>
        </div>

        <div className="mt-4 space-y-1.5">
          {segments.map((s) => (
            <SegmentRow key={s.key} segment={s} total={total} />
          ))}
        </div>
      </div>

      {/* Overrides */}
      <div className="px-4 mt-5 pt-4 border-t border-edge">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-[10px] uppercase tracking-[0.18em] text-ch-hint font-mono">
            hypotheticals
          </span>
          {!overridesAreDefault && (
            <button
              onClick={onResetOverrides}
              className="text-[10px] font-mono text-accent hover:text-accent-bright transition-colors"
            >
              reset
            </button>
          )}
        </div>

        <div className="space-y-2">
          <Toggle
            label="TTL compaction"
            hint="Auto-summarize stale tool results"
            checked={overrides.ttl_compaction !== false}
            onChange={(v) => onUpdateOverrides({ ttl_compaction: v })}
          />
          <Toggle
            label="Dive elision"
            hint="Elide replayed Dive sub-turns"
            checked={overrides.dive_elision !== false}
            onChange={(v) => onUpdateOverrides({ dive_elision: v })}
          />
          <Toggle
            label="Use recorded compaction"
            hint='Off = show raw uncompacted cost (compaction_mode="none")'
            checked={overrides.compaction_mode !== "none"}
            onChange={(v) =>
              onUpdateOverrides({ compaction_mode: v ? "recorded" : "none" })
            }
          />
        </div>

        {!overridesAreDefault && (
          <div className="mt-3 px-2 py-1.5 rounded text-[10px] font-mono text-accent-bright bg-accent/5 border border-accent/20">
            ✻ recalculated under hypothetical settings
          </div>
        )}
      </div>

      <div className="px-4 mt-auto pt-4 pb-3 text-[10px] font-mono text-ch-hint border-t border-edge">
        <div className="flex justify-between">
          <span>measurement</span>
          <span className="text-ch-subtext">{b.measurement_state}</span>
        </div>
        {snapshot.plan?.diagnostics?.[0] && (
          <div className="mt-1 line-clamp-2 text-ch-hint/70">
            {snapshot.plan.diagnostics[0]}
          </div>
        )}
      </div>
    </div>
  );
}

function SegmentRow({ segment, total }: { segment: Segment; total: number }) {
  const pct = total > 0 ? (segment.tokens / total) * 100 : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span
        className="w-1.5 h-1.5 rounded-sm flex-shrink-0"
        style={{ background: segment.color }}
      />
      <span className="text-ch-text font-medium flex-1 min-w-0 truncate">{segment.label}</span>
      <span className="text-ch-hint font-mono text-[10px] tabular-nums w-8 text-right">
        {pct.toFixed(0)}%
      </span>
      <span className="text-ch-subtext font-mono text-[11px] tabular-nums w-12 text-right">
        {formatTokens(segment.tokens)}
      </span>
    </div>
  );
}

function ThresholdMark({
  value,
  max,
  label,
  color,
}: {
  value: number;
  max: number;
  label: string;
  color: string;
}) {
  const left = (value / max) * 100;
  if (left > 100 || left < 0) return null;
  return (
    <div className="absolute top-0" style={{ left: `${left}%`, transform: "translateX(-50%)" }}>
      <div className="w-px h-1.5" style={{ background: color, opacity: 0.5 }} />
      <div
        className="text-[8px] font-mono mt-0.5 tracking-wider"
        style={{ color, opacity: 0.7 }}
      >
        {label}
      </div>
    </div>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="w-full flex items-start gap-3 p-2 rounded-md hover:bg-elevated/60 transition-colors text-left group"
    >
      <span
        className={cn(
          "mt-0.5 flex-shrink-0 w-7 h-4 rounded-full relative transition-colors",
          checked ? "bg-accent/80" : "bg-edge",
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 w-3 h-3 rounded-full bg-ch-text transition-all",
            checked ? "left-3.5" : "left-0.5",
          )}
        />
      </span>
      <span className="flex-1 min-w-0">
        <span className="block text-xs text-ch-text font-medium">{label}</span>
        {hint && (
          <span className="block text-[10px] text-ch-hint mt-0.5 leading-snug">{hint}</span>
        )}
      </span>
    </button>
  );
}

function PanelHeader({
  title,
  subtitle,
  loading,
}: {
  title: string;
  subtitle?: string;
  loading?: boolean;
}) {
  return (
    <div className="px-4 py-2.5 border-b border-edge flex items-center justify-between">
      <div className="flex items-baseline gap-2">
        <h2 className="text-xs font-semibold text-ch-text tracking-tight">{title}</h2>
        {subtitle && (
          <span className="text-[10px] font-mono text-ch-hint">{subtitle}</span>
        )}
      </div>
      {loading && (
        <span className="w-1 h-1 rounded-full bg-accent animate-pulse-status" />
      )}
    </div>
  );
}
