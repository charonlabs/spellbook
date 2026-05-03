"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { formatTokens } from "@/lib/playground/fidelity";
import type { BootstrapResponse, TimelinePoint } from "@/lib/playground/types";

interface TimelineScrubberProps {
  bootstrap: BootstrapResponse;
  timeline: TimelinePoint[];
  loading: boolean;
  selectedTurn: number;
  onSelectTurn: (turn: number) => void;
}

export function TimelineScrubber({
  bootstrap,
  timeline,
  loading,
  selectedTurn,
  onSelectTurn,
}: TimelineScrubberProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [width, setWidth] = useState(800);
  const height = 110;
  const padTop = 8;
  const padBottom = 18;
  const padX = 8;
  const innerW = Math.max(1, width - padX * 2);
  const innerH = Math.max(1, height - padTop - padBottom);

  const latest = bootstrap.latest_completed_turn;
  const warning = bootstrap.warning_message_tokens;
  const forced = bootstrap.forced_message_tokens;

  // Resize observer.
  useEffect(() => {
    const el = svgRef.current?.parentElement;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 800;
      setWidth(Math.max(200, Math.floor(w)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Y scale: cap at max(forced, observed peak * 1.05) so warning/forced lines fit.
  const yMax = useMemo(() => {
    const peak = timeline.reduce((m, p) => Math.max(m, p.estimated_memory_tokens), 0);
    return Math.max(forced * 1.05, peak * 1.05, 1);
  }, [timeline, forced]);

  const x = (turn: number) => padX + (latest === 0 ? 0 : (turn / latest) * innerW);
  const y = (tokens: number) => padTop + innerH - (tokens / yMax) * innerH;

  // Build sawtooth area path from points.
  const areaPath = useMemo(() => {
    if (timeline.length === 0) return "";
    const sorted = [...timeline].sort((a, b) => a.turn - b.turn);
    let d = `M ${x(sorted[0].turn)} ${padTop + innerH}`;
    for (const p of sorted) {
      d += ` L ${x(p.turn)} ${y(p.estimated_memory_tokens)}`;
    }
    d += ` L ${x(sorted[sorted.length - 1].turn)} ${padTop + innerH} Z`;
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeline, width, yMax, latest]);

  const linePath = useMemo(() => {
    if (timeline.length === 0) return "";
    const sorted = [...timeline].sort((a, b) => a.turn - b.turn);
    return sorted
      .map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.turn)} ${y(p.estimated_memory_tokens)}`)
      .join(" ");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeline, width, yMax, latest]);

  // Drag interaction.
  const [dragging, setDragging] = useState(false);
  const handlePointer = (clientX: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = clientX - rect.left - padX;
    const ratio = Math.max(0, Math.min(1, px / innerW));
    const turn = Math.round(ratio * latest);
    onSelectTurn(turn);
  };

  // Y axis ticks for token thresholds.
  const ticks = useMemo(() => {
    const vals = [warning, forced];
    return vals
      .filter((v) => v > 0 && v < yMax)
      .map((v) => ({ tokens: v, y: y(v) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [warning, forced, yMax, height]);

  const selectedX = x(selectedTurn);
  const selectedPoint = useMemo(() => {
    if (timeline.length === 0) return null;
    let best = timeline[0];
    let bestDist = Math.abs(best.turn - selectedTurn);
    for (const p of timeline) {
      const d = Math.abs(p.turn - selectedTurn);
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    return best;
  }, [timeline, selectedTurn]);

  return (
    <div className="w-full select-none">
      <div className="flex items-baseline justify-between px-1 mb-2">
        <div className="flex items-center gap-3">
          <span className="text-[10px] uppercase tracking-[0.18em] text-ch-hint font-mono">
            timeline
          </span>
          <span className="text-[10px] text-ch-hint font-mono">
            turn {selectedTurn}
            <span className="text-ch-hint/40"> / {latest}</span>
          </span>
          {selectedPoint && (
            <span className="text-[10px] text-ch-subtext font-mono">
              ≈ {formatTokens(selectedPoint.estimated_memory_tokens)} tokens
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono text-ch-hint">
          {loading && (
            <span
              className="flex items-center gap-1.5"
              title="strided snapshots — chart fills in as samples arrive"
            >
              <span className="w-1 h-1 rounded-full bg-accent animate-pulse-status" />
              sampling…
            </span>
          )}
          <span>
            {timeline.length} samples
          </span>
        </div>
      </div>

      <svg
        ref={svgRef}
        width={width}
        height={height}
        className={cn(
          "block w-full rounded-md bg-elevated/40 border border-edge cursor-crosshair",
          dragging && "cursor-grabbing",
        )}
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture(e.pointerId);
          setDragging(true);
          handlePointer(e.clientX);
        }}
        onPointerMove={(e) => {
          if (!dragging) return;
          handlePointer(e.clientX);
        }}
        onPointerUp={(e) => {
          (e.target as Element).releasePointerCapture(e.pointerId);
          setDragging(false);
        }}
      >
        {/* Threshold lines */}
        {ticks.map((t, i) => (
          <g key={i}>
            <line
              x1={padX}
              x2={padX + innerW}
              y1={t.y}
              y2={t.y}
              stroke={t.tokens === forced ? "#e86060" : "#e8b040"}
              strokeWidth={1}
              strokeDasharray="3 3"
              opacity={0.4}
            />
            <text
              x={padX + innerW - 4}
              y={t.y - 3}
              fontSize={9}
              fontFamily="var(--font-mono)"
              fill={t.tokens === forced ? "#e86060" : "#e8b040"}
              opacity={0.7}
              textAnchor="end"
            >
              {formatTokens(t.tokens)}
            </text>
          </g>
        ))}

        {/* Sawtooth area */}
        <defs>
          <linearGradient id="ctxArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.04" />
          </linearGradient>
        </defs>
        {areaPath && <path d={areaPath} fill="url(#ctxArea)" />}
        {linePath && (
          <path
            d={linePath}
            fill="none"
            stroke="#a78bfa"
            strokeWidth={1.25}
            strokeLinejoin="round"
          />
        )}

        {/* Sample dots */}
        {timeline.map((p) => (
          <circle
            key={p.turn}
            cx={x(p.turn)}
            cy={y(p.estimated_memory_tokens)}
            r={1.5}
            fill="#c4b5fd"
            opacity={0.7}
          />
        ))}

        {/* Scrubber line */}
        <line
          x1={selectedX}
          x2={selectedX}
          y1={padTop}
          y2={padTop + innerH}
          stroke="#e4e2de"
          strokeWidth={1}
          opacity={0.85}
        />
        <circle
          cx={selectedX}
          cy={selectedPoint ? y(selectedPoint.estimated_memory_tokens) : padTop + innerH}
          r={3.5}
          fill="#e4e2de"
          stroke="#0a0a0c"
          strokeWidth={1.5}
        />

        {/* X axis turn marks */}
        {[0, 0.25, 0.5, 0.75, 1].map((r, i) => {
          const turn = Math.round(r * latest);
          return (
            <text
              key={i}
              x={padX + r * innerW}
              y={height - 4}
              fontSize={9}
              fontFamily="var(--font-mono)"
              fill="#5c5a52"
              textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}
            >
              {turn}
            </text>
          );
        })}
      </svg>
    </div>
  );
}
