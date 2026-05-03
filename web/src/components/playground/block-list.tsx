"use client";

import { cn } from "@/lib/utils";
import {
  FIDELITY_BG,
  FIDELITY_DOT,
  FIDELITY_LABEL,
  FIDELITY_ORDER,
  formatTokens,
} from "@/lib/playground/fidelity";
import type { BlockListItem, FidelityMode } from "@/lib/playground/types";

interface BlockListProps {
  blocks: BlockListItem[];
  selectedBlockId: string | null;
  onSelectBlock: (blockId: string) => void;
  loading: boolean;
}

export function BlockList({
  blocks,
  selectedBlockId,
  onSelectBlock,
  loading,
}: BlockListProps) {
  // Token bar scaling — use the largest block as 100%.
  const maxTokens = blocks.reduce((m, b) => Math.max(m, b.estimated_tokens), 0);

  // Tally for the legend.
  const tally: Record<FidelityMode, number> = {
    full: 0,
    summary: 0,
    index: 0,
    headline: 0,
  };
  for (const b of blocks) tally[b.mode] = (tally[b.mode] ?? 0) + 1;

  return (
    <div className={cn("flex flex-col h-full min-h-0", loading && "opacity-80")}>
      <div className="px-4 py-2.5 border-b border-edge flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <h2 className="text-xs font-semibold text-ch-text tracking-tight">Blocks</h2>
          <span className="text-[10px] font-mono text-ch-hint">{blocks.length}</span>
        </div>
        {loading && (
          <span className="w-1 h-1 rounded-full bg-accent animate-pulse-status" />
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {blocks.length === 0 ? (
          <div className="px-4 py-8 text-center text-xs text-ch-hint font-mono">
            no blocks at this turn
          </div>
        ) : (
          <ul className="py-1">
            {blocks.map((block) => (
              <BlockRow
                key={block.block_id}
                block={block}
                maxTokens={maxTokens}
                selected={block.block_id === selectedBlockId}
                onClick={() => onSelectBlock(block.block_id)}
              />
            ))}
          </ul>
        )}
      </div>

      {/* Legend */}
      <div className="px-4 py-2 border-t border-edge bg-elevated/30">
        <div className="flex items-center gap-3 flex-wrap text-[10px] font-mono text-ch-subtext">
          {FIDELITY_ORDER.map((mode) => (
            <span key={mode} className="flex items-center gap-1.5">
              <span className={cn("w-1.5 h-1.5 rounded-sm", FIDELITY_DOT[mode])} />
              <span>{FIDELITY_LABEL[mode].toLowerCase()}</span>
              <span className="text-ch-hint/60">{tally[mode] || 0}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function BlockRow({
  block,
  maxTokens,
  selected,
  onClick,
}: {
  block: BlockListItem;
  maxTokens: number;
  selected: boolean;
  onClick: () => void;
}) {
  const widthPct =
    maxTokens > 0 ? Math.max(2, (block.estimated_tokens / maxTokens) * 100) : 0;

  return (
    <li>
      <button
        onClick={onClick}
        className={cn(
          "w-full text-left px-3 py-2 border-l-2 transition-all relative group",
          selected
            ? "bg-accent/10 border-l-accent"
            : "border-l-transparent hover:bg-elevated/50 hover:border-l-edge-bright",
        )}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[10px] font-mono text-ch-hint w-5 text-right tabular-nums flex-shrink-0">
            {block.ordinal}
          </span>
          <span
            className={cn(
              "px-1.5 py-0.5 rounded text-[9px] uppercase tracking-wider font-mono border flex-shrink-0",
              FIDELITY_BG[block.mode],
            )}
          >
            {block.mode}
          </span>
          <span className="flex-1 min-w-0 text-xs text-ch-text truncate">
            {block.title}
          </span>
          <div className="flex items-center gap-1 flex-shrink-0">
            {block.pinned && (
              <span title="pinned" className="text-[11px]">📌</span>
            )}
            {block.open_thread && (
              <span title="open thread" className="w-1.5 h-1.5 rounded-full bg-attention/80" />
            )}
            {block.proposed && (
              <span
                title="proposed for compaction"
                className="text-[9px] text-idle font-mono uppercase tracking-wider"
              >
                ⇣
              </span>
            )}
          </div>
        </div>
        <div className="mt-1 ml-7 flex items-center gap-2">
          <div className="flex-1 h-0.5 rounded-full bg-edge/50 overflow-hidden">
            <div
              className={cn("h-full rounded-full", FIDELITY_DOT[block.mode])}
              style={{ width: `${widthPct}%`, opacity: 0.7 }}
            />
          </div>
          <span className="text-[10px] font-mono text-ch-hint tabular-nums w-12 text-right">
            {formatTokens(block.estimated_tokens)}
          </span>
          <span className="text-[10px] font-mono text-ch-hint/50 tabular-nums w-16 text-right">
            t{block.turn_range[0]}–{block.turn_range[1]}
          </span>
        </div>
      </button>
    </li>
  );
}
