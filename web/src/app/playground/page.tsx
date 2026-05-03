"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { usePlayground } from "@/lib/playground/use-playground";
import { TimelineScrubber } from "@/components/playground/timeline-scrubber";
import { BudgetPanel } from "@/components/playground/budget-panel";
import { BlockList } from "@/components/playground/block-list";
import { Inspector } from "@/components/playground/inspector";

export default function PlaygroundPage() {
  const {
    bootstrap,
    bootstrapError,
    snapshot,
    snapshotLoading,
    snapshotError,
    timeline,
    timelineLoading,
    selectedTurn,
    overrides,
    overridesAreDefault,
    setSelectedTurn,
    updateOverrides,
    resetOverrides,
  } = usePlayground();

  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);

  // Clear selection when scrubbing to a turn that doesn't contain the block.
  useEffect(() => {
    if (!selectedBlockId || !snapshot) return;
    const stillExists = snapshot.block_list.some((b) => b.block_id === selectedBlockId);
    if (!stillExists) setSelectedBlockId(null);
  }, [snapshot, selectedBlockId]);

  if (bootstrapError) {
    return (
      <div className="h-screen flex flex-col items-center justify-center bg-ground text-center px-6">
        <div className="text-xs uppercase tracking-[0.2em] font-mono text-attention mb-2">
          could not connect
        </div>
        <p className="text-sm text-ch-subtext max-w-md">{bootstrapError}</p>
        <p className="text-[11px] font-mono text-ch-hint mt-4">
          is the spellbook server running on :8101?
        </p>
      </div>
    );
  }

  if (!bootstrap || selectedTurn == null) {
    return (
      <div className="h-screen flex items-center justify-center bg-ground">
        <div className="flex items-center gap-2 text-xs font-mono text-ch-hint">
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-status" />
          loading session…
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-ground text-ch-text overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-5 py-2.5 border-b border-edge bg-surface/60 backdrop-blur-sm flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-accent" />
          <div>
            <h1 className="font-display text-sm font-semibold tracking-tight">
              context playground
            </h1>
            <div className="text-[10px] font-mono text-ch-hint truncate max-w-[440px]">
              {bootstrap.session_dir}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3 text-[10px] font-mono text-ch-hint">
            <span>
              <span className="text-ch-subtext">{bootstrap.model}</span>
              <span className="text-ch-hint/50"> · {bootstrap.provider}</span>
            </span>
            <span className="w-px h-3 bg-edge" />
            <span>
              <span className="text-ch-subtext tabular-nums">
                {bootstrap.latest_completed_turn}
              </span>{" "}
              turns
            </span>
          </div>
          <Link
            href="/"
            className="text-[11px] font-mono text-ch-hint hover:text-accent transition-colors"
          >
            ← chat
          </Link>
        </div>
      </header>

      {/* Timeline */}
      <div className="px-5 pt-3 pb-3 flex-shrink-0 border-b border-edge bg-surface/40">
        <TimelineScrubber
          bootstrap={bootstrap}
          timeline={timeline}
          loading={timelineLoading}
          selectedTurn={selectedTurn}
          onSelectTurn={setSelectedTurn}
        />
      </div>

      {/* Errors */}
      {snapshotError && (
        <div className="px-4 py-1.5 bg-attention/10 text-attention text-[11px] font-mono border-b border-attention/30">
          snapshot: {snapshotError}
        </div>
      )}

      {/* Main grid */}
      <div className="flex-1 min-h-0 grid grid-cols-[320px_1fr] divide-x divide-edge">
        {/* Left: budget */}
        <aside className="bg-surface/40 overflow-y-auto">
          {snapshot ? (
            <BudgetPanel
              bootstrap={bootstrap}
              snapshot={snapshot}
              loading={snapshotLoading}
              overrides={overrides}
              overridesAreDefault={overridesAreDefault}
              onUpdateOverrides={updateOverrides}
              onResetOverrides={resetOverrides}
            />
          ) : (
            <PanelSkeleton label="budget" />
          )}
        </aside>

        {/* Right: blocks (top) + inspector (bottom) */}
        <section className={cn("grid min-h-0", selectedBlockId ? "grid-rows-[minmax(180px,40%)_1fr]" : "grid-rows-1")}>
          <div className="min-h-0 bg-surface/30 border-b border-edge overflow-hidden">
            {snapshot ? (
              <BlockList
                blocks={snapshot.block_list}
                selectedBlockId={selectedBlockId}
                onSelectBlock={(id) =>
                  setSelectedBlockId((prev) => (prev === id ? null : id))
                }
                loading={snapshotLoading}
              />
            ) : (
              <PanelSkeleton label="blocks" />
            )}
          </div>

          {selectedBlockId && (
            <div className="min-h-0 overflow-hidden">
              <Inspector
                blockId={selectedBlockId}
                turn={selectedTurn}
                overrides={overrides}
                onClose={() => setSelectedBlockId(null)}
              />
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function PanelSkeleton({ label }: { label: string }) {
  return (
    <div className="h-full flex items-center justify-center text-xs font-mono text-ch-hint">
      <span className="w-1 h-1 rounded-full bg-accent animate-pulse-status mr-2" />
      loading {label}…
    </div>
  );
}
