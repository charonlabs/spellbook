"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { fetchBlockDetail, fetchBlockPreview } from "@/lib/playground/api";
import {
  FIDELITY_BG,
  FIDELITY_DOT,
  FIDELITY_LABEL,
  FIDELITY_ORDER,
  formatTokens,
  formatTokensLong,
} from "@/lib/playground/fidelity";
import type {
  BlockDetail,
  BlockPreviewResponse,
  FidelityMode,
  PlaygroundOverrides,
} from "@/lib/playground/types";

interface InspectorProps {
  blockId: string | null;
  turn: number;
  overrides: PlaygroundOverrides;
  onClose: () => void;
}

export function Inspector({ blockId, turn, overrides, onClose }: InspectorProps) {
  const [detail, setDetail] = useState<BlockDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<FidelityMode>("full");
  const [previews, setPreviews] = useState<
    Partial<Record<FidelityMode, BlockPreviewResponse | "loading" | { error: string }>>
  >({});
  const previewCache = useRef<Map<string, BlockPreviewResponse>>(new Map());

  // Reset and fetch detail when block changes.
  useEffect(() => {
    if (!blockId) return;
    setDetail(null);
    setDetailError(null);
    setPreviews({});
    setDetailLoading(true);
    const ctrl = new AbortController();
    fetchBlockDetail({ block_id: blockId, turn, overrides }, ctrl.signal)
      .then((r) => {
        setDetail(r.detail);
        // Default to recorded mode if available, else first available.
        const defaultTab =
          r.detail.available_modes.includes(r.detail.mode)
            ? r.detail.mode
            : r.detail.available_modes[0] ?? "full";
        setActiveTab(defaultTab);
        setDetailLoading(false);
      })
      .catch((err) => {
        if (err?.name === "AbortError") return;
        setDetailError(err instanceof Error ? err.message : String(err));
        setDetailLoading(false);
      });
    return () => ctrl.abort();
  }, [blockId, turn, overrides]);

  // Lazy-fetch the active fidelity preview.
  useEffect(() => {
    if (!detail || !blockId) return;
    if (!detail.available_modes.includes(activeTab)) return;
    const cacheKey = `${blockId}|${turn}|${activeTab}|${overrides.compaction_mode}|${overrides.ttl_compaction}|${overrides.dive_elision}`;
    const cached = previewCache.current.get(cacheKey);
    if (cached) {
      setPreviews((p) => ({ ...p, [activeTab]: cached }));
      return;
    }
    setPreviews((p) => ({ ...p, [activeTab]: "loading" }));
    const ctrl = new AbortController();
    fetchBlockPreview(
      { block_id: blockId, mode: activeTab, turn, overrides },
      ctrl.signal,
    )
      .then((r) => {
        previewCache.current.set(cacheKey, r);
        setPreviews((p) => ({ ...p, [activeTab]: r }));
      })
      .catch((err) => {
        if (err?.name === "AbortError") return;
        setPreviews((p) => ({
          ...p,
          [activeTab]: { error: err instanceof Error ? err.message : String(err) },
        }));
      });
    return () => ctrl.abort();
  }, [blockId, turn, overrides, activeTab, detail]);

  const tabs = useMemo(
    () => FIDELITY_ORDER.filter((m) => detail?.available_modes.includes(m)),
    [detail],
  );

  if (!blockId) return null;

  return (
    <div className="flex flex-col h-full min-h-0 bg-surface">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-edge flex items-center gap-3">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className="text-[10px] uppercase tracking-[0.18em] text-ch-hint font-mono">
            inspector
          </span>
          {detail && (
            <>
              <span className="text-ch-hint/40">·</span>
              <span className="text-xs text-ch-text truncate font-medium">
                {detail.title}
              </span>
              <span
                className={cn(
                  "px-1.5 py-0.5 rounded text-[9px] uppercase tracking-wider font-mono border",
                  FIDELITY_BG[detail.mode],
                )}
              >
                {detail.mode}
              </span>
              {detail.pinned && <span title="pinned">📌</span>}
            </>
          )}
          {detailLoading && !detail && (
            <span className="text-[10px] font-mono text-ch-hint">loading…</span>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-[11px] font-mono text-ch-hint hover:text-ch-text px-2 py-0.5 rounded hover:bg-elevated transition-colors"
          aria-label="close inspector"
        >
          ✕
        </button>
      </div>

      {detailError && (
        <div className="px-4 py-3 text-xs font-mono text-attention">
          {detailError}
        </div>
      )}

      {detail && (
        <div className="flex-1 min-h-0 grid grid-cols-[260px_1fr] divide-x divide-edge">
          {/* Metadata pane */}
          <div className="overflow-y-auto p-4 space-y-4">
            <MetaRow
              label="Block ID"
              value={
                <span className="font-mono text-[10px] text-ch-subtext break-all">
                  {detail.block_id}
                </span>
              }
            />
            <MetaRow
              label="Turn range"
              value={
                <span className="font-mono text-xs text-ch-text">
                  {detail.turn_range[0]} – {detail.turn_range[1]}
                </span>
              }
            />
            <MetaRow
              label="Available modes"
              value={
                <div className="flex flex-wrap gap-1">
                  {FIDELITY_ORDER.filter((m) => detail.available_modes.includes(m)).map(
                    (m) => (
                      <span
                        key={m}
                        className={cn(
                          "px-1.5 py-0.5 rounded text-[9px] uppercase tracking-wider font-mono border",
                          FIDELITY_BG[m],
                        )}
                      >
                        {m}
                      </span>
                    ),
                  )}
                </div>
              }
            />
            <MetaRow
              label="Token cost"
              value={
                <div className="space-y-0.5">
                  {Object.entries(detail.estimated_tokens_by_mode).map(([m, t]) => (
                    <div
                      key={m}
                      className="flex items-center justify-between text-[11px] font-mono"
                    >
                      <span className="text-ch-subtext">{m}</span>
                      <span className="text-ch-text tabular-nums">
                        {formatTokensLong(Number(t))}
                      </span>
                    </div>
                  ))}
                </div>
              }
            />

            {detail.headline && (
              <MetaRow
                label="Headline"
                value={
                  <p className="text-xs text-ch-text leading-relaxed italic">
                    {detail.headline}
                  </p>
                }
              />
            )}

            {detail.open_thread && (
              <MetaRow
                label="Open thread"
                value={
                  <div className="flex gap-2">
                    <span className="text-attention">●</span>
                    <p className="text-xs text-ch-text leading-relaxed">
                      {detail.open_thread}
                    </p>
                  </div>
                }
              />
            )}

            {detail.facets.length > 0 && (
              <MetaRow
                label={`Facets (${detail.facets.length})`}
                value={
                  <div className="space-y-2 mt-1">
                    {detail.facets.map((f) => (
                      <div
                        key={f.id}
                        className={cn(
                          "p-2 rounded border text-[11px]",
                          f.pinned
                            ? "bg-accent/5 border-accent/30"
                            : "bg-elevated/40 border-edge",
                        )}
                      >
                        <div className="flex items-start gap-1.5">
                          {f.pinned && <span className="flex-shrink-0">📌</span>}
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-ch-text">{f.headline}</div>
                            <div className="text-ch-hint mt-0.5 text-[10px] font-mono">
                              t{f.turn_range[0]}–{f.turn_range[1]}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                }
              />
            )}
          </div>

          {/* Fidelity tabs + preview */}
          <div className="flex flex-col min-h-0">
            <div className="flex items-center border-b border-edge bg-elevated/20 px-2 gap-1 overflow-x-auto">
              {tabs.map((m) => {
                const tokens = detail.estimated_tokens_by_mode[m];
                return (
                  <button
                    key={m}
                    onClick={() => setActiveTab(m)}
                    className={cn(
                      "flex items-center gap-2 px-3 py-2 text-xs font-medium border-b-2 transition-all whitespace-nowrap",
                      activeTab === m
                        ? "border-accent text-ch-text"
                        : "border-transparent text-ch-subtext hover:text-ch-text",
                    )}
                  >
                    <span className={cn("w-1.5 h-1.5 rounded-sm", FIDELITY_DOT[m])} />
                    {FIDELITY_LABEL[m]}
                    {tokens != null && (
                      <span className="text-[10px] font-mono text-ch-hint tabular-nums">
                        {formatTokens(Number(tokens))}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            <PreviewPane
              state={previews[activeTab]}
              fallbackSummary={activeTab === "summary" ? detail.summary : null}
              fallbackHeadline={activeTab === "headline" ? detail.headline : null}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.18em] text-ch-hint font-mono mb-1">
        {label}
      </div>
      <div>{value}</div>
    </div>
  );
}

function PreviewPane({
  state,
  fallbackSummary,
  fallbackHeadline,
}: {
  state: BlockPreviewResponse | "loading" | { error: string } | undefined;
  fallbackSummary: string | null;
  fallbackHeadline: string | null;
}) {
  if (!state) {
    const fallback = fallbackHeadline ?? fallbackSummary;
    if (fallback) {
      return (
        <div className="flex-1 min-h-0 overflow-y-auto p-4 text-xs text-ch-subtext italic">
          {fallback}
        </div>
      );
    }
    return (
      <div className="flex-1 min-h-0 flex items-center justify-center text-xs text-ch-hint font-mono">
        select a fidelity
      </div>
    );
  }
  if (state === "loading") {
    return (
      <div className="flex-1 min-h-0 flex items-center justify-center text-xs text-ch-hint font-mono">
        <span className="w-1 h-1 rounded-full bg-accent animate-pulse-status mr-2" />
        rendering preview…
      </div>
    );
  }
  if ("error" in state) {
    return (
      <div className="flex-1 min-h-0 p-4 text-xs font-mono text-attention">
        {state.error}
      </div>
    );
  }
  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="px-4 py-1.5 text-[10px] font-mono text-ch-hint border-b border-edge bg-elevated/10 flex justify-between">
        <span>{state.formatted_messages.split("\n").length} lines · {state.formatted_messages.length} chars</span>
        <span className="text-ch-subtext">≈ {formatTokensLong(state.estimated_tokens)} tokens</span>
      </div>
      <pre className="flex-1 min-h-0 overflow-auto p-4 text-[11px] leading-relaxed font-mono text-ch-text whitespace-pre-wrap break-words">
        {state.formatted_messages}
      </pre>
    </div>
  );
}
