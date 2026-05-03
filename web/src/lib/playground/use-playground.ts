"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchBootstrap,
  fetchSnapshot,
} from "./api";
import type {
  BootstrapResponse,
  PlaygroundOverrides,
  SnapshotResponse,
  TimelinePoint,
} from "./types";

// Aim for ~30 chart samples regardless of session length. With 2k+ turns and
// stride ~70 that's a serial pass of ~30 short snapshots; each one is cheap
// after the first compute, and the chart fills in progressively.
const TIMELINE_TARGET_POINTS = 30;

const DEFAULT_OVERRIDES: PlaygroundOverrides = {
  ttl_compaction: true,
  dive_elision: true,
  compaction_mode: "recorded",
};

interface PlaygroundState {
  bootstrap: BootstrapResponse | null;
  bootstrapError: string | null;
  snapshot: SnapshotResponse | null;
  snapshotLoading: boolean;
  snapshotError: string | null;
  timeline: TimelinePoint[];
  timelineLoading: boolean;
  selectedTurn: number | null;
  overrides: PlaygroundOverrides;
  overridesAreDefault: boolean;
  setSelectedTurn: (turn: number) => void;
  updateOverrides: (patch: Partial<PlaygroundOverrides>) => void;
  resetOverrides: () => void;
}

export function usePlayground(): PlaygroundState {
  const [bootstrap, setBootstrap] = useState<BootstrapResponse | null>(null);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<SnapshotResponse | null>(null);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);
  const [selectedTurn, setSelectedTurnState] = useState<number | null>(null);
  const [overrides, setOverrides] = useState<PlaygroundOverrides>(DEFAULT_OVERRIDES);
  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);
  const [timelineLoading, setTimelineLoading] = useState(false);

  // Bootstrap once on mount.
  useEffect(() => {
    let alive = true;
    fetchBootstrap()
      .then((data) => {
        if (!alive) return;
        setBootstrap(data);
        setSelectedTurnState(data.latest_completed_turn);
      })
      .catch((err) => {
        if (!alive) return;
        setBootstrapError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      alive = false;
    };
  }, []);

  // Snapshot fetch — debounced when scrubbing, immediate on overrides change.
  const snapshotReqId = useRef(0);
  useEffect(() => {
    if (selectedTurn == null) return;
    const myId = ++snapshotReqId.current;
    setSnapshotLoading(true);
    setSnapshotError(null);
    const controller = new AbortController();

    fetchSnapshot({ turn: selectedTurn, overrides }, controller.signal)
      .then((data) => {
        if (myId !== snapshotReqId.current) return;
        setSnapshot(data);
        setSnapshotLoading(false);
      })
      .catch((err) => {
        if (myId !== snapshotReqId.current) return;
        if (err?.name === "AbortError") return;
        setSnapshotError(err instanceof Error ? err.message : String(err));
        setSnapshotLoading(false);
      });

    return () => controller.abort();
  }, [selectedTurn, overrides]);

  // Build the timeline client-side by issuing strided POST /playground/snapshot
  // calls. We deliberately avoid the dedicated /playground/timeline endpoint
  // because it stalls the serial app server on large sessions; many short,
  // cacheable snapshots give better progressive rendering and don't block
  // other requests.
  //
  // We wait for the first (latest-turn) snapshot to land so the budget and
  // block panels render immediately, then march through the rest sequentially
  // — each one updates the chart as it arrives. The Spellbook server caches
  // snapshots per turn+overrides, so subsequent visits to the same turn (e.g.
  // when the scrubber lands) reuse the same bundle.
  //
  // Overrides are passed through so the timeline reflects the same hypothetical
  // world as everything else. Changing overrides invalidates the timeline and
  // rebuilds it.
  const haveSnapshot = snapshot != null;
  useEffect(() => {
    if (!bootstrap || !haveSnapshot) return;
    const latest = bootstrap.latest_completed_turn;
    if (latest <= 0) return;

    const stride = Math.max(1, Math.ceil(latest / TIMELINE_TARGET_POINTS));
    const turns: number[] = [];
    for (let t = 1; t <= latest; t += stride) turns.push(t);
    if (turns[turns.length - 1] !== latest) turns.push(latest);

    let alive = true;
    setTimeline([]);
    setTimelineLoading(true);

    (async () => {
      const points: TimelinePoint[] = [];
      for (const turn of turns) {
        if (!alive) return;
        try {
          const snap = await fetchSnapshot({ turn, overrides });
          if (!alive) return;
          const blocks = snap.block_list ?? [];
          points.push({
            turn,
            estimated_memory_tokens: snap.budget_breakdown.estimated_input_tokens,
            regime: snap.budget_breakdown.regime,
            block_count: blocks.length,
            full_count: blocks.filter((b) => b.mode === "full").length,
          });
          // Sort so progressive insertion paints a coherent line even if a
          // future request beats an earlier one (shouldn't happen with the
          // sequential loop, but cheap insurance).
          points.sort((a, b) => a.turn - b.turn);
          setTimeline([...points]);
        } catch {
          // Skip individual point failures — partial timelines are still useful.
        }
      }
      if (alive) setTimelineLoading(false);
    })();

    return () => {
      alive = false;
    };
  }, [bootstrap, haveSnapshot, overrides]);

  const setSelectedTurn = useCallback((turn: number) => {
    setSelectedTurnState(turn);
  }, []);

  const updateOverrides = useCallback((patch: Partial<PlaygroundOverrides>) => {
    setOverrides((prev) => ({ ...prev, ...patch }));
  }, []);

  const resetOverrides = useCallback(() => {
    setOverrides(DEFAULT_OVERRIDES);
  }, []);

  const overridesAreDefault = useMemo(
    () =>
      overrides.ttl_compaction === DEFAULT_OVERRIDES.ttl_compaction &&
      overrides.dive_elision === DEFAULT_OVERRIDES.dive_elision &&
      overrides.compaction_mode === DEFAULT_OVERRIDES.compaction_mode,
    [overrides],
  );

  return {
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
  };
}
