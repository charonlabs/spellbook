export type FidelityMode = "full" | "summary" | "index" | "headline";

export type CompactionMode = "recorded" | "none";

export interface PlaygroundOverrides {
  ttl_compaction?: boolean;
  dive_elision?: boolean;
  compaction_mode?: CompactionMode;
}

export interface BootstrapResponse {
  session_dir: string;
  cwd: string;
  model: string;
  provider: string;
  latest_completed_turn: number;
  warning_message_tokens: number;
  forced_message_tokens: number;
  target_message_tokens: number;
  gauge: string;
}

export interface BudgetBreakdown {
  frame_tokens: number;
  compacted_block_tokens: number;
  full_block_tokens: number;
  tail_tokens: number;
  overlay_tokens: number;
  estimated_input_tokens: number;
  measured_message_tokens: number | null;
  measurement_state: string;
  regime: string;
}

export interface BlockListItem {
  ordinal: number;
  block_id: string;
  title: string;
  mode: FidelityMode;
  estimated_tokens: number;
  pinned: boolean;
  proposed: boolean;
  open_thread: boolean;
  available_modes: FidelityMode[];
  turn_range: [number, number];
}

export interface PressureInfo {
  regime: string;
  warning_threshold: number;
  forced_threshold: number;
  target_tokens: number;
  measured_message_tokens: number | null;
  projected_message_tokens: number | null;
  protected_segment_ids: string[];
  candidates: unknown[];
  proposed: unknown[];
}

export interface SnapshotResponse {
  turn: number;
  latest_completed_turn: number;
  overrides: PlaygroundOverrides;
  budget_breakdown: BudgetBreakdown;
  block_list: BlockListItem[];
  plan: {
    pressure: PressureInfo;
    diagnostics: string[];
    [k: string]: unknown;
  };
  formatted_plan: string;
}

export interface FacetAnchor {
  turn: number;
  seq: number;
  context: string;
}

export interface Facet {
  id: string;
  headline: string;
  summary: string;
  turn_range: [number, number];
  anchors: FacetAnchor[];
  resources: unknown[];
  pinned: boolean;
}

export interface BlockDetail {
  block_id: string;
  title: string;
  turn_range: [number, number];
  mode: FidelityMode;
  available_modes: FidelityMode[];
  pinned: boolean;
  headline: string | null;
  summary: string | null;
  open_thread: string | null;
  facets: Facet[];
  estimated_tokens_by_mode: Partial<Record<FidelityMode, number>>;
}

export interface BlockDetailResponse {
  turn: number;
  detail: BlockDetail;
}

export interface BlockPreviewResponse {
  turn: number;
  block_id: string;
  mode: FidelityMode;
  estimated_tokens: number;
  measured_tokens: number | null;
  formatted_messages: string;
}

// Built client-side from /playground/snapshot calls at strided turns. We
// don't use the dedicated /playground/timeline endpoint because it blocks
// the serial app server on large sessions; many short snapshot requests give
// us better caching, progressive rendering, and don't stall the server.
export interface TimelinePoint {
  turn: number;
  estimated_memory_tokens: number;
  regime: string;
  block_count: number;
  full_count: number;
}
