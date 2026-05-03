import type {
  BlockDetailResponse,
  BlockPreviewResponse,
  BootstrapResponse,
  FidelityMode,
  PlaygroundOverrides,
  SnapshotResponse,
} from "./types";

const BASE = "/api/sb";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

export async function fetchBootstrap(signal?: AbortSignal): Promise<BootstrapResponse> {
  const res = await fetch(`${BASE}/playground/bootstrap`, { signal });
  return jsonOrThrow(res);
}

export interface SnapshotArgs {
  turn?: number;
  measure?: boolean;
  overrides?: PlaygroundOverrides;
  block_modes?: Record<string, FidelityMode>;
}

export async function fetchSnapshot(
  args: SnapshotArgs,
  signal?: AbortSignal,
): Promise<SnapshotResponse> {
  const res = await fetch(`${BASE}/playground/snapshot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
    signal,
  });
  return jsonOrThrow(res);
}

export async function fetchBlockDetail(
  args: { block_id: string; turn?: number; overrides?: PlaygroundOverrides },
  signal?: AbortSignal,
): Promise<BlockDetailResponse> {
  const res = await fetch(`${BASE}/playground/block-detail`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
    signal,
  });
  return jsonOrThrow(res);
}

export async function fetchBlockPreview(
  args: {
    block_id: string;
    mode: FidelityMode;
    turn?: number;
    overrides?: PlaygroundOverrides;
  },
  signal?: AbortSignal,
): Promise<BlockPreviewResponse> {
  const res = await fetch(`${BASE}/playground/block-preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
    signal,
  });
  return jsonOrThrow(res);
}
