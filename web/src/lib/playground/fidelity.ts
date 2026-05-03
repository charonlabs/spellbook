import type { FidelityMode } from "./types";

export const FIDELITY_ORDER: FidelityMode[] = ["full", "summary", "index", "headline"];

export const FIDELITY_LABEL: Record<FidelityMode, string> = {
  full: "Full",
  summary: "Summary",
  index: "Index",
  headline: "Headline",
};

export const FIDELITY_HEX: Record<FidelityMode, string> = {
  full: "#5890e8",      // builder blue — most material
  summary: "#a78bfa",   // accent violet
  index: "#3dbfa0",     // reader teal
  headline: "#908e85",  // subtext grey — lightest
};

export const FIDELITY_BG: Record<FidelityMode, string> = {
  full: "bg-[#5890e8]/15 text-[#9ab8f3] border-[#5890e8]/30",
  summary: "bg-accent/15 text-accent-bright border-accent/30",
  index: "bg-[#3dbfa0]/15 text-[#7fdbc4] border-[#3dbfa0]/30",
  headline: "bg-ch-subtext/10 text-ch-subtext border-ch-subtext/20",
};

export const FIDELITY_DOT: Record<FidelityMode, string> = {
  full: "bg-[#5890e8]",
  summary: "bg-accent",
  index: "bg-[#3dbfa0]",
  headline: "bg-ch-subtext",
};

export function formatTokens(tokens: number): string {
  if (tokens === 0) return "0";
  if (tokens < 1000) return `${tokens}`;
  if (tokens < 10000) return `${(tokens / 1000).toFixed(2)}k`;
  if (tokens < 100000) return `${(tokens / 1000).toFixed(1)}k`;
  return `${Math.round(tokens / 1000)}k`;
}

export function formatTokensLong(tokens: number): string {
  return tokens.toLocaleString();
}
