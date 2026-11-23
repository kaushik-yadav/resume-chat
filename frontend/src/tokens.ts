import { useEffect, useRef, useState } from "react";
import type { Flow, FlowNode } from "./types";

// Rough live-estimate parameters (until the provider reports real usage at the end of a call).
const PROMPT_ESTIMATE = 700; // typical prompt size of one step, in tokens
const OUTPUT_RATE = 55; // tokens per second while a non-streaming step runs
const CHARS_PER_TOKEN = 4;

/** Real tokens of completed calls (input + output). */
export function nodeTokens(node: FlowNode): number {
  return node.llm
    .filter((c) => (c.status ?? "ok") === "ok")
    .reduce((sum, c) => sum + c.input_tokens + c.output_tokens, 0);
}

export function flowTokens(flow: Flow | null | undefined): number {
  return (flow?.nodes ?? []).reduce((sum, n) => sum + nodeTokens(n), 0);
}

/** Estimate for a step that is still running (shown with "≈"). */
export function runningEstimate(node: FlowNode, elapsedMs: number, replyChars: number): number {
  if (node.kind === "respond" && replyChars > 0) {
    return nodeTokens(node) + PROMPT_ESTIMATE + Math.round(replyChars / CHARS_PER_TOKEN);
  }
  return nodeTokens(node) + PROMPT_ESTIMATE + Math.round((elapsedMs / 1000) * OUTPUT_RATE);
}

/**
 * Server time of the turn, ticking locally between flow snapshots, so running steps can show
 * a moving elapsed time and token estimate.
 */
export function useServerClock(flow: Flow | null, live: boolean): () => number {
  const anchor = useRef<{ client: number; server: number } | null>(null);
  const [, setTick] = useState(0);
  useEffect(() => {
    if (flow) anchor.current = { client: performance.now(), server: flow.total_ms };
  }, [flow]);
  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => setTick((t) => t + 1), 200);
    return () => window.clearInterval(id);
  }, [live]);
  return () => (anchor.current ? anchor.current.server + (performance.now() - anchor.current.client) : 0);
}

/** Total tokens for a turn: exact for finished steps, estimated for running ones. */
export function useLiveTokens(flow: Flow | null, replyChars: number, live: boolean): number {
  const now = useServerClock(flow, live);
  if (!flow) return Math.round(replyChars / CHARS_PER_TOKEN);
  const serverNow = now();
  return flow.nodes.reduce((sum, node) => {
    if (node.status === "running" && node.start_ms != null) {
      return sum + runningEstimate(node, Math.max(0, serverNow - node.start_ms), replyChars);
    }
    return sum + nodeTokens(node);
  }, 0);
}
