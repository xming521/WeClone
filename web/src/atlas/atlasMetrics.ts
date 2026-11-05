// Evidence behind every node of the map (design: WC-exp/design, "A 纸本 · 浅色克制"): how
// important and how certain its facts are, and how many chats support them. Importance sets how
// far a node's colour spreads on the paper, support the width of the line into it and the strength
// of the ink that flows along that line. Only type imports are allowed here so the tests can load
// this file directly with node's type stripping.
import type { ProfileModel } from '../data';

export type NodeMetrics = {
  /** Mean source importance (1–4) of the scored facts below the node, null when none is scored. */
  importance: number | null;
  /** Mean source confidence (2–4) of those facts, null when none is scored. */
  confidence: number | null;
  /** Chats supporting the facts below the node, each counted once. */
  support: number;
  facts: number;
  /** 0 for the centre, 1 for dimensions, and one more for every level below. */
  depth: number;
  /** Importance on 0…1, 0.25 when unscored. */
  weight: number;
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

/** Width of the line into a node: grows with the log of its chat support. */
export const supportWidth = (support: number) => 1 + 0.42 * Math.log2(1 + Math.max(0, support));

/** Strength 0…1 of the ink along a line of width `width`. */
export const inkStrength = (width: number) => clamp01((width - 1.4) / 3.4);

type Totals = {
  importance: number;
  importanceCount: number;
  confidence: number;
  confidenceCount: number;
  sources: Set<string>;
  /** Support of facts without source ids, which cannot be deduplicated. */
  unlisted: number;
  facts: number;
};

/**
 * Metrics of every node in the model. Manual facts have no source scores, so they add support
 * only; their sources still count.
 */
export function nodeMetrics(model: ProfileModel): Map<string, NodeMetrics> {
  const metrics = new Map<string, NodeMetrics>();
  function visit(id: string, depth: number): Totals {
    const node = model.nodes.get(id)!;
    const totals: Totals = {
      importance: 0,
      importanceCount: 0,
      confidence: 0,
      confidenceCount: 0,
      sources: new Set(),
      unlisted: 0,
      facts: 0,
    };
    for (const index of node.factIndices) {
      const fact = model.facts[index];
      if (!fact) continue;
      totals.facts += 1;
      for (const sourceId of fact.source_ids) totals.sources.add(sourceId);
      if (!fact.source_ids.length) totals.unlisted += fact.support_count ?? 0;
      if (fact.origin === 'manual') continue;
      if (Number.isFinite(fact.source_importance_mean)) {
        totals.importance += fact.source_importance_mean!;
        totals.importanceCount += 1;
      }
      if (Number.isFinite(fact.source_confidence_mean)) {
        totals.confidence += fact.source_confidence_mean!;
        totals.confidenceCount += 1;
      }
    }
    for (const childId of node.childIds) {
      const child = visit(childId, depth + 1);
      totals.importance += child.importance;
      totals.importanceCount += child.importanceCount;
      totals.confidence += child.confidence;
      totals.confidenceCount += child.confidenceCount;
      for (const sourceId of child.sources) totals.sources.add(sourceId);
      totals.unlisted += child.unlisted;
      totals.facts += child.facts;
    }
    const importance = totals.importanceCount ? totals.importance / totals.importanceCount : null;
    metrics.set(id, {
      importance,
      confidence: totals.confidenceCount ? totals.confidence / totals.confidenceCount : null,
      support: totals.sources.size + totals.unlisted,
      facts: totals.facts,
      depth,
      weight: importance === null ? 0.25 : clamp01((importance - 1) / 3),
    });
    return totals;
  }
  visit('root', 0);
  return metrics;
}
