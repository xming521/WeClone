// Visual weight of every profile node (design: WC-exp/design, "画像入场 A"). Salience sets the
// size and glow, confidence the fill opacity and a dashed outline: well supported nodes read as
// bright solid stars, vague inferences as faint dashed ones. Only type imports are allowed here so
// the tests can load this file directly with node's type stripping.
import type { NodeKind, ProfileFact, ProfileModel } from '../data';

/**
 * Thresholds on the source confidence scale (2–4). At or above `solid` a node is a solid disc,
 * below `dash` it gets a dashed outline, and at `low` or below it is at its most translucent.
 */
export const CONFIDENCE = { low: 2.6, dash: 2.95, solid: 3.25 };

const BASE_SIZE: Record<NodeKind, number> = { root: 76, dimension: 54, topic: 31, subtopic: 24, attribute: 12 };

export type NodeLook = {
  size: number;
  /** Log-salience normalised within the node's level: 0 = least, 1 = most salient. */
  sal01: number;
  /** 0 = vague inference, 1 = clearly supported. */
  clarity: number;
  fillOpacity: number;
  strokeOpacity: number;
  lineWidth: number;
  lineDash: number[] | null;
  glowBlur: number;
  /** Alpha of the glow colour. The canvas multiplies it by the fill opacity. */
  glowAlpha: number;
};

type Evidence = { salience: number; confidence: number };

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const smooth = (edge0: number, edge1: number, x: number) => {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
};

/**
 * Salience = importance × ln(1 + support). Importance and confidence are means over the facts
 * weighted by their support. Manual facts have no source scores: they count as support only, and
 * an attribute made of manual facts alone gets the lowest importance and full confidence.
 */
function attributeEvidence(facts: ProfileFact[]): Evidence {
  let support = 0;
  let importance = 0;
  let importanceWeight = 0;
  let confidence = 0;
  let confidenceWeight = 0;
  for (const fact of facts) {
    const count = Math.max(1, fact.support_count ?? fact.source_ids.length);
    support += count;
    if (fact.origin === 'manual') continue;
    if (Number.isFinite(fact.source_importance_mean)) {
      importance += fact.source_importance_mean! * count;
      importanceWeight += count;
    }
    if (Number.isFinite(fact.source_confidence_mean)) {
      confidence += fact.source_confidence_mean! * count;
      confidenceWeight += count;
    }
  }
  return {
    salience: (importanceWeight ? importance / importanceWeight : 1) * Math.log(1 + Math.max(1, support)),
    confidence: confidenceWeight ? confidence / confidenceWeight : 4,
  };
}

function encode(kind: NodeKind, sal01: number, confidence: number): NodeLook {
  if (kind === 'root' || kind === 'dimension') {
    const root = kind === 'root';
    return {
      size: BASE_SIZE[kind],
      sal01: 1,
      clarity: 1,
      fillOpacity: 1,
      strokeOpacity: 1,
      lineWidth: 1.5,
      lineDash: null,
      glowBlur: root ? 16 : 24,
      glowAlpha: 1,
    };
  }
  const clarity = smooth(CONFIDENCE.low, CONFIDENCE.solid, confidence);
  const dashed = confidence < CONFIDENCE.dash;
  const size = BASE_SIZE[kind] * (0.7 + 0.6 * sal01);
  if (kind === 'attribute') {
    // A white body; only the more salient attributes glow at all.
    const fillOpacity = 0.32 + 0.68 * clarity;
    const bright = clamp((sal01 - 0.55) / 0.45, 0, 1);
    return {
      size,
      sal01,
      clarity,
      fillOpacity,
      strokeOpacity: (0.5 + 0.5 * sal01) * (0.62 + 0.38 * clarity),
      lineWidth: 2.5,
      lineDash: dashed ? [2.3, 1.9] : null,
      glowBlur: bright > 0 ? 3 + 9 * bright : 0,
      glowAlpha: bright > 0 ? Math.min(1, (0.22 + 0.3 * bright) / fillOpacity) : 0,
    };
  }
  // A translucent star also glows less, but not in full proportion to its fill.
  const fillOpacity = 0.2 + 0.8 * clarity;
  const glow = (0.42 + 0.58 * sal01) * (0.3 + 0.7 * fillOpacity);
  return {
    size,
    sal01,
    clarity,
    fillOpacity,
    strokeOpacity: 1,
    lineWidth: 1.5,
    lineDash: dashed ? [3.4, 2.6] : null,
    glowBlur: 6 + 16 * sal01,
    glowAlpha: Math.min(1, glow / fillOpacity),
  };
}

/**
 * Look of every node in the model. A topic or subtopic sums the salience of its attributes and
 * averages their confidence. Salience is normalised on a log scale within each level, so sizes
 * stay the same whichever nodes are expanded.
 */
export function nodeLooks(model: ProfileModel): Map<string, NodeLook> {
  const evidence = new Map<string, Evidence>();
  // Returns the summed salience and confidence of the attributes below `id`, and their count.
  function collect(id: string): Evidence & { attributes: number } {
    const node = model.nodes.get(id)!;
    if (node.kind === 'attribute') {
      const result = attributeEvidence(node.factIndices.flatMap((index) => model.facts[index] ?? []));
      evidence.set(id, result);
      return { ...result, attributes: 1 };
    }
    let salience = 0;
    let confidence = 0;
    let attributes = 0;
    for (const childId of node.childIds) {
      const child = collect(childId);
      salience += child.salience;
      confidence += child.confidence;
      attributes += child.attributes;
    }
    evidence.set(id, { salience, confidence: attributes ? confidence / attributes : 4 });
    return { salience, confidence, attributes };
  }
  collect('root');

  const range = new Map<NodeKind, [number, number]>();
  for (const [id, { salience }] of evidence) {
    const kind = model.nodes.get(id)!.kind;
    const log = Math.log(Math.max(salience, 1e-6));
    const [low, high] = range.get(kind) ?? [Infinity, -Infinity];
    range.set(kind, [Math.min(low, log), Math.max(high, log)]);
  }
  const looks = new Map<string, NodeLook>();
  for (const [id, { salience, confidence }] of evidence) {
    const kind = model.nodes.get(id)!.kind;
    const [low, high] = range.get(kind)!;
    const sal01 = high > low ? (Math.log(Math.max(salience, 1e-6)) - low) / (high - low) : 0.5;
    looks.set(id, encode(kind, sal01, confidence));
  }
  return looks;
}
