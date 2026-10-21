// Schedule of the map's entrance (design: WC-exp/design, "画像入场 A", with the sweep of
// "另一种展开动画"). The root appears and the dimensions spring out of it. Then the dimensions'
// stars come in one dimension after another, in the order the dimensions sprang out, so only one
// region moves at a time. Within a dimension the brightest stars show first, the best supported
// quickly and vague inferences slowly. A star glides in over the last stretch of the way from its
// parent, which shows where it belongs while the layout stays nearly still.
// Only type imports are allowed here so the tests can load this file directly.
import type { NodeKind } from '../data';
import type { NodeLook } from './nodeLook';

/**
 * Times in seconds. Phase 1 keeps the design's timing. Phase 2 is compressed so the entrance takes
 * about 2 s instead of the design's 5.5 s: stars start as the first dimensions land instead of
 * after all have settled (design 1.22), and with shorter durations (design 0.3–0.95, lines
 * 0.14 + 0.34, names 0.45).
 */
export const TIMING = {
  rootAt: 0.06,
  rootDur: 0.62,
  rootLabelAt: 0.34,
  dimAt: 0.3,
  dimStagger: 0.065,
  dimDur: 0.8,
  labelFade: 0.34,
  // A dimension reaches its place after this share of its spring; the rest settles the overshoot.
  landed: 0.35,
  // Second phase, starting as the first dimension lands. Each dimension's stars start once it has
  // landed and at least `sweep` after the previous dimension's, and come in over `clusterWindow`.
  starsAt: 0.62,
  sweep: 0.14,
  clusterWindow: 0.32,
  // appear(u) = clusterWindow · ((u + c)^p − c^p) / ((1 + c)^p − c^p), u = salience rank 0…1 within
  // the dimension. Convex: many stars early and a slow trickle at the end; the shift c keeps the
  // first ones distinct.
  curve: 1.8,
  curveShift: 0.25,
  // Nodes of equal salience tier appear in random order inside their time slot.
  tiers: 9,
  // ± share of the gap to the neighbouring slots, so a tier never queues up.
  jitter: 0.9,
  // The brightest snap in, the faintest surface slowly.
  durFirst: 0.18,
  durLast: 0.34,
  // Low-confidence stars take up to 30% longer to resolve.
  blurSlow: 0.3,
  // Share of the way from its parent that a star travels while it appears.
  approach: 0.35,
  // The line to the parent follows the star.
  edgeDelay: 0.08,
  edgeDur: 0.22,
  // A star's name fades in as the star comes to rest, at this share of its duration.
  starLabelAt: 0.6,
  starLabelFade: 0.22,
};

export type EntranceInput = {
  id: string;
  kind: NodeKind;
  parentId: string | null;
  x: number;
  y: number;
  look: NodeLook;
};

export type EntranceNode<T extends EntranceInput = EntranceInput> = T & {
  parent: EntranceNode<T> | null;
  r: number;
  appearAt: number;
  dur: number;
  end: number;
  /** When the line from the parent starts and finishes; dimensions draw theirs while springing out. */
  edgeAt: number;
  edgeEnd: number;
  /** The label fades in from `labelAt` to `labelEnd`. */
  labelAt: number;
  labelEnd: number;
};

export type EntrancePlan<T extends EntranceInput = EntranceInput> = {
  root: EntranceNode<T>;
  dimensions: EntranceNode<T>[];
  stars: EntranceNode<T>[];
  nodes: EntranceNode<T>[];
  /** Nodes by appearance time, and nodes with a parent line by line start time. */
  nodeQueue: EntranceNode<T>[];
  edgeQueue: EntranceNode<T>[];
  end: number;
};

/** State of a node at time t: position, alpha, scale and glow factor (1 = settled). */
export type NodeFrame = { x: number; y: number; alpha: number; scale: number; glow: number };

export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
export const smooth = (edge0: number, edge1: number, x: number) => {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
};
export const easeOutCubic = (t: number) => 1 - (1 - t) ** 3;
export const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const easeOutBack = (t: number, s: number) => 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2;
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Damped spring from 0 to 1 with one overshoot of about 7%, settled by t = 1. */
function spring(t: number): number {
  if (t >= 1) return 1;
  if (t <= 0) return 0;
  const zeta = 0.65;
  const omega = 8.5;
  const damped = omega * Math.sqrt(1 - zeta * zeta);
  return 1 - Math.exp(-zeta * omega * t) * (Math.cos(damped * t) + ((zeta * omega) / damped) * Math.sin(damped * t));
}

export function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Needs exactly one root. Dimensions keep their input order; every other node becomes a star. */
export function buildEntrancePlan<I extends EntranceInput>(inputs: I[]): EntrancePlan<I> {
  const T = TIMING;
  const rand = mulberry32(20261018);
  const nodes = inputs.map((input): EntranceNode<I> => ({
    ...input,
    parent: null,
    r: input.look.size / 2,
    appearAt: 0,
    dur: 1,
    end: 1,
    edgeAt: 0,
    edgeEnd: 0,
    labelAt: 0,
    labelEnd: 0,
  }));
  const byId = new Map(nodes.map((node) => [node.id, node]));
  for (const node of nodes) node.parent = node.parentId ? (byId.get(node.parentId) ?? null) : null;
  const root = nodes.find((node) => node.kind === 'root')!;
  const dimensions = nodes.filter((node) => node.kind === 'dimension');
  const stars = nodes.filter((node) => node.kind !== 'root' && node.kind !== 'dimension');

  // Phase 1: the root, then the dimensions spring out of it drawing their lines.
  root.appearAt = T.rootAt;
  root.dur = T.rootDur;
  root.end = root.appearAt + root.dur;
  root.labelAt = T.rootLabelAt;
  root.labelEnd = root.labelAt + T.labelFade;
  dimensions.forEach((node, index) => {
    node.appearAt = T.dimAt + index * T.dimStagger;
    node.dur = T.dimDur;
    node.end = node.appearAt + node.dur;
    node.edgeAt = node.appearAt;
    node.edgeEnd = node.end;
    node.labelAt = node.appearAt + node.dur * 0.6;
    node.labelEnd = node.labelAt + T.labelFade;
  });

  // Phase 2: the dimensions' stars in turn, the most salient first within each dimension.
  const clusters = new Map<EntranceNode<I> | null, EntranceNode<I>[]>(dimensions.map((dimension) => [dimension, []]));
  for (const node of stars) {
    const dimension = dimensionOf(node);
    if (!clusters.has(dimension)) clusters.set(dimension, []);
    clusters.get(dimension)!.push(node);
  }
  const shift = T.curveShift;
  const span = (1 + shift) ** T.curve - shift ** T.curve;
  let previous = T.starsAt - T.sweep;
  for (const [dimension, members] of clusters) {
    if (!members.length) continue;
    const start = Math.max(previous + T.sweep, dimension ? dimension.appearAt + dimension.dur * T.landed : 0);
    previous = start;
    const keyed = members.map((node) => ({
      node,
      tier: Math.min(T.tiers - 1, Math.floor((1 - node.look.sal01) * T.tiers)),
      key: rand(),
    }));
    keyed.sort((a, b) => a.tier - b.tier || a.key - b.key);
    keyed.forEach(({ node }, index) => {
      const u = (index + 0.5) / keyed.length;
      const base = (T.clusterWindow * ((u + shift) ** T.curve - shift ** T.curve)) / span;
      const gap = (T.clusterWindow * T.curve * (u + shift) ** (T.curve - 1)) / span / keyed.length;
      node.appearAt = Math.max(start - 0.02, start + base + (rand() - 0.5) * T.jitter * gap + (rand() - 0.5) * 0.02);
      node.dur = lerp(T.durFirst, T.durLast, u ** 0.8) * (1 + T.blurSlow * (1 - node.look.clarity));
    });
  }
  // A node below another star (only when deeper levels start expanded) never shows before its parent.
  for (const node of [...stars].sort((a, b) => depth(a) - depth(b))) {
    const parent = node.parent;
    if (parent && parent.kind !== 'root' && parent.kind !== 'dimension') {
      node.appearAt = Math.max(node.appearAt, parent.appearAt + T.edgeDelay);
    }
    node.end = node.appearAt + node.dur;
    node.edgeAt = node.appearAt + T.edgeDelay;
    node.edgeEnd = node.edgeAt + T.edgeDur;
    node.labelAt = node.appearAt + node.dur * T.starLabelAt;
    node.labelEnd = node.labelAt + T.starLabelFade;
  }

  let end = 0;
  for (const node of nodes) end = Math.max(end, node.end, node.labelEnd, node.parent ? node.edgeEnd : 0);
  return {
    root,
    dimensions,
    stars,
    nodes,
    nodeQueue: [...nodes].sort((a, b) => a.appearAt - b.appearAt),
    edgeQueue: nodes.filter((node) => node.parent).sort((a, b) => a.edgeAt - b.edgeAt),
    end: end + 0.05,
  };
}

function depth(node: EntranceNode): number {
  return node.parent ? depth(node.parent) + 1 : 0;
}

function dimensionOf<T extends EntranceInput>(node: EntranceNode<T>): EntranceNode<T> | null {
  let current = node.parent;
  while (current && current.kind !== 'dimension') current = current.parent;
  return current;
}

/** Writes the node's state at time t into `out`; false while it has not appeared yet. */
export function nodeFrame(node: EntranceNode, t: number, root: EntranceNode, out: NodeFrame): boolean {
  const p = (t - node.appearAt) / node.dur;
  if (p <= 0) return false;
  const q = Math.min(1, p);
  out.x = node.x;
  out.y = node.y;
  if (node.kind === 'root') {
    out.alpha = easeOutCubic(q);
    out.scale = 0.6 + 0.4 * easeOutBack(q, 1.4);
    out.glow = 1 + 0.6 * (1 - q) * (1 - q);
  } else if (node.kind === 'dimension') {
    const k = spring(q);
    out.x = root.x + (node.x - root.x) * k;
    out.y = root.y + (node.y - root.y) * k;
    out.alpha = smooth(0, 0.28, q);
    out.scale = 0.55 + 0.45 * easeOutCubic(Math.min(1, q * 1.35));
    out.glow = 1;
  } else {
    // Glides in from its parent's side, landing exactly where the layout put it.
    const parent = node.parent;
    if (parent && q < 1) {
      const k = 1 - TIMING.approach * (1 - easeOutCubic(q));
      out.x = parent.x + (node.x - parent.x) * k;
      out.y = parent.y + (node.y - parent.y) * k;
    }
    out.alpha = easeOutCubic(q);
    out.scale = node.look.sal01 > 0.6 ? 0.5 + 0.5 * easeOutBack(q, 1.5) : 0.62 + 0.38 * easeOutCubic(q);
    // Twinkle: a bright star flares as it appears, then settles at its own brightness.
    out.glow = 1 + 0.9 * node.look.sal01 * (1 - q) * (1 - q);
  }
  return true;
}
