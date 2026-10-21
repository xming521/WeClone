// Schedule and particle plan of the first-visit intro (design: WC-exp/design, "从夜到昼"). At night
// a few of the owner's own lines appear and dissolve into stars. The title counts what they said and
// what was kept. The stars then stream into clusters around the map's nodes while dawn spreads out
// from 「我」, and every cluster collapses into its node as the map fades in.
// Only type imports are allowed here so the tests can load this file directly.
import type { NodeKind, ProfileModel } from '../data';

const TAU = Math.PI * 2;

// ───────────────────────── timeline ─────────────────────────

/** Times in seconds. With opening lines everything but the lines themselves starts WORDS.shift later. */
const BASE = {
  total: 6,
  textIn: [0.12, 0.5, 0.88],
  textOut: [2, 2.06, 2.12],
  flowStart: 1.5,
  flowEnd: 4.5,
  settleStart: 4.45,
  edgesStart: 5,
  particlesFadeStart: 5.3,
  particlesFadeEnd: 5.95,
  dayStart: 2.2,
  dayEnd: 4.6,
  skipFadeStart: 5.55,
};

/** The opening lines. Their times are not shifted. */
export const WORDS = {
  shift: 1.2,
  lineIn: 0.1,
  lineStagger: 0.13,
  lineFade: 0.42,
  minRead: 0.42,
  faintIn: 0,
  faintSpread: 0.3,
  faintFade: 0.5,
  dissolveStart: 0.9,
  dissolveSpread: 0.38,
  sweepMax: 0.36,
  dotTravel: 0.62,
  dotTravelJitter: 0.16,
  // Stars that are not dots of a line light up from left to right while the lines dissolve,
  // except this share, which shines from the start.
  igniteStart: 0.85,
  igniteSpread: 0.9,
  litAtStart: 0.16,
};

export type Timeline = {
  total: number;
  /** The three title lines fade in and out at these times. */
  textIn: [number, number, number];
  textOut: [number, number, number];
  textInDuration: number;
  textOutDuration: number;
  flowStart: number;
  flowEnd: number;
  settleStart: number;
  edgesStart: number;
  particlesFadeStart: number;
  particlesFadeEnd: number;
  dayStart: number;
  dayEnd: number;
  skipFadeStart: number;
  words: boolean;
  /** When the last line has dissolved; 0 without lines. */
  wordsEnd: number;
};

export function timeline(words: boolean): Timeline {
  const s = words ? WORDS.shift : 0;
  const shift3 = (values: number[]) => values.map((value) => value + s) as [number, number, number];
  const read = WORDS.lineIn + 3 * WORDS.lineStagger + WORDS.lineFade + WORDS.minRead;
  return {
    total: BASE.total + s,
    textIn: shift3(BASE.textIn),
    textOut: shift3(BASE.textOut),
    textInDuration: 0.55,
    textOutDuration: 0.45,
    flowStart: BASE.flowStart + s,
    flowEnd: BASE.flowEnd + s,
    settleStart: BASE.settleStart + s,
    edgesStart: BASE.edgesStart + s,
    particlesFadeStart: BASE.particlesFadeStart + s,
    particlesFadeEnd: BASE.particlesFadeEnd + s,
    dayStart: BASE.dayStart + s,
    dayEnd: BASE.dayEnd + s,
    skipFadeStart: BASE.skipFadeStart + s,
    words,
    wordsEnd: words ? Math.max(WORDS.dissolveStart + WORDS.dissolveSpread + 0.06, read) + WORDS.sweepMax + 0.05 : 0,
  };
}

export const clamp01 = (value: number) => (value < 0 ? 0 : value > 1 ? 1 : value);
/** Progress 0…1 of a stretch that starts at `start` and lasts `duration`. */
export const progress = (t: number, start: number, duration: number) => clamp01((t - start) / duration);
export const easeOutCubic = (x: number) => 1 - (1 - x) ** 3;
export const easeInOutCubic = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
const easeInOutSine = (x: number) => -(Math.cos(Math.PI * x) - 1) / 2;
export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/** Opacity and vertical offset in px of title line `index` (0–2) at time t. */
export function titleLine(T: Timeline, index: number, t: number): { opacity: number; y: number } {
  const shown = easeOutCubic(progress(t, T.textIn[index], T.textInDuration));
  const gone = easeInOutSine(progress(t, T.textOut[index], T.textOutDuration));
  return { opacity: shown * (1 - gone), y: (1 - shown) * 10 - gone * 6 };
}

/** Dawn runs ahead near the root by this share of the way to the farthest corner. */
export const DAY_LEAD = 0.5;

export const dayProgress = (T: Timeline, t: number) => easeInOutCubic(progress(t, T.dayStart, T.dayEnd - T.dayStart));

/** Daylight 0…1 at `distance` px from the root, `maxDistance` being the farthest corner. */
export const dayAt = (T: Timeline, t: number, distance: number, maxDistance: number) =>
  clamp01(dayProgress(T, t) * (1 + DAY_LEAD) - (DAY_LEAD * distance) / maxDistance);

/**
 * The sky turns around the root, faster near it. While the stars set off it speeds up by `boost`,
 * so the vortex reads before the stars leave their places.
 */
export const SPIN = { rate: 0.075, boost: 2.6, falloff: 420 };
export const spinWindow = (T: Timeline): [number, number] => [T.flowStart - 0.2, T.flowStart + 1.5];

/** Turned time: the angle of a star at radius r is its start angle + spinRate(r) · spinPhase(t). */
export function spinPhase(T: Timeline, t: number): number {
  const [from, to] = spinWindow(T);
  const length = to - from;
  let extra = 0;
  if (t > from) {
    const x = Math.min(1, (t - from) / length);
    extra = length * (x * x * x - (x * x * x * x) / 2) + Math.max(0, t - to);
  }
  return t + SPIN.boost * extra;
}

export const spinRate = (r: number) => SPIN.rate * (0.3 + 0.7 * Math.exp(-r / SPIN.falloff));

// ───────────────────────── randomness ─────────────────────────

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let n = state;
    n = Math.imul(n ^ (n >>> 15), n | 1);
    n ^= n + Math.imul(n ^ (n >>> 7), n | 61);
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
}

function hash(x: number, y: number): number {
  const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return n - Math.floor(n);
}

/** Smooth value noise in 0…1 with features about one unit apart. */
export function valueNoise(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const a = hash(ix, iy);
  const b = hash(ix + 1, iy);
  const c = hash(ix, iy + 1);
  const d = hash(ix + 1, iy + 1);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

// ───────────────────────── opening lines ─────────────────────────

export type IntroSentence = { text: string; time: string; dim: number | null };

/** `count` items evenly spaced through `items`. */
export function spreadEvenly<T>(items: T[], count: number): T[] {
  if (items.length <= count) return items.slice();
  return Array.from({ length: count }, (_, i) => items[Math.floor(((i + 0.5) * items.length) / count)]);
}

/** `count` lines closest to evenly spaced moments between the first and the last, oldest first. */
export function spreadInTime(lines: IntroSentence[], count: number): IntroSentence[] {
  const dated = lines.map((line) => ({ line, at: Date.parse(line.time) })).filter((entry) => Number.isFinite(entry.at));
  if (dated.length <= count) return spreadEvenly(lines, count);
  let first = Infinity;
  let last = -Infinity;
  for (const entry of dated) {
    first = Math.min(first, entry.at);
    last = Math.max(last, entry.at);
  }
  const taken = new Set<number>();
  const picked: (typeof dated)[number][] = [];
  for (let i = 0; i < count; i++) {
    const target = first + ((last - first) * (i + 0.5)) / count;
    let best = -1;
    dated.forEach((entry, index) => {
      if (taken.has(index)) return;
      if (best < 0 || Math.abs(entry.at - target) < Math.abs(dated[best].at - target)) best = index;
    });
    taken.add(best);
    picked.push(dated[best]);
  }
  return picked.sort((a, b) => a.at - b.at).map((entry) => entry.line);
}

export type WordRole = 'line' | 'date' | 'faint';

/** A line, its date, or a faint background line, placed with its left edge at x and its middle at y. */
export type WordSpan = {
  text: string;
  role: WordRole;
  x: number;
  y: number;
  size: number;
  weight: number;
  /** Letter spacing in px. */
  tracking: number;
  alpha: number;
  width: number;
  /** Fades in from `appear` over `fade`; from `dissolve` it breaks into dots left to right over `sweep`. */
  appear: number;
  fade: number;
  dissolve: number;
  sweep: number;
  /** Diameter of its dots. */
  dot: number;
};

/** Width in px of `text` at the given size and weight, letter spacing included. */
export type Measure = (text: string, size: number, weight: number, tracking: number) => number;

type Box = { x0: number; y0: number; x1: number; y1: number };
const overlaps = (a: Box, b: Box) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
const yearMonth = (time: string) => (time.length >= 7 ? `${time.slice(0, 4)}.${time.slice(5, 7)}` : '');
const charCount = (text: string) => [...text].length;

/**
 * Up to four lines (two when narrow) spread over the years, each with its month, in the corners
 * around the title, and up to 18 faint lines (8) on a shuffled grid where they overlap nothing.
 * Returns null without sentences.
 */
export function layoutWords(
  sentences: IntroSentence[],
  width: number,
  height: number,
  measure: Measure,
  seed = 20261018,
): WordSpan[] | null {
  if (!sentences.length) return null;
  const rand = mulberry32((seed ^ 1540483477) >>> 0);
  const narrow = width < 760;
  const readable = sentences.filter((line) => charCount(line.text) >= 8 && charCount(line.text) <= 16);
  // Interests are the least personal; use them when there are enough.
  const interests = readable.filter((line) => line.dim === 8);
  const main = spreadInTime(interests.length >= 6 ? interests : readable, narrow ? 2 : 4);
  const used = new Set(main.map((line) => line.text));
  const faint = spreadEvenly(
    sentences.filter((line) => !used.has(line.text) && charCount(line.text) >= 6 && charCount(line.text) <= 18),
    narrow ? 8 : 18,
  );
  // The title block, then every placed line, is kept clear.
  const blocked: Box[] = [
    {
      x0: width / 2 - Math.min(410, width * 0.36),
      x1: width / 2 + Math.min(410, width * 0.36),
      y0: height / 2 - Math.min(150, height * 0.2),
      y1: height / 2 + Math.min(150, height * 0.2),
    },
  ];
  const spans: WordSpan[] = [];
  const size = narrow ? 16 : width < 1200 ? 18 : 21;
  const regions = narrow
    ? [
        [0.08, 0.1, 0.84, 0.2],
        [0.08, 0.72, 0.84, 0.82],
      ]
    : [
        [0.06, 0.14, 0.3, 0.24],
        [0.57, 0.17, 0.8, 0.27],
        [0.09, 0.7, 0.33, 0.79],
        [0.55, 0.66, 0.78, 0.75],
      ];
  main.forEach((line, index) => {
    const lineWidth = measure(line.text, size, 400, 0.4);
    const [rx0, ry0, rx1, ry1] = regions[index % regions.length];
    const x = Math.min(width - 32 - lineWidth, width * (rx0 + (rx1 - rx0) * rand()));
    const y = height * (ry0 + (ry1 - ry0) * rand());
    const appear = WORDS.lineIn + WORDS.lineStagger * index;
    const breakup = WORDS.dissolveStart + WORDS.dissolveSpread * ((x + lineWidth / 2) / width) + rand() * 0.05;
    const dissolve = Math.max(breakup, appear + WORDS.lineFade + WORDS.minRead);
    spans.push({
      text: line.text,
      role: 'line',
      x,
      y,
      size,
      weight: 400,
      tracking: 0.4,
      alpha: 0.95,
      width: lineWidth,
      appear,
      fade: WORDS.lineFade,
      dissolve,
      sweep: Math.min(WORDS.sweepMax, 0.14 + lineWidth / 1400),
      dot: 1.6,
    });
    const month = yearMonth(line.time);
    if (month) {
      const monthWidth = measure(month, 11, 400, 1.6);
      spans.push({
        text: month,
        role: 'date',
        x,
        y: y - size * 0.5 - 13,
        size: 11,
        weight: 400,
        tracking: 1.6,
        alpha: 0.8,
        width: monthWidth,
        appear: appear - 0.04,
        fade: WORDS.lineFade,
        dissolve: dissolve - 0.02,
        sweep: Math.min(WORDS.sweepMax, 0.1 + monthWidth / 1400),
        dot: 1.2,
      });
    }
    blocked.push({ x0: x - 24, y0: y - size - 26, x1: x + lineWidth + 24, y1: y + size * 0.6 + 14 });
  });

  const margin = Math.min(56, width * 0.05);
  const top = 48;
  const bottom = 96;
  const cols = Math.max(1, Math.floor((width - 2 * margin) / 250));
  const rows = Math.max(2, Math.floor((height - top - bottom) / 64));
  const cellWidth = (width - 2 * margin) / cols;
  const cellHeight = (height - top - bottom) / rows;
  const cells = Array.from({ length: cols * rows }, (_, i) => i);
  for (let i = cells.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  let next = 0;
  for (const line of faint) {
    const faintSize = rand() < 0.5 ? 12 : 13;
    const lineWidth = measure(line.text, faintSize, 400, 0.2);
    let placed = false;
    while (next < cells.length && !placed) {
      const cell = cells[next++];
      const cx = margin + (cell % cols) * cellWidth;
      const cy = top + Math.floor(cell / cols) * cellHeight;
      const x = cx + rand() * Math.max(0, cellWidth - lineWidth);
      const y = cy + cellHeight * (0.3 + 0.4 * rand());
      const box = { x0: x - 10, y0: y - faintSize, x1: x + lineWidth + 10, y1: y + faintSize };
      if (x + lineWidth > width - margin || blocked.some((other) => overlaps(box, other))) continue;
      blocked.push(box);
      const appear = WORDS.faintIn + WORDS.faintSpread * rand();
      const breakup = WORDS.dissolveStart + WORDS.dissolveSpread * ((x + lineWidth / 2) / width) + rand() * 0.06;
      spans.push({
        text: line.text,
        role: 'faint',
        x,
        y,
        size: faintSize,
        weight: 400,
        tracking: 0.2,
        alpha: 0.42,
        width: lineWidth,
        appear,
        fade: WORDS.faintFade,
        dissolve: Math.max(breakup, appear + WORDS.faintFade + 0.15),
        sweep: Math.min(WORDS.sweepMax, 0.12 + lineWidth / 1400),
        dot: 1.15,
      });
      placed = true;
    }
  }
  return spans;
}

/** Pixels of one span rendered in white: RGBA rows of `width` starting at (x0, y0). */
export type GlyphRaster = { x0: number; y0: number; width: number; height: number; data: Uint8ClampedArray };

/** Dots on the glyphs of the opening lines, each with the time it breaks away and arrives in the sky. */
export type WordDots = {
  spans: WordSpan[];
  count: number;
  x: Float32Array;
  y: Float32Array;
  start: Float32Array;
  end: Float32Array;
  alpha: Float32Array;
  size: Float32Array;
  sentence: Uint16Array;
};

/**
 * A dot on every other pixel (checkerboard, 2 px apart) of each span's glyphs. A dot breaks away
 * when the dissolving sweep passes it and reaches the sky by `latest` at the latest.
 */
export function sampleGlyphs(
  spans: WordSpan[],
  raster: (span: WordSpan) => GlyphRaster,
  latest: number,
  seed = 20261018,
): WordDots {
  const rand = mulberry32((seed ^ 0x2545f491) >>> 0);
  const x: number[] = [];
  const y: number[] = [];
  const start: number[] = [];
  const end: number[] = [];
  const alpha: number[] = [];
  const size: number[] = [];
  const sentence: number[] = [];
  const step = 2;
  spans.forEach((span, index) => {
    const { x0, y0, width, height, data } = raster(span);
    for (let py = y0; py < y0 + height; py += step) {
      for (let px = x0 + (((py - y0) / step) % 2); px < x0 + width; px += step) {
        if (data[((py - y0) * width + (px - x0)) * 4 + 3] < 96) continue;
        const along = clamp01((px - span.x) / Math.max(1, span.width));
        const breaks = span.dissolve + along * span.sweep;
        end.push(Math.min(latest, breaks + WORDS.dotTravel + rand() * WORDS.dotTravelJitter));
        x.push(px + 0.5 + (rand() - 0.5) * 0.6);
        y.push(py + 0.5 + (rand() - 0.5) * 0.6);
        start.push(breaks);
        alpha.push(Math.min(1, span.alpha * (span.role === 'faint' ? 1.15 : 1)));
        size.push(span.dot);
        sentence.push(index);
      }
    }
  });
  return {
    spans,
    count: x.length,
    x: Float32Array.from(x),
    y: Float32Array.from(y),
    start: Float32Array.from(start),
    end: Float32Array.from(end),
    alpha: Float32Array.from(alpha),
    size: Float32Array.from(size),
    sentence: Uint16Array.from(sentence),
  };
}

// ───────────────────────── particles ─────────────────────────

/** A node on screen at the start of the intro: centre and radius in px. */
export type PlanNode = { id: string; kind: NodeKind; dim: number | null; x: number; y: number; r: number };

export type NodeTiming = {
  /** Its stars collapse into it from `collapse` over `collapseDuration`. */
  collapse: number;
  collapseDuration: number;
  /** The node fades in from `fadeIn` to `fadeOut`. */
  fadeIn: number;
  fadeOut: number;
  edgeStart: number;
  edgeEnd: number;
  labelStart: number;
  labelEnd: number;
};

/** Vertex attributes, four or two floats per particle; see the vertex shader in nightIntro.ts. */
export type ParticlePlan = {
  count: number;
  /** Particles that end up in a cluster; the rest drift out and fade as the clusters form. */
  allocated: number;
  center: { x: number; y: number };
  a: Float32Array;
  b: Float32Array;
  c: Float32Array;
  d: Float32Array;
  e: Float32Array;
  f: Float32Array;
  g: Float32Array;
  words: boolean;
  wordDots: number;
};

export type ParticleOptions = {
  timeline: Timeline;
  width: number;
  height: number;
  count: number;
  zoom: number;
  /** Dimensions in palette order; the root takes slot 5 and drifting stars slot 6. */
  dims: number[];
  /** Share of the particles that end up in a cluster. */
  allocation?: number;
  words?: WordDots | null;
  seed?: number;
};

/** Shares of the clustered particles kept by the root, the dimensions and the groups; facts share the rest. */
const SHARE = { root: 0.045, dimension: 0.115, group: 0.09 };

/** Integer shares of `total` proportional to `weights`, by largest remainder. */
export function apportion(weights: number[], total: number): number[] {
  const sum = weights.reduce((a, b) => a + b, 0) || 1;
  const exact = weights.map((weight) => (weight / sum) * total);
  const shares = exact.map(Math.floor);
  let left = total - shares.reduce((a, b) => a + b, 0);
  const order = exact.map((value, index) => [value - Math.floor(value), index]).sort((a, b) => b[0] - a[0]);
  for (let i = 0; left > 0 && i < order.length; i++, left--) shares[order[i][1]] += 1;
  return shares;
}

const wrap = (angle: number) => angle - TAU * Math.floor((angle + Math.PI) / TAU);

type Cluster = { x: number; y: number; count: number; radius: number; nodeId: string; slot: number; level: number };

/**
 * Where every particle starts in the sky, which cluster it joins and when, and when every node
 * appears. Each fact draws stars in proportion to its support × importance and gathers them around
 * its nearest shown node; the root, dimensions and groups keep a share of their own. Stars are
 * matched to cluster places by angle around the root, after the turn of the sky while they set off,
 * so they flow along the vortex instead of across it. Needs the root among `nodes` and at least one fact.
 */
export function planParticles(
  model: ProfileModel,
  nodes: PlanNode[],
  options: ParticleOptions,
): { particles: ParticlePlan; timings: Map<string, NodeTiming> } {
  const T = options.timeline;
  const rand = mulberry32(options.seed ?? 20261018);
  const { width, height, count, zoom } = options;
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const root = byId.get('root')!;
  const slotOf = (dim: number | null) => {
    const index = dim === null ? -1 : options.dims.indexOf(dim);
    return index < 0 ? 5 : index;
  };
  const sparse = count < 15000;
  const facts = model.facts;
  const weights = facts.map((fact) =>
    Math.max(0.25, (fact.support_count ?? fact.source_ids.length) * (fact.source_importance_mean ?? 2.5)),
  );
  const attributeOf: (string | undefined)[] = new Array(facts.length);
  for (const id of model.attributeIds) for (const index of model.nodes.get(id)!.factIndices) attributeOf[index] = id;
  const shownAncestor = (id: string | undefined) => {
    let node = id ? model.nodes.get(id) : undefined;
    while (node && !byId.has(node.id)) node = node.parentId ? model.nodes.get(node.parentId) : undefined;
    return node ?? model.nodes.get('root')!;
  };
  const below = new Map<string, number>();
  facts.forEach((_, index) => {
    let node = attributeOf[index] ? model.nodes.get(attributeOf[index]!) : undefined;
    while (node) {
      below.set(node.id, (below.get(node.id) ?? 0) + weights[index]);
      node = node.parentId ? model.nodes.get(node.parentId) : undefined;
    }
  });

  const clustered = Math.round(count * Math.min(0.95, Math.max(0.05, options.allocation ?? 0.6)));
  const groups = nodes.filter((node) => node.kind === 'topic' || node.kind === 'subtopic');
  const dimensions = nodes.filter((node) => node.kind === 'dimension');
  const rootCount = Math.round(clustered * SHARE.root);
  const dimensionCount = Math.round(clustered * SHARE.dimension);
  const groupCount = Math.round(clustered * SHARE.group);
  const factCount = clustered - rootCount - dimensionCount - groupCount;
  // Every fact keeps a few stars; the scale is refined so the floors still add up to the total.
  const floor = sparse ? 2 : 4;
  let scale = factCount / weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < 6; i++) scale *= factCount / weights.reduce((sum, w) => sum + Math.max(floor, w * scale), 0);
  const perFact = apportion(
    weights.map((w) => Math.max(floor, w * scale)),
    factCount,
  );
  const share = (list: PlanNode[], total: number) =>
    apportion(
      list.map((node) => below.get(node.id) ?? 1e-4),
      total,
    );
  const perDimension = share(dimensions, dimensionCount);
  const perGroup = share(groups, groupCount);
  const density = sparse ? 13 : 5.2;
  const clusterRadius = (n: number, min: number) => Math.max(min, Math.sqrt((n * density) / Math.PI));
  const clusters: Cluster[] = [
    {
      x: root.x,
      y: root.y,
      count: rootCount,
      radius: clusterRadius(rootCount, root.r),
      nodeId: 'root',
      slot: 5,
      level: 0,
    },
  ];
  dimensions.forEach((node, i) =>
    clusters.push({
      x: node.x,
      y: node.y,
      count: perDimension[i],
      radius: clusterRadius(perDimension[i], node.r),
      nodeId: node.id,
      slot: slotOf(node.dim),
      level: 1,
    }),
  );
  groups.forEach((node, i) =>
    clusters.push({
      x: node.x,
      y: node.y,
      count: perGroup[i],
      radius: clusterRadius(perGroup[i], node.r),
      nodeId: node.id,
      slot: slotOf(node.dim),
      level: 2,
    }),
  );
  // A fact's stars gather on its attribute when that is shown, otherwise in a fan beside the
  // nearest shown ancestor, on the side away from its dimension.
  const factsAt = new Map<string, number[]>();
  facts.forEach((_, index) => {
    const host = shownAncestor(attributeOf[index]);
    const list = factsAt.get(host.id) ?? [];
    list.push(index);
    factsAt.set(host.id, list);
  });
  for (const [id, indices] of factsAt) {
    const host = model.nodes.get(id)!;
    const at = byId.get(id) ?? root;
    if (host.kind === 'attribute') {
      for (const index of indices)
        clusters.push({
          x: at.x,
          y: at.y,
          count: perFact[index],
          radius: clusterRadius(perFact[index], 2.2),
          nodeId: at.id,
          slot: slotOf(host.dim),
          level: 3,
        });
      continue;
    }
    const from = host.dim === null ? root : (byId.get(`dimension:${host.dim}`) ?? root);
    let angle = Math.atan2(at.y - from.y, at.x - from.x);
    if (host.kind === 'dimension' || host.kind === 'root') angle = Math.atan2(at.y - root.y, at.x - root.x);
    const base = at.r + 7;
    const step = 6.4 * Math.max(0.75, Math.min(1.4, zoom / 0.7));
    const fan = Math.min(TAU * 0.92, 1.6 + indices.length * 0.11);
    indices.forEach((index, i) => {
      const distance = base + step * Math.sqrt(i + 0.6) * 1.15;
      const direction = angle + (((i * 0.6180339887) % 1) - 0.5) * fan;
      clusters.push({
        x: at.x + distance * Math.cos(direction),
        y: at.y + distance * Math.sin(direction),
        count: perFact[index],
        radius: clusterRadius(perFact[index], 1.8),
        nodeId: at.id,
        slot: slotOf(host.dim),
        level: 3,
      });
    });
  }

  // Places inside the clusters: a sunflower, denser towards the centre.
  const places = clusters.reduce((sum, cluster) => sum + cluster.count, 0);
  const placeCluster = new Int32Array(places);
  const placeOffset = new Float32Array(places * 2);
  let p = 0;
  clusters.forEach((cluster, index) => {
    const turn = rand() * TAU;
    for (let i = 0; i < cluster.count; i++) {
      const r = cluster.radius * Math.pow((i + 0.5) / cluster.count, 0.62) * (0.92 + 0.16 * rand());
      const a = turn + i * 2.399963 + (rand() - 0.5) * 0.5;
      placeCluster[p] = index;
      placeOffset[p * 2] = r * Math.cos(a);
      placeOffset[p * 2 + 1] = r * Math.sin(a);
      p++;
    }
  });

  // Stars in the sky, a little past the edges, thinned in patches by value noise.
  const sky = new Float32Array(count * 2);
  const bleed = 0.03;
  for (let i = 0; i < count; i++) {
    let x = 0;
    let y = 0;
    for (let tries = 0; tries < 12; tries++) {
      x = (-bleed + (1 + 2 * bleed) * rand()) * width;
      y = (-bleed + (1 + 2 * bleed) * rand()) * height;
      const n = 0.62 * valueNoise(x / 210, y / 210) + 0.38 * valueNoise(x / 70 + 17, y / 70 + 5);
      if (rand() < Math.min(1, Math.max(0.12, (n - 0.18) * 1.55))) break;
    }
    sky[i * 2] = x;
    sky[i * 2 + 1] = y;
  }

  const wordAttr = new Float32Array(count * 4);
  const wordLook = new Float32Array(count * 2);
  const words = options.words && options.words.count ? options.words : null;
  const isDot = new Uint8Array(count);
  const wordDots = words ? assignWordDots(words, sky, count, width, height, wordAttr, wordLook, isDot) : 0;
  for (let i = 0; i < count; i++) {
    if (isDot[i]) continue;
    wordAttr[i * 4 + 2] = -1;
    if (!words) continue;
    const across = clamp01(sky[i * 2] / width);
    wordAttr[i * 4 + 3] =
      rand() < WORDS.litAtStart ? 0 : WORDS.igniteStart + WORDS.igniteSpread * (0.55 * across + 0.45 * rand());
  }

  // Which stars join a cluster: a random subset, matched to places by angle around the root.
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const kept = order[i];
    order[i] = order[j];
    order[j] = kept;
  }
  const matched = Math.min(places, count);
  const chosen = order.slice(0, matched);
  const setOff = T.flowStart + 0.45;
  // A star lands this far ahead (rad) of where it set off, in the sense of the turn.
  const lead = 0.8;
  const skyAngle = new Float64Array(matched);
  const skyRadius = new Float64Array(matched);
  for (let i = 0; i < matched; i++) {
    const star = chosen[i];
    const dx = sky[star * 2] - root.x;
    const dy = sky[star * 2 + 1] - root.y;
    const r = Math.hypot(dx, dy);
    skyRadius[i] = r;
    skyAngle[i] = Math.atan2(dy, dx) + spinRate(r) * spinPhase(T, setOff);
  }
  const placeAngle = new Float64Array(places);
  const placeRadius = new Float64Array(places);
  for (let i = 0; i < places; i++) {
    const cluster = clusters[placeCluster[i]];
    const dx = cluster.x + placeOffset[i * 2] - root.x;
    const dy = cluster.y + placeOffset[i * 2 + 1] - root.y;
    placeAngle[i] = Math.atan2(dy, dx);
    placeRadius[i] = Math.hypot(dx, dy);
  }
  const skyOrder = Array.from({ length: matched }, (_, i) => i).sort((a, b) => wrap(skyAngle[a]) - wrap(skyAngle[b]));
  const placeOrder = Array.from({ length: matched }, (_, i) => i).sort((a, b) => placeAngle[a] - placeAngle[b]);
  const n = matched;
  // The rotation between the two angular orders that best keeps every star `lead` behind its place.
  const cost = (offset: number, stride: number) => {
    let sum = 0;
    for (let i = 0; i < n; i += stride) {
      const d = wrap(placeAngle[placeOrder[(i + offset) % n]] - wrap(skyAngle[skyOrder[i]]) - lead);
      sum += d * d;
    }
    return sum;
  };
  let best = 0;
  let bestCost = Infinity;
  const tries = 240;
  if (n > 0) {
    for (let i = 0; i < tries; i++) {
      const offset = Math.floor((i * n) / tries);
      const c = cost(offset, 23);
      if (c < bestCost) {
        bestCost = c;
        best = offset;
      }
    }
    const near = Math.max(1, Math.floor(n / tries));
    for (let i = best - near; i <= best + near; i += Math.max(1, Math.floor(near / 24))) {
      const offset = ((i % n) + n) % n;
      const c = cost(offset, 7);
      if (c < bestCost) {
        bestCost = c;
        best = offset;
      }
    }
  }
  // Within blocks of neighbouring angles, the nearest stars take the nearest places.
  const placeOfChosen = new Uint32Array(n);
  const block = 48;
  for (let i = 0; i < n; i += block) {
    const stop = Math.min(n, i + block);
    const stars = skyOrder.slice(i, stop).sort((a, b) => skyRadius[a] - skyRadius[b]);
    const spots: number[] = [];
    for (let j = i; j < stop; j++) spots.push(placeOrder[(j + best) % n]);
    spots.sort((a, b) => placeRadius[a] - placeRadius[b]);
    for (let j = 0; j < stars.length; j++) placeOfChosen[stars[j]] = spots[j];
  }

  // Nodes appear from the root outwards.
  let farthest = 1;
  for (const node of nodes) farthest = Math.max(farthest, Math.hypot(node.x - root.x, node.y - root.y));
  const timings = new Map<string, NodeTiming>();
  for (const node of nodes) {
    const reach = Math.hypot(node.x - root.x, node.y - root.y) / farthest;
    const collapse = T.settleStart + 0.42 * reach + 0.05 * rand();
    const collapseDuration = 0.4;
    const fadeIn = collapse + collapseDuration * 0.45;
    const fadeOut = fadeIn + 0.38;
    const edgeStart = T.edgesStart + 0.34 * reach;
    const labelStart = Math.max(fadeOut - 0.1, edgeStart);
    timings.set(node.id, {
      collapse,
      collapseDuration,
      fadeIn,
      fadeOut,
      edgeStart,
      edgeEnd: edgeStart + 0.42,
      labelStart,
      labelEnd: labelStart + 0.4,
    });
  }

  const a = new Float32Array(count * 4);
  const b = new Float32Array(count * 4);
  const c = new Float32Array(count * 4);
  const d = new Float32Array(count * 4);
  const e = new Float32Array(count * 2);
  const placeOfStar = new Int32Array(count).fill(-1);
  for (let i = 0; i < n; i++) placeOfStar[chosen[i]] = placeOfChosen[i];
  for (let i = 0; i < count; i++) {
    const dx = sky[i * 2] - root.x;
    const dy = sky[i * 2 + 1] - root.y;
    const r0 = Math.hypot(dx, dy);
    const angle0 = Math.atan2(dy, dx);
    d[i * 4] = rand();
    const sizeRoll = rand();
    d[i * 4 + 1] = (sizeRoll < 0.7 ? 1.15 : sizeRoll < 0.94 ? 1.65 : 2.35) * (0.9 + 0.2 * rand()) * (sparse ? 1.45 : 1);
    d[i * 4 + 3] = 0.55 + 0.45 * rand();
    a[i * 4] = r0;
    a[i * 4 + 1] = angle0;
    const place = placeOfStar[i];
    if (place < 0) {
      // Drifts outwards and fades while the others gather.
      const depart = T.flowStart + 0.7 * rand();
      a[i * 4 + 2] = sky[i * 2];
      a[i * 4 + 3] = sky[i * 2 + 1];
      c[i * 4] = depart;
      c[i * 4 + 1] = depart + 0.45 + 0.35 * rand();
      c[i * 4 + 2] = -1;
      c[i * 4 + 3] = 0;
      d[i * 4 + 2] = 6;
      continue;
    }
    const cluster = clusters[placeCluster[place]];
    const node = byId.get(cluster.nodeId)!;
    const timing = timings.get(cluster.nodeId)!;
    a[i * 4 + 2] = cluster.x;
    a[i * 4 + 3] = cluster.y;
    b[i * 4] = placeOffset[place * 2];
    b[i * 4 + 1] = placeOffset[place * 2 + 1];
    b[i * 4 + 2] = node.x;
    b[i * 4 + 3] = node.y;
    d[i * 4 + 2] = cluster.slot;
    const cx = cluster.x - root.x;
    const cy = cluster.y - root.y;
    const clusterDistance = Math.hypot(cx, cy);
    const travel = Math.hypot(
      cluster.x + placeOffset[place * 2] - sky[i * 2],
      cluster.y + placeOffset[place * 2 + 1] - sky[i * 2 + 1],
    );
    // Inner levels and nearer clusters set off first; everyone has arrived before the collapse.
    const depart = T.flowStart - 0.05 + 0.05 * cluster.level + 0.25 * (clusterDistance / farthest) + 0.35 * rand();
    let duration = 1.7 + 6e-4 * travel + 0.3 * rand();
    const latest = Math.min(T.flowEnd - 0.1, timing.collapse - 0.05);
    if (depart + duration > latest) duration = Math.max(0.7, latest - depart);
    c[i * 4] = depart;
    c[i * 4 + 1] = depart + duration;
    c[i * 4 + 2] = timing.collapse + (rand() - 0.5) * 0.06;
    c[i * 4 + 3] = timing.collapseDuration * (0.85 + 0.3 * rand());
    // The cluster's angle, unwrapped to lie just ahead of where the star is when it sets off.
    const swept = angle0 + spinRate(r0) * spinPhase(T, depart);
    let home = clusterDistance < 0.001 ? swept + lead : Math.atan2(cy, cx);
    home += TAU * Math.round((swept + lead - home) / TAU);
    e[i * 2] = home;
    e[i * 2 + 1] = 0.9 * Math.sqrt(10 / (cluster.radius + 10)) * (0.85 + 0.3 * rand());
  }
  return {
    particles: {
      count,
      allocated: n,
      center: { x: root.x, y: root.y },
      a,
      b,
      c,
      d,
      e,
      f: wordAttr,
      g: wordLook,
      words: !!words,
      wordDots,
    },
    timings,
  };
}

/**
 * Hands every glyph dot to a star near its line, so a line breaks up into stars that were already
 * around it. Stars and dots are paired in angular order around the line, nearest with nearest.
 * Marks the stars in `used` and returns how many dots found one.
 */
function assignWordDots(
  dots: WordDots,
  sky: Float32Array,
  count: number,
  width: number,
  height: number,
  attr: Float32Array,
  look: Float32Array,
  used: Uint8Array,
): number {
  const cell = 32;
  const cols = Math.ceil(width / cell) + 2;
  const rows = Math.ceil(height / cell) + 2;
  const grid: number[][] = Array.from({ length: cols * rows }, () => []);
  const cellOf = (x: number, y: number) => [
    Math.min(cols - 1, Math.max(0, Math.floor(x / cell) + 1)),
    Math.min(rows - 1, Math.max(0, Math.floor(y / cell) + 1)),
  ];
  for (let i = 0; i < count; i++) {
    const [cx, cy] = cellOf(sky[i * 2], sky[i * 2 + 1]);
    grid[cy * cols + cx].push(i);
  }
  const bySpan = new Map<number, number[]>();
  for (let i = 0; i < dots.count; i++) {
    const list = bySpan.get(dots.sentence[i]) ?? [];
    list.push(i);
    bySpan.set(dots.sentence[i], list);
  }
  let assigned = 0;
  for (const [index, members] of bySpan) {
    const span = dots.spans[index];
    const mx = span.x + span.width / 2;
    const my = span.y;
    const rx = span.width / 2 + 30;
    const ry = span.size + 30;
    const ellipse = (x: number, y: number) => ((x - mx) / rx) ** 2 + ((y - my) / ry) ** 2;
    const [gx, gy] = cellOf(mx, my);
    let pool: number[] = [];
    for (let reach = Math.ceil(rx / cell) + 1; reach < Math.max(cols, rows); reach += 2) {
      pool = [];
      const reachY = Math.ceil((reach * ry) / rx) + 1;
      for (let y = Math.max(0, gy - reachY); y <= Math.min(rows - 1, gy + reachY); y++)
        for (let x = Math.max(0, gx - reach); x <= Math.min(cols - 1, gx + reach); x++)
          for (const star of grid[y * cols + x]) if (!used[star]) pool.push(star);
      if (pool.length >= members.length * 1.5) break;
    }
    pool.sort((p, q) => ellipse(sky[p * 2], sky[p * 2 + 1]) - ellipse(sky[q * 2], sky[q * 2 + 1]));
    const stars = pool.slice(0, members.length);
    const angle = (x: number, y: number) => Math.atan2((y - my) / ry, (x - mx) / rx);
    const glyphs = members
      .slice(0, stars.length)
      .sort((p, q) => angle(dots.x[p], dots.y[p]) - angle(dots.x[q], dots.y[q]));
    stars.sort((p, q) => angle(sky[p * 2], sky[p * 2 + 1]) - angle(sky[q * 2], sky[q * 2 + 1]));
    const chunk = 24;
    for (let i = 0; i < glyphs.length; i += chunk) {
      const g = glyphs
        .slice(i, i + chunk)
        .sort((p, q) => ellipse(dots.x[p], dots.y[p]) - ellipse(dots.x[q], dots.y[q]));
      const s = stars
        .slice(i, i + chunk)
        .sort((p, q) => ellipse(sky[p * 2], sky[p * 2 + 1]) - ellipse(sky[q * 2], sky[q * 2 + 1]));
      for (let j = 0; j < g.length; j++) {
        const star = s[j];
        const dot = g[j];
        used[star] = 1;
        attr[star * 4] = dots.x[dot];
        attr[star * 4 + 1] = dots.y[dot];
        attr[star * 4 + 2] = dots.start[dot];
        attr[star * 4 + 3] = Math.max(dots.start[dot] + 0.05, dots.end[dot]);
        look[star * 2] = dots.alpha[dot];
        look[star * 2 + 1] = dots.size[dot];
        assigned++;
      }
    }
  }
  return assigned;
}
