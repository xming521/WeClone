import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildModel } from '../src/data.ts';
import type { ProfileFact, ProfileInput } from '../src/data.ts';
import {
  WORDS,
  apportion,
  dayAt,
  layoutWords,
  planParticles,
  sampleGlyphs,
  spinPhase,
  spreadInTime,
  timeline,
  titleLine,
} from '../src/atlas/nightIntroPlan.ts';
import type { GlyphRaster, IntroSentence, PlanNode, WordSpan } from '../src/atlas/nightIntroPlan.ts';

const close = (actual: number, expected: number, eps = 1e-9) =>
  assert.ok(Math.abs(actual - expected) < eps, `${actual} ≉ ${expected}`);

test('the opening lines push everything but themselves back by 1.2 s', () => {
  const plain = timeline(false);
  const T = timeline(true);
  close(plain.total, 6);
  close(T.total, 7.2);
  close(T.flowStart, 2.7);
  close(T.settleStart, 5.65);
  assert.deepEqual(
    T.textIn.map((v) => +v.toFixed(2)),
    [1.32, 1.7, 2.08],
  );
  // The last line has broken up and its dots have reached the sky before the stars set off.
  close(T.wordsEnd, 1.75);
  assert.equal(plain.wordsEnd, 0);
  assert.ok(T.wordsEnd < T.flowStart);
});

test('title lines rise in, hold and lift out', () => {
  const T = timeline(true);
  assert.deepEqual(titleLine(T, 1, 0), { opacity: 0, y: 10 });
  const held = titleLine(T, 1, 2.6);
  close(held.opacity, 1);
  close(held.y, 0);
  const gone = titleLine(T, 1, T.textOut[1] + T.textOutDuration);
  close(gone.opacity, 0);
  close(gone.y, -6);
});

test('dawn starts at the root and has reached every corner by dayEnd', () => {
  const T = timeline(true);
  const max = 1000;
  assert.equal(dayAt(T, T.dayStart, 0, max), 0);
  const mid = (T.dayStart + T.dayEnd) / 2;
  assert.ok(dayAt(T, mid, 0, max) > dayAt(T, mid, 500, max));
  assert.ok(dayAt(T, mid, 500, max) > dayAt(T, mid, max, max));
  assert.equal(dayAt(T, T.dayEnd, max, max), 1);
});

test('the sky turns faster while the stars set off', () => {
  const T = timeline(true);
  const [from, to] = [T.flowStart - 0.2, T.flowStart + 1.5];
  close(spinPhase(T, from - 0.5), from - 0.5);
  const slope = (t: number) => (spinPhase(T, t + 1e-4) - spinPhase(T, t)) / 1e-4;
  close(slope(from - 0.1), 1, 1e-6);
  close(slope(to + 0.5), 3.6, 1e-3);
  assert.ok(slope((from + to) / 2) > 1);
});

test('apportion splits a total by largest remainder', () => {
  assert.deepEqual(apportion([1, 1, 1], 10), [4, 3, 3]);
  assert.deepEqual(apportion([3, 1], 4), [3, 1]);
  assert.equal(
    apportion([0.3, 2.2, 7.1, 0.4], 97).reduce((a, b) => a + b, 0),
    97,
  );
});

const line = (text: string, time: string, dim: number | null = 8): IntroSentence => ({ text, time, dim });

test('spreadInTime picks lines near evenly spaced moments, oldest first', () => {
  const lines = [
    line('a', '2020-01-01T00:00:00'),
    line('b', '2020-01-05T00:00:00'),
    line('c', '2021-06-01T00:00:00'),
    line('d', '2022-12-20T00:00:00'),
    line('e', '2023-01-01T00:00:00'),
  ];
  assert.deepEqual(
    spreadInTime(lines, 2).map((l) => l.text),
    ['c', 'd'],
  );
  assert.deepEqual(
    spreadInTime(lines.slice(0, 2), 4).map((l) => l.text),
    ['a', 'b'],
  );
});

// Every character 10 px wide at size 10, letter spacing included.
const measure = (text: string, size: number, _weight: number, tracking: number) =>
  [...text].length * size + tracking * Math.max(0, [...text].length - 1);

const sentences = Array.from({ length: 40 }, (_, i) =>
  line(
    `${'这是第几句话'.repeat(2).slice(0, 6 + (i % 9))}${i}`,
    `20${20 + (i % 5)}-0${1 + (i % 9)}-15T12:00:00`,
    i % 3 ? 8 : 5,
  ),
);

test('layoutWords keeps lines out of the title block and off each other', () => {
  const width = 1440;
  const height = 900;
  const spans = layoutWords(sentences, width, height, measure)!;
  const lines = spans.filter((span) => span.role === 'line');
  const faint = spans.filter((span) => span.role === 'faint');
  assert.equal(lines.length, 4);
  assert.equal(spans.filter((span) => span.role === 'date').length, 4);
  assert.ok(faint.length > 0 && faint.length <= 18);
  const title = { x0: width / 2 - 410, x1: width / 2 + 410, y0: height / 2 - 150, y1: height / 2 + 150 };
  for (const span of [...lines, ...faint]) {
    const inside = span.x < title.x1 && span.x + span.width > title.x0 && span.y < title.y1 && span.y > title.y0;
    assert.ok(!inside, `${span.text} overlaps the title`);
    assert.ok(span.x >= 0 && span.x + span.width <= width);
  }
  // A line is read before it breaks up, and has broken up before the stars set off.
  for (const span of lines) {
    assert.ok(span.dissolve >= span.appear + WORDS.lineFade + WORDS.minRead - 1e-9);
    assert.ok(span.dissolve + span.sweep < timeline(true).flowStart);
  }
  // Lines of interests are preferred when there are enough.
  const chosen = new Set(lines.map((span) => span.text));
  assert.ok(sentences.filter((s) => chosen.has(s.text)).every((s) => s.dim === 8));
  assert.equal(layoutWords([], width, height, measure), null);
  assert.equal(layoutWords(sentences, 600, 800, measure)!.filter((span) => span.role === 'line').length, 2);
});

test('sampleGlyphs puts dots on lit pixels and breaks them off left to right', () => {
  const span: WordSpan = {
    text: 'ab',
    role: 'line',
    x: 10,
    y: 20,
    size: 10,
    weight: 400,
    tracking: 0,
    alpha: 0.95,
    width: 20,
    appear: 0,
    fade: 0.4,
    dissolve: 1,
    sweep: 0.2,
    dot: 1.6,
  };
  // A 20 × 4 raster lit everywhere but the first two columns.
  const raster = (): GlyphRaster => {
    const data = new Uint8ClampedArray(20 * 4 * 4);
    for (let y = 0; y < 4; y++) for (let x = 2; x < 20; x++) data[(y * 20 + x) * 4 + 3] = 255;
    return { x0: 10, y0: 18, width: 20, height: 4, data };
  };
  const dots = sampleGlyphs([span], raster, 1.5);
  // Rows 18 and 20; columns 12, 14, … 28 on the first and 13, … 29 on the second.
  assert.equal(dots.count, 18);
  for (let i = 0; i < dots.count; i++) {
    assert.ok(dots.x[i] >= 12 && dots.x[i] < 30.5);
    close(dots.start[i], 1 + ((Math.floor(dots.x[i]) - 10) / 20) * 0.2, 1e-5);
    assert.ok(dots.end[i] <= 1.5 + 1e-6 && dots.end[i] > dots.start[i]);
    assert.equal(dots.sentence[i], 0);
  }
});

function fact(index: number, support: number, importance: number): ProfileFact {
  return {
    id: `f${index}`,
    node_id: '',
    group_id: '',
    status: 'pending',
    version: 1,
    origin: 'extracted',
    original: { value: '', attr: '', source_ids: [] },
    dim: 1,
    attr: '',
    value: '',
    source_ids: Array.from({ length: support }, (_, i) => `s${index}-${i}`),
    support_count: support,
    source_importance_mean: importance,
  };
}

const input: ProfileInput = {
  domains: [],
  dimensions: [
    {
      dim: 1,
      name: '身份',
      groups: [
        { id: 't1', name: '名字', items: [{ id: 'a1', attr: '姓名', fact_indices: [0, 1] }] },
        { id: 't2', name: '年龄', items: [{ id: 'a2', attr: '年龄', fact_indices: [2] }] },
      ],
    },
    {
      dim: 8,
      name: '兴趣',
      groups: [{ id: 't3', name: '运动', items: [{ id: 'a3', attr: '跑步', fact_indices: [3, 4, 5] }] }],
    },
  ],
  facts: [fact(0, 6, 4), fact(1, 1, 2), fact(2, 2, 3), fact(3, 1, 1), fact(4, 3, 3), fact(5, 1, 2)],
  sources: {},
  locations: [],
  snapshot_sha256: 'fixture',
};

// The overview: root, dimensions and topics, attributes hidden.
const nodes: PlanNode[] = [
  { id: 'root', kind: 'root', dim: null, x: 700, y: 450, r: 30 },
  { id: 'dimension:1', kind: 'dimension', dim: 1, x: 500, y: 450, r: 20 },
  { id: 'dimension:8', kind: 'dimension', dim: 8, x: 900, y: 450, r: 20 },
  { id: 't1', kind: 'topic', dim: 1, x: 380, y: 360, r: 12 },
  { id: 't2', kind: 'topic', dim: 1, x: 380, y: 540, r: 10 },
  { id: 't3', kind: 'topic', dim: 8, x: 1060, y: 450, r: 12 },
];

const options = (count: number) => ({
  timeline: timeline(true),
  width: 1400,
  height: 900,
  count,
  zoom: 1,
  dims: [1, 3, 4, 5, 8],
});

test('planParticles sends a share of the stars to every node and collapses them from the root out', () => {
  const model = buildModel(input);
  const T = timeline(true);
  const count = 3000;
  const { particles, timings } = planParticles(model, nodes, options(count));
  assert.equal(particles.count, count);
  assert.equal(particles.allocated, Math.round(count * 0.6));
  const perNode = new Map<string, number>();
  for (let i = 0; i < count; i++) {
    const depart = particles.c[i * 4];
    const arrive = particles.c[i * 4 + 1];
    const collapse = particles.c[i * 4 + 2];
    assert.ok(arrive > depart);
    if (collapse < 0) {
      assert.equal(particles.d[i * 4 + 2], 6);
      continue;
    }
    // The node a star collapses into is one of the shown nodes.
    const node = nodes.find((n) => n.x === particles.b[i * 4 + 2] && n.y === particles.b[i * 4 + 3])!;
    assert.ok(node);
    perNode.set(node.id, (perNode.get(node.id) ?? 0) + 1);
    assert.ok(depart >= T.flowStart - 0.05);
    assert.ok(arrive <= Math.max(T.flowEnd - 0.1, depart + 0.7) + 1e-6);
    assert.ok(arrive <= timings.get(node.id)!.collapse - 0.05 + 1e-6 || arrive - depart <= 0.7 + 1e-6);
  }
  // Root 4.5%, dimensions 11.5%, groups 9%; the facts' 75% gather around their topics.
  close(perNode.get('root')!, Math.round(1800 * 0.045));
  assert.equal(perNode.get('dimension:1')! + perNode.get('dimension:8')!, Math.round(1800 * 0.115));
  assert.ok(perNode.get('t1')! > perNode.get('t2')!);
  assert.equal(
    [...perNode.values()].reduce((a, b) => a + b, 0),
    1800,
  );
  // Nodes collapse in order of their distance from the root, all well before the end.
  assert.ok(timings.get('root')!.collapse < timings.get('dimension:1')!.collapse);
  assert.ok(timings.get('dimension:1')!.collapse < timings.get('t1')!.collapse);
  for (const timing of timings.values()) {
    assert.ok(timing.fadeOut < T.total && timing.edgeEnd < T.total && timing.labelEnd < T.total);
  }
  // Same inputs, same plan.
  assert.deepEqual(planParticles(model, nodes, options(count)).particles.c, particles.c);
});

test('planParticles hands glyph dots to stars near their line', () => {
  const model = buildModel(input);
  const span: WordSpan = {
    text: '一二三四五',
    role: 'line',
    x: 100,
    y: 120,
    size: 20,
    weight: 400,
    tracking: 0,
    alpha: 0.95,
    width: 100,
    appear: 0.1,
    fade: 0.42,
    dissolve: 1,
    sweep: 0.2,
    dot: 1.6,
  };
  const raster = (): GlyphRaster => {
    const data = new Uint8ClampedArray(100 * 20 * 4).fill(255);
    return { x0: 100, y0: 110, width: 100, height: 20, data };
  };
  const words = sampleGlyphs([span], raster, timeline(true).flowStart - 0.12);
  const { particles } = planParticles(model, nodes, { ...options(4000), words });
  assert.equal(particles.wordDots, words.count);
  let dots = 0;
  for (let i = 0; i < particles.count; i++) {
    const start = particles.f[i * 4 + 2];
    if (start < 0) {
      // Other stars shine from the start or light up while the lines dissolve.
      const lit = particles.f[i * 4 + 3];
      assert.ok(lit === 0 || (lit >= WORDS.igniteStart && lit <= WORDS.igniteStart + WORDS.igniteSpread));
      continue;
    }
    dots++;
    assert.ok(particles.f[i * 4] >= 100 && particles.f[i * 4] <= 201);
    assert.ok(particles.f[i * 4 + 3] > start);
    close(particles.g[i * 2 + 1], 1.6, 1e-6);
  }
  assert.equal(dots, words.count);
});
