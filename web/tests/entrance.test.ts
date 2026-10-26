import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildModel } from '../src/data.ts';
import type { ProfileFact, ProfileInput } from '../src/data.ts';
import { TIMING, buildEntrancePlan, nodeFrame } from '../src/atlas/entrancePlan.ts';
import type { EntranceInput, NodeFrame } from '../src/atlas/entrancePlan.ts';
import { nodeLooks } from '../src/atlas/nodeLook.ts';
import type { NodeLook } from '../src/atlas/nodeLook.ts';

function fact(
  index: number,
  scores: { support: number; importance: number; confidence: number } | null,
  origin: ProfileFact['origin'] = 'extracted',
): ProfileFact {
  return {
    id: `f${index}`,
    node_id: '',
    group_id: '',
    status: 'approved',
    version: 1,
    origin,
    original: { value: '', attr: '', source_ids: [] },
    dim: 1,
    attr: '',
    value: '',
    source_ids: scores ? Array.from({ length: scores.support }, (_, i) => `s${index}-${i}`) : [],
    ...(scores && {
      support_count: scores.support,
      source_importance_mean: scores.importance,
      source_confidence_mean: scores.confidence,
    }),
  };
}

const input: ProfileInput = {
  domains: [],
  dimensions: [
    {
      dim: 1,
      name: '身份',
      groups: [
        {
          id: 'strong',
          name: '证据多',
          items: [
            { id: 'a-strong', attr: '姓名', fact_indices: [0, 1] },
            { id: 'a-solid', attr: '年龄', fact_indices: [2] },
          ],
        },
        { id: 'weak', name: '证据少', items: [{ id: 'a-weak', attr: '猜测', fact_indices: [3] }] },
        { id: 'manual', name: '手写', items: [{ id: 'a-manual', attr: '补充', fact_indices: [4] }] },
      ],
    },
  ],
  facts: [
    fact(0, { support: 3, importance: 4, confidence: 3 }),
    fact(1, { support: 1, importance: 2, confidence: 4 }),
    fact(2, { support: 1, importance: 3, confidence: 3.5 }),
    fact(3, { support: 1, importance: 1, confidence: 2.5 }),
    fact(4, null, 'manual'),
  ],
  sources: {},
  locations: [],
  snapshot_sha256: 'fixture',
};

test('nodeLooks sizes nodes by log salience per level and fades low-confidence ones', () => {
  const looks = nodeLooks(buildModel(input));
  // importance weighted by support: (4·3 + 2·1) / 4 = 3.5, salience 3.5 · ln 5; the solid attribute adds 3 · ln 2
  // The weak topic: 1 · ln 2. The manual topic: lowest importance, 1 · ln 2, and full confidence.
  const strong = looks.get('strong')!;
  const weak = looks.get('weak')!;
  const manual = looks.get('manual')!;
  assert.equal(strong.sal01, 1);
  assert.equal(weak.sal01, 0);
  assert.equal(manual.sal01, 0);
  assert.ok(Math.abs(strong.size - 31 * 1.3) < 1e-9);
  assert.ok(Math.abs(weak.size - 31 * 0.7) < 1e-9);
  // Confidence 2.5 is below every threshold: the most translucent fill and a dashed outline.
  assert.equal(weak.clarity, 0);
  assert.ok(Math.abs(weak.fillOpacity - 0.2) < 1e-9);
  assert.ok(weak.lineDash);
  // Topic confidence is the mean over its attributes: ((3·3 + 4·1) / 4 + 3.5) / 2 = 3.375, above `solid`.
  assert.equal(strong.clarity, 1);
  assert.equal(strong.lineDash, null);
  assert.equal(manual.clarity, 1);
  // Attributes are normalised among attributes, independently of topics.
  assert.equal(looks.get('a-strong')!.sal01, 1);
  assert.ok(Math.abs(looks.get('a-solid')!.sal01 - Math.log(3) / Math.log((3.5 * Math.log(5)) / Math.log(2))) < 1e-9);
  assert.equal(looks.get('root')!.size, 76);
  assert.equal(looks.get('dimension:1')!.size, 54);
});

const look = (sal01: number, clarity = 1): NodeLook => ({
  size: 30,
  sal01,
  clarity,
  fillOpacity: 1,
  strokeOpacity: 1,
  lineWidth: 1.5,
  lineDash: null,
  glowBlur: 10,
  glowAlpha: 1,
});

function inputs(): EntranceInput[] {
  const nodes: EntranceInput[] = [{ id: 'root', kind: 'root', parentId: null, x: 0, y: 0, look: look(1) }];
  for (let d = 0; d < 5; d++) {
    nodes.push({
      id: `d${d}`,
      kind: 'dimension',
      parentId: 'root',
      x: 300 * Math.cos(d),
      y: 300 * Math.sin(d),
      look: look(1),
    });
    for (let t = 0; t < 12; t++) {
      const id = `d${d}t${t}`;
      nodes.push({
        id,
        kind: 'topic',
        parentId: `d${d}`,
        x: d * 100 + t,
        y: t * 10,
        look: look(((d * 12 + t) % 10) / 9, t % 3 ? 1 : 0),
      });
    }
  }
  nodes.push({ id: 'deep', kind: 'attribute', parentId: 'd0t0', x: 5, y: 5, look: look(1) });
  return nodes;
}

test('buildEntrancePlan sweeps the dimensions in turn and brings their stars in by salience', () => {
  const plan = buildEntrancePlan(inputs());
  assert.equal(plan.root.appearAt, TIMING.rootAt);
  plan.dimensions.forEach((node, index) => {
    assert.ok(Math.abs(node.appearAt - (TIMING.dimAt + index * TIMING.dimStagger)) < 1e-9);
    assert.equal(node.dur, TIMING.dimDur);
  });
  const topics = plan.stars.filter((node) => node.kind === 'topic');
  const meanAppear = (items: typeof topics) => items.reduce((sum, node) => sum + node.appearAt, 0) / items.length;
  let previousStart = -Infinity;
  for (const dimension of plan.dimensions) {
    const members = topics.filter((node) => node.parent === dimension);
    const times = members.map((node) => node.appearAt);
    const start = Math.min(...times);
    // No star shows before its dimension has landed, and the dimensions take turns.
    assert.ok(start >= dimension.appearAt + dimension.dur * TIMING.landed - 0.02);
    assert.ok(start >= previousStart + TIMING.sweep - 0.03);
    assert.ok(Math.max(...times) - start <= TIMING.clusterWindow + 0.03);
    previousStart = start;
    // Within a dimension the bright stars come first.
    const bright = members.filter((node) => node.look.sal01 > 0.8);
    const faint = members.filter((node) => node.look.sal01 < 0.2);
    assert.ok(meanAppear(bright) < meanAppear(faint));
    // Bright stars snap in, faint ones surface slowly; vague ones take longer still.
    assert.ok(Math.max(...bright.map((node) => node.dur)) < Math.min(...faint.map((node) => node.dur)));
  }
  const deep = plan.nodes.find((node) => node.id === 'deep')!;
  assert.ok(deep.appearAt > deep.parent!.appearAt);
  // A star's name fades in as the star comes to rest.
  for (const node of plan.stars) assert.ok(node.labelAt > node.appearAt && node.labelAt < node.end);
  assert.ok(plan.end < 2.1, `entrance lasts ${plan.end.toFixed(2)}s`);
  for (const node of plan.nodes) assert.ok(plan.end >= node.end && plan.end >= node.labelEnd);
});

test('nodeFrame springs dimensions out of the root, glides stars in and settles every node at rest', () => {
  const plan = buildEntrancePlan(inputs());
  const frame: NodeFrame = { x: 0, y: 0, alpha: 0, scale: 0, glow: 0 };
  const dimension = plan.dimensions[2];
  assert.equal(nodeFrame(dimension, dimension.appearAt - 0.01, plan.root, frame), false);
  nodeFrame(dimension, dimension.appearAt + 0.02, plan.root, frame);
  assert.ok(Math.hypot(frame.x - plan.root.x, frame.y - plan.root.y) < Math.hypot(dimension.x, dimension.y) / 2);
  // A star starts on the line from its parent, the last `approach` share of the way short of its place.
  const star = plan.stars.find((node) => node.kind === 'topic')!;
  const parent = star.parent!;
  nodeFrame(star, star.appearAt + 1e-6, plan.root, frame);
  const way = Math.hypot(star.x - parent.x, star.y - parent.y);
  assert.ok(Math.abs(Math.hypot(star.x - frame.x, star.y - frame.y) - TIMING.approach * way) < 1e-3 * way);
  assert.ok(Math.abs(Math.hypot(frame.x - parent.x, frame.y - parent.y) - (1 - TIMING.approach) * way) < 1e-3 * way);
  for (const node of plan.nodes) {
    assert.equal(nodeFrame(node, node.end + 1e-6, plan.root, frame), true);
    assert.deepEqual(frame, { x: node.x, y: node.y, alpha: 1, scale: 1, glow: 1 });
  }
});
