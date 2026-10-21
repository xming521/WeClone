import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildModel, visibleNodes } from '../src/data.ts';
import type { ProfileInput } from '../src/data.ts';
import {
  clusterCenters,
  computeSeeds,
  expansionPositions,
  overviewPositions,
  placeNodes,
  sameTree,
  shiftClusters,
} from '../src/atlas/graphLayout.ts';

const input: ProfileInput = {
  dimensions: [
    {
      dim: 8,
      name: '兴趣',
      groups: [
        { id: 'topic-a', name: '运动', items: [{ id: 'attr-a', attr: '偏好', fact_indices: [0] }] },
        { id: 'topic-b', name: '阅读', items: [{ id: 'attr-b', attr: '书籍', fact_indices: [1] }] },
      ],
    },
    {
      dim: 1,
      name: '性格',
      groups: [{ id: 'topic-c', name: '社交', items: [{ id: 'attr-c', attr: '外向', fact_indices: [2] }] }],
    },
  ],
  facts: [],
  sources: {},
  locations: [],
  snapshot_sha256: 'fixture',
};

const model = buildModel(input);

test('overviewPositions places every node with the root at the origin', () => {
  const positions = overviewPositions(model);
  assert.deepEqual(positions.get('root'), { x: 0, y: 0 });
  assert.deepEqual([...positions.keys()].sort(), [...model.nodes.keys()].sort());
  for (const point of positions.values()) {
    assert.ok(Number.isFinite(point.x) && Number.isFinite(point.y));
  }
});

test('expansionPositions keeps existing coordinates and only places new nodes', () => {
  const previous = new Map([
    ['root', { x: 10, y: 20 }],
    ['dimension:8', { x: 310, y: 20 }],
    ['dimension:1', { x: -290, y: 20 }],
  ]);
  const visible = visibleNodes(model, new Set(['root', 'dimension:8']));
  const next = expansionPositions(model, visible, previous);
  for (const [id, point] of previous) assert.deepEqual(next.get(id), point);
  assert.deepEqual([...next.keys()].filter((id) => !previous.has(id)).sort(), ['topic-a', 'topic-b']);
  assert.equal(previous.size, 3);
  for (const id of ['topic-a', 'topic-b']) {
    const point = next.get(id)!;
    assert.ok(Number.isFinite(point.x) && Number.isFinite(point.y));
  }
});

test('sameTree compares ids and parents in order', () => {
  const visible = visibleNodes(model, new Set(['root', 'dimension:8']));
  assert.equal(sameTree(visible, [...visible]), true);
  assert.equal(sameTree(visible, visible.slice(1)), false);
  assert.equal(sameTree(visible, [...visible].reverse()), false);
  assert.equal(
    sameTree(
      visible,
      visible.map((node) => ({ ...node, parentId: node.parentId && 'root' })),
    ),
    false,
  );
});

test('computeSeeds reuses current positions in the same view and aligns the root when switching', () => {
  const previous = new Map([
    ['root', { x: 100, y: -50 }],
    ['dimension:8', { x: 400, y: -50 }],
  ]);
  const overview = overviewPositions(model);
  assert.equal(computeSeeds(false, previous, overview, undefined), previous);

  const restored = new Map([
    ['root', { x: 5, y: 7 }],
    ['dimension:8', { x: 25, y: 7 }],
  ]);
  const seeds = computeSeeds(true, previous, overview, restored);
  assert.deepEqual(seeds.get('root'), previous.get('root'));
  assert.deepEqual(seeds.get('dimension:8'), { x: 120, y: -50 });
  assert.deepEqual(seeds.get('dimension:1'), {
    x: overview.get('dimension:1')!.x + 95,
    y: overview.get('dimension:1')!.y - 57,
  });
  assert.deepEqual(restored.get('root'), { x: 5, y: 7 });

  const fallback = computeSeeds(true, previous, overview, undefined);
  assert.deepEqual(fallback.get('root'), previous.get('root'));
});

test('placeNodes prefers seeds, then initial positions, then a spot around the parent', () => {
  const visible = visibleNodes(model, new Set(['root', 'dimension:8']));
  const seeds = new Map([['root', { x: 5, y: 6 }]]);
  const initial = new Map([
    ['root', { x: 50, y: 60 }],
    ['dimension:8', { x: 300, y: 0 }],
  ]);
  const positions = placeNodes(visible, seeds, initial);
  assert.deepEqual(
    [...positions.keys()],
    visible.map((node) => node.id),
  );
  assert.deepEqual(positions.get('root'), { x: 5, y: 6 });
  assert.deepEqual(positions.get('dimension:8'), { x: 300, y: 0 });
  const topic = positions.get('topic-a')!;
  assert.ok(Math.abs(Math.hypot(topic.x - 300, topic.y) - 135) < 1e-9);
  assert.deepEqual(placeNodes(visible, new Map(), new Map()).get('root'), { x: 0, y: 0 });
});

test('clusterCenters keeps each dimension direction and moves a larger dimension farther out', () => {
  const root = { x: 10, y: 20 };
  const collapsed = clusterCenters(model, visibleNodes(model, new Set(['root'])), root);
  // Eighty visible nodes in dimension 8 stand in for a dimension whose topics are all expanded.
  const crowded = [...visibleNodes(model, new Set(['root'])), ...Array(80).fill(model.nodes.get('topic-a')!)];
  const expanded = clusterCenters(model, crowded, root);
  const distance = (point: { x: number; y: number }) => Math.hypot(point.x - root.x, point.y - root.y);
  const direction = (point: { x: number; y: number }) => Math.atan2(point.y - root.y, point.x - root.x);
  for (const id of model.dimensionIds) {
    assert.ok(distance(collapsed.get(id)!) >= 330);
    assert.ok(Math.abs(direction(collapsed.get(id)!) - direction(expanded.get(id)!)) < 1e-9);
  }
  assert.ok(distance(expanded.get('dimension:8')!) > distance(collapsed.get('dimension:8')!));
});

test('shiftClusters moves a whole dimension with its dimension node and ignores small offsets', () => {
  const positions = new Map([
    ['root', { x: 0, y: 0 }],
    ['dimension:8', { x: 300, y: 0 }],
    ['topic-a', { x: 400, y: 50 }],
    ['dimension:1', { x: -300, y: 0 }],
    ['topic-c', { x: -400, y: 0 }],
  ]);
  const centers = new Map([
    ['dimension:8', { x: 500, y: 0 }],
    ['dimension:1', { x: -320, y: 0 }],
  ]);
  const shifted = shiftClusters(model, positions, centers);
  assert.deepEqual(shifted.get('dimension:8'), { x: 500, y: 0 });
  assert.deepEqual(shifted.get('topic-a'), { x: 600, y: 50 });
  assert.deepEqual(shifted.get('root'), { x: 0, y: 0 });
  assert.deepEqual(shifted.get('dimension:1'), { x: -300, y: 0 });
  assert.deepEqual(shifted.get('topic-c'), { x: -400, y: 0 });
  const unchanged = new Map([['dimension:1', { x: -320, y: 0 }]]);
  assert.equal(shiftClusters(model, positions, unchanged), positions);
});
