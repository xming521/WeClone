import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildModel } from '../src/data.ts';
import type { ProfileFact, ProfileInput } from '../src/data.ts';
import { inkStrength, nodeMetrics, supportWidth } from '../src/atlas/atlasMetrics.ts';

function fact(
  index: number,
  sourceIds: string[],
  scores: { importance: number; confidence: number } | null,
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
    source_ids: sourceIds,
    support_count: sourceIds.length,
    ...(scores && { source_importance_mean: scores.importance, source_confidence_mean: scores.confidence }),
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
          id: 'topic-a',
          name: '工作',
          items: [
            { id: 'attr-a', attr: '职业', fact_indices: [0, 1] },
            { id: 'sub-a', name: '经历', items: [{ id: 'attr-b', attr: '公司', fact_indices: [2] }] },
          ],
        },
      ],
    },
    {
      dim: 3,
      name: '兴趣',
      groups: [{ id: 'topic-b', name: '运动', items: [{ id: 'attr-c', attr: '爱好', fact_indices: [3] }] }],
    },
  ],
  facts: [
    fact(0, ['s1', 's2'], { importance: 4, confidence: 4 }),
    fact(1, ['s2'], { importance: 2, confidence: 3 }),
    fact(2, ['s3'], null),
    fact(3, ['s4'], { importance: 3, confidence: 2 }, 'manual'),
  ],
  sources: {},
  locations: [],
  snapshot_sha256: 'fixture',
};

const metrics = nodeMetrics(buildModel(input));

test('nodeMetrics averages the scored facts below a node and counts each chat once', () => {
  const attribute = metrics.get('attr-a')!;
  assert.equal(attribute.importance, 3);
  assert.equal(attribute.confidence, 3.5);
  assert.equal(attribute.support, 2);
  assert.equal(attribute.facts, 2);
  assert.equal(attribute.weight, 2 / 3);

  const topic = metrics.get('topic-a')!;
  assert.equal(topic.importance, 3);
  assert.equal(topic.support, 3);
  assert.equal(topic.facts, 3);
  assert.deepEqual(
    ['root', 'dimension:1', 'topic-a', 'sub-a', 'attr-b'].map((id) => metrics.get(id)!.depth),
    [0, 1, 2, 3, 4],
  );
  assert.equal(metrics.get('root')!.support, 4);
});

test('nodeMetrics leaves unscored nodes null and counts manual facts as support only', () => {
  const unscored = metrics.get('attr-b')!;
  assert.equal(unscored.importance, null);
  assert.equal(unscored.confidence, null);
  assert.equal(unscored.weight, 0.25);
  assert.equal(unscored.support, 1);

  const manual = metrics.get('attr-c')!;
  assert.equal(manual.importance, null);
  assert.equal(manual.support, 1);
  assert.equal(metrics.get('dimension:3')!.importance, null);
});

test('supportWidth grows with the log of support and inkStrength maps it onto 0…1', () => {
  assert.equal(supportWidth(0), 1);
  assert.equal(supportWidth(1), 1.42);
  assert.equal(supportWidth(-3), 1);
  assert.ok(supportWidth(100) > supportWidth(10));
  assert.equal(inkStrength(1), 0);
  assert.equal(inkStrength(6), 1);
  assert.ok(Math.abs(inkStrength(3.1) - 0.5) < 1e-9);
});
