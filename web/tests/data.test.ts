import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  buildModel,
  DOMAIN_COLORS,
  chineseNumeral,
  collectFacts,
  dimensionColor,
  exportProfile,
  filterProfile,
  levelIsExpanded,
  levelOf,
  reconcileNavigation,
  retainedSelection,
  toggleLevelExpansion,
  withinLevel,
} from '../src/data.ts';
import type { ProfileInput } from '../src/data.ts';

const input: ProfileInput = {
  dimensions: [
    {
      dim: 8,
      name: '兴趣',
      groups: [
        { id: 'topic-a', name: '运动', items: [{ id: 'attr-a', attr: '偏好', fact_indices: [0, 1] }] },
        { id: 'topic-b', name: '阅读', items: [{ id: 'attr-b', attr: '书籍', fact_indices: [2] }] },
      ],
    },
  ],
  facts: ['pending', 'approved', 'rejected'].map((status, i) => ({
    id: `fact-${i}`,
    node_id: i === 2 ? 'attr-b' : 'attr-a',
    group_id: i === 2 ? 'topic-b' : 'topic-a',
    dim: 8,
    attr: i === 2 ? '书籍' : '偏好',
    value: `内容${i}`,
    status: status as 'pending' | 'approved' | 'rejected',
    source_ids: [],
    origin: 'extracted',
    version: 1,
    original: { attr: '偏好', value: `内容${i}`, source_ids: [] },
  })),
  sources: {},
  locations: [],
  snapshot_sha256: 'fixture',
};

test('status filtering prunes branches and remaps fact indices without changing node identity', () => {
  const model = buildModel(filterProfile(input, 'approved'));
  assert.deepEqual(model.attributeIds, ['attr-a']);
  assert.deepEqual(model.nodes.get('attr-a')!.factIndices, [0]);
  assert.deepEqual(
    collectFacts(model, 'root').map((fact) => fact.id),
    ['fact-1'],
  );
  assert.equal(model.nodes.has('topic-b'), false);
  assert.equal(model.nodes.get('root')!.attributeCount, 1);
  assert.equal(input.facts.length, 3);
  assert.deepEqual(input.dimensions[0].groups[0].items[0], { id: 'attr-a', attr: '偏好', fact_indices: [0, 1] });
});

test('all view retains independently reviewed facts under the same attribute', () => {
  const model = buildModel(filterProfile(input, 'all'));
  assert.deepEqual(
    collectFacts(model, 'attr-a').map((fact) => fact.status),
    ['pending', 'approved'],
  );
  assert.equal(model.nodes.get('root')!.attributeCount, 2);
  assert.equal(collectFacts(model, 'root').length, 3);
});

test('export carries each fact review status and the approved view exports only approved facts', () => {
  const all = exportProfile(input, 'all');
  assert.deepEqual(
    all.facts.map((fact) => [fact.id, fact.status]),
    [
      ['fact-0', 'pending'],
      ['fact-1', 'approved'],
      ['fact-2', 'rejected'],
    ],
  );
  const approved = exportProfile(input, 'approved');
  assert.deepEqual(
    approved.facts.map((fact) => [fact.id, fact.status]),
    [['fact-1', 'approved']],
  );
});

test('empty review view yields a valid root without empty dimensions or stale indices', () => {
  const model = buildModel(
    filterProfile({ ...input, facts: input.facts.map((fact) => ({ ...fact, status: 'pending' })) }, 'approved'),
  );
  assert.deepEqual(model.dimensionIds, []);
  assert.deepEqual(collectFacts(model, 'root'), []);
  assert.equal(model.nodes.size, 1);
});

test('review removal returns to the nearest surviving ancestor without selecting another branch', () => {
  const previous = buildModel(input);
  const next = buildModel(filterProfile(input, 'approved'));
  assert.equal(retainedSelection(previous, next, 'attr-a'), 'attr-a');
  assert.equal(retainedSelection(previous, next, 'attr-b'), 'dimension:8');
  assert.equal(retainedSelection(previous, next, 'topic-b'), 'dimension:8');
  assert.equal(retainedSelection(previous, next, null), null);

  const withoutAttribute = { ...next, nodes: new Map(next.nodes) };
  withoutAttribute.nodes.delete('attr-a');
  assert.equal(retainedSelection(next, withoutAttribute, 'attr-a'), 'topic-a');

  const empty = buildModel({ ...input, dimensions: [], facts: [] });
  assert.equal(retainedSelection(previous, empty, 'attr-b'), 'root');
});

function reconcileFixture(): ProfileInput {
  const fact = (i: number, status: 'pending' | 'approved', dim: number, node: string, group: string) => ({
    id: `fact-${i}`,
    node_id: node,
    group_id: group,
    dim,
    attr: node,
    value: `内容${i}`,
    status,
    source_ids: [],
    origin: 'extracted' as const,
    version: 1,
    original: { attr: node, value: `内容${i}`, source_ids: [] },
  });
  return {
    dimensions: [
      {
        dim: 8,
        name: '兴趣',
        groups: [{ id: 'topic-a', name: '运动', items: [{ id: 'attr-a', attr: '偏好', fact_indices: [0] }] }],
      },
      {
        dim: 3,
        name: '目标',
        groups: [{ id: 'topic-c', name: '学习', items: [{ id: 'attr-c', attr: '计划', fact_indices: [1] }] }],
      },
    ],
    facts: [fact(0, 'pending', 8, 'attr-a', 'topic-a'), fact(1, 'approved', 3, 'attr-c', 'topic-c')],
    sources: {},
    locations: [],
    snapshot_sha256: 'fixture',
  };
}

test('navigation drops expanded nodes that no longer exist and keeps the rest', () => {
  const full = buildModel(reconcileFixture());
  const pendingOnly = buildModel(filterProfile(reconcileFixture(), 'pending'));
  const expandedIds = new Set(['root', 'dimension:8', 'dimension:3', 'topic-c', 'topic-a']);
  const next = reconcileNavigation(full, pendingOnly, { expandedIds, selectedId: 'root' });
  assert.deepEqual([...next.expandedIds].sort(), ['dimension:8', 'root', 'topic-a']);
  assert.equal(expandedIds.size, 5);
});

test('navigation expands dimensions that newly appear and always keeps root expanded', () => {
  const full = buildModel(reconcileFixture());
  const pendingOnly = buildModel(filterProfile(reconcileFixture(), 'pending'));
  const next = reconcileNavigation(pendingOnly, full, { expandedIds: new Set(['dimension:8']), selectedId: null });
  assert.deepEqual([...next.expandedIds].sort(), ['dimension:3', 'dimension:8', 'root']);
  assert.equal(next.selectedId, null);
  assert.equal(next.focusedDimension, null);
});

test('navigation selection falls back to the nearest surviving ancestor and refocuses its dimension', () => {
  const full = buildModel(reconcileFixture());
  const pendingOnly = buildModel(filterProfile(reconcileFixture(), 'pending'));
  const kept = reconcileNavigation(full, pendingOnly, { expandedIds: new Set(), selectedId: 'attr-a' });
  assert.equal(kept.selectedId, 'attr-a');
  assert.equal(kept.focusedDimension, 8);
  const nearest = reconcileNavigation(buildModel(input), buildModel(filterProfile(input, 'approved')), {
    expandedIds: new Set(),
    selectedId: 'attr-b',
  });
  assert.equal(nearest.selectedId, 'dimension:8');
  assert.equal(nearest.focusedDimension, 8);
  const fallback = reconcileNavigation(full, pendingOnly, { expandedIds: new Set(), selectedId: 'attr-c' });
  assert.equal(fallback.selectedId, 'root');
  assert.equal(fallback.focusedDimension, null);
  const empty = buildModel(filterProfile(reconcileFixture(), 'rejected'));
  const toRoot = reconcileNavigation(full, empty, { expandedIds: new Set(['dimension:8']), selectedId: 'attr-a' });
  assert.equal(toRoot.selectedId, 'root');
  assert.deepEqual([...toRoot.expandedIds], ['root']);
});

function deepFixture(): ProfileInput {
  return {
    ...input,
    dimensions: [
      {
        dim: 8,
        name: '兴趣',
        groups: [
          {
            id: 'topic-a',
            name: '运动',
            items: [
              { id: 'sub-a', name: '球类', items: [{ id: 'attr-a', attr: '偏好', fact_indices: [0] }] },
              { id: 'attr-top', attr: '频率', fact_indices: [1] },
            ],
          },
        ],
      },
      {
        dim: 3,
        name: '目标',
        groups: [{ id: 'topic-c', name: '学习', items: [{ id: 'attr-c', attr: '计划', fact_indices: [2] }] }],
      },
    ],
  };
}

test('levelOf and withinLevel place only dimension, topic and subtopic nodes in the hierarchy', () => {
  const model = buildModel(deepFixture());
  assert.deepEqual(
    ['root', 'dimension:8', 'topic-a', 'sub-a', 'attr-a'].map((id) => levelOf(model.nodes.get(id)!.kind)),
    [null, 1, 2, 3, null],
  );
  assert.equal(withinLevel(model.nodes.get('topic-a')!, 1), false);
  assert.equal(withinLevel(model.nodes.get('topic-a')!, 2), true);
  assert.equal(withinLevel(model.nodes.get('sub-a')!, 2), false);
  assert.equal(withinLevel(model.nodes.get('sub-a')!, 3), true);
  assert.equal(withinLevel(model.nodes.get('root')!, 3), false);
  assert.equal(withinLevel(model.nodes.get('attr-a')!, 3), false);
});

test('levelIsExpanded requires every branching node of the dimension down to that level', () => {
  const model = buildModel(deepFixture());
  const expanded = (...ids: string[]) => new Set(['root', ...ids]);
  assert.equal(levelIsExpanded(model, expanded(), 1, 8), false);
  assert.equal(levelIsExpanded(model, expanded('dimension:8'), 1, 8), true);
  assert.equal(levelIsExpanded(model, expanded('dimension:8'), 2, 8), false);
  assert.equal(levelIsExpanded(model, expanded('dimension:8', 'topic-a'), 2, 8), true);
  assert.equal(levelIsExpanded(model, expanded('dimension:8', 'topic-a'), 3, 8), false);
  assert.equal(levelIsExpanded(model, expanded('dimension:8', 'topic-a', 'sub-a'), 3, 8), true);
  // Other dimensions do not count.
  assert.equal(levelIsExpanded(model, expanded('dimension:3'), 1, 8), false);
  assert.equal(levelIsExpanded(model, expanded('dimension:8', 'topic-a', 'sub-a'), 3, 3), false);
});

test('levelIsExpanded is false without a focused dimension or an expanded root', () => {
  const model = buildModel(deepFixture());
  const all = new Set(['root', 'dimension:8', 'dimension:3', 'topic-a', 'topic-c', 'sub-a']);
  assert.equal(levelIsExpanded(model, all, 3, null), false);
  all.delete('root');
  for (const level of [1, 2, 3] as const) assert.equal(levelIsExpanded(model, all, level, 8), false);
});

test('toggling a level expands branching nodes down to it, then collapses it and the deeper levels', () => {
  const model = buildModel(deepFixture());
  const start = new Set(['dimension:3']);
  const two = toggleLevelExpansion(model, start, 2, 8);
  assert.deepEqual([...two].sort(), ['dimension:3', 'dimension:8', 'root', 'topic-a']);
  assert.deepEqual([...start], ['dimension:3']);
  const three = toggleLevelExpansion(model, two, 3, 8);
  assert.deepEqual([...three].sort(), ['dimension:3', 'dimension:8', 'root', 'sub-a', 'topic-a']);
  // Collapsing level 3 leaves the shallower levels, collapsing level 2 also drops subtopics.
  assert.deepEqual([...toggleLevelExpansion(model, three, 3, 8)].sort(), [
    'dimension:3',
    'dimension:8',
    'root',
    'topic-a',
  ]);
  assert.deepEqual([...toggleLevelExpansion(model, three, 2, 8)].sort(), ['dimension:3', 'dimension:8', 'root']);
  assert.deepEqual([...toggleLevelExpansion(model, three, 1, 8)].sort(), ['dimension:3', 'root']);
});

test('dimensionColor returns configured colours and falls back to dimension 1', () => {
  assert.equal(dimensionColor(1), '#008f85');
  assert.equal(dimensionColor(8), '#3775b4');
  assert.equal(dimensionColor(2), '#008f85');
  assert.equal(dimensionColor(null), '#008f85');
});

test('chineseNumeral renders small counts', () => {
  assert.equal(chineseNumeral(5), '五');
  assert.equal(chineseNumeral(10), '十');
  assert.equal(chineseNumeral(16), '十六');
  assert.equal(chineseNumeral(20), '二十');
  assert.equal(chineseNumeral(100), '100');
});

test('buildModel skips dimensions that are not configured', () => {
  const model = buildModel({
    ...input,
    dimensions: [
      ...input.dimensions,
      {
        dim: 2,
        name: '未配置',
        groups: [{ id: 'topic-x', name: '其他', items: [{ id: 'attr-x', attr: 'x', fact_indices: [] }] }],
      },
    ],
  });
  assert.deepEqual(model.dimensionIds, ['dimension:8']);
  assert.equal(model.nodes.has('dimension:2'), false);
  assert.equal(model.nodes.has('attr-x'), false);
});

// Two dimensions organized by domain, as build-profile writes them, plus a manual fact on dimension 8.
const domainInput: ProfileInput = {
  ...input,
  domains: [
    { name: '工作', description: 'B的工作。', topics: ['求职', '职场'] },
    { name: '空领域', description: '', topics: [] },
    { name: '兴趣', description: 'B的爱好。', topics: ['运动'] },
  ],
  dimensions: [
    {
      dim: 3,
      name: '目标',
      groups: [
        {
          id: 'd3-work',
          name: '工作',
          items: [
            { id: 'd3-work-office', name: '职场', items: [{ id: 'attr-goal', attr: '目标', fact_indices: [0] }] },
            { id: 'd3-work-jobs', name: '求职', items: [{ id: 'attr-plan', attr: '计划', fact_indices: [1] }] },
          ],
        },
      ],
    },
    {
      dim: 8,
      name: '兴趣',
      groups: [
        {
          id: 'd8-work',
          name: '工作',
          items: [
            { id: 'd8-work-office', name: '职场', items: [{ id: 'attr-like', attr: '喜好', fact_indices: [2] }] },
          ],
        },
        {
          id: 'd8-hobby',
          name: '兴趣',
          items: [
            { id: 'd8-hobby-sport', name: '运动', items: [{ id: 'attr-sport', attr: '运动', fact_indices: [3] }] },
          ],
        },
        { id: 'dimension:8:direct', name: '直属属性', items: [{ id: 'attr-manual', attr: '手作', fact_indices: [4] }] },
      ],
    },
  ],
  facts: Array.from({ length: 5 }, (_, i) => ({
    ...input.facts[0],
    id: `fact-${i}`,
    status: i === 3 ? ('approved' as const) : ('pending' as const),
  })),
};

test('domain grouping merges topics across dimensions and keeps the stored attributes', () => {
  const model = buildModel(domainInput, 'domain');
  assert.equal(model.grouping, 'domain');
  assert.deepEqual(
    model.dimensionIds.map((id) => model.nodes.get(id)!.name),
    ['工作', '兴趣', '未归入领域'],
  );
  const work = model.nodes.get(model.dimensionIds[0])!;
  assert.equal(work.kind, 'dimension');
  assert.equal(work.description, 'B的工作。');
  // Listed topic order, with the topic shared by two dimensions merged into one node.
  assert.deepEqual(
    work.childIds.map((id) => model.nodes.get(id)!.name),
    ['求职', '职场'],
  );
  const office = model.nodes.get(work.childIds[1])!;
  assert.deepEqual(office.childIds, ['attr-goal', 'attr-like']);
  assert.equal(model.nodes.get('attr-like')!.parentId, office.id);
  assert.equal(model.nodes.get('attr-like')!.dim, work.dim);
  // Domains keep their list position as key, so colours do not shift when an empty domain is hidden.
  assert.equal(model.nodes.get(model.dimensionIds[1])!.dim, 102);
  assert.equal(dimensionColor(102), DOMAIN_COLORS[2]);
  const unassigned = model.nodes.get(model.dimensionIds[2])!;
  assert.deepEqual(
    collectFacts(model, unassigned.id).map((fact) => fact.id),
    ['fact-4'],
  );
  assert.equal(model.attributeIds.length, 5);
  assert.equal(model.nodes.get('root')!.attributeCount, 5);
});

test('domain grouping keeps node identity across review views and needs a domain list', () => {
  const all = buildModel(domainInput, 'domain');
  const approved = buildModel(filterProfile(domainInput, 'approved'), 'domain');
  assert.deepEqual(approved.dimensionIds, [all.dimensionIds[1]]);
  assert.equal(approved.nodes.get('attr-sport')!.parentId, all.nodes.get('attr-sport')!.parentId);
  const plain = buildModel({ ...domainInput, domains: undefined }, 'domain');
  assert.equal(plain.grouping, 'dimension');
  assert.deepEqual(plain.dimensionIds, ['dimension:3', 'dimension:8']);
});
