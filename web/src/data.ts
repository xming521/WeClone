export type ReviewStatus = 'pending' | 'approved' | 'rejected';
export type ReviewView = 'all' | ReviewStatus;

export interface ProfileFact {
  id: string;
  node_id: string;
  group_id: string;
  status: ReviewStatus;
  version: number;
  origin: 'extracted' | 'manual';
  original: { value: string; attr: string; dim?: number; source_ids: string[] };
  dim: number;
  attr: string;
  value: string;
  source_ids: string[];
  support_count?: number;
  source_confidence_mean?: number;
  source_importance_mean?: number;
}

export interface ProfileSource {
  id?: string;
  sample_time?: string;
  content?: string;
  confidence?: number;
  importance?: number;
}

export interface SourceChat {
  source_id: string;
  sample_id: string;
  chat_with: string;
  sample_time: string | null;
  messages: { id: string; role: 'user' | 'assistant'; speaker: string; content: string; time: string | null }[];
}

interface AttributeInput {
  id: string;
  attr: string;
  fact_indices: number[];
}

interface GroupInput {
  id: string;
  name: string;
  items: Array<GroupInput | AttributeInput>;
}

export interface ProfileInput {
  locations: { id: string; parent_id: string | null; kind: string; name: string; dim: number }[];
  snapshot_sha256: string;
  dimensions: { dim: number; name: string; groups: GroupInput[] }[];
  facts: ProfileFact[];
  sources: Record<string, ProfileSource>;
}

export type NodeKind = 'root' | 'dimension' | 'topic' | 'subtopic' | 'attribute';

export interface ProfileNode {
  id: string;
  name: string;
  kind: NodeKind;
  dim: number | null;
  parentId: string | null;
  childIds: string[];
  factIndices: number[];
  attributeCount: number;
}

export interface ProfileModel {
  nodes: Map<string, ProfileNode>;
  dimensionIds: string[];
  attributeIds: string[];
  facts: ProfileFact[];
  sources: Record<string, ProfileSource>;
}

export const DIMENSION_COLORS: Record<number, string> = {
  1: '#008f85',
  3: '#7859cf',
  4: '#d28b32',
  5: '#c35f72',
  8: '#3775b4',
};

// Match the uppercase role label, excluding Latin identifiers and hyphenated model names.
// Chinese exceptions are common terms, independent of profile samples; extend for new terms.
const PROFILE_ROLE_B = /(?<![A-Za-z0-9_]|[A-Za-z0-9][-－]|维生素)B(?![A-Za-z0-9_]|[-－][A-Za-z0-9]|站|超|型|股|细胞|族)/g;

export function displayProfileText(text: string): string {
  return text.replace(PROFILE_ROLE_B, '我');
}

export function buildModel(input: ProfileInput): ProfileModel {
  const nodes = new Map<string, ProfileNode>();
  const attributeIds: string[] = [];
  const dimensionIds: string[] = [];

  function addGroup(group: GroupInput, id: string, parentId: string, dim: number, kind: 'topic' | 'subtopic'): number {
    const childIds: string[] = [];
    let attributeCount = 0;
    for (const item of group.items) {
      const childId = item.id;
      childIds.push(childId);
      if ('attr' in item) {
        nodes.set(childId, {
          id: childId,
          name: item.attr,
          kind: 'attribute',
          dim,
          parentId: id,
          childIds: [],
          factIndices: item.fact_indices,
          attributeCount: 1,
        });
        attributeIds.push(childId);
        attributeCount += 1;
      } else {
        attributeCount += addGroup(item, childId, id, dim, 'subtopic');
      }
    }
    nodes.set(id, {
      id,
      name: group.name,
      kind,
      dim,
      parentId,
      childIds,
      factIndices: [],
      attributeCount,
    });
    return attributeCount;
  }

  for (const dimension of input.dimensions) {
    if (!(dimension.dim in DIMENSION_COLORS)) continue;
    const id = `dimension:${dimension.dim}`;
    const childIds = dimension.groups.map((group) => group.id);
    const attributeCount = dimension.groups.reduce(
      (count, group, index) => count + addGroup(group, childIds[index], id, dimension.dim, 'topic'),
      0,
    );
    nodes.set(id, {
      id,
      name: dimension.name,
      kind: 'dimension',
      dim: dimension.dim,
      parentId: 'root',
      childIds,
      factIndices: [],
      attributeCount,
    });
    dimensionIds.push(id);
  }

  nodes.set('root', {
    id: 'root',
    name: '我的画像',
    kind: 'root',
    dim: null,
    parentId: null,
    childIds: dimensionIds,
    factIndices: [],
    attributeCount: attributeIds.length,
  });

  return { nodes, dimensionIds, attributeIds, facts: input.facts, sources: input.sources };
}

export function ancestors(model: ProfileModel, id: string): ProfileNode[] {
  const path: ProfileNode[] = [];
  let current = model.nodes.get(id);
  while (current) {
    path.unshift(current);
    current = current.parentId ? model.nodes.get(current.parentId) : undefined;
  }
  return path;
}

export function retainedSelection(previous: ProfileModel, model: ProfileModel, id: string | null): string | null {
  if (id === null || model.nodes.has(id)) return id;
  return ancestors(previous, id).reverse().find((node) => model.nodes.has(node.id))?.id ?? 'root';
}

export function visibleNodes(model: ProfileModel, expandedIds: Set<string>): ProfileNode[] {
  const visible: ProfileNode[] = [];
  function visit(id: string) {
    const node = model.nodes.get(id)!;
    visible.push(node);
    if (expandedIds.has(id)) {
      for (const childId of node.childIds) visit(childId);
    }
  }

  visit('root');
  return visible;
}

export function filterProfile(input: ProfileInput, view: ReviewView): ProfileInput {
  const facts = input.facts.filter((fact) => view === 'all' || fact.status === view);
  const indices = new Map(facts.map((fact, index) => [fact.id, index]));
  function filter(group: GroupInput): GroupInput | null {
    const items: GroupInput['items'] = [];
    for (const item of group.items) {
      if ('attr' in item) {
        const refs = item.fact_indices.flatMap((index) => {
          const next = indices.get(input.facts[index].id);
          return next === undefined ? [] : [next];
        });
        if (refs.length) items.push({ ...item, fact_indices: refs });
      } else {
        const next = filter(item);
        if (next) items.push(next);
      }
    }
    return items.length ? { ...group, items } : null;
  }
  return { ...input, facts, dimensions: input.dimensions.flatMap((dimension) => {
    const groups = dimension.groups.map(filter).filter((group): group is GroupInput => group !== null);
    return groups.length ? [{ ...dimension, groups }] : [];
  }) };
}

export function exportProfile(input: ProfileInput, view: ReviewView) {
  const profile = filterProfile(input, view);
  const sourceIds = new Set(profile.facts.flatMap((fact) => fact.source_ids));
  return {
    dimensions: profile.dimensions,
    facts: profile.facts.map(({ id, dim, attr, value, source_ids, origin }) => ({ id, dim, attr, value, source_ids, origin })),
    sources: Object.fromEntries(Object.entries(profile.sources).filter(([id]) => sourceIds.has(id))),
  };
}

export function collectFacts(model: ProfileModel, id: string): ProfileFact[] {
  const node = model.nodes.get(id);
  if (!node) return [];
  return node.kind === 'attribute' ? node.factIndices.map((index) => model.facts[index])
    : node.childIds.flatMap((child) => collectFacts(model, child));
}
