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

export interface ChatSelection {
  id: string;
  label: string;
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

// Single source of truth for profile dimensions (ordered). Icons live in LeftPanel because data.ts must stay import-free.
export const DIMENSIONS = [
  { dim: 1, color: '#008f85' },
  { dim: 3, color: '#7859cf' },
  { dim: 4, color: '#d28b32' },
  { dim: 5, color: '#c35f72' },
  { dim: 8, color: '#3775b4' },
] as const;

export type DimensionId = (typeof DIMENSIONS)[number]['dim'];

export const ROOT_COLOR = '#59616d';

// Unknown or missing dimensions fall back to the first dimension's colour.
export function dimensionColor(dim: number | null | undefined): string {
  return (DIMENSIONS.find((entry) => entry.dim === dim) ?? DIMENSIONS[0]).color;
}

const CHINESE_NUMERALS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'];

export function chineseNumeral(n: number): string {
  return CHINESE_NUMERALS[n] ?? String(n);
}

// Match the uppercase role label, excluding Latin identifiers and hyphenated model names.
// Chinese exceptions are common terms, independent of profile samples; extend for new terms.
const PROFILE_ROLE_B =
  /(?<![A-Za-z0-9_]|[A-Za-z0-9][-－]|维生素)B(?![A-Za-z0-9_]|[-－][A-Za-z0-9]|站|超|型|股|细胞|族)/g;

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
    // Dimensions not listed in DIMENSIONS are skipped. Keep in sync with `dims` in
    // weclone/data/agent/llm_profile_hierarchy.py.
    if (!DIMENSIONS.some((entry) => entry.dim === dimension.dim)) continue;
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
  return (
    ancestors(previous, id)
      .reverse()
      .find((node) => model.nodes.has(node.id))?.id ?? 'root'
  );
}

export interface NavigationState {
  expandedIds: Set<string>;
  selectedId: string | null;
  focusedDimension: number | null;
}

// Keep the expanded nodes and selection that still exist after the model changes.
export function reconcileNavigation(
  previous: ProfileModel,
  next: ProfileModel,
  current: Pick<NavigationState, 'expandedIds' | 'selectedId'>,
): NavigationState {
  const expandedIds = new Set([...current.expandedIds].filter((id) => next.nodes.has(id)));
  expandedIds.add('root');
  next.dimensionIds.forEach((id) => {
    if (!previous.nodes.has(id)) expandedIds.add(id);
  });
  const selectedId = retainedSelection(previous, next, current.selectedId);
  const focusedDimension = selectedId ? (next.nodes.get(selectedId)?.dim ?? null) : null;
  return { expandedIds, selectedId, focusedDimension };
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
  return {
    ...input,
    facts,
    dimensions: input.dimensions.flatMap((dimension) => {
      const groups = dimension.groups.map(filter).filter((group): group is GroupInput => group !== null);
      return groups.length ? [{ ...dimension, groups }] : [];
    }),
  };
}

export function exportProfile(input: ProfileInput, view: ReviewView) {
  const profile = filterProfile(input, view);
  const sourceIds = new Set(profile.facts.flatMap((fact) => fact.source_ids));
  return {
    dimensions: profile.dimensions,
    facts: profile.facts.map(({ id, dim, attr, value, source_ids, origin, status }) => ({
      id,
      dim,
      attr,
      value,
      source_ids,
      origin,
      status,
    })),
    sources: Object.fromEntries(Object.entries(profile.sources).filter(([id]) => sourceIds.has(id))),
  };
}

export function collectFacts(model: ProfileModel, id: string): ProfileFact[] {
  const node = model.nodes.get(id);
  if (!node) return [];
  return node.kind === 'attribute'
    ? node.factIndices.map((index) => model.facts[index])
    : node.childIds.flatMap((child) => collectFacts(model, child));
}

export type Level = 1 | 2 | 3;

/** Depth of a node in the dimension > topic > subtopic hierarchy, null for root and attributes. */
export function levelOf(kind: NodeKind): Level | null {
  return kind === 'dimension' ? 1 : kind === 'topic' ? 2 : kind === 'subtopic' ? 3 : null;
}

/** Whether the node is a dimension, topic or subtopic at depth `level` or shallower. */
export function withinLevel(node: ProfileNode, level: Level): boolean {
  const depth = levelOf(node.kind);
  return depth !== null && depth <= level;
}

export function levelIsExpanded(
  model: ProfileModel,
  expandedIds: Set<string>,
  level: Level,
  dim: number | null,
): boolean {
  return (
    dim !== null &&
    expandedIds.has('root') &&
    [...model.nodes.values()]
      .filter((node) => node.dim === dim && node.childIds.length && withinLevel(node, level))
      .every((node) => expandedIds.has(node.id))
  );
}

/** Expanded set after toggling `level` of dimension `dim`: collapse it when fully expanded, else expand it. */
export function toggleLevelExpansion(
  model: ProfileModel,
  expandedIds: Set<string>,
  level: Level,
  dim: number,
): Set<string> {
  const next = new Set(expandedIds);
  if (levelIsExpanded(model, next, level, dim)) {
    for (const node of model.nodes.values()) {
      const depth = levelOf(node.kind);
      if (node.dim === dim && depth !== null && depth >= level) next.delete(node.id);
    }
  } else {
    next.add('root');
    for (const node of model.nodes.values()) {
      if (node.dim === dim && node.childIds.length && withinLevel(node, level)) next.add(node.id);
    }
  }
  return next;
}
