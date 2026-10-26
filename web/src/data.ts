export type ReviewStatus = 'pending' | 'approved' | 'rejected';
export type ReviewView = 'all' | ReviewStatus;
/** What the first level of the map holds: the five dimensions, or the person's life domains. */
export type Grouping = 'dimension' | 'domain';

/**
 * Privacy level of a fact or a source memory: whom the person would tell. Scale from
 * `SENSITIVITY_RULES` in weclone/prompts/memory_organization.py; profiles built before it have none.
 */
export type Sensitivity = 'public' | 'normal' | 'private' | 'sensitive';
/** The levels, least private first. */
export const SENSITIVITY_LEVELS: Sensitivity[] = ['public', 'normal', 'private', 'sensitive'];
/** What the review list filters on: a fact's level, or 'unrated' when it has none. */
export type PrivacyLevel = Sensitivity | 'unrated';

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
  sensitivity?: Sensitivity;
}

export interface ProfileSource {
  id?: string;
  sample_time?: string;
  content?: string;
  confidence?: number;
  importance?: number;
  /** Retrieval tags the extraction gave the memory. */
  tags?: string[];
  sensitivity?: Sensitivity;
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

export interface DomainInput {
  name: string;
  description: string;
  topics: string[];
}

export interface ProfileInput {
  locations: { id: string; parent_id: string | null; kind: string; name: string; dim: number }[];
  snapshot_sha256: string;
  dimensions: { dim: number; name: string; groups: GroupInput[] }[];
  /** Each dimension groups by domain, then topic. This list gives the domains' order, descriptions and topics. */
  domains: DomainInput[];
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
  description?: string;
}

export interface ProfileModel {
  grouping: Grouping;
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

// The domain view puts the domains where the dimensions are: a domain node has kind 'dimension',
// the id `dimension:<key>` and its key in `dim`, so the map clusters and colours it like a dimension.
// Keys follow the domain list and start above every dimension number.
const DOMAIN_KEY = 100;

// Colours in domain-list order, at one chroma in OKLCH. The hue walks from teal through blue,
// violet and rose to gold, so neighbouring domains, which the list keeps related, get related
// colours; it skips yellow-green, which turns muddy at this lightness. Lightness alternates
// between about 0.59 and 0.68, as the domains alternate between the inner and the outer ring.
// Lists longer than this repeat.
export const DOMAIN_COLORS = [
  '#058f80',
  '#1cabb1',
  '#058aa4',
  '#49a1d1',
  '#4a7ebb',
  '#7e92db',
  '#7a70b9',
  '#a984cc',
  '#9c64a0',
  '#c67ba9',
  '#b05e78',
  '#d27a7c',
  '#b3624c',
  '#cf8755',
  '#ae7a30',
  '#c2a44f',
];

// Unknown or missing dimensions fall back to the first dimension's colour.
export function dimensionColor(dim: number | null | undefined): string {
  if (dim != null && dim >= DOMAIN_KEY) return DOMAIN_COLORS[(dim - DOMAIN_KEY) % DOMAIN_COLORS.length];
  return (DIMENSIONS.find((entry) => entry.dim === dim) ?? DIMENSIONS[0]).color;
}

const CHINESE_NUMERALS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'];

export function chineseNumeral(n: number): string {
  if (!Number.isInteger(n) || n < 0 || n >= 100) return String(n);
  if (n <= 10) return CHINESE_NUMERALS[n];
  const tens = Math.floor(n / 10);
  return `${tens > 1 ? CHINESE_NUMERALS[tens] : ''}十${n % 10 ? CHINESE_NUMERALS[n % 10] : ''}`;
}

// Match the uppercase role label, excluding Latin identifiers and hyphenated model names.
// Chinese exceptions are common terms, independent of profile samples; extend for new terms.
const PROFILE_ROLE_B =
  /(?<![A-Za-z0-9_]|[A-Za-z0-9][-－]|维生素)B(?![A-Za-z0-9_]|[-－][A-Za-z0-9]|站|超|型|股|细胞|族)/g;

export function displayProfileText(text: string): string {
  return text.replace(PROFILE_ROLE_B, '我');
}

interface TopLevelInput {
  dim: number;
  name: string;
  description?: string;
  groups: GroupInput[];
}

const UNASSIGNED_DOMAIN = '未归入领域';
const DIRECT_TOPIC = '直属属性';

/**
 * Domain → topic → attribute, regrouped from the dimension tree: the topics of one domain merge
 * across dimensions by name, and their attributes stay the stored nodes. Domains and topics follow
 * the domain list; ones missing from it come after, in the order they appear.
 */
function domainLevel(input: ProfileInput): TopLevelInput[] {
  interface Domain {
    key: number;
    name: string;
    description?: string;
    topics: Map<string, GroupInput>;
    direct: AttributeInput[];
  }
  const domains = new Map<string, Domain>();
  const domainOf = (name: string, description?: string) => {
    let domain = domains.get(name);
    if (!domain) {
      domain = { key: DOMAIN_KEY + domains.size, name, description, topics: new Map(), direct: [] };
      domains.set(name, domain);
    }
    return domain;
  };
  // Listed domains take their keys first, so a domain keeps its id and colour in every review view.
  for (const domain of input.domains) domainOf(domain.name, domain.description);
  for (const dimension of input.dimensions) {
    if (!DIMENSIONS.some((entry) => entry.dim === dimension.dim)) continue;
    for (const group of dimension.groups) {
      // The server wraps attributes added straight to a dimension in a `:direct` group: they have no domain.
      const domain = domainOf(group.id.endsWith(':direct') ? UNASSIGNED_DOMAIN : group.name);
      for (const item of group.items) {
        if ('attr' in item) {
          domain.direct.push(item);
          continue;
        }
        const topic = domain.topics.get(item.name);
        if (topic) topic.items.push(...item.items);
        else
          domain.topics.set(item.name, {
            id: `dimension:${domain.key}/${item.name}`,
            name: item.name,
            items: [...item.items],
          });
      }
    }
  }
  const listedTopics = new Map(input.domains.map((domain) => [domain.name, domain.topics]));
  return [...domains.values()]
    .filter((domain) => domain.topics.size || domain.direct.length)
    .map((domain) => {
      const order = listedTopics.get(domain.name) ?? [];
      const rank = (name: string) => (order.includes(name) ? order.indexOf(name) : order.length);
      const groups = [...domain.topics.values()].sort((a, b) => rank(a.name) - rank(b.name));
      if (domain.direct.length)
        groups.push({ id: `dimension:${domain.key}:direct`, name: DIRECT_TOPIC, items: domain.direct });
      return { dim: domain.key, name: domain.name, description: domain.description, groups };
    });
}

export function buildModel(input: ProfileInput, grouping: Grouping = 'dimension'): ProfileModel {
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

  const topLevel: TopLevelInput[] =
    grouping === 'domain'
      ? domainLevel(input)
      : // Dimensions not listed in DIMENSIONS are skipped. Keep in sync with `PROFILE_DIMS` in
        // weclone/prompts/memory_organization.py.
        input.dimensions.filter((dimension) => DIMENSIONS.some((entry) => entry.dim === dimension.dim));
  for (const dimension of topLevel) {
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
      description: dimension.description,
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

  return { grouping, nodes, dimensionIds, attributeIds, facts: input.facts, sources: input.sources };
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

/** Tags of the fact's sources, those more of its sources carry first; ties keep their first appearance. */
export function factTags(fact: ProfileFact, sources: Record<string, ProfileSource> | undefined): string[] {
  const counts = new Map<string, number>();
  for (const id of fact.source_ids) {
    for (const tag of new Set(sources?.[id]?.tags ?? [])) counts.set(tag, (counts.get(tag) ?? 0) + 1);
  }
  return [...counts].sort((a, b) => b[1] - a[1]).map(([tag]) => tag);
}

export function privacyLevel(fact: ProfileFact): PrivacyLevel {
  return fact.sensitivity && SENSITIVITY_LEVELS.includes(fact.sensitivity) ? fact.sensitivity : 'unrated';
}

/** Facts whose level is one of `levels`; no levels keeps them all. */
export function filterPrivacy(facts: ProfileFact[], levels: PrivacyLevel[]): ProfileFact[] {
  return levels.length ? facts.filter((fact) => levels.includes(privacyLevel(fact))) : facts;
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
