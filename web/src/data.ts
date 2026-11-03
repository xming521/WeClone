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
  messages: { id: string; role: 'user' | 'assistant'; content: string; time: string | null }[];
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
// `token` is the design system's name for a dimension (`data-dim`, `--dim-<token>`). The colours are those
// tokens (web/src/ds/tokens.css, light theme); a canvas and the intro need plain values, so they are
// copied here and tests/graphTokens.test.ts checks them.
export const DIMENSIONS = [
  { dim: 1, token: 'identity', color: '#118f83' },
  { dim: 3, token: 'goal', color: '#7860c4' },
  { dim: 4, token: 'condition', color: '#c68331' },
  { dim: 5, token: 'plan', color: '#be5d6f' },
  { dim: 8, token: 'preference', color: '#3a76b4' },
] as const;

export type DimensionId = (typeof DIMENSIONS)[number]['dim'];

export function isDimensionId(dim: number | null | undefined): dim is DimensionId {
  return DIMENSIONS.some((entry) => entry.dim === dim);
}

export const ROOT_COLOR = '#626871'; // --ink-3: the centre has no colour of its own

// The domain view puts the domains where the dimensions are: a domain node has kind 'dimension',
// the id `dimension:<key>` and its key in `dim`, so the map clusters and colours it like a dimension.
// Keys follow the domain list and start above every dimension number.
const DOMAIN_KEY = 100;

// Section colours follow one rule, the design system's `[data-sec]` (web/src/ds/components.css), so
// CSS and canvas agree for any number of sections. Section i of n starts from --sec-base,
// oklch(0.585 0.11 182), and turns its hue 268° (--wc-sec-span) through teal, blue, violet and rose to
// gold, in equal steps, stopping before yellow-green. Odd sections are 0.09 lighter (--wc-sec-dl), as
// they sit on the outer ring; over the last 20% of the walk, which ends on warm yellow, lightness
// rises by up to 0.05 so gold does not turn grey. Chroma stays. The numbers were fitted to the
// 16-colour palette this view had before, which the rule reproduces at n = 16 to within 5/255 a
// channel, and are the only thing to change to tune it.
const SECTION = { l: 0.585, c: 0.11, h: 182, span: 268, odd: 0.09, lift: 0.05 };

/** Colour (#rrggbb, clipped to sRGB as browsers do) of section `index` of `count`; an index past the count extends it. */
export function sectionColor(index: number, count: number): string {
  const n = Math.max(count, index + 1);
  const t = index / Math.max(n - 1, 1);
  const l = SECTION.l + SECTION.odd * (index % 2) + SECTION.lift * Math.min(1, Math.max(0, (t - 0.8) / 0.2));
  const h = ((SECTION.h + SECTION.span * t) * Math.PI) / 180;
  const a = SECTION.c * Math.cos(h);
  const b = SECTION.c * Math.sin(h);
  // OKLab to linear sRGB (Björn Ottosson), then gamma.
  const l3 = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m3 = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s3 = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const linear = [
    4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
    -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
    -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3,
  ];
  const channel = (v: number) => {
    const c = Math.min(1, Math.max(0, v));
    return Math.round((c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055) * 255);
  };
  return `#${linear.map((v) => channel(v).toString(16).padStart(2, '0')).join('')}`;
}

/** The palette of 16 sections the rule was fitted to. */
export const DOMAIN_COLORS = Array.from({ length: 16 }, (_, index) => sectionColor(index, 16));

// How many sections the profile being shown lists. A section's colour depends on it, and
// `dimensionColor` has no model to ask, so `buildModel` records it when it builds the domain view.
let sectionCount = DOMAIN_COLORS.length;

/** A domain's place in the section palette (the `--i` and `--n` of `[data-sec]`); null for anything else. */
export function sectionPlace(dim: number | null | undefined): { index: number; count: number } | null {
  if (dim == null || dim < DOMAIN_KEY) return null;
  const index = dim - DOMAIN_KEY;
  return { index, count: Math.max(sectionCount, index + 1) };
}

// Unknown or missing dimensions fall back to the first dimension's colour.
export function dimensionColor(dim: number | null | undefined): string {
  const place = sectionPlace(dim);
  if (place) return sectionColor(place.index, place.count);
  return (DIMENSIONS.find((entry) => entry.dim === dim) ?? DIMENSIONS[0]).color;
}

// Match the uppercase role label, excluding Latin identifiers and hyphenated model names.
// Chinese exceptions are common terms, independent of profile samples; extend for new terms.
const PROFILE_ROLE_B =
  /(?<![A-Za-z0-9_]|[A-Za-z0-9][-－]|维生素)B(?![A-Za-z0-9_]|[-－][A-Za-z0-9]|站|超|型|股|细胞|族)/g;

export function displayProfileText(text: string): string {
  // eslint-disable-next-line no-restricted-syntax -- the profile's own wording, not UI copy
  return text.replace(PROFILE_ROLE_B, '我');
}

interface TopLevelInput {
  dim: number;
  name: string;
  description?: string;
  groups: GroupInput[];
}

/** Names the model gives its own nodes, in the UI language. Passed in because this module stays import-free. */
export interface ModelLabels {
  root: string;
  dimensions: Record<DimensionId, string>;
  /** The domain of attributes added straight to a dimension. */
  unassignedDomain: string;
  /** The topic that holds a domain's attributes without a topic. */
  directTopic: string;
}

/**
 * Domain → topic → attribute, regrouped from the dimension tree: the topics of one domain merge
 * across dimensions by name, and their attributes stay the stored nodes. Domains and topics follow
 * the domain list; ones missing from it come after, in the order they appear.
 */
function domainLevel(input: ProfileInput, labels: ModelLabels): TopLevelInput[] {
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
    if (!isDimensionId(dimension.dim)) continue;
    for (const group of dimension.groups) {
      // The server wraps attributes added straight to a dimension in a `:direct` group: they have no domain.
      const domain = domainOf(group.id.endsWith(':direct') ? labels.unassignedDomain : group.name);
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
  // The listed domains fix the palette, so colours do not shift between review views when an extra
  // one (attributes without a domain) is missing from one of them.
  sectionCount = input.domains.length || domains.size;
  const listedTopics = new Map(input.domains.map((domain) => [domain.name, domain.topics]));
  return [...domains.values()]
    .filter((domain) => domain.topics.size || domain.direct.length)
    .map((domain) => {
      const order = listedTopics.get(domain.name) ?? [];
      const rank = (name: string) => (order.includes(name) ? order.indexOf(name) : order.length);
      const groups = [...domain.topics.values()].sort((a, b) => rank(a.name) - rank(b.name));
      if (domain.direct.length)
        groups.push({ id: `dimension:${domain.key}:direct`, name: labels.directTopic, items: domain.direct });
      return { dim: domain.key, name: domain.name, description: domain.description, groups };
    });
}

export function buildModel(input: ProfileInput, labels: ModelLabels, grouping: Grouping = 'dimension'): ProfileModel {
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
      ? domainLevel(input, labels)
      : // Dimensions not listed in DIMENSIONS are skipped. Keep in sync with `PROFILE_DIMS` in
        // weclone/prompts/memory_organization.py. Their names and the `:direct` group's come from the labels,
        // not the server.
        input.dimensions.flatMap((dimension) =>
          isDimensionId(dimension.dim)
            ? [
                {
                  ...dimension,
                  name: labels.dimensions[dimension.dim],
                  groups: dimension.groups.map((group) =>
                    group.id.endsWith(':direct') ? { ...group, name: labels.directTopic } : group,
                  ),
                },
              ]
            : [],
        );
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
    name: labels.root,
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

/** Every privacy level a fact can have, the unrated ones last. */
export const PRIVACY_LEVELS: PrivacyLevel[] = [...SENSITIVITY_LEVELS, 'unrated'];

export function privacyLevel(fact: ProfileFact): PrivacyLevel {
  return fact.sensitivity && SENSITIVITY_LEVELS.includes(fact.sensitivity) ? fact.sensitivity : 'unrated';
}

/** Orders of the review list: as stored, least confident first, most important first, newest source first. */
export type ReviewSort = 'default' | 'confidence' | 'importance' | 'recent';
export const REVIEW_SORTS: ReviewSort[] = ['default', 'confidence', 'importance', 'recent'];

/** The newest source time of a fact; ISO times sort as text. */
export function latestSourceTime(fact: ProfileFact, sources: Record<string, ProfileSource> | undefined) {
  return fact.source_ids
    .map((id) => sources?.[id]?.sample_time)
    .filter((time): time is string => !!time)
    .sort()
    .at(-1);
}

/** The facts in the chosen order. Facts without the value sorted on go last; ties keep the stored order. */
export function sortFacts(
  facts: ProfileFact[],
  sort: ReviewSort,
  sources: Record<string, ProfileSource> | undefined,
): ProfileFact[] {
  if (sort === 'default') return facts;
  const key = (fact: ProfileFact): number | string | undefined =>
    sort === 'confidence'
      ? fact.source_confidence_mean
      : sort === 'importance'
        ? fact.source_importance_mean
        : latestSourceTime(fact, sources);
  // Ascending for confidence, descending for the others.
  const sign = sort === 'confidence' ? 1 : -1;
  const keyed = facts.map((fact) => ({ fact, key: key(fact) }));
  return keyed
    .sort((a, b) => {
      if (a.key === undefined || b.key === undefined) return a.key === b.key ? 0 : a.key === undefined ? 1 : -1;
      return a.key < b.key ? -sign : a.key > b.key ? sign : 0;
    })
    .map(({ fact }) => fact);
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
