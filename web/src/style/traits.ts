import type zhCN from '../i18n/locales/zh-CN/translation.json';
import type { StyleContact, StyleInput, StyleLeaf, VariationClass } from './types';

export type TraitGroupId = 'typing' | 'words' | 'emotion' | 'manner';

// Names, notes and units are in the catalog under `style`, by these ids.
export const TRAIT_GROUPS: { id: TraitGroupId; color: string }[] = [
  { id: 'typing', color: '#008f85' },
  { id: 'words', color: '#d28b32' },
  { id: 'emotion', color: '#c35f72' },
  { id: 'manner', color: '#3775b4' },
];

/** A trait is shown only when it appears at least once in 25 messages. */
export const MIN_RATE = 0.04;
/** Above this share of undecided messages the rate could sit almost anywhere. */
const MAX_MISSING = 0.3;

// Bubbles are shaded by how often the trait shows up.
export const TIERS = [
  { id: 'most', min: 0.5 },
  { id: 'often', min: 0.2 },
  { id: 'sometimes', min: 0.1 },
  { id: 'rarely', min: 0 },
] as const;

export type Tier = (typeof TIERS)[number];

export function tierOf(rate: number): Tier {
  return TIERS.find((tier) => rate >= tier.min) ?? TIERS[TIERS.length - 1];
}

export type TraitId = keyof typeof zhCN.style.traits;
/** What the denominator counts: whole messages unless a trait says otherwise. */
export type TraitUnit = 'messages' | 'boundaries' | 'bubbles' | 'stances' | 'requests';

// Titles name the habit without saying how often, since the rate differs from person to person.
interface TraitSpec {
  /** Its title and note in the catalog. */
  id: TraitId;
  item: string;
  /** Rate leaves added into one rate; they share one denominator. */
  paths: string[];
  group: TraitGroupId;
  /** The situation the rate is counted in, when it is not every message. */
  when?: keyof typeof zhCN.style.when;
  unit?: TraitUnit;
  /** Distribution whose most frequent entries are shown as examples. */
  forms?: string;
}

const SPECS: TraitSpec[] = [
  {
    id: 'lineBreakPunctuation',
    item: '088',
    paths: ['surface/unpunctuated_boundary_rate'],
    group: 'typing',
    unit: 'boundaries',
  },
  {
    id: 'noPunctuation',
    item: '086',
    paths: ['surface/bubble_no_punctuation_rate'],
    group: 'typing',
    unit: 'bubbles',
  },
  {
    id: 'splitMessages',
    item: 'burst_pattern',
    paths: ['surface/message_newline_burst_rate'],
    group: 'typing',
  },
  {
    id: 'shortMessages',
    item: '218',
    paths: [
      'surface/message_length_shares/1-2',
      'surface/message_length_shares/3-4',
      'surface/message_length_shares/5-8',
    ],
    group: 'typing',
  },
  {
    id: 'modalParticles',
    item: '097-098',
    paths: ['surface/message_rate'],
    group: 'words',
    forms: 'surface/forms',
  },
  {
    id: 'repeatedCharacters',
    item: '106-107,109',
    paths: ['surface/message_rate'],
    group: 'words',
    forms: 'surface/forms',
  },
  {
    id: 'internetSlang',
    item: '007',
    paths: ['contextual/message_occurrence_rate'],
    group: 'words',
    forms: 'contextual/expressions',
  },
  {
    id: 'jargon',
    item: '009',
    paths: ['contextual/message_occurrence_rate'],
    group: 'words',
    forms: 'contextual/expressions',
  },
  {
    id: 'degreeWords',
    item: '163',
    paths: ['contextual/message_rate'],
    group: 'words',
    forms: 'surface/forms',
  },
  {
    id: 'reduplication',
    item: '016',
    paths: ['contextual/message_occurrence_rate'],
    group: 'words',
    forms: 'contextual/expressions',
  },
  {
    id: 'profanity',
    item: '245',
    paths: ['contextual/message_occurrence_rate'],
    group: 'words',
    forms: 'contextual/forms',
  },
  {
    id: 'emotional',
    item: '153,160',
    paths: ['contextual/message_rate'],
    group: 'emotion',
  },
  {
    id: 'implicitEmotion',
    item: '153,160',
    paths: ['contextual/modes/message_rates/implicit'],
    group: 'emotion',
  },
  {
    id: 'symbolicEmotion',
    item: '153,160',
    paths: ['contextual/modes/message_rates/symbolic'],
    group: 'emotion',
    forms: 'contextual/forms',
  },
  {
    id: 'laughter',
    item: '099-105',
    paths: ['contextual/message_rate'],
    group: 'emotion',
    forms: 'contextual/forms',
  },
  {
    id: 'grumbling',
    item: '158',
    paths: ['contextual/message_rate'],
    group: 'emotion',
    forms: 'contextual/forms',
  },
  {
    id: 'snark',
    item: '159',
    paths: ['contextual/message_rate'],
    group: 'emotion',
    forms: 'contextual/forms',
  },
  {
    id: 'teasing',
    item: '174-175',
    paths: ['175/contextual/message_occurrence_rate'],
    group: 'emotion',
  },
  {
    id: 'direct',
    item: '138',
    paths: ['contextual/attributes/directness/message_rates/explicit'],
    group: 'manner',
    when: 'stance',
    unit: 'stances',
  },
  {
    id: 'matterOfFact',
    item: '228',
    paths: ['contextual/label_message_rates/matter'],
    group: 'manner',
  },
  {
    id: 'suggestions',
    item: '136-137',
    paths: ['contextual/attributes/act/message_rates/suggestion'],
    group: 'manner',
    when: 'request',
    unit: 'requests',
  },
  {
    id: 'questions',
    item: '052,211',
    paths: ['contextual/message_occurrence_rate'],
    group: 'manner',
  },
  {
    id: 'followUps',
    item: '210',
    paths: ['contextual/message_occurrence_rate'],
    group: 'manner',
  },
  {
    id: 'elaborates',
    item: '219',
    paths: ['contextual/message_occurrence_rate'],
    group: 'manner',
  },
  {
    id: 'advice',
    item: '256',
    paths: ['contextual/message_advice_occurrence_rate'],
    group: 'manner',
  },
  {
    id: 'hedging',
    item: '120-127',
    paths: ['contextual/message_rate'],
    group: 'manner',
    forms: 'contextual/forms',
  },
  {
    id: 'emphasis',
    item: '146-150',
    paths: ['contextual/message_rate'],
    group: 'manner',
    forms: 'contextual/forms',
  },
  {
    id: 'agreement',
    item: '231',
    paths: ['contextual/message_occurrence_rate'],
    group: 'manner',
  },
  {
    id: 'metaphors',
    item: '166',
    paths: ['contextual/message_occurrence_rate'],
    group: 'manner',
  },
];

export interface ContactRate {
  contact: StyleContact;
  rate: number;
}

export interface Trait extends TraitSpec {
  key: string;
  rate: number;
  messages: number;
  sampled: boolean;
  examples: string[];
  variation: VariationClass | null;
  /** Tested contacts, most frequent first, rates adjusted for message length. */
  contacts: ContactRate[];
}

// Single punctuation marks and multi-line quotes make poor examples.
const LONE_MARK = /^[\p{P}\p{S}]$/u;

function examplesOf(data: StyleInput, spec: TraitSpec) {
  const distribution = data.items[spec.item]?.distributions.find((entry) => entry.path === spec.forms);
  return (distribution?.top ?? [])
    .map(([form]) => form)
    .filter((form) => form.length <= 8 && !form.includes('\n') && !LONE_MARK.test(form))
    .slice(0, 4);
}

function contactRates(data: StyleInput, leaf: StyleLeaf, rate: number): ContactRate[] {
  return data.contacts
    .flatMap((contact) => {
      const cell = leaf.groups[contact.id];
      if (!cell?.tested || !cell.messages) return [];
      const adjusted = cell.ratio != null ? Math.min(1, rate * cell.ratio) : cell.hits / cell.messages;
      return [{ contact, rate: adjusted }];
    })
    .sort((left, right) => right.rate - left.rate);
}

/** Traits the person shows often enough, most frequent first. */
export function selectTraits(data: StyleInput): Trait[] {
  return SPECS.flatMap((spec) => {
    const leaves = spec.paths.map((path) => data.items[spec.item]?.leaves.find((leaf) => leaf.path === path));
    if (!leaves.every(Boolean)) return [];
    const found = leaves as StyleLeaf[];
    const messages = found[0].person.messages;
    if (!messages || found.some((leaf) => leaf.person.messages !== messages)) return [];
    const rate = found.reduce((sum, leaf) => sum + leaf.person.hits, 0) / messages;
    // Summed shares carry no reliability record of their own.
    const { person } = found[0];
    const reliable = found.length > 1 || (person.class === 'estimable' && (person.missing_share ?? 0) <= MAX_MISSING);
    if (!reliable || rate < MIN_RATE) return [];
    const single = found.length === 1 ? found[0] : null;
    return [
      {
        ...spec,
        key: `${spec.item}:${spec.paths.join('+')}`,
        rate,
        messages,
        sampled: found[0].source === 'contextual',
        examples: examplesOf(data, spec),
        variation: single?.test?.class ?? null,
        contacts: single ? contactRates(data, single, rate) : [],
      },
    ];
  }).sort((left, right) => right.rate - left.rate);
}

export function formatPercent(rate: number) {
  const percent = rate * 100;
  return `${percent >= 10 ? Math.round(percent) : percent.toFixed(1)}%`;
}
