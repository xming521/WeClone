import type { StyleContact, StyleInput, StyleLeaf, VariationClass } from './types';

export type TraitGroupId = 'typing' | 'words' | 'emotion' | 'manner';

export const TRAIT_GROUPS: { id: TraitGroupId; name: string; hint: string; color: string }[] = [
  { id: 'typing', name: '打字习惯', hint: '标点、换行、长短', color: '#008f85' },
  { id: 'words', name: '用词', hint: '语气词、网络语、粗口', color: '#d28b32' },
  { id: 'emotion', name: '情绪表达', hint: '笑声、表情、吐槽', color: '#c35f72' },
  { id: 'manner', name: '说话方式', hint: '直说、商量、发问', color: '#3775b4' },
];

/** A trait is shown only when it appears at least once in 25 messages. */
export const MIN_RATE = 0.04;
/** Above this share of undecided messages the rate could sit almost anywhere. */
const MAX_MISSING = 0.3;

// Bubbles are shaded by how often the trait shows up.
export const TIERS = [
  { id: 'most', min: 0.5, label: '多数时候' },
  { id: 'often', min: 0.2, label: '常常' },
  { id: 'sometimes', min: 0.1, label: '有时' },
  { id: 'rarely', min: 0, label: '偶尔' },
] as const;

export type Tier = (typeof TIERS)[number];

export function tierOf(rate: number): Tier {
  return TIERS.find((tier) => rate >= tier.min) ?? TIERS[TIERS.length - 1];
}

// Titles name the habit without saying how often, since the rate differs from person to person.
interface TraitSpec {
  item: string;
  /** Rate leaves added into one rate; they share one denominator. */
  paths: string[];
  title: string;
  note: string;
  group: TraitGroupId;
  /** The situation the rate is counted in, when it is not every message. */
  when?: string;
  /** What the denominator counts, when it is not whole messages. */
  unit?: string;
  /** Distribution whose most frequent entries are shown as examples. */
  forms?: string;
}

const SPECS: TraitSpec[] = [
  {
    item: '088',
    paths: ['surface/unpunctuated_boundary_rate'],
    title: '换行当标点',
    note: '该打逗号、句号的地方，直接换行接着发',
    group: 'typing',
    unit: '处断句',
  },
  {
    item: '086',
    paths: ['surface/bubble_no_punctuation_rate'],
    title: '不打标点',
    note: '发出去的一条里，一个标点都没有',
    group: 'typing',
    unit: '条气泡',
  },
  {
    item: 'burst_pattern',
    paths: ['surface/message_newline_burst_rate'],
    title: '一句话分几条发',
    note: '一次回复拆成好几条，连着发出去',
    group: 'typing',
  },
  {
    item: '218',
    paths: [
      'surface/message_length_shares/1-2',
      'surface/message_length_shares/3-4',
      'surface/message_length_shares/5-8',
    ],
    title: '短消息',
    note: '一次回复不超过 8 个字',
    group: 'typing',
  },
  {
    item: '097-098',
    paths: ['surface/message_rate'],
    title: '语气词',
    note: '句子里带「啊」「吧」「嘛」这类词',
    group: 'words',
    forms: 'surface/forms',
  },
  {
    item: '106-107,109',
    paths: ['surface/message_rate'],
    title: '连打重复',
    note: '「哈哈哈」「？？？」这样，同一个字或符号连着打',
    group: 'words',
    forms: 'surface/forms',
  },
  {
    item: '007',
    paths: ['contextual/message_occurrence_rate'],
    title: '网络用语',
    note: '聊天里夹着网络流行语和梗',
    group: 'words',
    forms: 'contextual/expressions',
  },
  {
    item: '009',
    paths: ['contextual/message_occurrence_rate'],
    title: '专业词',
    note: '工作、学习领域的词直接带进聊天',
    group: 'words',
    forms: 'contextual/expressions',
  },
  {
    item: '163',
    paths: ['contextual/message_rate'],
    title: '程度词',
    note: '用「很」「太」「挺」来加重或减轻',
    group: 'words',
    forms: 'surface/forms',
  },
  {
    item: '016',
    paths: ['contextual/message_occurrence_rate'],
    title: '叠词',
    note: '「看看」「慢慢」这样把字叠起来说',
    group: 'words',
    forms: 'contextual/expressions',
  },
  {
    item: '245',
    paths: ['contextual/message_occurrence_rate'],
    title: '粗口',
    note: '话里夹着粗口，包括缩写和谐音',
    group: 'words',
    forms: 'contextual/forms',
  },
  {
    item: '153,160',
    paths: ['contextual/message_rate'],
    title: '说话带情绪',
    note: '能从字里行间看出开心、无奈或不满',
    group: 'emotion',
  },
  {
    item: '153,160',
    paths: ['contextual/modes/message_rates/implicit'],
    title: '情绪藏在话里',
    note: '不直接说出感受，情绪藏在语气和说法里',
    group: 'emotion',
  },
  {
    item: '153,160',
    paths: ['contextual/modes/message_rates/symbolic'],
    title: '用表情符号带情绪',
    note: '用表情、符号或拟声来传递情绪',
    group: 'emotion',
    forms: 'contextual/forms',
  },
  {
    item: '099-105',
    paths: ['contextual/message_rate'],
    title: '笑声和语气声',
    note: '用文字打出笑声、惊讶、犹豫',
    group: 'emotion',
    forms: 'contextual/forms',
  },
  {
    item: '158',
    paths: ['contextual/message_rate'],
    title: '发牢骚',
    note: '对烦心事发几句牢骚、诉诉苦',
    group: 'emotion',
    forms: 'contextual/forms',
  },
  {
    item: '159',
    paths: ['contextual/message_rate'],
    title: '吐槽',
    note: '用揶揄、夸张的话表达不满',
    group: 'emotion',
    forms: 'contextual/forms',
  },
  {
    item: '174-175',
    paths: ['175/contextual/message_occurrence_rate'],
    title: '调侃',
    note: '开玩笑地打趣人和事，也包括自嘲',
    group: 'emotion',
  },
  {
    item: '138',
    paths: ['contextual/attributes/directness/message_rates/explicit'],
    title: '有话直说',
    note: '表达看法和打算时，直接说出来，不绕弯子',
    group: 'manner',
    when: '表态的时候',
    unit: '条表态的消息',
  },
  {
    item: '228',
    paths: ['contextual/label_message_rates/matter'],
    title: '就事论事',
    note: '接对方的话时，回应的是事情本身',
    group: 'manner',
  },
  {
    item: '136-137',
    paths: ['contextual/attributes/act/message_rates/suggestion'],
    title: '商量的口吻',
    note: '想让对方做什么时，用建议的语气说',
    group: 'manner',
    when: '想让对方做什么的时候',
    unit: '条让对方做事的消息',
  },
  {
    item: '052,211',
    paths: ['contextual/message_occurrence_rate'],
    title: '提问',
    note: '回复里带着问题',
    group: 'manner',
  },
  {
    item: '210',
    paths: ['contextual/message_occurrence_rate'],
    title: '追问',
    note: '顺着对方的话继续问下去',
    group: 'manner',
  },
  {
    item: '219',
    paths: ['contextual/message_occurrence_rate'],
    title: '主动多聊几句',
    note: '不只回答问题，还会顺着往下说',
    group: 'manner',
  },
  {
    item: '256',
    paths: ['contextual/message_advice_occurrence_rate'],
    title: '出主意',
    note: '聊着聊着就给对方提建议',
    group: 'manner',
  },
  {
    item: '120-127',
    paths: ['contextual/message_rate'],
    title: '话不说满',
    note: '用「好像」「应该」「可能」给自己留余地',
    group: 'manner',
    forms: 'contextual/forms',
  },
  {
    item: '146-150',
    paths: ['contextual/message_rate'],
    title: '加重语气',
    note: '用重复、程度词或符号把话说重',
    group: 'manner',
    forms: 'contextual/forms',
  },
  {
    item: '231',
    paths: ['contextual/message_occurrence_rate'],
    title: '附和对方',
    note: '对方表态时，说「对」「确实」表示认同',
    group: 'manner',
  },
  {
    item: '166',
    paths: ['contextual/message_occurrence_rate'],
    title: '打比方',
    note: '借别的事物来形容想说的东西',
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

export function frequencyText(rate: number) {
  if (rate >= 0.95) return '几乎每条都这样';
  if (rate >= 0.3) return `大约 10 条里有 ${Math.round(rate * 10)} 条`;
  return `大约每 ${Math.round(1 / rate)} 条有 1 条`;
}

export const VARIATION_TEXT: Partial<Record<VariationClass, string>> = {
  relationship_dependent: '跟不同的人聊，差别很明显。',
  uncertain: '跟不同的人聊，可能有些差别。',
  stable: '跟谁聊都差不多。',
};
