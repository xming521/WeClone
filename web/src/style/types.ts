// Shape of GET /api/style, built by weclone/server/style_view.py from one style extraction run.

export type ReliabilityClass = 'estimable' | 'rare' | 'undetermined';
export type VariationClass = 'relationship_dependent' | 'uncertain' | 'stable' | 'insufficient';

export interface RateCell {
  hits: number;
  messages: number;
  /** Per length bucket: [hits, messages]. */
  by_length?: Record<string, [number, number]>;
  interval?: [number, number];
  class?: ReliabilityClass;
  rare_outcome?: 'occurrence' | 'non_occurrence';
  rare_upper?: number;
  effective_events?: number | null;
  bounds?: [number, number];
  missing_share?: number;
  windows?: number;
}

export interface GroupCell extends RateCell {
  /** Length-adjusted multiplier of the person rate: (hits + 0.5) / (expected hits + 0.5). */
  ratio?: number;
  expected?: number;
  tested: boolean;
}

export interface RateTest {
  test?: 'length_stratified_cmh' | 'chi_square';
  class: VariationClass;
  rate_ratio?: number;
  q_value?: number;
  p_value?: number;
  chi2?: number;
  dof?: number;
  highest_group?: string;
  lowest_group?: string;
  testable_groups?: number;
  reason?: string;
}

export interface StyleLeaf {
  path: string;
  source: 'surface' | 'contextual' | 'other';
  person: RateCell;
  groups: Record<string, GroupCell>;
  test: RateTest | null;
}

export interface Distribution {
  path: string;
  total: number;
  size: number;
  top: [string, number][];
}

export interface StyleItem {
  id: string;
  name: string;
  category: string;
  state: string;
  reason?: string | null;
  extractor?: string | null;
  variation_class?: VariationClass | null;
  leaves: StyleLeaf[];
  distributions: Distribution[];
  numbers: Record<string, number>;
}

export interface StyleContact {
  id: string;
  label: string;
  messages: number;
  windows: number;
  semantic_windows: number | null;
}

export interface StyleInput {
  run: {
    id: string;
    protocol: string;
    window: { days?: number; start?: string; end?: string };
    scope: { surface?: { windows: number; messages: number }; semantic?: { windows: number; messages: number } };
    sampling: { budget?: number; selected?: number; pooled_selected?: number; group_floor?: number };
  };
  categories: { id: string; name: string; items: string[] }[];
  contacts: StyleContact[];
  items: Record<string, StyleItem>;
}
