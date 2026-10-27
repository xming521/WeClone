// Shapes of GET /api/summaries, served by weclone/server/profile_summary.py.
export type SummaryLevel = 'full' | 'normal' | 'public';
export type SummaryStatus = 'pending' | 'approved' | 'rejected';

export interface DomainSummary {
  title: string;
  summary: string;
  origin: 'generated' | 'edited';
  status: SummaryStatus;
  version: number;
  stale: boolean;
}

export interface SummaryDomain {
  domain: string;
  facts: number;
  as_of: string;
  summary: DomainSummary | null;
}

export interface SummaryView {
  level: SummaryLevel;
  generating: boolean;
  /** Why the last generation left domains without a summary, while they still wait for one. */
  error: string;
  limits: { title: number; summary: number; facts: number };
  domains: SummaryDomain[];
  text: string;
}

export const SUMMARY_LEVELS: SummaryLevel[] = ['public', 'normal', 'full'];

/** Where one domain stands. Only approved summaries reach the export, and generation fills missing and stale ones. */
export type SummaryState = 'empty' | 'missing' | 'stale' | 'pending' | 'approved' | 'rejected';

export const STATE_LABELS: Record<SummaryState, string> = {
  empty: '没有可用条目',
  missing: '未生成',
  stale: '需重新生成',
  pending: '待确认',
  approved: '已确认',
  rejected: '不导出',
};

export function summaryState({ facts, summary }: SummaryDomain): SummaryState {
  if (summary?.status === 'rejected') return 'rejected';
  if (!facts) return 'empty';
  if (!summary) return 'missing';
  if (summary.stale) return 'stale';
  return summary.status;
}

/** How many domains are in each state. */
export function stateCounts(domains: SummaryDomain[]): Record<SummaryState, number> {
  const counts = { empty: 0, missing: 0, stale: 0, pending: 0, approved: 0, rejected: 0 };
  for (const domain of domains) counts[summaryState(domain)] += 1;
  return counts;
}
