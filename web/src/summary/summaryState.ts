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
  /** How many domains the last generation left without a summary, while they still wait for one, and why. The
   * reason is the error's own text, empty when the model calls gave up after their retries. */
  failure: { failed: number; reason: string } | null;
  limits: { title: number; summary: number; facts: number };
  domains: SummaryDomain[];
  text: string;
}

export const SUMMARY_LEVELS: SummaryLevel[] = ['public', 'normal', 'full'];

/** Where one domain stands. Only approved summaries reach the export, and generation fills missing and stale ones. */
export type SummaryState = 'empty' | 'missing' | 'stale' | 'pending' | 'approved' | 'rejected';

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
