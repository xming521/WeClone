import assert from 'node:assert/strict';
import { test } from 'node:test';
import { stateCounts, summaryState } from '../src/summary/summaryState.ts';
import type { DomainSummary, SummaryDomain } from '../src/summary/summaryState.ts';

const summary: DomainSummary = {
  title: '学业',
  summary: '在读博士',
  origin: 'generated',
  status: 'pending',
  version: 1,
  stale: false,
};

function domain(facts: number, row: Partial<DomainSummary> | null): SummaryDomain {
  return { domain: '学业', facts, as_of: '2024-07', summary: row ? { ...summary, ...row } : null };
}

test('summary state follows what generation and export do with a domain', () => {
  assert.equal(summaryState(domain(0, null)), 'empty');
  assert.equal(summaryState(domain(0, { stale: true })), 'empty');
  assert.equal(summaryState(domain(3, null)), 'missing');
  assert.equal(summaryState(domain(3, { stale: true, status: 'approved' })), 'stale');
  assert.equal(summaryState(domain(3, {})), 'pending');
  assert.equal(summaryState(domain(3, { status: 'approved' })), 'approved');
  // Left out of the export, even once its facts change or none are left.
  assert.equal(summaryState(domain(3, { status: 'rejected', stale: true })), 'rejected');
  assert.equal(summaryState(domain(0, { status: 'rejected' })), 'rejected');
});

test('state counts cover every domain once', () => {
  const counts = stateCounts([domain(3, null), domain(3, {}), domain(3, { status: 'approved' }), domain(0, null)]);
  assert.deepEqual(counts, { empty: 1, missing: 1, stale: 0, pending: 1, approved: 1, rejected: 0 });
});
