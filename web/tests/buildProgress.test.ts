import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildStep, clock, type BuildStage, type BuildStatus } from '../src/build/progress.ts';

function stage(step: number, key: BuildStage['key'], state: BuildStage['state']): BuildStage {
  return { step, key, state, started_at: null, finished_at: null, done: null, total: null };
}

function status(stages: BuildStage[]): BuildStatus {
  return {
    purpose: 'assistant',
    state: 'running',
    started_at: '2026-11-05T00:00:00+00:00',
    finished_at: null,
    chats: null,
    stages,
    error: null,
    run: 'r',
    current: false,
    log: 'r/build.log',
  };
}

test('buildStep counts the steps of the command line, where both extraction stages are step 2', () => {
  const stages = [
    stage(1, 'snapshot', 'succeeded'),
    stage(2, 'distill_profile', 'succeeded'),
    stage(2, 'distill_event', 'running'),
    stage(3, 'organize', 'pending'),
    stage(4, 'hierarchy', 'pending'),
    stage(5, 'summaries', 'pending'),
  ];
  assert.deepEqual(buildStep(status(stages)), { step: 2, steps: 5 });
});

test('buildStep shows the first step before any stage starts, and the last one done between stages', () => {
  const pending = [stage(1, 'snapshot', 'pending'), stage(2, 'distill_profile', 'pending')];
  assert.deepEqual(buildStep(status(pending)), { step: 1, steps: 2 });
  const between = [stage(1, 'snapshot', 'succeeded'), stage(2, 'distill_profile', 'pending')];
  assert.deepEqual(buildStep(status(between)), { step: 1, steps: 2 });
});

test('clock shows minutes and seconds, and hours from an hour on', () => {
  assert.equal(clock(0), '0:00');
  assert.equal(clock(65.4), '1:05');
  assert.equal(clock(3725), '1:02:05');
  assert.equal(clock(-3), '0:00');
});
