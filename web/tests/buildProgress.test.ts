import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildProgress, buildStep, clock, type BuildStage, type BuildStatus } from '../src/build/progress.ts';

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

test('buildStep counts the stages in order, also in a status file where both extraction stages are step 2', () => {
  const stages = [
    stage(1, 'snapshot', 'succeeded'),
    stage(2, 'distill_profile', 'succeeded'),
    stage(2, 'distill_event', 'running'),
    stage(3, 'organize', 'pending'),
    stage(4, 'hierarchy', 'pending'),
    stage(5, 'summaries', 'pending'),
  ];
  assert.deepEqual(buildStep(status(stages)), { step: 3, steps: 6 });
});

test('buildStep shows the first step before any stage starts, the next one between stages, the last one at the end', () => {
  const pending = [stage(1, 'snapshot', 'pending'), stage(2, 'distill_profile', 'pending')];
  assert.deepEqual(buildStep(status(pending)), { step: 1, steps: 2 });
  const between = [stage(1, 'snapshot', 'succeeded'), stage(2, 'distill_profile', 'pending')];
  assert.deepEqual(buildStep(status(between)), { step: 2, steps: 2 });
  const finishing = [stage(1, 'snapshot', 'succeeded'), stage(2, 'distill_profile', 'succeeded')];
  assert.deepEqual(buildStep(status(finishing)), { step: 2, steps: 2 });
});

test('clock shows minutes and seconds, and hours from an hour on', () => {
  assert.equal(clock(0), '0:00');
  assert.equal(clock(65.4), '1:05');
  assert.equal(clock(3725), '1:02:05');
  assert.equal(clock(-3), '0:00');
});

test('buildProgress gives each stage an equal share, by chats when counted and closing in by time when not', () => {
  const counted = { ...stage(2, 'distill_profile', 'running'), done: 7, total: 28 };
  const progress = buildProgress(status([stage(1, 'snapshot', 'succeeded'), counted]), 0);
  assert.equal(progress, (1 + 0.25) / 2);
  const started = { ...stage(4, 'organize', 'running'), started_at: '2026-11-05T00:00:00+00:00' };
  const at = (seconds: number) => buildProgress(status([started]), Date.parse(started.started_at) + seconds * 1000);
  assert.equal(at(0), 0);
  assert.ok(at(60) > 0.5 && at(60) < at(600) && at(600) < 0.9);
});
