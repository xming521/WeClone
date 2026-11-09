import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  buildProgress,
  buildStep,
  clock,
  overspent,
  projectBuild,
  type BuildStage,
  type BuildStatus,
  type Estimate,
  type Usage,
} from '../src/build/progress.ts';

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

const START = '2026-11-05T00:00:00+00:00';
const at = (seconds: number) => Date.parse(START) + seconds * 1000;

function usage(requests: number, cost: number | null): Usage {
  return { requests, unmetered: 0, input_tokens: 0, cached_tokens: 0, output_tokens: 0, reasoning_tokens: 0, cost };
}

function estimate(cost: number | null = 10): Estimate {
  return {
    backend: 'api',
    requests: 4510,
    probe_requests: 10,
    plan: { distill_profile: 2000, distill_event: 2000, later: 450, style: 60 },
    probe: null,
    unmetered: false,
    input_tokens: null,
    output_tokens: null,
    cost:
      cost === null
        ? null
        : {
            low: cost / 2,
            high: cost,
            per_request: { low: 0, high: 0 },
            thinking_share: 0,
            off_peak_saving: null,
            source: 'official',
            vendor: 'deepseek',
            prices_date: '2026-11-08',
          },
    seconds: null,
  };
}

/** A build in the middle of the profile extraction, 1000 requests in after 1000 seconds, at 0.01 dollars each. */
function building(stages: BuildStage[], spent: Usage | null = usage(1000, 10), planned: Estimate | null = estimate()) {
  return { ...status(stages), started_at: START, usage: spent, estimate: planned };
}

const extracting = [
  stage(1, 'snapshot', 'succeeded'),
  { ...stage(2, 'distill_profile', 'running'), started_at: START, done: 800, total: 2000 },
  stage(2, 'distill_event', 'pending'),
  stage(3, 'organize', 'pending'),
  stage(4, 'hierarchy', 'pending'),
  stage(5, 'summaries', 'pending'),
];

test('projectBuild waits for 20 requests and for a usage and an estimate', () => {
  assert.equal(projectBuild(building(extracting, usage(19, 1)), at(100)), null);
  assert.equal(projectBuild(building(extracting, null), at(100)), null);
  assert.equal(projectBuild(building(extracting, usage(1000, 10), null), at(100)), null);
  assert.ok(projectBuild(building(extracting, usage(20, 1)), at(100)));
});

test('projectBuild counts what is left of the running stage, the planned rest and the later stages at twice the time', () => {
  const projection = projectBuild(building(extracting), at(1000))!;
  // 1200 left in the profile stage, 2000 events, 450 later: 3650 requests in all.
  assert.equal(projection.requests, 3650);
  // One second a request: the extraction ones at that, the later ones at two.
  assert.equal(projection.seconds, 1200 + 2000 + 450 * 2);
  assert.ok(Math.abs(projection.cost! - 36.5) < 1e-9);
  assert.ok(Math.abs(projection.totalCost! - 46.5) < 1e-9);
});

test('projectBuild takes a finished stage as done, the later stages by their usage, and the clone style by its plan', () => {
  const stages = [
    stage(1, 'snapshot', 'succeeded'),
    { ...stage(2, 'distill_profile', 'succeeded'), started_at: START },
    { ...stage(2, 'distill_event', 'succeeded'), started_at: START },
    { ...stage(3, 'organize', 'succeeded'), usage: usage(300, 3) },
    { ...stage(4, 'hierarchy', 'running'), usage: usage(100, 1) },
    stage(5, 'summaries', 'pending'),
    { ...stage(6, 'style', 'pending') },
  ];
  const projection = projectBuild(building(stages, usage(4400, 44)), at(4400))!;
  // 450 planned, 400 sent: 50 later ones, then 60 of the style.
  assert.equal(projection.requests, 110);
  assert.equal(projection.seconds, 60 * 1 + 50 * 2);
  const done = stages.map((item) => (item.key === 'summaries' ? { ...item, state: 'succeeded' as const } : item));
  assert.equal(projectBuild(building(done, usage(4400, 44)), at(4400))!.requests, 60);
});

test('projectBuild has no cost for a model without a price', () => {
  const projection = projectBuild(building(extracting, usage(1000, null)), at(1000))!;
  assert.equal(projection.cost, null);
  assert.equal(projection.totalCost, null);
  assert.ok(projection.seconds > 0);
});

test('overspent flags a build whose total passes twice the top of its estimate', () => {
  const projection = projectBuild(building(extracting), at(1000))!;
  // The total is 46.5 against a top of 10.
  assert.equal(overspent(building(extracting), projection), true);
  assert.equal(overspent(building(extracting, usage(1000, 10), estimate(30)), projection), false);
  assert.equal(overspent(building(extracting, usage(1000, 10), estimate(null)), projection), false);
  assert.equal(overspent(building(extracting), null), false);
});
