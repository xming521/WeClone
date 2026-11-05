import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  WAVE,
  arrivalTime,
  hubSwell,
  swellProgress,
  swellSpan,
  wavePeriod,
  waveRunning,
} from '../src/atlas/atlasFlow.ts';

test('a wave starts on the deepest lines and moves up one level per step', () => {
  const maxDepth = 3;
  assert.equal(wavePeriod(WAVE, maxDepth), 2 * WAVE.step + WAVE.seg + WAVE.rest);
  assert.equal(swellProgress(WAVE, maxDepth, 3, 0), 0);
  assert.equal(swellProgress(WAVE, maxDepth, 2, 0), -1);
  assert.equal(swellProgress(WAVE, maxDepth, 2, WAVE.step), 0);
  assert.equal(swellProgress(WAVE, maxDepth, 1, 2 * WAVE.step + WAVE.seg / 2), 0.5);
  assert.equal(swellProgress(WAVE, maxDepth, 1, 2 * WAVE.step + WAVE.seg + 1), -1);
  assert.equal(swellProgress(WAVE, maxDepth, 3, -1), -1);
  assert.ok(waveRunning(WAVE, maxDepth, 2 * WAVE.step + WAVE.seg));
  assert.ok(!waveRunning(WAVE, maxDepth, 2 * WAVE.step + WAVE.seg + 1));
  assert.ok(!waveRunning(WAVE, maxDepth, -1));
});

test('a node swells when the ink from its children arrives, and the centre last', () => {
  const maxDepth = 3;
  const childrenArrive = (depth: number) => (maxDepth - (depth + 1)) * WAVE.step + WAVE.seg * 0.88;
  assert.equal(hubSwell(WAVE, maxDepth, 1, childrenArrive(1), 230), 1);
  assert.ok(hubSwell(WAVE, maxDepth, 1, childrenArrive(1) + 600, 230) < 0.01);
  assert.equal(hubSwell(WAVE, maxDepth, 0, -1, 230), 0);
  assert.ok(arrivalTime(WAVE, maxDepth) > childrenArrive(0));
  assert.ok(arrivalTime(WAVE, maxDepth) < wavePeriod(WAVE, maxDepth));
});

test('swellSpan stays on the line and runs from the child to the parent', () => {
  assert.equal(swellSpan(0, 10, 110), null);
  const middle = swellSpan(0.5, 10, 110)!;
  assert.ok(middle.from >= 10 && middle.to <= 110 && middle.from < middle.to);
  assert.equal(middle.tail, 85);
  const late = swellSpan(0.95, 10, 110)!;
  assert.equal(late.to, 110);
  assert.ok(late.from > middle.from);
  assert.equal(swellSpan(1, 10, 110), null);
  assert.equal(swellSpan(0.5, 10, 5), null);
});
