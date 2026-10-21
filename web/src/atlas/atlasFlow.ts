// Timing of the ink that flows along the map's lines into 「我」 (design: WC-exp/design, "A 纸本 ·
// 浅色克制"). A wave starts at the deepest visible level and moves up one level per step: on every
// line a soft swell enters at the child, runs along the line and drains into the parent, whose halo
// swells as it arrives. Only type imports are allowed here so the tests can load this file directly
// with node's type stripping.

export type WaveTiming = {
  /** Time a swell takes along one line, in ms. */
  seg: number;
  /** Delay between two levels. */
  step: number;
  /** Pause after the swell has reached the centre. */
  rest: number;
};

/** The wave through the whole map. */
export const WAVE: WaveTiming = { seg: 950, step: 720, rest: 2600 };
/** The faster wave along the branch under the pointer. */
export const LOCAL_WAVE: WaveTiming = { seg: 760, step: 560, rest: 420 };

/** Length of one cycle through a tree whose deepest node has depth `maxDepth` (the centre has 0). */
export const wavePeriod = (timing: WaveTiming, maxDepth: number) =>
  (Math.max(1, maxDepth) - 1) * timing.step + timing.seg + timing.rest;

/** Time into the cycle at which the swell on the lines into the dimensions reaches the centre. */
export const arrivalTime = (timing: WaveTiming, maxDepth: number) =>
  (Math.max(1, maxDepth) - 1) * timing.step + timing.seg * 0.9;

/** Whether the swells are still running `clock` ms into the cycle, as opposed to resting. */
export const waveRunning = (timing: WaveTiming, maxDepth: number, clock: number) =>
  clock >= 0 && clock <= (Math.max(1, maxDepth) - 1) * timing.step + timing.seg;

/**
 * Progress 0…1 of the swell on the line into a node of depth `depth`, `clock` ms into the cycle,
 * or -1 while there is none. Deeper lines go first.
 */
export function swellProgress(timing: WaveTiming, maxDepth: number, depth: number, clock: number): number {
  if (clock < 0) return -1;
  const progress = (clock - (Math.max(1, maxDepth) - depth) * timing.step) / timing.seg;
  return progress >= 0 && progress <= 1 ? progress : -1;
}

/**
 * 0…1 swell of the halo of a node of depth `depth` as the ink from its children drains into it,
 * a bell `width` ms wide.
 */
export function hubSwell(timing: WaveTiming, maxDepth: number, depth: number, clock: number, width: number): number {
  if (clock < 0) return 0;
  const d = (clock - ((Math.max(1, maxDepth) - (depth + 1)) * timing.step + timing.seg * 0.88)) / width;
  return Math.exp(-d * d);
}

const easeInOutQuad = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

/**
 * Part of a line lit by a swell at `progress`, as distances from the child's centre: the swell runs
 * from `start` (the child's rim) to `end` (the parent's rim) with a fading tail behind its head.
 * Null when nothing of it is on the line.
 */
export function swellSpan(
  progress: number,
  start: number,
  end: number,
): { head: number; tail: number; from: number; to: number } | null {
  const run = end - start;
  if (run <= 0) return null;
  const tail = Math.max(30, Math.min(170, run * 0.85));
  const head = start + (run + tail) * easeInOutQuad(progress);
  const from = Math.max(start, head - tail);
  const to = Math.min(end, head);
  return to - from < 0.5 ? null : { head, tail, from, to };
}
