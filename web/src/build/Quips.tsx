import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { StageKey } from './progress';

/** What a line is about, so that it goes with what the faces are doing (Workers.tsx): looking through the chats,
 * sorting, writing, typing, a break, or anything at all. */
export type QuipKind = 'look' | 'sort' | 'write' | 'type' | 'rest' | 'any';

/** A line, with the group it comes from: the stage's own, 0, or the ones for any stage, 1. */
type Line = { group: number; kind: string; text: string };
/** The line said at the faces' `step`, by its place among the lines, and when each line was last said. */
type Said = { step: number; index: number; last: ReadonlyMap<number, number> };

/** Of the lines that go with the faces' work, the one said longest ago, or never, the first of them on a tie. Lines of
 * one group and kind sound alike, so two of them don't come one after the other while there is another to say. */
function next(lines: Line[], kinds: readonly string[], step: number, said?: Said): Said {
  const going = lines.flatMap((line, index) => (kinds.includes(line.kind) ? [index] : []));
  const before = said && lines[said.index];
  const other = going.filter((index) => lines[index].group !== before?.group || lines[index].kind !== before.kind);
  const last = said?.last ?? new Map<number, number>();
  const pool = other.length ? other : going;
  const index = pool.reduce(
    (best, index) => ((last.get(index) ?? -1) < (last.get(best) ?? -1) ? index : best),
    pool[0] ?? 0,
  );
  return { step, index, last: new Map(last).set(index, step) };
}

/** A light line about what the build is doing, a new one each time the faces start on something (`step`), going with
 * what they do: the stage's own lines and the ones for any stage, in turn. Mount it with the stage as its key, so a new
 * stage starts on its own lines. */
export function Quips({
  stage,
  kinds,
  step,
  className,
}: {
  stage: StageKey;
  kinds: readonly QuipKind[];
  step: number;
  className: string;
}) {
  const { t } = useTranslation();
  const groups: Record<string, readonly string[]>[] = [
    t(($) => $.build.quips[stage], { returnObjects: true }),
    t(($) => $.build.quips.common, { returnObjects: true }),
  ];
  const lines = groups.flatMap((lines, group) =>
    Object.entries(lines).flatMap(([kind, texts]) => texts.map((text) => ({ group, kind, text }))),
  );
  const [said, setSaid] = useState(() => next(lines, kinds, step));
  if (said.step !== step) setSaid(next(lines, kinds, step, said));
  return (
    <p className={className} aria-hidden="true">
      <span key={said.step}>{lines[said.index]?.text}</span>
    </p>
  );
}
