import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { StageKey } from './progress';

const QUIP_MS = 4000;

/** A light line about what the build is doing, a new one every few seconds: the stage's own lines, then the ones for
 * any stage. Mount it with the stage as its key, so a new stage starts on its own first line. */
export function Quips({ stage, className }: { stage: StageKey; className: string }) {
  const { t } = useTranslation();
  const lines = [
    ...t(($) => $.build.quips[stage], { returnObjects: true }),
    ...t(($) => $.build.quips.any, { returnObjects: true }),
  ];
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setIndex((value) => value + 1), QUIP_MS);
    return () => window.clearInterval(timer);
  }, []);
  return (
    <p className={className} aria-hidden="true">
      <span key={index}>{lines[index % lines.length]}</span>
    </p>
  );
}
