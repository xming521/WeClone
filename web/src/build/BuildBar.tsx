import type { CSSProperties } from 'react';
import { buildProgress, type BuildStatus, type StageKey } from './progress';
import { Workers } from './Workers';

/** A running build's progress as a bar, each stage an equal stretch of it, under the two faces of the mark taking
 * turns at work and saying a line about the stage it is on. */
export function BuildBar({ status, stage, now }: { status: BuildStatus; stage: StageKey; now: number }) {
  const count = status.stages.length;
  const style = { '--progress': (buildProgress(status, now) * 100).toFixed(2) } as CSSProperties;
  return (
    <div className="build-bar" style={style} aria-hidden="true">
      <Workers stage={stage} />
      <div className="build-bar__track">
        <span className="build-bar__fill" />
        {Array.from({ length: count - 1 }, (_, index) => (
          <span key={index} className="build-bar__cut" style={{ left: `${((index + 1) * 100) / count}%` }} />
        ))}
      </div>
    </div>
  );
}
