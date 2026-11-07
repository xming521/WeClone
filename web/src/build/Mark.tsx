import { useId, type ReactNode } from 'react';

// weclone-mark.svg drawn inline, so the faces can move: each blinks, the clone a moment after you as a copy would,
// and when they are happy they smile with their eyes shut and their cheeks flushed.
export const MARK_VIEW = { x: -12.4, y: -12.64, w: 40.36, h: 28.95 };
export const FACES = [
  {
    who: 'you',
    gradient: { x1: -8.64, y1: -8.64, x2: 8.64, y2: 8.64, stops: ['#70d1fc', '#3ea1f6'] },
    body: 'M-12 0a12 12 0 1 0 24 0a12 12 0 1 0 -24 0zM-11.93 13.27a2.64 2.64 0 1 0 5.28 0a2.64 2.64 0 1 0 -5.28 0z',
    shine:
      'M-9.22 -3.92A10.02 10.02 0 0 1 -1.57 -9.9A0.78 0.78 0 0 1 -1.32 -8.36A8.46 8.46 0 0 0 -7.79 -3.31A0.78 0.78 0 0 1 -9.22 -3.92z',
    ink: '#075690',
    eyes: 'M-5.46 0.48a1.43 2.05 0 1 0 2.86 0a1.43 2.05 0 1 0 -2.86 0zM3.54 0.48a1.43 2.05 0 1 0 2.86 0a1.43 2.05 0 1 0 -2.86 0z',
    glints:
      'M-5.03 -0.19a0.57 0.57 0 1 0 1.15 0a0.57 0.57 0 1 0 -1.15 0zM3.97 -0.19a0.57 0.57 0 1 0 1.15 0a0.57 0.57 0 1 0 -1.15 0z',
    mouth:
      'M2.26 3.11A2 2 0 0 1 -1.35 3.11A0.35 0.35 0 0 1 -0.72 2.81A1.31 1.31 0 0 0 1.63 2.81A0.35 0.35 0 0 1 2.26 3.11z',
    // Shut eyes over the open ones, an open smile, and the cheeks beside them.
    smile: { eyes: 'M-5.55 1.3Q-4.03 -1.2 -2.51 1.3M3.45 1.3Q4.97 -1.2 6.49 1.3', width: 0.95 },
    grin: 'M-1.7 2.6H2.6Q2.4 5.7 0.45 5.7Q-1.5 5.7 -1.7 2.6Z',
    cheeks: [
      { x: -6.6, y: 3.6 },
      { x: 7.5, y: 3.6 },
    ],
  },
  {
    who: 'clone',
    gradient: { x1: 16.44, y1: -12.24, x2: 16.44, y2: 8.4, stops: ['#fd9f77', '#fec394'] },
    body: 'M9.39 -9.45A10.32 10.32 0 1 1 11.31 7.04A13.32 13.32 0 0 0 9.39 -9.45zM22.76 9.24a2.4 2.4 0 1 0 4.8 0a2.4 2.4 0 1 0 -4.8 0z',
    shine:
      'M13.08 -9.46A8.26 8.26 0 0 1 17.3 -10.13A0.62 0.62 0 0 1 17.17 -8.9A7.02 7.02 0 0 0 13.59 -8.33A0.62 0.62 0 0 1 13.08 -9.46z',
    ink: '#843b19',
    eyes: 'M15.91 -2.23a1.15 1.64 0 1 0 2.3 0a1.15 1.64 0 1 0 -2.3 0zM22.1 -2.23a1.15 1.64 0 1 0 2.3 0a1.15 1.64 0 1 0 -2.3 0z',
    glints:
      'M16.26 -2.77a0.46 0.46 0 1 0 0.92 0a0.46 0.46 0 1 0 -0.92 0zM22.45 -2.77a0.46 0.46 0 1 0 0.92 0a0.46 0.46 0 1 0 -0.92 0z',
    mouth:
      'M21.66 0.22A1.7 1.7 0 0 1 18.65 0.22A0.31 0.31 0 0 1 19.2 -0.07A1.08 1.08 0 0 0 21.11 -0.07A0.31 0.31 0 0 1 21.66 0.22z',
    smile: { eyes: 'M15.85 -1.5Q17.06 -3.6 18.27 -1.5M22.04 -1.5Q23.25 -3.6 24.46 -1.5', width: 0.8 },
    grin: null,
    cheeks: [
      { x: 15.2, y: 0.6 },
      { x: 25.4, y: 0.6 },
    ],
  },
];

/** The WeClone mark with faces that blink, or smile when `happy`. `view` widens the picture for what `children` draw
 * round it, in the mark's own units. */
export function Mark({
  className,
  view = MARK_VIEW,
  happy = false,
  children,
}: {
  className?: string;
  view?: { x: number; y: number; w: number; h: number };
  happy?: boolean;
  children?: ReactNode;
}) {
  const id = `mark${useId().replace(/[^\w-]/g, '')}`;
  return (
    <svg
      className={className ? `build-mark ${className}` : 'build-mark'}
      viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
    >
      <defs>
        {FACES.map(({ who, gradient: { stops, ...line } }) => (
          <linearGradient key={who} id={`${id}-${who}`} gradientUnits="userSpaceOnUse" {...line}>
            <stop offset="0" stopColor={stops[0]} />
            <stop offset="1" stopColor={stops[1]} />
          </linearGradient>
        ))}
      </defs>
      {FACES.map((face) => (
        <g key={face.who} className={`build-mark__face build-mark__face--${face.who}`}>
          <path fill={`url(#${id}-${face.who})`} d={face.body} />
          <path fill="#fff" fillOpacity=".8" d={face.shine} />
          {happy ? (
            <>
              {face.cheeks.map((cheek) => (
                <ellipse key={cheek.x} className="build-mark__cheek" cx={cheek.x} cy={cheek.y} rx="1.7" ry="1" />
              ))}
              {face.grin && <path fill={face.ink} d={face.grin} />}
              <path
                d={face.smile.eyes}
                fill="none"
                stroke={face.ink}
                strokeWidth={face.smile.width}
                strokeLinecap="round"
              />
            </>
          ) : (
            <>
              <path fill={face.ink} d={face.mouth} />
              <g className="build-mark__eyes">
                <path fill={face.ink} d={face.eyes} />
                <path fill="#fff" d={face.glints} />
              </g>
            </>
          )}
        </g>
      ))}
      {children}
    </svg>
  );
}
