import { useEffect, useId, useState, type ReactNode } from 'react';
import { FACES } from './Mark';
import { Quips } from './Quips';
import type { StageKey } from './progress';

// The two faces of the mark taking turns at work while a build runs, one thing after another. Each face is a layer of
// its own so that the one at work can come to the front (build.css). Units are the mark's (Mark.tsx): you round 0 0
// with radius 12, the clone round 16.44 -1.91 with radius 10.32. The view takes in both faces and what either holds.
const VIEW = '-27.5 -15.5 74 33';
const HOLD_MS = 6400;
const LEAVE_MS = 420;

// The props' colours, each a light-to-deep diagonal like the faces' own gradients, and the hands' in the faces'.
const SHADES = {
  yellow: ['#ffe590', '#ffc23e'],
  folder: ['#ffcf55', '#f2a21f'],
  cream: ['#fffdf7', '#f1e6d2'],
  edge: ['#efe3cc', '#d9c8a8'],
  blue: ['#8fdcff', '#3ea1f6'],
  peach: ['#fec8a0', '#fd9a70'],
  mint: ['#a4f0d6', '#36c193'],
  glass: ['#f2fbff', '#c4e8ff'],
  phone: ['#d3d9e2', '#909bac'],
  silver: ['#f4f7fb', '#c7d1de'],
  deck: ['#e6ecf3', '#b3bfcd'],
  navy: ['#4a5b82', '#27324f'],
  youHand: ['#70d1fc', '#3ea1f6'],
  cloneHand: ['#fdb489', '#fec394'],
};
// The lines of writing on them.
const INK = { blue: '#5bbcf8', peach: '#fda27a', mint: '#4fcfa6', yellow: '#ffcf5c', grey: '#dccfb8' };

type Who = 'you' | 'clone';
type Paint = (name: keyof typeof SHADES | Who | 'screen') => string;
/** One thing done by one face: the prop, round 0 0 at the middle of its foot, and anything held in the hand. */
type Act = { key: string; who: Who; prop: (paint: Paint) => ReactNode; held?: ReactNode };

/** A line of writing; `drawn` names the animation that writes it out. */
function Ink({ d, ink, bold = false, drawn }: { d: string; ink: keyof typeof INK; bold?: boolean; drawn?: string }) {
  const className = ['build-prop__ink', bold && 'build-prop__ink--bold', drawn && `build-prop__drawn ${drawn}`]
    .filter(Boolean)
    .join(' ');
  return <path className={className} stroke={INK[ink]} d={d} pathLength={drawn ? 1 : undefined} />;
}

/** A page of the open book with its spine at 0, the right one; mirrored, the left. */
function Page({ paint }: { paint: Paint }) {
  return (
    <>
      <path fill={paint('cream')} d="M0 -10.4Q3.5 -11.8 6.8 -10.8V-0.9Q3.5 -1.9 0 -0.5Z" />
      <Ink ink="blue" d="M1.5 -8.3H5.2" />
      <Ink ink="grey" d="M1.5 -6.1H5.2" />
      <Ink ink="peach" d="M1.5 -3.9H4.2" />
    </>
  );
}

// The chat on the phone, longer than its screen so it can scroll: the other side in peach on the left, you in blue
// on the right.
const CHAT = Array.from({ length: 11 }, (_, index) => {
  const [ink, width] = (
    [
      ['peach', 4],
      ['blue', 3],
      ['peach', 2.6],
      ['blue', 4.2],
      ['peach', 3.4],
    ] as const
  )[index % 5];
  return { ink, width, x: ink === 'peach' ? -3 : 3 - width, y: -10.9 + index * 2.4 };
});

// In turn: getting out the chats, scrolling them, looking through them, reading, typing, taking notes.
const ACTS: Act[] = [
  {
    key: 'folder',
    who: 'clone',
    prop: (paint) => (
      <>
        <path
          fill={paint('folder')}
          d="M-7 -9.4Q-7 -11 -5.4 -11H-2.8Q-1.9 -11 -1.4 -10.3L-0.9 -9.6H5.4Q7 -9.6 7 -8V-1.6Q7 0 5.4 0H-5.4Q-7 0 -7 -1.6Z"
        />
        <g className="build-prop__sheet">
          <rect fill={paint('cream')} x="-5.2" y="-12.6" width="10.4" height="9" rx="1.3" />
          <Ink ink="blue" bold d="M-3.4 -10.7H0.6" />
          <Ink ink="peach" bold d="M-0.4 -8.5H3.4" />
          <Ink ink="blue" bold d="M-3.4 -6.3H-0.6" />
        </g>
        <path
          fill={paint('yellow')}
          d="M-7.6 -6.2Q-7.6 -7.4 -6.4 -7.4H6.4Q7.6 -7.4 7.6 -6.2L7.1 -1.6Q6.9 0 5.3 0H-5.3Q-6.9 0 -7.1 -1.6Z"
        />
        <path className="build-prop__shine" d="M-5.8 -6.1H-2" />
      </>
    ),
  },
  {
    key: 'phone',
    who: 'you',
    prop: (paint) => (
      <>
        <rect fill={paint('phone')} x="-4.6" y="-13" width="9.2" height="13.4" rx="2.2" />
        <rect fill="#fffdf8" x="-3.5" y="-11.7" width="7" height="10.2" rx="1.2" />
        <g clipPath={paint('screen')}>
          <g className="build-prop__chat">
            {CHAT.map(({ ink, width, x, y }) => (
              <rect key={y} fill={paint(ink)} x={x} y={y} width={width} height="1.7" rx="0.85" />
            ))}
          </g>
        </g>
        <path className="build-prop__shine" d="M-4.05 -3.4V-9.6" />
        <path className="build-prop__shine" strokeOpacity="0.55" d="M-0.9 -12.35H0.9" />
      </>
    ),
  },
  {
    key: 'lens',
    who: 'clone',
    prop: (paint) => (
      <g className="build-prop__sweep">
        <path className="build-prop__handle" stroke="#97a2b2" d="M0.1 -0.4L1 -3.6" />
        <circle className="build-prop__glass" cx="2.4" cy="-8.8" r="4.3" fill={paint('glass')} stroke={paint('mint')} />
        <path className="build-prop__shine" d="M-0.33 -9.79A2.9 2.9 0 0 1 1.41 -11.53" />
        <g transform="translate(7.8 -13.6)">
          <path
            className="build-prop__spark"
            fill="#ffc93d"
            d="M0 -2Q0.3 -0.3 2 0Q0.3 0.3 0 2Q-0.3 0.3 -2 0Q-0.3 -0.3 0 -2Z"
          />
        </g>
      </g>
    ),
  },
  {
    key: 'book',
    who: 'you',
    prop: (paint) => (
      <>
        <path
          fill={paint('mint')}
          d="M0 -9.4Q-3.8 -10.9 -7.8 -9.8V0.4Q-3.8 -0.7 0 0.7Q3.8 -0.7 7.8 0.4V-9.8Q3.8 -10.9 0 -9.4Z"
        />
        <g transform="scale(-1 1)">
          <Page paint={paint} />
        </g>
        <Page paint={paint} />
        <g className="build-prop__turn">
          <Page paint={paint} />
        </g>
      </>
    ),
  },
  {
    key: 'laptop',
    who: 'clone',
    prop: (paint) => (
      <>
        <rect fill={paint('silver')} x="-6.8" y="-12.6" width="13.6" height="9.4" rx="1.6" />
        <rect fill={paint('navy')} x="-5.8" y="-11.6" width="11.6" height="7.4" rx="0.9" />
        <Ink ink="blue" drawn="build-prop__type--1" d="M-4.5 -10H0.3" />
        <Ink ink="peach" drawn="build-prop__type--2" d="M-3.2 -8.3H2.4" />
        <Ink ink="mint" drawn="build-prop__type--3" d="M-3.2 -6.6H3.8" />
        <Ink ink="yellow" drawn="build-prop__type--4" d="M-4.5 -4.9H-1.6" />
        <path
          fill={paint('deck')}
          d="M-7.8 -3.2H7.8Q8.9 -3.2 8.6 -2.1L8.2 -0.6Q7.9 0.4 6.9 0.4H-6.9Q-7.9 0.4 -8.2 -0.6L-8.6 -2.1Q-8.9 -3.2 -7.8 -3.2Z"
        />
        <path className="build-prop__shine" d="M-6.4 -2.4H-2.8" />
      </>
    ),
  },
  {
    key: 'notes',
    who: 'you',
    // A pad lying flat, a little out to your left, and a pencil held round the middle with its point down.
    prop: (paint) => (
      <g transform="translate(-2.5 0)">
        <path fill={paint('edge')} d="M-5.8 0.4H5.8V0.9Q5.8 1.5 5.2 1.5H-5.2Q-5.8 1.5 -5.8 0.9Z" />
        <path
          fill={paint('cream')}
          d="M-3.6 -6.2H3.6Q4.4 -6.2 4.57 -5.42L5.63 -0.38Q5.8 0.4 5 0.4H-5Q-5.8 0.4 -5.63 -0.38L-4.57 -5.42Q-4.4 -6.2 -3.6 -6.2Z"
        />
        <path
          fill={paint('mint')}
          d="M-3.6 -6.2H3.6Q4.4 -6.2 4.57 -5.42L4.65 -5H-4.65L-4.57 -5.42Q-4.4 -6.2 -3.6 -6.2Z"
        />
        <path className="build-prop__shine" strokeOpacity="0.9" d="M-2.4 -5.6h0M0 -5.6h0M2.4 -5.6h0" />
        <Ink ink="blue" drawn="build-prop__write--1" d="M-3 -3.9H2.8" />
        <Ink ink="grey" drawn="build-prop__write--2" d="M-3.3 -2.3H3.1" />
        <Ink ink="peach" drawn="build-prop__write--3" d="M-3.6 -0.7H0.6" />
      </g>
    ),
    held: (
      <>
        <path className="build-prop__pencil" stroke="#ffc947" strokeLinecap="butt" d="M-1.49 3.62L1.6 -3.89" />
        <polygon fill="#fbe0b0" points="-2.1,5.1 -2.55,3.18 -0.43,4.06" />
        <circle fill="#3d4660" cx="-2" cy="4.87" r="0.5" />
        <path className="build-prop__pencil" stroke="#f68aa4" d="M1.6 -3.89L1.98 -4.81" />
      </>
    ),
  },
];

// Where each face holds things: you out on your left, the clone out on its right, both clear of the faces. The
// clone's are drawn at 0.86, so that at its full size they match yours. The hand rests `rest` in from there, where
// the mark has its tail dot.
const SEATS = {
  you: { centre: [0, 0], r: 12, grip: 'translate(-16.29 13.27)', rest: 7, hand: 2.64 },
  clone: { centre: [16.44, -1.91], r: 10.32, grip: 'translate(31.6 9.24) scale(0.86)', rest: -7.49, hand: 2.79 },
};

/** One face, with the prop and the hand of the act it is at, if any. */
function Layer({ who, act, paint, children }: { who: Who; act?: Act; paint: Paint; children?: ReactNode }) {
  const face = FACES[who === 'you' ? 0 : 1];
  const { centre, r, grip, rest, hand } = SEATS[who];
  const [x, y] = centre;
  return (
    <svg className="build-workers__layer" data-who={who} data-act={act?.key} viewBox={VIEW}>
      {children}
      <g transform={`translate(${x} ${y})`}>
        <g className="build-workers__face">
          <g transform={`translate(${-x} ${-y})`}>
            <circle className="build-workers__body" cx={x} cy={y} r={r} fill={paint(who)} />
            <path fill="#fff" fillOpacity=".8" d={face.shine} />
            <g className="build-workers__feat">
              <path fill={face.ink} d={face.mouth} />
              <g className="build-workers__look">
                <g className="build-workers__scan">
                  <g className="build-workers__eyes">
                    <path fill={face.ink} d={face.eyes} />
                    <path fill="#fff" d={face.glints} />
                  </g>
                </g>
              </g>
            </g>
            <g transform={grip}>
              {act && (
                <g transform="scale(1.15)">
                  <g key={act.key} className="build-prop">
                    {act.prop(paint)}
                  </g>
                </g>
              )}
              <g className="build-workers__hand">
                <g>
                  <g transform={`translate(${rest} 0)`}>
                    {act?.held && (
                      <g key={act.key} className="build-prop">
                        {act.held}
                      </g>
                    )}
                    <circle r={hand} fill={paint(`${who}Hand`)} />
                  </g>
                </g>
              </g>
            </g>
          </g>
        </g>
      </g>
    </svg>
  );
}

/** The faces at work, saying a line about the stage the build is on: a few seconds at each thing, then the other face
 * takes over with the next. */
export function Workers({ stage }: { stage: StageKey }) {
  const id = `workers${useId().replace(/[^\w-]/g, '')}`;
  const paint: Paint = (name) => `url(#${id}-${name})`;
  const [step, setStep] = useState(0);
  const [leaving, setLeaving] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(
      () => {
        if (leaving) setStep((value) => value + 1);
        setLeaving(!leaving);
      },
      leaving ? LEAVE_MS : HOLD_MS,
    );
    return () => window.clearTimeout(timer);
  }, [step, leaving]);
  const act = ACTS[step % ACTS.length];
  return (
    <div className="build-workers" data-front={act.who} data-leaving={leaving || undefined}>
      <Quips key={stage} stage={stage} className="build-workers__say" />
      <div className="build-workers__duo">
        <Layer who="you" act={act.who === 'you' ? act : undefined} paint={paint}>
          {/* The paints for both layers. */}
          <defs>
            {FACES.map(({ who, gradient: { stops, ...line } }) => (
              <linearGradient key={who} id={`${id}-${who}`} gradientUnits="userSpaceOnUse" {...line}>
                <stop offset="0" stopColor={stops[0]} />
                <stop offset="1" stopColor={stops[1]} />
              </linearGradient>
            ))}
            {Object.entries(SHADES).map(([name, [from, to]]) => (
              <linearGradient key={name} id={`${id}-${name}`} x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor={from} />
                <stop offset="1" stopColor={to} />
              </linearGradient>
            ))}
            <clipPath id={`${id}-screen`}>
              <rect x="-3.5" y="-11.7" width="7" height="10.2" rx="1.2" />
            </clipPath>
          </defs>
        </Layer>
        <Layer who="clone" act={act.who === 'clone' ? act : undefined} paint={paint} />
      </div>
    </div>
  );
}
