import { useId, type CSSProperties } from 'react';
import { Mark, MARK_VIEW } from './Mark';

// The mark celebrating a finished build: you in a party hat, the clone blowing a party horn that unrolls every few
// seconds and puffs out confetti, and on arrival a burst of confetti from behind both. Everything is in the mark's
// units; the confetti flies in the page, from points given as shares of the picture. The picture is centred on the
// mark, so the two of them sit in the middle and the horn reaches out to the right.
const VIEW = { x: MARK_VIEW.x + MARK_VIEW.w / 2 - 36, y: -29, w: 72, h: 46 };
// The clone's mouth, where the horn starts, and the horn's angle below level.
const MOUTH = { x: 20.15, y: 0.4 };
const HORN_ANGLE = 14;
// The confetti's colours: the ring's, then yours and the clone's.
const PALETTE = ['--ring-1', '--ring-2', '--ring-3', '--ring-4', '--ring-5', '--you', '--clone'];
const SHAPES = ['strip', 'square', 'dot'] as const;

const share = (x: number, y: number) => ({
  left: `${((x - VIEW.x) / VIEW.w) * 100}%`,
  top: `${((y - VIEW.y) / VIEW.h) * 100}%`,
});
const MIDDLE = share(MARK_VIEW.x + MARK_VIEW.w / 2, MARK_VIEW.y + MARK_VIEW.h / 2);
const radians = (degrees: number) => (degrees * Math.PI) / 180;
// The horn's end, unrolled: its paper reaches 22 units along it.
const TIP = share(MOUTH.x + 22 * Math.cos(radians(HORN_ANGLE)), MOUTH.y + 22 * Math.sin(radians(HORN_ANGLE)) - 1);

/** A number in [0, 1) that is the same for the same arguments, so the confetti falls the same way every time. */
function noise(index: number, salt: number): number {
  const value = Math.sin(index * 127.1 + salt * 311.7) * 43758.5453;
  return value - Math.floor(value);
}

interface Bit {
  dx: number;
  dy: number;
  fall: number;
  spin: number;
  delay: number;
  time: number;
}

/** `count` bits of confetti thrown from `from` within `spread` degrees of `aim` (0 is right, -90 up). */
function throwBits(count: number, aim: number, spread: number, reach: [number, number], salt: number): Bit[] {
  return Array.from({ length: count }, (_, index) => {
    const angle = radians(aim + (noise(index, salt) - 0.5) * spread);
    const distance = reach[0] + noise(index, salt + 1) * (reach[1] - reach[0]);
    return {
      dx: Math.cos(angle) * distance,
      dy: Math.sin(angle) * distance,
      fall: distance * (0.4 + noise(index, salt + 2) * 0.5),
      spin: (noise(index, salt + 3) - 0.5) * 1080,
      delay: noise(index, salt + 4) * 0.15,
      time: 1.8 + noise(index, salt + 5) * 1.2,
    };
  });
}

const BURST = throwBits(44, -90, 230, [80, 190], 1);
const PUFF = throwBits(9, HORN_ANGLE - 25, 80, [28, 70], 7);

function Confetti({ bits, from, kind }: { bits: Bit[]; from: { left: string; top: string }; kind: 'burst' | 'puff' }) {
  return bits.map((bit, index) => (
    <span
      key={index}
      className={`build-party__bit build-party__bit--${kind}`}
      data-shape={SHAPES[index % SHAPES.length]}
      style={
        {
          ...from,
          '--c': `var(${PALETTE[index % PALETTE.length]})`,
          '--dx': `${bit.dx.toFixed(1)}px`,
          '--dy': `${bit.dy.toFixed(1)}px`,
          '--fall': `${bit.fall.toFixed(1)}px`,
          '--spin': `${bit.spin.toFixed(0)}deg`,
          '--delay': `${bit.delay.toFixed(2)}s`,
          '--time': `${bit.time.toFixed(2)}s`,
        } as CSSProperties
      }
    />
  ));
}

/** The horn rolled up at rest: a spiral from the mouthpiece, a turn and a quarter, narrowing inwards. */
function coil(): string {
  const center = { x: 5.8, y: -2 };
  const start = { x: 3.4, y: 0 };
  const from = Math.atan2(start.y - center.y, start.x - center.x);
  const r0 = Math.hypot(start.x - center.x, start.y - center.y);
  const steps = 40;
  return Array.from({ length: steps + 1 }, (_, step) => {
    const t = step / steps;
    const angle = from - t * radians(450);
    const r = r0 - t * (r0 - 1);
    return `${step ? 'L' : 'M'}${(center.x + r * Math.cos(angle)).toFixed(2)} ${(center.y + r * Math.sin(angle)).toFixed(2)}`;
  }).join('');
}

const STRIP = 'M3.4 0H19.6C21.6 0 22.6 -1.3 22.3 -2.7C22 -4 20.4 -4.2 19.8 -3';
// The mouthpiece: a tube narrowing into the clone's pursed lips, its top catching the light.
const MOUTHPIECE =
  'M0.2 -0.5L3.5 -0.95Q4.1 -0.95 4.1 -0.4V0.4Q4.1 0.95 3.5 0.95L0.2 0.5Q-0.15 0.5 -0.15 0Q-0.15 -0.5 0.2 -0.5Z';

/** The mark celebrating, with confetti. Mount it again to celebrate again. */
export function Party() {
  const id = `party${useId().replace(/[^\w-]/g, '')}`;
  return (
    <span className="build-party" style={{ aspectRatio: `${VIEW.w} / ${VIEW.h}` }} aria-hidden="true">
      <span className="build-party__behind">
        <Confetti bits={BURST} from={MIDDLE} kind="burst" />
      </span>
      <Mark className="build-party__mark" view={VIEW} happy>
        <defs>
          <linearGradient id={`${id}-hat`} x1="0" y1="1" x2="0" y2="0">
            <stop offset="0" style={{ stopColor: 'var(--ring-3)' }} />
            <stop offset="1" style={{ stopColor: 'var(--ring-4)' }} />
          </linearGradient>
          <linearGradient id={`${id}-tube`} gradientUnits="userSpaceOnUse" x1="0" y1="-1" x2="0" y2="1">
            <stop offset="0" style={{ stopColor: 'oklch(from var(--ring-5) calc(l + 0.14) calc(c * 0.8) h)' }} />
            <stop offset="1" style={{ stopColor: 'oklch(from var(--ring-5) calc(l - 0.08) c h)' }} />
          </linearGradient>
          <linearGradient id={`${id}-paper`} gradientUnits="userSpaceOnUse" x1="3" y1="0" x2="22" y2="0">
            <stop offset="0" style={{ stopColor: 'var(--ring-1)' }} />
            <stop offset="0.5" style={{ stopColor: 'var(--ring-2)' }} />
            <stop offset="1" style={{ stopColor: 'var(--ring-4)' }} />
          </linearGradient>
          <clipPath id={`${id}-cone`}>
            <path d="M-6.5 -10.1L-7 -25.5L2.5 -11.7Q-2 -12.76 -6.5 -10.1Z" />
          </clipPath>
        </defs>
        <g className="build-party__hat">
          <path d="M-6.5 -10.1L-7 -25.5L2.5 -11.7Q-2 -12.76 -6.5 -10.1Z" fill={`url(#${id}-hat)`} />
          <g clipPath={`url(#${id}-cone)`} className="build-party__stripes">
            <path d="M-12 -13.5L8 -18M-12 -19.5L8 -24" />
          </g>
          <circle cx="-7" cy="-26" r="2" fill="var(--ring-5)" />
        </g>
        <ellipse className="build-party__lips" cx={MOUTH.x} cy={MOUTH.y} rx="1.3" ry="1.1" />
        <g transform={`translate(${MOUTH.x} ${MOUTH.y}) rotate(${HORN_ANGLE})`}>
          <path className="build-party__coil" d={coil()} stroke={`url(#${id}-paper)`} />
          <g className="build-party__strip">
            <path className="build-party__paper" d={STRIP} stroke={`url(#${id}-paper)`} />
            <path className="build-party__bands" d={STRIP} />
          </g>
          <path d={MOUTHPIECE} fill={`url(#${id}-tube)`} />
          <path className="build-party__sheen" d="M0.7 -0.32L3.4 -0.68" />
        </g>
      </Mark>
      <span className="build-party__front">
        <Confetti bits={PUFF} from={TIP} kind="puff" />
      </span>
    </span>
  );
}
