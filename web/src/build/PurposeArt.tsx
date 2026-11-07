import { useId, type CSSProperties, type ReactNode } from 'react';
import favicon from '../assets/logo/weclone-favicon.svg';
import mark from '../assets/logo/weclone-mark.svg';
import type { Purpose } from './progress';

// Both pictures start from the same chats, in a card on the left. In the assistant's, your portrait starts empty and
// each message flies out to one of its five dimensions and fills it; once all five are full, what they hold flows
// into you at the centre. In the clone's, your messages fly into the clone, which then answers a friend in a chat of
// its own. Every colour is a token, so they follow the theme.
//
// They move on one loop, the assistant's first and then the clone's. Each moving part starts at its --at, in seconds
// into the loop.

// The Logo's highlight on a bubble of radius 12, as the map's balls carry it (atlas/paint.ts).
const GLOSS =
  'M-9.22 -3.92A10.02 10.02 0 0 1 -1.57 -9.9A0.78 0.78 0 0 1 -1.32 -8.36A8.46 8.46 0 0 0 -7.79 -3.31A0.78 0.78 0 0 1 -9.22 -3.92z';
// radius-tail: the corner of a chat bubble nearest its speaker.
const TAIL = 4;
const LINE = 18;
const ROW = 22;
const PAD = 10;
// The loop's length (build.css, --build-art-loop) and a flight's (build-art-fly-x), in seconds.
const LOOP = 10;
const FLIGHT = 0.5;

type Point = { x: number; y: number };
type Box = Point & { w: number; h: number };
interface Line {
  by: 'you' | 'other' | 'clone';
  w: number;
  text: number;
  end?: 'tilde' | 'dots';
}
type Placed = Line & Point;

const vars = (values: Record<`--${string}`, string | number>) => values as CSSProperties;

/** A chat bubble: a box with round ends, the corner nearest the speaker the tail corner. */
function bubble(x: number, y: number, w: number, h: number, tail: 'left' | 'right'): string {
  const r = h / 2;
  const bl = tail === 'left' ? TAIL : r;
  const br = tail === 'right' ? TAIL : r;
  return (
    `M${x + r} ${y}H${x + w - r}A${r} ${r} 0 0 1 ${x + w} ${y + r}V${y + h - br}` +
    `A${br} ${br} 0 0 1 ${x + w - br} ${y + h}H${x + bl}A${bl} ${bl} 0 0 1 ${x} ${y + h - bl}V${y + r}` +
    `A${r} ${r} 0 0 1 ${x + r} ${y}Z`
  );
}

/** The point that bends a line from a to b a little, always to the same side, as the map's edges bend. */
function bend(a: Point, b: Point, amount: number): Point {
  return { x: (a.x + b.x) / 2 - (b.y - a.y) * amount, y: (a.y + b.y) / 2 + (b.x - a.x) * amount };
}

function polar(from: Point, degrees: number, distance: number): Point {
  const angle = (degrees * Math.PI) / 180;
  return { x: from.x + Math.cos(angle) * distance, y: from.y + Math.sin(angle) * distance };
}

/** Messages down a card under its header: a friend's on the left, yours and the clone's on the right. */
function place(card: Box, lines: Line[]): Placed[] {
  return lines.map((line, index) => ({
    ...line,
    x: line.by === 'other' ? card.x + PAD : card.x + card.w - PAD - line.w,
    y: card.y + 32 + index * ROW,
  }));
}

// Where a flight leaves a message, and where one lands in it.
const exit = (line: Placed): Point => ({ x: line.x + line.w - 8, y: line.y + LINE / 2 });
const entry = (line: Placed): Point => ({ x: line.x + 10, y: line.y + LINE / 2 });

/** The timing that moves one coordinate along a quadratic curve: the curve's own polynomial as a cubic-bezier,
 * whose x values of 1/3 and 2/3 keep time linear. */
function along(from: number, control: number, to: number): string {
  const pull = Math.abs(to - from) < 1 ? 0.5 : (control - from) / (to - from);
  return `cubic-bezier(0.3333, ${((2 * pull) / 3).toFixed(3)}, 0.6667, ${((1 + 2 * pull) / 3).toFixed(3)})`;
}

/** A bubble and its tail dot leaving `from` at `at` and curving towards `control` on the way to `to`. Whatever is
 * drawn after it covers it at either end. */
function Flight({
  from,
  control,
  to,
  at,
  color,
}: {
  from: Point;
  control: Point;
  to: Point;
  at: number;
  color: string;
}) {
  const style = {
    '--c': color,
    '--dx': `${(to.x - from.x).toFixed(2)}px`,
    '--dy': `${(to.y - from.y).toFixed(2)}px`,
    '--fly-x': along(from.x, control.x, to.x),
    '--fly-y': along(from.y, control.y, to.y),
  };
  return [
    { lag: 0.07, r: 1.7 },
    { lag: 0, r: 4 },
  ].map(({ lag, r }) => (
    <g key={r} className="build-art__fly" style={vars({ ...style, '--at': at + lag })}>
      <g>
        <circle className="build-art__particle" cx={from.x} cy={from.y} r={r} />
      </g>
    </g>
  ));
}

/** A ring that spreads from a circle and fades as something arrives in it. */
function Ripple({
  center,
  r,
  color,
  at,
  spread,
}: {
  center: Point;
  r: number;
  color: string;
  at: number;
  spread?: number;
}) {
  return (
    <circle
      className="build-art__ripple"
      cx={center.x}
      cy={center.y}
      r={r}
      style={vars({ '--c': color, '--at': at, ...(spread && { '--spread': spread }) })}
    />
  );
}

/** One message: its bubble, its text and the habit it ends on. A message something is read from lights up in that
 * thing's colour as it leaves. */
function Said({ line, read }: { line: Placed; read?: { at: number; color: string } }) {
  const { by, x, y, w, text, end } = line;
  const left = x + 12;
  const after = left + text + 7;
  const mid = y + LINE / 2;
  return (
    <g className={`build-art__${by}`}>
      <path
        className={read ? 'build-art__bubble build-art__read' : 'build-art__bubble'}
        style={read && vars({ '--at': read.at, '--c': read.color })}
        d={bubble(x, y, w, LINE, by === 'other' ? 'left' : 'right')}
      />
      <rect className="build-art__text" x={left} y={mid - 2.5} width={text} height="5" rx="2.5" />
      {end === 'tilde' && <path className="build-art__tilde" d={`M${after} ${mid + 0.6}q2.5 -3.4 5 0t5 0`} />}
      {end === 'dots' &&
        [0, 1, 2].map((dot) => (
          <circle key={dot} className="build-art__dot" cx={after + 1.5 + dot * 5} cy={mid} r="1.5" />
        ))}
    </g>
  );
}

/** A chat window: the friend's avatar and name on top, then the messages, and on a live chat the box to type in. */
function Card({ box, id, live, children }: { box: Box; id: string; live?: boolean; children: ReactNode }) {
  const avatar = { x: box.x + 16, y: box.y + 16 };
  return (
    <>
      <rect
        className="build-art__card"
        x={box.x}
        y={box.y}
        width={box.w}
        height={box.h}
        rx="14"
        filter={`url(#${id}-lift)`}
      />
      <circle className="build-art__avatar" cx={avatar.x} cy={avatar.y} r="8" />
      <rect className="build-art__name" x={avatar.x + 12} y={avatar.y - 3} width="40" height="6" rx="3" />
      {live && (
        <rect
          className="build-art__input"
          x={box.x + PAD}
          y={box.y + box.h - PAD - 16}
          width={box.w - 2 * PAD}
          height="16"
          rx="8"
        />
      )}
      {children}
    </>
  );
}

/** A dimension's ball as the map draws it: its colour, the sheen from the upper left, the Logo's highlight and a rim.
 * It grows and shrinks on `grow`. */
function Ball({ pos, r, color, sheen, grow }: { pos: Point; r: number; color: string; sheen: string; grow: string }) {
  return (
    <g transform={`translate(${pos.x} ${pos.y})`} style={vars({ '--c': color })}>
      <g className="build-art__grow" style={vars({ '--at': 0, '--grow': grow })}>
        <circle className="build-art__fill" r={r} />
        <circle r={r} fill={`url(#${sheen})`} />
        <path className="build-art__gloss" d={GLOSS} transform={`scale(${r / 12})`} />
        <circle className="build-art__rim" r={r - 0.9} />
      </g>
    </g>
  );
}

function Defs({ id }: { id: string }) {
  return (
    <defs>
      <radialGradient id={`${id}-sheen`} cx="0.32" cy="0.24" r="0.7">
        <stop offset="0" stopColor="#fff" stopOpacity="0.3" />
        <stop offset="0.55" stopColor="#fff" stopOpacity="0" />
      </radialGradient>
      <filter id={`${id}-lift`} x="-30%" y="-30%" width="160%" height="170%">
        <feDropShadow className="build-art__shadow" dx="0" dy="6" stdDeviation="8" />
      </filter>
      {/* Blurs over the whole picture: a straight line has no width or no height of its own to measure a region by. */}
      {[
        { name: 'soft', blur: 1.6 },
        { name: 'glow', blur: 4 },
      ].map(({ name, blur }) => (
        <filter key={name} id={`${id}-${name}`} filterUnits="userSpaceOnUse" x="0" y="0" width="400" height="190">
          <feGaussianBlur stdDeviation={blur} />
        </filter>
      ))}
    </defs>
  );
}

// The chats both pictures start from: a friend's messages and yours, with the habits you end yours on.
const HISTORY_CARD: Box = { x: 16, y: 20, w: 120, h: 150 };
const HISTORY = place(HISTORY_CARD, [
  { by: 'other', w: 74, text: 48 },
  { by: 'you', w: 78, text: 38, end: 'dots' },
  { by: 'other', w: 58, text: 34 },
  { by: 'you', w: 86, text: 48, end: 'tilde' },
  { by: 'you', w: 50, text: 26 },
]);

// Your portrait as the map draws it: you at the centre and the five dimensions round you in their colours, the first
// facing the chats. The chats fill them in turn, top to bottom and clockwise, so each flight passes between the
// dimensions already full and you.
const CENTER = { x: 274, y: 95 };
const DIM_R = 11;
const DIMS = ['preference', 'plan', 'condition', 'goal', 'identity'].map((name, index, all) => ({
  color: `var(--dim-${name})`,
  pos: polar(CENTER, 180 + (index * 360) / all.length, 60),
}));
const readAt = (index: number) => 0.4 + 0.55 * index;
const landAt = (index: number) => readAt(index) + FLIGHT;
// How far every flight arcs up on its way, as if tossed.
const TOSS = -0.06;

// The portrait empties down to you as the loop starts, and each dimension and its line come back as its message lands. Once the last one
// is full, a swell of light runs along every line into you, and you glow as it arrives.
const EMPTIED = 0.3;
const FLOW = landAt(DIMS.length - 1) + 0.3;
const GLOW = FLOW + 0.45;
// The swell's soft tail and bright head, in hundredths of a line (pathLength).
const SWELL = [
  { part: 'tail', dash: 34 },
  { part: 'head', dash: 12 },
];

/** build-art-grow's timing over the loop as a linear() easing: full, emptied, empty until `at`, then swelling past
 * full for a moment as it fills. */
function filling(at: number): string {
  const share = (seconds: number) => `${((seconds / LOOP) * 100).toFixed(2)}%`;
  return `linear(1, 0 ${share(EMPTIED)}, 0 ${share(at)}, 1.25 ${share(at + 0.15)}, 1 ${share(at + 0.45)}, 1)`;
}

/** The chats, read one message at a time into the empty dimensions of your portrait, which then flow into you. */
function AssistantArt({ id }: { id: string }) {
  const sheen = `${id}-sheen`;
  return (
    <>
      <Card box={HISTORY_CARD} id={id}>
        {HISTORY.map((line, index) => (
          <Said key={index} line={line} read={{ at: readAt(index) - 0.15, color: DIMS[index].color }} />
        ))}
      </Card>
      {DIMS.map((dim, index) => (
        <circle
          key={dim.color}
          className="build-art__halo build-art__grow"
          cx={dim.pos.x}
          cy={dim.pos.y}
          r="22"
          style={vars({ '--c': dim.color, '--at': 0, '--grow': filling(landAt(index)) })}
        />
      ))}
      {DIMS.map((dim, index) => {
        const line = `M${dim.pos.x} ${dim.pos.y}L${CENTER.x} ${CENTER.y}`;
        return (
          <g key={dim.color}>
            <path className="build-art__edge" d={line} style={vars({ '--at': 0, '--grow': filling(landAt(index)) })} />
            {SWELL.map(({ part, dash }) => (
              <path
                key={part}
                className={`build-art__flow build-art__flow--${part}`}
                d={line}
                pathLength="100"
                filter={part === 'tail' ? `url(#${id}-soft)` : undefined}
                style={vars({
                  '--c': dim.color,
                  '--at': FLOW,
                  '--dash': dash,
                  // Both heads run from the dimension's centre until the longest tail has gone into you.
                  '--to': dash - 100 - SWELL[0].dash,
                })}
              />
            ))}
            <Ripple center={dim.pos} r={DIM_R} color={dim.color} at={landAt(index)} />
          </g>
        );
      })}
      {DIMS.map((dim, index) => {
        const from = exit(HISTORY[index]);
        return (
          <Flight
            key={dim.color}
            from={from}
            control={bend(from, dim.pos, TOSS)}
            to={dim.pos}
            at={readAt(index)}
            color={dim.color}
          />
        );
      })}
      {DIMS.map((dim, index) => (
        <Ball key={dim.color} pos={dim.pos} r={DIM_R} color={dim.color} sheen={sheen} grow={filling(landAt(index))} />
      ))}
      <circle
        className="build-art__glow"
        cx={CENTER.x}
        cy={CENTER.y}
        r="19"
        filter={`url(#${id}-glow)`}
        style={vars({ '--c': 'var(--you)', '--at': GLOW })}
      />
      <g className="build-art__beat" style={vars({ '--at': GLOW })}>
        <circle className="build-art__root" cx={CENTER.x} cy={CENTER.y} r="19" filter={`url(#${id}-lift)`} />
        <image href={favicon} x={CENTER.x - 11} y={CENTER.y - 11} width="22" height="22" />
      </g>
    </>
  );
}

// The clone's own chat on the right: a friend asks, the clone answers with your habits. The box to type in shows the
// chat is live.
const CHAT_CARD: Box = { x: 264, y: 20, w: 120, h: 150 };
const CHAT = place(CHAT_CARD, [
  { by: 'other', w: 72, text: 46 },
  { by: 'clone', w: 80, text: 40, end: 'dots' },
  { by: 'clone', w: 54, text: 30 },
  { by: 'clone', w: 76, text: 38, end: 'tilde' },
]);
// The mark between the cards, and the clone's bubble in it: weclone-mark.svg's viewBox, and the clone's circle in
// those units.
const MARK_VIEW = { x: -12.4, y: -12.64, w: 40.36, h: 28.95 };
const MARK_CLONE = { x: 16.44, y: -1.92, r: 10.32 };
const MARK_SCALE = 80 / MARK_VIEW.w;
const MARK: Box = { x: 160, y: 95 - (MARK_VIEW.h * MARK_SCALE) / 2, w: 80, h: MARK_VIEW.h * MARK_SCALE };
const CLONE = {
  x: MARK.x + (MARK_CLONE.x - MARK_VIEW.x) * MARK_SCALE,
  y: MARK.y + (MARK_CLONE.y - MARK_VIEW.y) * MARK_SCALE,
};
// The clone's turn on the loop comes after the assistant's: your messages go in, then its replies come out.
const TURN = 5;
const inAt = (index: number) => TURN + 0.4 + 0.35 * index;
// Your messages drop into the clone from above, over your own bubble in the mark.
const ABOVE_CLONE = { x: CLONE.x - 10, y: CLONE.y - 70 };
const outAt = (index: number) => TURN + 1.9 + 0.45 * index;

/** Your messages taken in by the clone, which then answers a friend in your habits. */
function CloneArt({ id }: { id: string }) {
  const yours = HISTORY.filter((line) => line.by === 'you');
  const replies = CHAT.filter((line) => line.by === 'clone');
  return (
    <>
      <Card box={HISTORY_CARD} id={id}>
        {HISTORY.map((line, index) => {
          const taken = yours.indexOf(line);
          return (
            <Said
              key={index}
              line={line}
              read={taken < 0 ? undefined : { at: inAt(taken) - 0.15, color: 'var(--you)' }}
            />
          );
        })}
      </Card>
      <Card box={CHAT_CARD} id={id} live>
        {CHAT.filter((line) => line.by === 'other').map((line, index) => (
          <Said key={index} line={line} />
        ))}
        <g className="build-art__clear" style={vars({ '--at': TURN })}>
          {replies.map((line, index) => (
            <g key={index} className="build-art__pop" style={vars({ '--at': outAt(index) + FLIGHT })}>
              <Said line={line} />
            </g>
          ))}
        </g>
      </Card>
      {yours.map((_, index) => (
        <Ripple
          key={index}
          center={CLONE}
          r={MARK_CLONE.r * MARK_SCALE}
          color="var(--clone)"
          at={inAt(index) + FLIGHT}
          spread={1.5}
        />
      ))}
      {yours.map((line, index) => (
        <Flight key={index} from={exit(line)} control={ABOVE_CLONE} to={CLONE} at={inAt(index)} color="var(--you)" />
      ))}
      {replies.map((line, index) => {
        const to = entry(line);
        return (
          <Flight
            key={index}
            from={CLONE}
            control={bend(CLONE, to, -0.2)}
            to={to}
            at={outAt(index)}
            color="var(--clone)"
          />
        );
      })}
      <image href={mark} x={MARK.x} y={MARK.y} width={MARK.w} height={MARK.h} />
    </>
  );
}

/** The picture on a purpose's card: what the build is for. */
export function PurposeArt({ purpose }: { purpose: Purpose }) {
  const id = `art${useId().replace(/[^\w-]/g, '')}`;
  return (
    <span className="build-art" data-purpose={purpose} aria-hidden="true">
      <svg viewBox="0 0 400 190" preserveAspectRatio="xMidYMid meet">
        <Defs id={id} />
        {purpose === 'assistant' ? <AssistantArt id={id} /> : <CloneArt id={id} />}
      </svg>
    </span>
  );
}
