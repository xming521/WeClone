// Drawing primitives shared by the map (atlasEngine.ts) and its entrance overlay (entrance.ts). The
// overlay hands its last frame to the map in a single frame, so both draw nodes, lines and labels
// with these functions and the tokens in graphStyle.ts.
import { ROOT_COLOR } from '../data';
import type { NodeKind } from '../data';
import {
  EDGE_OPACITY,
  GRAPH_FONT,
  LABEL_FILL_OPACITY,
  LABEL_HALO,
  LABEL_HALO_WIDTH,
  LABEL_LINE_HEIGHT,
  LABEL_OFFSET_Y,
  PEARL,
  SELECTED_NODE,
  edgeStroke,
  edgeWidth,
  glowColor,
  labelFill,
  labelFontSize,
  labelFontWeight,
  nodeFill,
  nodeStroke,
  rgba,
} from './graphStyle';
import type { NodeLook } from './nodeLook';

export type Rgb = [number, number, number];

/** A dimension as seen from the centre: the direction its branch leaves in, and its colour. */
export type PearlSlice = { angle: number; rgb: Rgb };

export type PaintNode = {
  kind: NodeKind;
  dim: number | null;
  look: NodeLook;
  r: number;
  label: string;
  selected: boolean;
  /** The centre's dimensions, which tint the pearl. */
  slices?: PearlSlice[];
};

/** Device pixels of a map point: point × k + (ox, oy). */
export type Transform = { k: number; ox: number; oy: number };

type Circle = { x: number; y: number; r: number };

const TAU = Math.PI * 2;

export const rgbaOf = (rgb: Rgb, alpha: number) =>
  `rgba(${Math.round(rgb[0])},${Math.round(rgb[1])},${Math.round(rgb[2])},${Math.min(1, Math.max(0, alpha)).toFixed(3)})`;

/** Pixel ratio of the map's canvases: rounded up so lines and text stay sharp, at most 2 to bound the cost. */
export function canvasDpr(): number {
  const raw = window.devicePixelRatio || 1;
  return raw >= 1 ? Math.min(2, Math.ceil(raw)) : 1;
}

// ───────────────────────── nodes ─────────────────────────

/** A four-point star centred on (x, y) with arms of length `size`. */
export function star4(c: CanvasRenderingContext2D, x: number, y: number, size: number) {
  const q = size * 0.13;
  c.beginPath();
  c.moveTo(x, y - size);
  c.quadraticCurveTo(x + q, y - q, x + size, y);
  c.quadraticCurveTo(x + q, y + q, x, y + size);
  c.quadraticCurveTo(x - q, y + q, x - size, y);
  c.quadraticCurveTo(x - q, y - q, x, y - size);
  c.fill();
}

/** A conic gradient around (x, y) through the colours of `slices`, each at the angle of its branch. */
export function sliceGradient(
  c: CanvasRenderingContext2D,
  slices: PearlSlice[],
  x: number,
  y: number,
  alpha: number,
): CanvasGradient | string {
  if (!slices.length || typeof c.createConicGradient !== 'function') return rgba(ROOT_COLOR, alpha);
  const turn = (angle: number) => (((angle % TAU) + TAU) % TAU) / TAU;
  const stops = slices.map((slice) => ({ at: turn(slice.angle), rgb: slice.rgb })).sort((a, b) => a.at - b.at);
  const first = stops[0];
  const last = stops[stops.length - 1];
  // Colour at angle 0, between the last stop and the first one after it.
  const gap = first.at + 1 - last.at;
  const share = gap > 0 ? (1 - last.at) / gap : 0;
  const seam = last.rgb.map((value, i) => value + (first.rgb[i] - value) * share) as Rgb;
  const gradient = c.createConicGradient(0, x, y);
  gradient.addColorStop(0, rgbaOf(seam, alpha));
  for (const stop of stops) gradient.addColorStop(stop.at, rgbaOf(stop.rgb, alpha));
  gradient.addColorStop(1, rgbaOf(seam, alpha));
  return gradient;
}

// The body of every node but the centre: the glow is the shadow of the fill. Shadow blur is in
// device pixels.
function drawBody(c: CanvasRenderingContext2D, n: PaintNode, alpha: number, glow: number) {
  const look = n.look;
  c.beginPath();
  c.arc(0, 0, n.r, 0, TAU);
  const blur = n.selected ? SELECTED_NODE.shadowBlur : look.glowBlur;
  const shadow = glow > 0 && blur > 0 && look.glowAlpha > 0;
  c.globalAlpha = alpha * look.fillOpacity;
  if (shadow) {
    c.shadowColor = rgba(glowColor(n.kind, n.dim), look.glowAlpha * glow);
    c.shadowBlur = blur;
  }
  c.fillStyle = nodeFill(n.kind, n.dim);
  c.fill();
  if (shadow) {
    c.shadowColor = 'transparent';
    c.shadowBlur = 0;
  }
  c.globalAlpha = alpha * look.strokeOpacity;
  c.lineWidth = look.lineWidth;
  c.strokeStyle = nodeStroke(n.kind, n.dim);
  if (look.lineDash) c.setLineDash(look.lineDash);
  c.stroke();
  if (look.lineDash) c.setLineDash([]);
}

// The centre: a pearl lit from the upper left and tinted by each dimension from the side its
// branch leaves on. Its glow is drawn as rings fading outwards in the same colours. `k` is the
// device pixels per unit, to turn the glow's blur into map units.
function drawPearl(c: CanvasRenderingContext2D, n: PaintNode, alpha: number, glow: number, k: number) {
  const r = n.r;
  const slices = n.slices ?? [];
  const blur = n.selected ? SELECTED_NODE.shadowBlur : n.look.glowBlur;
  if (glow > 0 && blur > 0) {
    const spread = (blur * 1.1) / k;
    const rings = 8;
    c.strokeStyle = sliceGradient(c, slices, 0, 0, 1);
    c.lineWidth = (spread / rings) * 1.2;
    for (let i = 0; i < rings; i++) {
      const t = (i + 0.5) / rings;
      c.globalAlpha = alpha * Math.min(1, glow) * 0.6 * (1 - t) ** 2;
      c.beginPath();
      c.arc(0, 0, r + spread * t, 0, TAU);
      c.stroke();
    }
  }
  c.globalAlpha = alpha;
  c.beginPath();
  c.arc(0, 0, r, 0, TAU);
  c.fillStyle = PEARL.base;
  c.fill();
  c.save();
  c.clip();
  for (const slice of slices) {
    const cx = Math.cos(slice.angle) * r * 0.6;
    const cy = Math.sin(slice.angle) * r * 0.6;
    const tint = c.createRadialGradient(cx, cy, 0, cx, cy, r * 0.95);
    tint.addColorStop(0, rgbaOf(slice.rgb, PEARL.tint));
    tint.addColorStop(1, rgbaOf(slice.rgb, 0));
    c.fillStyle = tint;
    c.fillRect(-r, -r, 2 * r, 2 * r);
  }
  const light = c.createRadialGradient(-r * 0.32, -r * 0.38, 0, -r * 0.32, -r * 0.38, r * 0.85);
  light.addColorStop(0, 'rgba(255,255,255,0.7)');
  light.addColorStop(1, 'rgba(255,255,255,0)');
  c.fillStyle = light;
  c.fillRect(-r, -r, 2 * r, 2 * r);
  c.restore();
  c.beginPath();
  c.arc(0, 0, r, 0, TAU);
  c.lineWidth = 1.5;
  c.strokeStyle = sliceGradient(c, slices, 0, 0, PEARL.rimOpacity);
  c.stroke();
  c.fillStyle = PEARL.star;
  star4(c, 0, 0, r * 0.37);
}

// A wide translucent ring just outside the node, in its colour (the pearl's in all of them), and a white rim.
function drawSelection(c: CanvasRenderingContext2D, n: PaintNode, alpha: number) {
  const { ringWidth, ringOpacity, rimWidth } = SELECTED_NODE;
  c.globalAlpha = alpha * ringOpacity;
  c.lineWidth = ringWidth;
  c.strokeStyle = n.kind === 'root' ? sliceGradient(c, n.slices ?? [], 0, 0, 1) : glowColor(n.kind, n.dim);
  c.beginPath();
  c.arc(0, 0, n.r + ringWidth / 2, 0, TAU);
  c.stroke();
  c.globalAlpha = alpha;
  c.lineWidth = rimWidth;
  c.strokeStyle = '#ffffff';
  c.beginPath();
  c.arc(0, 0, n.r + 1, 0, TAU);
  c.stroke();
}

/** Draws a node at (x, y). `glow` scales its glow: 1 at rest, more while it flares, 0 for none. */
export function drawNode(
  c: CanvasRenderingContext2D,
  tf: Transform,
  n: PaintNode,
  x: number,
  y: number,
  scale = 1,
  alpha = 1,
  glow = 1,
) {
  if (alpha <= 0.002) return;
  const k = tf.k * scale;
  c.setTransform(k, 0, 0, k, tf.ox + tf.k * x, tf.oy + tf.k * y);
  if (n.kind === 'root') drawPearl(c, n, alpha, glow, k);
  else drawBody(c, n, alpha, glow);
  if (n.selected) drawSelection(c, n, alpha);
  c.globalAlpha = 1;
}

// ───────────────────────── lines ─────────────────────────

/** Ends of the line between two nodes, from rim to rim; null while they overlap. */
export function edgeEnds(parent: Circle, child: Circle): { sx: number; sy: number; ex: number; ey: number } | null {
  const dx = child.x - parent.x;
  const dy = child.y - parent.y;
  const d = Math.hypot(dx, dy) || 1;
  if (d <= parent.r + child.r) return null;
  const ux = dx / d;
  const uy = dy / d;
  return {
    sx: parent.x + ux * parent.r,
    sy: parent.y + uy * parent.r,
    ex: child.x - ux * child.r,
    ey: child.y - uy * child.r,
  };
}

/** A line from the parent's rim towards the child's; `grow` 0…1 extends it out of the parent. */
export function strokeEdge(
  c: CanvasRenderingContext2D,
  tf: Transform,
  parent: Circle,
  child: Circle,
  color: string,
  width: number,
  alpha: number,
  grow = 1,
) {
  const ends = edgeEnds(parent, child);
  if (!ends || grow <= 0 || alpha <= 0) return;
  c.setTransform(tf.k, 0, 0, tf.k, tf.ox, tf.oy);
  c.globalAlpha = alpha;
  c.strokeStyle = color;
  c.lineWidth = width;
  c.beginPath();
  c.moveTo(ends.sx, ends.sy);
  c.lineTo(ends.sx + (ends.ex - ends.sx) * grow, ends.sy + (ends.ey - ends.sy) * grow);
  c.stroke();
  c.globalAlpha = 1;
}

/** The line into node `n` before the map settles into its own look: width and shade by the node's kind. */
export function drawEdge(
  c: CanvasRenderingContext2D,
  tf: Transform,
  n: { kind: NodeKind },
  parent: Circle,
  child: Circle,
  grow = 1,
  alpha = 1,
) {
  if (alpha <= 0.002) return;
  strokeEdge(c, tf, parent, child, edgeStroke(n.kind), edgeWidth(n.kind), EDGE_OPACITY * alpha, grow);
}

// ───────────────────────── labels ─────────────────────────

export const labelFont = (n: PaintNode) =>
  `${n.selected ? SELECTED_NODE.labelFontWeight : labelFontWeight(n.kind)} ${labelFontSize(n.kind)}px ${GRAPH_FONT}`;

// Height of the box a single-line label is centred in: the line height, or more when the glyphs of
// "|ÉqÅM" plus the halo stroke are taller.
const labelBoxHeight = new Map<string, number>();
export function labelHeight(c: CanvasRenderingContext2D, font: string): number {
  let height = labelBoxHeight.get(font);
  if (height === undefined) {
    c.font = font;
    const metrics = c.measureText('|ÉqÅM');
    const glyphs = Math.ceil(metrics.actualBoundingBoxAscent) + Math.ceil(metrics.actualBoundingBoxDescent);
    height = Math.max(LABEL_LINE_HEIGHT, glyphs + LABEL_HALO_WIDTH);
    labelBoxHeight.set(font, height);
  }
  return height;
}

// The label sits below the node, its box starting LABEL_OFFSET_Y below the rim and shifted right
// by half the halo width.
export function drawLabel(
  c: CanvasRenderingContext2D,
  tf: Transform,
  n: PaintNode,
  x: number,
  y: number,
  alpha: number,
) {
  if (alpha <= 0.01 || !n.label) return;
  const font = labelFont(n);
  c.setTransform(tf.k, 0, 0, tf.k, tf.ox, tf.oy);
  const ly = y + n.r + LABEL_OFFSET_Y + labelHeight(c, font) / 2;
  c.font = font;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.lineJoin = 'round';
  c.lineWidth = LABEL_HALO_WIDTH;
  const lx = x + LABEL_HALO_WIDTH / 2;
  c.globalAlpha = alpha;
  c.strokeStyle = LABEL_HALO;
  c.strokeText(n.label, lx, ly);
  c.globalAlpha = alpha * LABEL_FILL_OPACITY;
  c.fillStyle = labelFill(n.kind);
  c.fillText(n.label, lx, ly);
  c.globalAlpha = 1;
}
