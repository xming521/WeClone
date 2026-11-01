// Drawing primitives shared by the map (atlasEngine.ts) and its entrance overlay (entrance.ts). The
// overlay hands its last frame to the map in a single frame, so both draw nodes, lines and labels
// with these functions and the tokens in graphStyle.ts. The look is the design system's AtlasGraph.
import type { NodeKind } from '../data';
import {
  BALL,
  EDGE_BEND,
  EDGE_GAP,
  LABEL_HALO_WIDTH,
  LABEL_LINE_HEIGHT,
  LABEL_OFFSET_Y,
  ROOT,
  SELECTED_NODE,
  edgeStyle,
  glowColor,
  labelFont,
  labelStyle,
  nodeFill,
  nodeStroke,
  palette,
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
  /** The centre's dimensions, which become the arcs round its edge. */
  slices?: PearlSlice[];
  /** A domain in the domain view: a section, labelled like one. */
  section?: boolean;
};

/** Device pixels of a map point: point × k + (ox, oy). */
export type Transform = { k: number; ox: number; oy: number };

type Circle = { x: number; y: number; r: number };

const TAU = Math.PI * 2;
const WHITE = 'rgba(255,255,255,';

export const rgbaOf = (rgb: Rgb, alpha: number) =>
  `rgba(${Math.round(rgb[0])},${Math.round(rgb[1])},${Math.round(rgb[2])},${Math.min(1, Math.max(0, alpha)).toFixed(3)})`;

/** Pixel ratio of the map's canvases: rounded up so lines and text stay sharp, at most 2 to bound the cost. */
export function canvasDpr(): number {
  const raw = window.devicePixelRatio || 1;
  return raw >= 1 ? Math.min(2, Math.ceil(raw)) : 1;
}

// ───────────────────────── nodes ─────────────────────────

// The Logo's shapes, in the coordinates of the design system's symbols (Assets › Logos). Built on
// first use: this module is loaded where there is no Path2D.
let shapes: { gloss: Path2D; you: Path2D; clone: Path2D; youEye: Path2D; cloneEye: Path2D } | null = null;
function logoShapes() {
  shapes ??= {
    // The front bubble's highlight, drawn for a bubble of radius 12.
    gloss: new Path2D(
      'M-9.22 -3.92A10.02 10.02 0 0 1 -1.57 -9.9A0.78 0.78 0 0 1 -1.32 -8.36A8.46 8.46 0 0 0 -7.79 -3.31A0.78 0.78 0 0 1 -9.22 -3.92z',
    ),
    // The favicon: the front bubble, the back bubble behind it, and their faces.
    you: new Path2D('M-12 2a12 12 0 1 0 24 0a12 12 0 1 0 -24 0z'),
    clone: new Path2D('M6.47 -9.85A10.6 10.6 0 1 1 12.19 7.81A13.5 13.5 0 0 0 6.47 -9.85z'),
    youEye: new Path2D(
      'M-5.48 2.5a1.58 2.02 0 1 0 3.17 0a1.58 2.02 0 1 0 -3.17 0zM3.42 2.5a1.58 2.02 0 1 0 3.17 0a1.58 2.02 0 1 0 -3.17 0z',
    ),
    cloneEye: new Path2D(
      'M14.69 -2.8a1.41 1.79 0 1 0 2.82 0a1.41 1.79 0 1 0 -2.82 0zM20.19 -2.8a1.41 1.79 0 1 0 2.82 0a1.41 1.79 0 1 0 -2.82 0z',
    ),
  };
  return shapes;
}
// The favicon symbol's viewBox: 37.4 units square, centred on (6.4, 0.4).
const MARK_BOX = 37.4;
const MARK_CENTER = { x: 6.4, y: 0.4 };

// The glow a node flares with while it enters (the shadow of its fill). Shadow blur is in device
// pixels. At rest `glow` is 0 and nothing glows.
function glowOn(c: CanvasRenderingContext2D, n: PaintNode, alpha: number, glow: number): boolean {
  const look = n.look;
  const blur = n.selected ? SELECTED_NODE.shadowBlur : look.glowBlur;
  if (!(glow > 0 && blur > 0 && look.glowAlpha > 0)) return false;
  c.shadowColor = rgba(glowColor(n.kind, n.dim), look.glowAlpha * glow * alpha);
  c.shadowBlur = blur;
  return true;
}

function glowOff(c: CanvasRenderingContext2D) {
  c.shadowColor = 'transparent';
  c.shadowBlur = 0;
}

// A section or topic: a solid disc of its colour, 20% to 100% filled by confidence.
function drawDisc(c: CanvasRenderingContext2D, n: PaintNode, alpha: number, glow: number) {
  c.beginPath();
  c.arc(0, 0, n.r, 0, TAU);
  c.globalAlpha = alpha * n.look.fillOpacity;
  const shadow = glowOn(c, n, 1, glow);
  c.fillStyle = nodeFill(n.kind, n.dim);
  c.fill();
  if (shadow) glowOff(c);
}

// An attribute: a hollow ring, its `card` body and its outline both fading with confidence.
function drawAttribute(c: CanvasRenderingContext2D, n: PaintNode, alpha: number, glow: number) {
  const look = n.look;
  c.beginPath();
  c.arc(0, 0, n.r, 0, TAU);
  c.globalAlpha = alpha * look.fillOpacity;
  const shadow = glowOn(c, n, 1, glow);
  c.fillStyle = nodeFill(n.kind, n.dim);
  c.fill();
  if (shadow) glowOff(c);
  c.globalAlpha = alpha * look.strokeOpacity;
  c.lineWidth = look.lineWidth;
  c.strokeStyle = nodeStroke(n.kind, n.dim);
  c.stroke();
}

// A dimension (or section, in the domain view): a solid ball with a sheen from the upper left, the
// Logo's highlight scaled to it and a white rim.
function drawBall(c: CanvasRenderingContext2D, n: PaintNode, alpha: number, glow: number) {
  const r = n.r;
  c.beginPath();
  c.arc(0, 0, r, 0, TAU);
  c.globalAlpha = alpha;
  const shadow = glowOn(c, n, 1, glow);
  c.fillStyle = nodeFill(n.kind, n.dim);
  c.fill();
  if (shadow) glowOff(c);
  // radialGradient cx 32%, cy 24%, r 70% of the ball's box.
  const sx = -r + 0.64 * r;
  const sy = -r + 0.48 * r;
  const sheen = c.createRadialGradient(sx, sy, 0, sx, sy, 1.4 * r);
  sheen.addColorStop(0, `${WHITE}${BALL.sheenOpacity})`);
  sheen.addColorStop(0.55, `${WHITE}0)`);
  sheen.addColorStop(1, `${WHITE}0)`);
  c.fillStyle = sheen;
  c.fill();
  c.save();
  c.scale(r / 12, r / 12);
  c.fillStyle = `${WHITE}${BALL.glossOpacity * alpha})`;
  c.fill(logoShapes().gloss);
  c.restore();
  c.beginPath();
  c.arc(0, 0, r - BALL.rimInset, 0, TAU);
  c.lineWidth = BALL.rimWidth;
  c.strokeStyle = `${WHITE}${BALL.rimOpacity * alpha})`;
  c.stroke();
}

// The Logo's favicon, as the design system sets it in the centre: its box is 28/30.4 of the disc's radius.
function drawMark(c: CanvasRenderingContext2D, r: number, alpha: number) {
  const { logo } = palette();
  const shape = logoShapes();
  const scale = (r * ROOT.markSize) / MARK_BOX;
  c.save();
  c.scale(scale, scale);
  c.translate(-MARK_CENTER.x, -MARK_CENTER.y);
  c.globalAlpha = alpha;
  const front = c.createLinearGradient(-8.64, -6.64, 8.64, 10.64);
  front.addColorStop(0, logo.you1);
  front.addColorStop(1, logo.you2);
  c.fillStyle = front;
  c.fill(shape.you);
  const back = c.createLinearGradient(14.2, -13.2, 14.2, 8);
  back.addColorStop(0, logo.clone1);
  back.addColorStop(1, logo.clone2);
  c.fillStyle = back;
  c.fill(shape.clone);
  c.fillStyle = logo.youEye;
  c.fill(shape.youEye);
  c.fillStyle = logo.cloneEye;
  c.fill(shape.cloneEye);
  c.restore();
}

// The centre: a `card` disc with a `line-1` edge, one arc per dimension round it in that
// dimension's colour, and the favicon. A translucent `card` backdrop lifts it off the ground.
function drawRoot(c: CanvasRenderingContext2D, n: PaintNode, alpha: number, glow: number) {
  const r = n.r;
  const p = palette();
  c.globalAlpha = alpha * ROOT.backdropOpacity;
  c.beginPath();
  c.arc(0, 0, r * ROOT.backdrop, 0, TAU);
  c.fillStyle = p.card;
  c.fill();
  if (n.selected) drawSelectionRing(c, n, alpha);
  c.globalAlpha = alpha;
  c.beginPath();
  c.arc(0, 0, r, 0, TAU);
  const shadow = glowOn(c, n, 1, glow);
  c.fillStyle = p.card;
  c.fill();
  if (shadow) glowOff(c);
  c.lineWidth = ROOT.edgeWidth;
  c.strokeStyle = p.line1;
  c.stroke();
  const slices = n.slices ?? [];
  if (slices.length) {
    const half = Math.max(0.02, Math.PI / slices.length - ROOT.arcGap);
    c.lineWidth = ROOT.arcWidth;
    c.lineCap = 'round';
    for (const slice of slices) {
      c.strokeStyle = rgbaOf(slice.rgb, 1);
      c.beginPath();
      c.arc(0, 0, r * ROOT.arcRadius, slice.angle - half, slice.angle + half);
      c.stroke();
    }
    c.lineCap = 'butt';
  }
  drawMark(c, r, alpha);
}

// A wide translucent ring in the node's colour just outside it.
function drawSelectionRing(c: CanvasRenderingContext2D, n: PaintNode, alpha: number) {
  c.globalAlpha = alpha * SELECTED_NODE.ringOpacity;
  c.lineWidth = SELECTED_NODE.ringWidth;
  c.strokeStyle = nodeStroke(n.kind, n.dim);
  c.beginPath();
  c.arc(0, 0, n.r + SELECTED_NODE.ringOffset, 0, TAU);
  c.stroke();
}

// A `card` ring against the selected node.
function drawSelectionRim(c: CanvasRenderingContext2D, n: PaintNode, alpha: number) {
  c.globalAlpha = alpha;
  c.lineWidth = SELECTED_NODE.rimWidth;
  c.strokeStyle = palette().card;
  c.beginPath();
  c.arc(0, 0, n.r + SELECTED_NODE.rimOffset, 0, TAU);
  c.stroke();
}

/** Draws a node at (x, y). `glow` scales the glow it flares with while it enters: 1 at rest of the entrance, more while it flares, 0 for none. */
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
  if (n.kind === 'root') drawRoot(c, n, alpha, glow);
  else {
    if (n.selected) drawSelectionRing(c, n, alpha);
    if (n.kind === 'dimension') drawBall(c, n, alpha, glow);
    else if (n.kind === 'attribute') drawAttribute(c, n, alpha, glow);
    else drawDisc(c, n, alpha, glow);
  }
  if (n.selected) drawSelectionRim(c, n, alpha);
  c.globalAlpha = 1;
}

// ───────────────────────── lines ─────────────────────────

/** A line as a quadratic curve: from the parent's end (x0, y0) through the control point to the child's end. */
export type EdgeCurve = { x0: number; y0: number; qx: number; qy: number; x1: number; y1: number };

/**
 * The curve between two nodes, null while they overlap. It leaves the straight line between their
 * centres, stopping EDGE_GAP short of each edge, and bends to the left of the direction from parent to child.
 */
export function edgeCurve(parent: Circle, child: Circle): EdgeCurve | null {
  const dx = child.x - parent.x;
  const dy = child.y - parent.y;
  const d = Math.hypot(dx, dy) || 1;
  if (d <= parent.r + child.r + 2 * EDGE_GAP) return null;
  const ux = dx / d;
  const uy = dy / d;
  const x0 = parent.x + ux * (parent.r + EDGE_GAP);
  const y0 = parent.y + uy * (parent.r + EDGE_GAP);
  const x1 = child.x - ux * (child.r + EDGE_GAP);
  const y1 = child.y - uy * (child.r + EDGE_GAP);
  const bend = EDGE_BEND * d;
  return { x0, y0, x1, y1, qx: (x0 + x1) / 2 - uy * bend, qy: (y0 + y1) / 2 + ux * bend };
}

/** The point `u` (0 at the parent's end, 1 at the child's) of the curve. Its distance along the straight line is linear in `u`. */
export function curvePoint(curve: EdgeCurve, u: number): { x: number; y: number } {
  const a = (1 - u) * (1 - u);
  const b = 2 * u * (1 - u);
  const e = u * u;
  return { x: a * curve.x0 + b * curve.qx + e * curve.x1, y: a * curve.y0 + b * curve.qy + e * curve.y1 };
}

/** A line from the parent's edge towards the child's; `grow` 0…1 extends it out of the parent. */
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
  const curve = edgeCurve(parent, child);
  if (!curve || grow <= 0 || alpha <= 0) return;
  c.setTransform(tf.k, 0, 0, tf.k, tf.ox, tf.oy);
  c.globalAlpha = alpha;
  c.strokeStyle = color;
  c.lineWidth = width;
  c.beginPath();
  c.moveTo(curve.x0, curve.y0);
  if (grow >= 1) c.quadraticCurveTo(curve.qx, curve.qy, curve.x1, curve.y1);
  else {
    // The part up to `grow` of a quadratic is a quadratic: its control point lies on the first half-tangent.
    const end = curvePoint(curve, grow);
    c.quadraticCurveTo(curve.x0 + (curve.qx - curve.x0) * grow, curve.y0 + (curve.qy - curve.y0) * grow, end.x, end.y);
  }
  c.stroke();
  c.globalAlpha = 1;
}

/** The line into node `n` before the map settles into its own look: its colour and width by the node's kind. */
export function drawEdge(
  c: CanvasRenderingContext2D,
  tf: Transform,
  n: { kind: NodeKind; dim: number | null },
  parent: Circle,
  child: Circle,
  grow = 1,
  alpha = 1,
) {
  if (alpha <= 0.002) return;
  const style = edgeStyle(n.kind, n.dim);
  strokeEdge(c, tf, parent, child, style.color, style.width, style.opacity * alpha, grow);
}

// ───────────────────────── labels ─────────────────────────

export const labelFontOf = (n: PaintNode) => labelFont(labelStyle(n, n.selected));

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

/** Forgets measured label heights, once the fonts they were measured in have loaded. */
export const resetLabelMetrics = () => labelBoxHeight.clear();

// The label sits below the node, its box starting LABEL_OFFSET_Y below the edge and shifted right
// by half the halo width. A `paper` halo keeps it readable where a line crosses it.
export function drawLabel(
  c: CanvasRenderingContext2D,
  tf: Transform,
  n: PaintNode,
  x: number,
  y: number,
  alpha: number,
) {
  if (alpha <= 0.01 || !n.label) return;
  const style = labelStyle(n, n.selected);
  const font = labelFont(style);
  c.setTransform(tf.k, 0, 0, tf.k, tf.ox, tf.oy);
  const ly = y + n.r + LABEL_OFFSET_Y + labelHeight(c, font) / 2;
  c.font = font;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.lineJoin = 'round';
  c.lineWidth = LABEL_HALO_WIDTH;
  const lx = x + LABEL_HALO_WIDTH / 2;
  c.globalAlpha = alpha;
  c.strokeStyle = palette().paper;
  c.strokeText(n.label, lx, ly);
  c.fillStyle = style.color;
  c.fillText(n.label, lx, ly);
  c.globalAlpha = 1;
}
