// Drawing tokens shared by the map (atlasEngine.ts) and its entrance overlay (entrance.ts). The
// overlay's last frame is replaced by the map underneath, so both draw with the primitives in paint.ts.
//
// The look is the design system's AtlasGraph (web/src/ds/components.css `.wc-atlas`). A canvas cannot
// use var(), so neutral colours and type are read from the same custom properties once and again when
// the theme changes; dimension and section colours come from data.ts, which keeps plain values.
import { ROOT_COLOR, dimensionColor } from '../data';
import type { NodeKind } from '../data';

/**
 * The design system draws the atlas at 0.8 of this map's node sizes (a dimension ball is 21.6 there and
 * 27 here, the centre 30.4 and 38). Lengths it gives that belong to a node (strokes, rings, the tail, the
 * centre's arcs) are multiplied by this, so they keep their proportion to the node. Type is not.
 */
export const DS_SCALE = 1.25;

type TypeToken = { weight: number; size: number; family: string };

/** The tokens the map paints with. */
export type Palette = {
  paper: string;
  card: string;
  line1: string;
  line2: string;
  ink1: string;
  ink2: string;
  ink3: string;
  ink4: string;
  edge1: string;
  edge2: string;
  pending: string;
  /** The Logo's two bubbles and their faces at 32px and below, for the mark in the centre. */
  logo: { you1: string; you2: string; youEye: string; clone1: string; clone2: string; cloneEye: string };
  /** --text-node-root / -dim / -section / -attr. */
  type: { root: TypeToken; dim: TypeToken; section: TypeToken; attr: TypeToken };
};

// A font shorthand token such as `600 15px/20px "Figtree", "PingFang SC", sans-serif`.
const TYPE_PATTERN = /^(\d+)\s+([\d.]+)px\s*\/\s*[\d.]+(?:px)?\s+(.+)$/s;

// Properties that are not defined yet read as '', which a canvas ignores: the stylesheet is loaded
// before the map is, so that only happens in a page without it.
function readPalette(style: CSSStyleDeclaration): Palette {
  const get = (name: string) => style.getPropertyValue(name).trim();
  const font = (name: string): TypeToken => {
    const match = TYPE_PATTERN.exec(get(name));
    return match
      ? { weight: Number(match[1]), size: Number(match[2]), family: match[3] }
      : { weight: 500, size: 12, family: get('--font-sans') || 'sans-serif' };
  };
  return {
    paper: get('--paper'),
    card: get('--card'),
    line1: get('--line-1'),
    line2: get('--line-2'),
    ink1: get('--ink-1'),
    ink2: get('--ink-2'),
    ink3: get('--ink-3'),
    ink4: get('--ink-4'),
    edge1: get('--edge-1'),
    edge2: get('--edge-2'),
    pending: get('--pending'),
    logo: {
      you1: get('--logo-you-1'),
      you2: get('--logo-you-2'),
      youEye: get('--logo-you-eye-sm'),
      clone1: get('--logo-clone-1'),
      clone2: get('--logo-clone-2'),
      cloneEye: get('--logo-clone-eye-sm'),
    },
    type: {
      root: font('--text-node-root'),
      dim: font('--text-node-dim'),
      section: font('--text-node-section'),
      attr: font('--text-node-attr'),
    },
  };
}

let cached: { theme: string | null; palette: Palette } | null = null;

/** The tokens' current values. */
export function palette(): Palette {
  const root = document.documentElement;
  const theme = root.getAttribute('data-theme');
  if (cached && cached.theme === theme) return cached.palette;
  const read = readPalette(getComputedStyle(root));
  if (read.paper) cached = { theme, palette: read };
  return read;
}

// ───────────────────────── labels ─────────────────────────

// Page colour behind label text (`paper`, 3.2px), so labels stay readable where they cross a line.
export const LABEL_HALO_WIDTH = 3.2;
// A label's box starts LABEL_OFFSET_Y below the node and is at least LABEL_LINE_HEIGHT tall.
export const LABEL_OFFSET_Y = 11;
export const LABEL_LINE_HEIGHT = 16;

/** What a node's label is set in: `section` for a domain in the domain view, which is a section there. */
export type LabelRole = { kind: NodeKind; section?: boolean };

export type LabelStyle = { weight: number; size: number; family: string; color: string };

/** Type and colour of a node's label: the centre and dimensions `ink-1`, sections and topics `ink-2`, attributes `ink-3`. */
export function labelStyle(role: LabelRole, selected = false): LabelStyle {
  const p = palette();
  const pick = (token: TypeToken, color: string): LabelStyle => ({ ...token, color });
  const style =
    role.kind === 'root'
      ? pick(p.type.root, p.ink1)
      : role.kind === 'dimension' && !role.section
        ? pick(p.type.dim, p.ink1)
        : role.kind === 'attribute'
          ? pick(p.type.attr, p.ink3)
          : pick(p.type.section, p.ink2);
  // The selected node's label is bold and ink-1 (`t-on`).
  return selected ? { ...style, weight: SELECTED_NODE.labelFontWeight, color: p.ink1 } : style;
}

export const labelFontSize = (role: LabelRole) => labelStyle(role).size;
export const labelFont = (style: Pick<LabelStyle, 'weight' | 'size' | 'family'>) =>
  `${style.weight} ${style.size}px ${style.family}`;

// ───────────────────────── nodes ─────────────────────────

/** A selected node: a 10px translucent ring in its colour, 2px outside the node's own edge, and a 2px `card` ring against the node. */
export const SELECTED_NODE = {
  ringWidth: 10 * DS_SCALE,
  ringOpacity: 0.2,
  /** Distance of the wide ring's centre line from the node's edge. */
  ringOffset: 7.4 * DS_SCALE,
  rimWidth: 2 * DS_SCALE,
  rimOffset: 1.2 * DS_SCALE,
  /** Blur of the glow a selected node flares with while it glows, in device pixels. */
  shadowBlur: 20,
  labelFontWeight: 700,
};

/** The pending tail: a dot about a quarter of the node's size, lower right at 45°, with a `paper` edge. */
export const TAIL = { ratio: 0.25, minRadius: 1.8 * DS_SCALE, gap: 1.5 * DS_SCALE, edge: 1.2 * DS_SCALE };

/** The centre: a `card` disc with a `line-1` edge, a translucent `card` backdrop, one arc per dimension and the Logo's favicon. */
export const ROOT = {
  backdrop: 44 / 30.4,
  backdropOpacity: 0.6,
  /** The arcs' radius as a share of the disc's, their width, and the gap left between neighbours (radians). */
  arcRadius: 27.4 / 30.4,
  arcWidth: 4.5 * DS_SCALE,
  arcGap: 0.05,
  edgeWidth: DS_SCALE,
  /** The favicon's box (28px on a disc of 30.4), as a share of the disc's radius. */
  markSize: 28 / 30.4,
};

/** A dimension ball: a solid colour, a sheen from the upper left, the Logo's highlight and a white rim. */
export const BALL = {
  sheenOpacity: 0.3,
  glossOpacity: 0.8,
  rimOpacity: 0.38,
  rimWidth: 1.5 * DS_SCALE,
  rimInset: 0.8 * DS_SCALE,
  /** The halo behind a ball is a flat disc of its colour. */
  haloOpacity: 0.09,
};

export const nodeFill = (kind: NodeKind, dim: number | null) =>
  kind === 'root' ? ROOT_COLOR : kind === 'attribute' ? palette().card : dimensionColor(dim);
export const nodeStroke = (kind: NodeKind, dim: number | null) => (kind === 'root' ? ROOT_COLOR : dimensionColor(dim));
export const glowColor = (kind: NodeKind, dim: number | null) =>
  kind === 'root' ? palette().ink4 : dimensionColor(dim);

// ───────────────────────── lines ─────────────────────────

/** How far a line bends: the control point sits this share of the nodes' distance off the straight line. */
export const EDGE_BEND = 0.1;
/** Clear space between a line's end and the node's edge. */
export const EDGE_GAP = 2 * DS_SCALE;

export type EdgeStyle = { color: string; width: number; opacity: number };

/** The line into a node of `kind`: `edge-1` into a dimension, `edge-2` into a topic, `edge-soft` into an attribute. */
export function edgeStyle(kind: NodeKind, dim: number | null): EdgeStyle {
  const p = palette();
  if (kind === 'dimension') return { color: p.edge1, width: 1.6 * DS_SCALE, opacity: 1 };
  if (kind === 'attribute') return { color: dimensionColor(dim), width: DS_SCALE, opacity: 0.28 };
  return { color: p.edge2, width: 1.1 * DS_SCALE, opacity: 1 };
}

/** A line on the path to the hovered or selected node (`edge-on`). */
export const edgeOnStyle = (dim: number | null): EdgeStyle => ({
  color: dimensionColor(dim),
  width: 1.6 * DS_SCALE,
  opacity: 0.55,
});

/** An orbit (`orbit`): `line-2` dots 1.5 long every 5 (1.5, 5 apart in `dash`). */
export const ORBIT = { width: DS_SCALE, dash: [1.5 * DS_SCALE, 5 * DS_SCALE] };

// ───────────────────────── colour helpers ─────────────────────────

export function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function rgba(hex: string, alpha: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${Math.min(1, Math.max(0, alpha)).toFixed(3)})`;
}
