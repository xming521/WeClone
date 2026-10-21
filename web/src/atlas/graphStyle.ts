// Drawing tokens shared by the map (atlasEngine.ts) and its entrance overlay (entrance.ts). The
// overlay's last frame is replaced by the map underneath, so both draw with the primitives in paint.ts.
import { ROOT_COLOR, dimensionColor } from '../data';
import type { NodeKind } from '../data';

export const GRAPH_FONT = "'Noto Sans', 'Noto Sans CJK SC', system-ui, sans-serif";
// Page colour behind label text, so labels stay readable where they cross a line.
export const LABEL_HALO = 'rgba(244,243,239,0.9)';
export const LABEL_HALO_WIDTH = 3;
// A label's box starts LABEL_OFFSET_Y below the node and is at least LABEL_LINE_HEIGHT tall.
export const LABEL_OFFSET_Y = 11;
export const LABEL_LINE_HEIGHT = 16;
export const LABEL_FILL_OPACITY = 0.85;
export const EDGE_OPACITY = 0.75;

/** A selected node: a wide translucent ring in its colour around it, a white rim, and a stronger glow while it glows. */
export const SELECTED_NODE = { ringWidth: 14, ringOpacity: 0.18, rimWidth: 2, shadowBlur: 20, labelFontWeight: 700 };

/** The centre 「我」: a pearl tinted by every dimension from the side its branch leaves on, with a white star. */
export const PEARL = { base: '#fdfcf8', tint: 0.5, rimOpacity: 0.6, star: '#ffffff' };

export const labelFontSize = (kind: NodeKind) =>
  kind === 'root' ? 16 : kind === 'dimension' ? 14 : kind === 'attribute' ? 11 : 12;
export const labelFontWeight = (kind: NodeKind) => (kind === 'root' || kind === 'dimension' ? 650 : 520);
export const labelFill = (kind: NodeKind) => (kind === 'root' ? '#4c5561' : '#34474b');

export const nodeFill = (kind: NodeKind, dim: number | null) =>
  kind === 'root' ? ROOT_COLOR : kind === 'attribute' ? '#ffffff' : dimensionColor(dim);
export const nodeStroke = (kind: NodeKind, dim: number | null) => (kind === 'root' ? '#707986' : dimensionColor(dim));
export const glowColor = (kind: NodeKind, dim: number | null) => (kind === 'root' ? '#a3aab4' : dimensionColor(dim));

export const edgeStroke = (kind: NodeKind) => (kind === 'dimension' ? '#b8c8c6' : '#cad4d2');
export const edgeWidth = (kind: NodeKind) => (kind === 'dimension' ? 2 : 1.2);

export function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function rgba(hex: string, alpha: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${Math.min(1, Math.max(0, alpha)).toFixed(3)})`;
}
