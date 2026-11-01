import type { CSSProperties } from 'react';
import { DIMENSIONS, sectionPlace } from '../data';

export interface ScopeColor {
  'data-dim'?: string;
  'data-sec'?: string;
  style?: CSSProperties;
}

/**
 * Attributes that give an element the colour the map paints a dimension or domain in: `data-dim`, or for a
 * domain `data-sec` with its place in the palette. Both set the design system's `--c` and `--ct`. Nothing
 * for the root.
 */
export function scopeColor(dim: number | null | undefined): ScopeColor {
  const dimension = DIMENSIONS.find((entry) => entry.dim === dim);
  if (dimension) return { 'data-dim': dimension.token };
  const place = sectionPlace(dim);
  if (!place) return {};
  return { 'data-sec': '', style: { '--i': place.index, '--n': place.count } as CSSProperties };
}
