import { forceCollide, forceSimulation, forceX, forceY, type SimulationNodeDatum } from 'd3-force';
import { useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { useTranslation } from 'react-i18next';
import { TRAIT_GROUPS, formatPercent, tierOf, type Trait, type TraitGroupId } from './traits';

// Each group gathers around its own point, so colours form clusters instead of a mix.
const ANCHORS: Record<TraitGroupId, [number, number]> = {
  typing: [-230, -70],
  words: [160, -150],
  emotion: [230, 130],
  manner: [-140, 160],
};
const GAP = 9;
const PADDING = 20;
const MAX_SCALE = 1.2;
/** How much the selected bubble swells. Keep in step with `.trait-bubble.is-selected` in styleView.css. */
const GROW = 1.2;
/** Space kept between the selected bubble and the detail panel. */
const MARGIN = 32;

interface Bubble extends SimulationNodeDatum {
  trait: Trait;
  r: number;
  x: number;
  y: number;
}

/** Area grows with the rate, with a floor so the smallest bubble still holds its title. */
function radius(rate: number) {
  return 30 + 74 * Math.sqrt(rate);
}

function simulate(nodes: Bubble[], ticks: number, selected: string | null, alpha = 1) {
  forceSimulation(nodes)
    .alpha(alpha)
    .force('x', forceX<Bubble>((node) => ANCHORS[node.trait.group][0]).strength(0.05))
    .force('y', forceY<Bubble>((node) => ANCHORS[node.trait.group][1]).strength(0.075))
    .force(
      'collide',
      forceCollide<Bubble>((node) => node.r * (node.trait.key === selected ? GROW : 1) + GAP).iterations(3),
    )
    .stop()
    .tick(ticks);
}

function layout(traits: Trait[]) {
  const placed: Record<string, number> = {};
  const nodes: Bubble[] = traits.map((trait) => {
    // Traits come most frequent first, so each group's biggest bubbles start at its centre.
    const order = (placed[trait.group] = (placed[trait.group] ?? -1) + 1);
    const [x, y] = ANCHORS[trait.group];
    const distance = 40 * Math.sqrt(order);
    return {
      trait,
      r: radius(trait.rate),
      x: x + Math.cos(order * 2.4) * distance,
      y: y + Math.sin(order * 2.4) * distance,
    };
  });
  simulate(nodes, 360, null);
  const left = Math.min(...nodes.map((node) => node.x - node.r)) - PADDING;
  const top = Math.min(...nodes.map((node) => node.y - node.r)) - PADDING;
  const width = Math.max(...nodes.map((node) => node.x + node.r)) + PADDING - left;
  const height = Math.max(...nodes.map((node) => node.y + node.r)) + PADDING - top;
  // Nodes stay in simulation coordinates, where the group anchors live, and are shifted onto the stage when drawn.
  return { width, height, left, top, nodes };
}

/** The resting layout with the selected bubble held in place and swollen, so its neighbours step aside. */
function makeRoom(nodes: Bubble[], selected: string | null) {
  if (!selected) return nodes;
  const moved = nodes.map((node) =>
    node.trait.key === selected ? { ...node, fx: node.x, fy: node.y } : { ...node, vx: 0, vy: 0 },
  );
  simulate(moved, 140, selected, 0.4);
  return moved;
}

const COLORS = Object.fromEntries(TRAIT_GROUPS.map((group) => [group.id, group.color]));

export function BubbleField({
  traits,
  selected,
  focus,
  onSelect,
}: {
  traits: Trait[];
  selected: string | null;
  focus: TraitGroupId | null;
  onSelect: (key: string | null) => void;
}) {
  const { t } = useTranslation();
  const { width, height, left, top, nodes } = useMemo(() => layout(traits), [traits]);
  const bubbles = useMemo(() => makeRoom(nodes, selected), [nodes, selected]);
  const frame = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ width: 0, height: 0, visibleRight: 0 });

  useLayoutEffect(() => {
    const element = frame.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      const box = element.getBoundingClientRect();
      const shell = element.closest('.app-shell')?.getBoundingClientRect();
      const panel = parseFloat(getComputedStyle(element).getPropertyValue('--detail-panel-width')) || 0;
      // Right edge of what stays visible once the detail panel slides in, in frame coordinates.
      setView({ width: box.width, height: box.height, visibleRight: (shell?.right ?? box.right) - panel - box.left });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const scale = view.width ? Math.min(view.width / width, view.height / height, MAX_SCALE) : 1;
  // Slide the field sideways, never zoom, so the open bubble stays clear of the detail panel.
  let pan = 0;
  const target = bubbles.find((node) => node.trait.key === selected);
  if (target && view.width) {
    const center = view.width / 2 + (target.x - left - width / 2) * scale;
    const reach = target.r * GROW * scale + MARGIN;
    const right = Math.min(view.width, view.visibleRight);
    if (center + reach > right) pan = right - reach - center;
    if (center + pan - reach < 0) pan = reach - center;
  }

  return (
    <div className={`bubble-frame${selected ? ' has-selection' : ''}`} ref={frame} onClick={() => onSelect(null)}>
      <div
        className="bubble-stage"
        style={{ width, height, transform: `translate(calc(-50% + ${pan}px), -50%) scale(${scale})` }}
      >
        {bubbles.map(({ trait, r, x, y }, index) => {
          const classes = ['trait-bubble', `tier-${tierOf(trait.rate).id}`];
          if (trait.key === selected) classes.push('is-selected');
          if (focus && trait.group !== focus) classes.push('is-dimmed');
          return (
            <button
              key={trait.key}
              type="button"
              className={classes.join(' ')}
              aria-label={t(($) => $.style.bubble, {
                title: t(($) => $.style.traits[trait.id].title),
                rate: formatPercent(trait.rate),
              })}
              aria-pressed={trait.key === selected}
              onClick={(event) => {
                event.stopPropagation();
                onSelect(trait.key === selected ? null : trait.key);
              }}
              style={
                {
                  left: x - left - r,
                  top: y - top - r,
                  width: r * 2,
                  height: r * 2,
                  '--r': r,
                  '--bubble-color': COLORS[trait.group],
                  '--enter-delay': `${index * 45}ms`,
                  '--float-delay': `${-((index * 1.7) % 9)}s`,
                  '--float-duration': `${8 + (index % 5)}s`,
                } as CSSProperties
              }
            >
              <span className="trait-bubble-float">
                <span className="trait-bubble-body">
                  <strong>{t(($) => $.style.traits[trait.id].title)}</strong>
                  <em>{formatPercent(trait.rate)}</em>
                  {r >= 70 && trait.examples.length > 0 && <small>{trait.examples.slice(0, 3).join(' · ')}</small>}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
