// Pure layout helpers. Only type imports are allowed here so the tests can load this file
// directly with node's type stripping.
import type { ProfileModel, ProfileNode } from '../data';

export type Point = {
  x: number;
  y: number;
};

export type Positions = Map<string, Point>;

/** Fixed direction of each dimension around the root, starting on the left and going clockwise. */
export function dimensionAngle(index: number, count: number): number {
  return Math.PI - (index * 2 * Math.PI) / count;
}

/** Above this many visible nodes attributes pack closer together and their labels show only when zoomed in. */
export const DENSE_NODE_COUNT = 600;

// Space one visible node takes inside its dimension cluster: a cluster of n nodes spans about this times sqrt(n).
const clusterSpacing = (visibleCount: number) => (visibleCount > DENSE_NODE_COUNT ? 19 : 31);

/**
 * From this many dimensions on (the domain view), one ring would leave the middle empty: they
 * alternate between an inner and an outer ring, and their few topics sit closer to them.
 */
export const STAGGER_COUNT = 8;
// Free space kept between two clusters, and between a cluster and the root.
const CLUSTER_GAP = 50;
const ROOT_CLEARANCE = 110;

/** Length of the line from a dimension node to each of its topics. */
export const topicDistance = (dimensionCount: number) => (dimensionCount >= STAGGER_COUNT ? 130 : 220);

/** Every cluster at the same distance, far enough out that neighbours do not touch. */
function ringDistances(radii: number[]): number[] {
  const count = radii.length;
  // Distance between two neighbouring directions at unit radius.
  const chord = count > 1 ? 2 * Math.sin(Math.PI / count) : 0;
  return radii.map((radius, index) => {
    const neighbour = Math.max(radii[(index + count - 1) % count], radii[(index + 1) % count]);
    const apart = chord ? (radius + neighbour + CLUSTER_GAP) / chord : 0;
    return Math.max(330, radius + ROOT_CLEARANCE, apart);
  });
}

/**
 * Even clusters on an inner ring, odd ones on an outer ring between them. Each cluster sits as
 * close to the root as its own size and its neighbours allow, so the distances differ.
 */
function staggeredDistances(radii: number[]): number[] {
  const count = radii.length;
  const step = (2 * Math.PI) / count;
  const angleBetween = (a: number, b: number) => {
    const turns = Math.abs(a - b) % count;
    return Math.min(turns, count - turns) * step;
  };
  const need = (a: number, b: number) => radii[a] + radii[b] + CLUSTER_GAP;
  // Distance at which clusters a and b just clear each other when both are that far out.
  const sharedRing = (a: number, b: number) => need(a, b) / (2 * Math.sin(angleBetween(a, b) / 2));
  // Distance along its own direction at which cluster b clears cluster a, which sits `away` out.
  const clearOf = (a: number, away: number, b: number) => {
    const angle = angleBetween(a, b);
    const closest = angle >= Math.PI / 2 ? away : away * Math.sin(angle);
    if (closest >= need(a, b)) return 0;
    return away * Math.cos(angle) + Math.sqrt(need(a, b) ** 2 - (away * Math.sin(angle)) ** 2);
  };
  const inner = radii.map((_, index) => index % 2 === 0);
  // The next cluster on the same ring in each direction.
  const ringNeighbours = (index: number) =>
    [1, -1].map((direction) => {
      let next = (index + direction + count) % count;
      while (inner[next] !== inner[index]) next = (next + direction + count) % count;
      return next;
    });
  const distances = radii.map((radius) => radius + ROOT_CLEARANCE);
  for (const outer of [false, true]) {
    radii.forEach((_, index) => {
      if (inner[index] === outer) return;
      for (const other of ringNeighbours(index)) {
        if (other !== index) distances[index] = Math.max(distances[index], sharedRing(index, other));
      }
      if (!outer) return;
      radii.forEach((_, other) => {
        if (inner[other]) distances[index] = Math.max(distances[index], clearOf(other, distances[other], index));
      });
    });
  }
  // Neighbours on one ring at different distances can still touch: the one farther out moves on.
  for (let pass = 0; pass < 8; pass++) {
    let moved = false;
    for (let a = 0; a < count; a++) {
      for (let b = a + 1; b < count; b++) {
        const [near, far] = distances[a] <= distances[b] ? [a, b] : [b, a];
        const reach = clearOf(near, distances[near], far);
        if (reach > distances[far] + 0.5) {
          distances[far] = reach;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }
  return distances;
}

/**
 * Target position of every dimension node. Dimensions keep their direction around the
 * root, and a dimension with more visible nodes sits farther out, so that expanded
 * dimensions get their own region instead of growing into their neighbours.
 */
export function clusterCenters(model: ProfileModel, visible: ProfileNode[], root: Point): Positions {
  const count = model.dimensionIds.length;
  const sizes = new Map(model.dimensionIds.map((id) => [model.nodes.get(id)!.dim, 0]));
  for (const node of visible) {
    if (node.kind !== 'root' && node.kind !== 'dimension') sizes.set(node.dim, (sizes.get(node.dim) ?? 0) + 1);
  }
  const spacing = clusterSpacing(visible.length);
  const staggered = count >= STAGGER_COUNT;
  // A staggered cluster with any topic shown reaches at least as far as its topics' labels.
  const reach = staggered ? topicDistance(count) + 40 : 0;
  const radii = model.dimensionIds.map((id) => {
    const size = sizes.get(model.nodes.get(id)!.dim)!;
    return Math.max(spacing * Math.sqrt(size) + 60, size ? reach : 0);
  });
  const distances = staggered ? staggeredDistances(radii) : ringDistances(radii);
  return new Map(
    model.dimensionIds.map((id, index) => {
      const angle = dimensionAngle(index, count);
      return [id, { x: root.x + distances[index] * Math.cos(angle), y: root.y + distances[index] * Math.sin(angle) }];
    }),
  );
}

/**
 * Moves each dimension's nodes together so the dimension node lands on its target from
 * `clusterCenters`. The cluster keeps its own arrangement, and offsets of up to 40px are
 * left to the layout forces so that unchanged clusters do not shift on every render.
 */
export function shiftClusters(model: ProfileModel, positions: Positions, centers: Positions): Positions {
  const offsets = new Map<number, Point>();
  for (const [id, center] of centers) {
    const current = positions.get(id);
    const dim = model.nodes.get(id)?.dim;
    if (!current || dim == null) continue;
    const offset = { x: center.x - current.x, y: center.y - current.y };
    if (Math.hypot(offset.x, offset.y) > 40) offsets.set(dim, offset);
  }
  if (!offsets.size) return positions;
  const shifted: Positions = new Map();
  for (const [id, position] of positions) {
    const dim = model.nodes.get(id)?.dim;
    const offset = dim == null ? undefined : offsets.get(dim);
    shifted.set(id, offset ? { x: position.x + offset.x, y: position.y + offset.y } : position);
  }
  return shifted;
}

export function overviewPositions(model: ProfileModel): Positions {
  const positions: Positions = new Map([['root', { x: 0, y: 0 }]]);
  function placeDescendants(parentId: string, direction: number) {
    const parent = model.nodes.get(parentId)!;
    const center = positions.get(parentId)!;
    parent.childIds.forEach((childId, index) => {
      const ring = Math.floor(index / 12);
      const count = Math.min(12, parent.childIds.length - ring * 12);
      const angle = direction + (count === 1 ? 0 : ((index % 12) / (count - 1) - 0.5) * 2.3);
      const radius = (parent.kind === 'topic' ? 120 : 85) + ring * 55;
      positions.set(childId, { x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) });
      placeDescendants(childId, angle);
    });
  }
  model.dimensionIds.forEach((id, dimensionIndex) => {
    const angle = dimensionAngle(dimensionIndex, model.dimensionIds.length);
    const center = { x: 330 * Math.cos(angle), y: 330 * Math.sin(angle) };
    positions.set(id, center);
    const topics = model.nodes.get(id)!.childIds;
    const spread = Math.min(2.3, 1.2 + topics.length * 0.07);
    const radius = Math.min(335, 250 + topics.length * 5);
    topics.forEach((topicId, topicIndex) => {
      const topicAngle = angle + (topics.length === 1 ? 0 : (topicIndex / (topics.length - 1) - 0.5) * spread);
      positions.set(topicId, {
        x: center.x + radius * Math.cos(topicAngle),
        y: center.y + radius * Math.sin(topicAngle),
      });
      placeDescendants(topicId, topicAngle);
    });
  });
  return positions;
}

export function expansionPositions(model: ProfileModel, visible: ProfileNode[], previous: Positions): Positions {
  const positions = new Map(previous);
  const root = previous.get('root')!;
  const halfWidth = (Math.PI / model.dimensionIds.length) * 0.9;
  const displayed = new Set(visible.map((node) => node.id));
  for (const node of visible) {
    if (positions.has(node.id)) continue;
    const parent = positions.get(node.parentId ?? 'root') ?? root;
    const dimension = previous.get(`dimension:${node.dim}`) ?? parent;
    const direction = Math.atan2(dimension.y - root.y, dimension.x - root.x);
    const siblings = model.nodes.get(node.parentId ?? 'root')!.childIds.filter((id) => displayed.has(id));
    const index = siblings.indexOf(node.id);
    const spread = siblings.length === 1 ? 0 : (index / (siblings.length - 1) - 0.5) * 1.4;
    const distance = (node.kind === 'attribute' ? 130 : 180) + Math.floor(index / 10) * 70;
    const x = parent.x + Math.cos(direction + spread) * distance - root.x;
    const y = parent.y + Math.sin(direction + spread) * distance - root.y;
    const relative = Math.atan2(Math.sin(Math.atan2(y, x) - direction), Math.cos(Math.atan2(y, x) - direction));
    const angle = direction + Math.max(-halfWidth, Math.min(halfWidth, relative));
    const radius = Math.hypot(x, y);
    positions.set(node.id, { x: root.x + radius * Math.cos(angle), y: root.y + radius * Math.sin(angle) });
  }
  return positions;
}

/**
 * Starting coordinates for a render. Staying in the same view reuses the current
 * positions. Switching views restores the cached coordinates of the target view
 * (falling back to the overview layout) and translates them so its root lands on
 * the root that is currently on screen.
 */
export function computeSeeds(
  switchingView: boolean,
  previous: Positions,
  initialPositions: Positions,
  restored: Positions | undefined,
): Positions {
  if (!switchingView) return previous;
  const seeds: Positions = new Map([...initialPositions, ...(restored ?? [])]);
  const currentRoot = previous.get('root')!;
  const targetRoot = seeds.get('root')!;
  const offset = { x: currentRoot.x - targetRoot.x, y: currentRoot.y - targetRoot.y };
  for (const [id, position] of seeds) {
    seeds.set(id, { x: position.x + offset.x, y: position.y + offset.y });
  }
  return seeds;
}

/**
 * Position of every visible node: its seed, else its initial position, else a spot around its
 * parent (nodes come parent first), else the origin.
 */
export function placeNodes(visible: ProfileNode[], seeds: Positions, initial: Positions): Positions {
  const positions: Positions = new Map();
  visible.forEach((node, index) => {
    const parent = node.parentId ? positions.get(node.parentId) : undefined;
    const angle = index * 2.39996;
    const distance = node.kind === 'dimension' ? 210 : node.kind === 'attribute' ? 90 : 135;
    positions.set(
      node.id,
      seeds.get(node.id) ??
        initial.get(node.id) ??
        (parent
          ? { x: parent.x + Math.cos(angle) * distance, y: parent.y + Math.sin(angle) * distance }
          : { x: 0, y: 0 }),
    );
  });
  return positions;
}

/** True when `next` lists the same nodes under the same parents in the same order as `current`. */
export function sameTree(
  current: Array<{ id: string; parentId: string | null }>,
  next: Array<{ id: string; parentId: string | null }>,
): boolean {
  return (
    current.length === next.length &&
    current.every((node, index) => node.id === next[index].id && node.parentId === next[index].parentId)
  );
}
