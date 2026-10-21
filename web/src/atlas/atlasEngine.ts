// The profile map on a canvas (design: WC-exp/design, "A 纸本 · 浅色克制"), laid out by d3-force.
// Every node's colour spreads on the paper by its importance and each dimension washes the ground
// under its cluster. Lines widen with chat support, and waves of ink run along them into 「我」.
// Hovering a node lights its path to the centre and dims the rest; clicking one rings it and brings
// its branch into view. The entrance (entrance.ts) plays over the blank map and hands over its last
// frame, so the map first draws that frame with the same primitives (paint.ts), then settles into
// its own look.
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force';
import { DIMENSIONS, ROOT_COLOR, dimensionColor } from '../data';
import type { NodeKind, ProfileModel, ProfileNode, ReviewView } from '../data';
import {
  LOCAL_WAVE,
  WAVE,
  arrivalTime,
  hubSwell,
  swellProgress,
  swellSpan,
  wavePeriod,
  waveRunning,
} from './atlasFlow';
import { inkStrength, supportWidth, type NodeMetrics } from './atlasMetrics';
import type { Camera, EntranceItem } from './entrance';
import {
  DENSE_NODE_COUNT,
  clusterCenters,
  computeSeeds,
  dimensionAngle,
  expansionPositions,
  overviewPositions,
  placeNodes,
  sameTree,
  shiftClusters,
  type Point,
  type Positions,
} from './graphLayout';
import {
  EDGE_OPACITY,
  LABEL_LINE_HEIGHT,
  LABEL_OFFSET_Y,
  edgeStroke,
  edgeWidth,
  hexToRgb,
  labelFontSize,
} from './graphStyle';
import type { NodeLook } from './nodeLook';
import { canvasDpr, drawLabel, drawNode, rgbaOf, strokeEdge, type PaintNode, type Rgb, type Transform } from './paint';

/** What the toolbar can do with the map. */
export interface AtlasControls {
  zoomBy(factor: number): void;
  fit(): void;
}

/** A node to preview at (x, y), in pixels from the map's top left, inside a `width` × `height` area. */
export type HoverInfo = { id: string; x: number; y: number; width: number; height: number };

/** Room other panels take from the map's right and bottom edges. */
export type Insets = { right: number; bottom: number };

export type AtlasCallbacks = {
  onNodeClick: (id: string) => void;
  onBackgroundClick: () => void;
  onHover: (hover: HoverInfo | null) => void;
};

type MapNode = SimulationNodeDatum &
  PaintNode & {
    id: string;
    node: ProfileNode;
    depth: number;
    metrics: NodeMetrics;
    color: string;
    rgb: Rgb;
    /** Width of the name at the weight the layout measures with. */
    labelWidth: number;
    x: number;
    y: number;
    /** Offset from the laid-out position, eased away after a new layout moves the node. */
    ox: number;
    oy: number;
    /** Where the node is drawn. */
    rx: number;
    ry: number;
    /** Index in `order`. */
    at: number;
    born: number;
    delay: number;
    /** Progress 0…1 of growing out of the parent after an expansion. */
    enterT: number;
    /** 1, or less while another branch is hovered or selected. */
    emph: number;
    scale: number;
    vis: number;
    hasKids: boolean;
  };

type MapLink = SimulationLinkDatum<MapNode> & { child: MapNode };
type MapEdge = { c: MapNode; p: MapNode; w: number; s: number };
/** A removed node shrinking into its nearest remaining ancestor. */
type Ghost = {
  n: MapNode;
  paint: PaintNode;
  into: MapNode | null;
  start: number;
  x: number;
  y: number;
  a: number;
  scale: number;
};
type Ring = { n: MapNode; start: number; soft: boolean };
type View = { x: number; y: number; k: number };
type CameraGoal = { until: number } & ({ fixed: View } | { follow: string } | { fit: true });
type Press = { id: number; x: number; y: number; lx: number; ly: number; n: MapNode | null; moved: boolean };
type Box = { x0: number; y0: number; x1: number; y1: number };

const TAU = Math.PI * 2;
const ZOOM_MIN = 0.28;
const ZOOM_MAX = 2.2;
/** Emphasis of nodes off the hovered branch, and off the selected one. */
const HOVER_DIM = 0.12;
const FOCUS_DIM = 0.42;
/** How fast the camera closes in on its goal, per second. */
const CAMERA_RATE = 5.2;
const ENTER_MS = 700;
const EXIT_MS = 380;
/** Time the map takes to turn from the entrance's look into its own. */
const SETTLE_MS = 700;
const TOOLTIP_DELAY = 250;
// The layout cools slowly during and after a drag, so the map eases to rest after release.
const DRAG_ALPHA_DECAY = 0.06;
// How strongly each node is pulled to its anchor: the root stays put, dimension nodes hold
// their position from `clusterCenters`, and everything else is kept near its dimension node.
const CLUSTER_PULL: Record<NodeKind, number> = {
  root: 1,
  dimension: 0.2,
  topic: 0.05,
  subtopic: 0.03,
  attribute: 0.03,
};
/** Room kept free for the top bar, the toolbar and the help line. */
const PADDING = { top: 92, right: 56, bottom: 86, left: 56 };
/** Closest zoom that shows a node's whole branch, and least zoom on a node shown alone. */
const BRANCH_ZOOM: Record<NodeKind, number> = { root: 0.95, dimension: 0.9, topic: 1.1, subtopic: 1.2, attribute: 1.2 };
const NODE_ZOOM: Record<NodeKind, number> = {
  root: 0.72,
  dimension: 0.76,
  topic: 0.92,
  subtopic: 1.04,
  attribute: 1.16,
};
/** Share of the frame a focused branch fills, so that some of its surroundings stay in view. */
const BRANCH_FILL = 0.88;
// While the map is dense, attribute labels show only when zoomed in far enough to read them.
const ATTRIBUTE_LABEL_MIN_ZOOM = 8 / labelFontSize('attribute');
const MEASURE_FONT = '"Noto Sans", "Noto Sans CJK SC", sans-serif';
const DIMENSION_INDEX = new Map<number, number>(DIMENSIONS.map((entry, index) => [entry.dim, index]));
const BLACK: Rgb = [0, 0, 0];

const clamp = (value: number, min = 0, max = 1) => (value < min ? min : value > max ? max : value);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const mix = (a: Rgb, b: Rgb, t: number): Rgb => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const easeOutCubic = (t: number) => 1 - (1 - t) ** 3;
const easeInCubic = (t: number) => t * t * t;
const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const easeOutBack = (t: number) => 1 + 2.5 * (t - 1) ** 3 + 1.5 * (t - 1) ** 2;

/** A stable number in [0, 1) for `text` (FNV-1a). */
function hash01(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 1000003) / 1000003;
}

export class AtlasEngine implements AtlasControls {
  private readonly canvas: HTMLCanvasElement;
  private readonly wash: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly washCtx: CanvasRenderingContext2D;
  private readonly callbacks: AtlasCallbacks;
  private readonly reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  private readonly measure = document.createElement('canvas').getContext('2d');
  private readonly sprites = new Map<string, HTMLCanvasElement>();

  private model: ProfileModel | null = null;
  private view: ReviewView = 'all';
  private looks = new Map<string, NodeLook>();
  private metrics = new Map<string, NodeMetrics>();
  private initialPositions: Positions = new Map();
  private visible: ProfileNode[] | null = null;

  private nodes = new Map<string, MapNode>();
  /** Visible nodes, every parent before its children. */
  private order: MapNode[] = [];
  private edges: MapEdge[] = [];
  private ghosts: Ghost[] = [];
  private rings: Ring[] = [];
  private sim: Simulation<MapNode, MapLink> | null = null;
  private layoutDecay = DRAG_ALPHA_DECAY;
  /** Targets of the root and of every dimension in the current layout. */
  private anchors: Positions = new Map();
  /** How far drags have moved the root and each dimension from where the layout puts them; kept across layouts. */
  private readonly anchorShifts: Positions = new Map();
  /** Positions in each review view, restored when switching back to it. */
  private readonly viewPositions = new Map<ReviewView, Positions>();
  private layoutView: ReviewView | null = null;
  private fitted = false;

  private selectedId: string | null = null;
  private hoverId: string | null = null;
  /** The hovered node and its ancestors, and with its descendants. */
  private hoverPath: Set<string> | null = null;
  private hoverSet: Set<string> | null = null;
  private hotRoot = false;
  /** The selected node and its ancestors, and with its descendants. */
  private routeSet: Set<string> | null = null;
  private focusSet: Set<string> | null = null;

  private waveStart = 0;
  /** Time into the current wave cycle, -1 before it starts. */
  private waveClock = -1;
  private waveDepth = 1;
  private localStart = 0;
  private localDepth = 1;

  /** The map point at the centre of the canvas, and the zoom. */
  private cam: View = { x: 0, y: 0, k: 1 };
  private goal: CameraGoal | null = null;
  private userMoved = false;
  /** The view before a click brought a node into view, restored by a click on the background. */
  private overview: { view: View; fitted: boolean } | null = null;
  private insets: Insets = { right: 0, bottom: 0 };
  /** Visible map area of the frame being drawn, with a margin. */
  private bounds: Box = { x0: 0, y0: 0, x1: 0, y1: 0 };

  private W = 0;
  private H = 0;
  private dpr = 1;
  private time = 0;
  private last = 0;
  private raf = 0;
  private running = false;
  private destroyed = false;
  private dirty = false;
  /** Until the entrance hands over, the map stays blank. */
  private hidden = true;
  private settleStart = 0;
  /** Progress 0…1 from the entrance's look to the map's own. */
  private settle = 1;

  private press: Press | null = null;
  private drag: MapNode | null = null;
  private readonly pointers = new Map<number, Point>();
  private pinch: { distance: number; k: number; anchor: Point } | null = null;
  private tipTimer = 0;
  private tipShown = false;

  constructor(canvas: HTMLCanvasElement, wash: HTMLCanvasElement, callbacks: AtlasCallbacks) {
    this.canvas = canvas;
    this.wash = wash;
    this.ctx = canvas.getContext('2d')!;
    this.washCtx = wash.getContext('2d')!;
    this.callbacks = callbacks;
    canvas.style.cursor = 'grab';
    canvas.addEventListener('pointerdown', this.onDown);
    canvas.addEventListener('pointermove', this.onMove);
    canvas.addEventListener('pointerup', this.onUp);
    canvas.addEventListener('pointercancel', this.onCancel);
    canvas.addEventListener('pointerleave', this.onLeave);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    document.addEventListener('visibilitychange', this.onVisibility);
    this.resize();
  }

  // ───────────────────────── public api ─────────────────────────

  setModel(model: ProfileModel, view: ReviewView, looks: Map<string, NodeLook>, metrics: Map<string, NodeMetrics>) {
    if (model !== this.model) this.initialPositions = overviewPositions(model);
    this.model = model;
    this.view = view;
    this.looks = looks;
    this.metrics = metrics;
  }

  /**
   * Shows `visible`, parents first. The first call lays the map out at once and fits it to the
   * canvas. Later calls keep the nodes that stay where they are, grow new ones out of their
   * parents and let the layout settle; switching review views restores that view's last layout.
   */
  setTree(visible: ProfileNode[]) {
    const model = this.model;
    if (!model || visible === this.visible) return;
    this.visible = visible;
    const switchingView = this.layoutView !== null && this.layoutView !== this.view;
    if (
      this.fitted &&
      !switchingView &&
      sameTree(
        this.order.map((n) => n.node),
        visible,
      )
    ) {
      // A new model with the same nodes: keep the layout and take over sizes, scores and names.
      for (const n of this.order) this.refresh(n, model.nodes.get(n.id)!);
      this.connect();
      this.sim?.nodes(this.order);
      this.requestDraw();
      return;
    }
    const initial = !this.fitted;
    const previous: Positions = new Map(this.order.map((n) => [n.id, { x: n.x, y: n.y }]));
    if (this.fitted && this.layoutView) this.viewPositions.set(this.layoutView, previous);
    const restored = switchingView ? this.viewPositions.get(this.view) : undefined;
    const root = previous.get('root') ?? this.initialPositions.get('root')!;
    // Clusters are laid out around the root's undragged place, then moved by their own drags.
    const rootShift = this.anchorShifts.get('root') ?? { x: 0, y: 0 };
    const centers = clusterCenters(model, visible, { x: root.x - rootShift.x, y: root.y - rootShift.y });
    for (const [id, center] of centers) {
      const shift = this.anchorShifts.get(id);
      if (shift) centers.set(id, { x: center.x + shift.x, y: center.y + shift.y });
    }
    const seeds = shiftClusters(model, computeSeeds(switchingView, previous, this.initialPositions, restored), centers);
    const fallback = initial
      ? shiftClusters(model, this.initialPositions, centers)
      : expansionPositions(model, visible, seeds);
    const positions = placeNodes(visible, seeds, fallback);

    const now = this.time;
    const next = new Map<string, MapNode>();
    const born: MapNode[] = [];
    visible.forEach((node, at) => {
      const position = positions.get(node.id)!;
      let n = this.nodes.get(node.id);
      if (n) {
        this.refresh(n, node);
        n.ox += n.x - position.x;
        n.oy += n.y - position.y;
        n.vx = 0;
        n.vy = 0;
        n.fx = null;
        n.fy = null;
      } else {
        n = this.makeNode(node);
        n.rx = position.x;
        n.ry = position.y;
        born.push(n);
      }
      n.x = position.x;
      n.y = position.y;
      n.at = at;
      next.set(node.id, n);
    });
    if (!this.reduced) {
      for (const [id, n] of this.nodes) {
        if (next.has(id)) continue;
        let into: MapNode | undefined;
        for (let up = n.node.parentId; up && !into; up = this.nodes.get(up)?.node.parentId ?? null) into = next.get(up);
        const paint: PaintNode = { kind: n.kind, dim: n.dim, look: n.look, r: n.r, label: '', selected: false };
        this.ghosts.push({ n, paint, into: into ?? null, start: now, x: n.rx, y: n.ry, a: n.vis * n.emph, scale: 1 });
      }
    }
    if (this.drag && !next.has(this.drag.id)) this.drag = null;
    this.nodes = next;
    this.order = visible.map((node) => next.get(node.id)!);
    if (!initial && !this.reduced && born.length) {
      born.sort((a, b) => a.depth - b.depth);
      const stagger = Math.min(38, 640 / born.length);
      born.forEach((n, i) => {
        n.born = now;
        n.delay = i * stagger;
        n.enterT = 0;
        n.vis = 0;
        n.scale = 0;
      });
    }
    this.connect();
    this.layoutView = this.view;
    this.anchors = new Map([...centers, ['root', root]]);
    this.buildSimulation();
    if (initial) {
      const sim = this.sim!;
      for (let i = 0; i < 500 && sim.alpha() > sim.alphaMin(); i++) sim.tick();
      for (const n of this.order) {
        n.rx = n.x;
        n.ry = n.y;
      }
      this.fitted = true;
      if (this.W >= 2) this.cam = this.fitView();
    } else {
      this.waveStart = this.time + 520;
      this.waveClock = -1;
    }
    if (this.hoverId && !next.has(this.hoverId)) this.setHover(null);
    this.recomputeSets();
    this.requestDraw();
  }

  /** Marks `id` as selected. With `focus` a newly selected node is ringed and its branch brought into view. */
  setSelected(id: string | null, focus: boolean) {
    const changed = id !== this.selectedId;
    this.selectedId = id;
    for (const n of this.order) n.selected = n.id === id;
    this.recomputeSets();
    this.hideTip();
    if (focus && changed && id) this.pulse(id);
    this.requestDraw();
  }

  setInsets(insets: Insets) {
    if (insets.right === this.insets.right && insets.bottom === this.insets.bottom) return;
    this.insets = insets;
    this.requestDraw();
  }

  /** Follows the canvas size. Until the user moves the map it stays fitted. */
  resize() {
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    const dpr = canvasDpr();
    if (W < 2 || H < 2 || (W === this.W && H === this.H && dpr === this.dpr)) return;
    this.W = W;
    this.H = H;
    this.dpr = dpr;
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(H * dpr);
    this.wash.width = Math.max(1, Math.ceil(W / 4));
    this.wash.height = Math.max(1, Math.ceil(H / 4));
    this.hideTip();
    if (this.fitted && !this.userMoved) this.cam = this.fitView();
    // Resizing cleared the canvas: draw again before the browser shows it.
    this.draw();
  }

  /** The current picture for the entrance to build up, in its terms. */
  entranceScene(): { items: EntranceItem[]; camera: Camera; width: number; height: number } {
    this.updateSlices();
    const { k } = this.cam;
    const attributeLabels = this.attributeLabelsShown();
    return {
      items: this.order.map((n) => ({
        id: n.id,
        kind: n.kind,
        dim: n.dim,
        parentId: n.node.parentId,
        x: n.rx,
        y: n.ry,
        look: n.look,
        label: this.labelOf(n, attributeLabels),
        selected: n.selected,
        slices: n.slices,
      })),
      camera: { zoom: k, x: this.W / 2 - this.cam.x * k, y: this.H / 2 - this.cam.y * k },
      width: this.W,
      height: this.H,
    };
  }

  /**
   * Shows the map, drawn at once. With `settle` it starts as the entrance's last frame and then
   * eases into its own look; without, it shows that look straight away.
   */
  reveal(settle: boolean) {
    if (!this.hidden) return;
    this.hidden = false;
    this.settle = settle && !this.reduced ? 0 : 1;
    this.settleStart = this.time;
    this.waveStart = this.time + 1500;
    this.waveClock = -1;
    this.draw();
    this.start();
  }

  zoomBy(factor: number) {
    this.goal = { fixed: { ...this.cam, k: clamp(this.cam.k * factor, ZOOM_MIN, ZOOM_MAX) }, until: this.time + 700 };
    this.userMoved = true;
    this.hideTip();
    this.requestDraw();
  }

  fit() {
    this.goal = { fit: true, until: this.time + 1400 };
    this.userMoved = false;
    this.overview = null;
    this.hideTip();
    this.requestDraw();
  }

  destroy() {
    this.stop();
    this.destroyed = true;
    window.clearTimeout(this.tipTimer);
    this.sim?.stop();
    const canvas = this.canvas;
    canvas.removeEventListener('pointerdown', this.onDown);
    canvas.removeEventListener('pointermove', this.onMove);
    canvas.removeEventListener('pointerup', this.onUp);
    canvas.removeEventListener('pointercancel', this.onCancel);
    canvas.removeEventListener('pointerleave', this.onLeave);
    canvas.removeEventListener('wheel', this.onWheel);
    document.removeEventListener('visibilitychange', this.onVisibility);
  }

  // ───────────────────────── nodes and layout ─────────────────────────

  private makeNode(node: ProfileNode): MapNode {
    const look = this.looks.get(node.id)!;
    const metrics = this.metrics.get(node.id)!;
    const color = node.kind === 'root' ? ROOT_COLOR : dimensionColor(node.dim);
    return {
      id: node.id,
      node,
      kind: node.kind,
      dim: node.dim,
      look,
      r: look.size / 2,
      label: node.name,
      selected: node.id === this.selectedId,
      depth: metrics.depth,
      metrics,
      color,
      rgb: hexToRgb(color),
      labelWidth: this.measureLabel(node),
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      ox: 0,
      oy: 0,
      rx: 0,
      ry: 0,
      at: 0,
      born: this.time,
      delay: 0,
      enterT: 1,
      emph: 1,
      scale: 1,
      vis: 1,
      hasKids: false,
    };
  }

  private refresh(n: MapNode, node: ProfileNode) {
    if (node.name !== n.node.name) n.labelWidth = this.measureLabel(node);
    n.node = node;
    n.look = this.looks.get(node.id)!;
    n.r = n.look.size / 2;
    n.metrics = this.metrics.get(node.id)!;
    n.depth = n.metrics.depth;
  }

  private measureLabel(node: ProfileNode): number {
    const size = labelFontSize(node.kind);
    if (!this.measure) return node.name.length * size;
    this.measure.font = `600 ${size}px ${MEASURE_FONT}`;
    return this.measure.measureText(node.name).width;
  }

  /** Lines from every visible node to its parent; their width follows the child's chat support. */
  private connect() {
    for (const n of this.order) n.hasKids = false;
    this.edges = [];
    let depth = 1;
    for (const c of this.order) {
      const p = c.node.parentId ? this.nodes.get(c.node.parentId) : undefined;
      if (!p) continue;
      const w = Math.round(supportWidth(c.metrics.support) * 4) / 4;
      this.edges.push({ c, p, w, s: inkStrength(w) });
      p.hasKids = true;
      depth = Math.max(depth, c.depth);
    }
    this.waveDepth = depth;
  }

  private collisionRadius(n: MapNode, dense: boolean): number {
    if (dense && n.kind === 'attribute') return n.r + 8;
    return Math.max(n.r + 8, Math.hypot(n.labelWidth / 2 + 4, n.r + 11 + labelFontSize(n.kind)));
  }

  /** The forces read `anchors` whenever they are initialised, so moving an anchor and re-initialising moves its cluster. */
  private buildSimulation() {
    const model = this.model!;
    const count = this.order.length;
    const dense = count > DENSE_NODE_COUNT;
    const busy = count > 200;
    const anchors = this.anchors;
    const dimensionIdOf = new Map(model.dimensionIds.map((id) => [model.nodes.get(id)!.dim, id]));
    const target = (n: MapNode) =>
      anchors.get(n.kind === 'root' ? 'root' : (dimensionIdOf.get(n.dim) ?? 'root')) ?? anchors.get('root')!;
    const pull = (n: MapNode) => CLUSTER_PULL[n.kind];
    const links: MapLink[] = this.edges.map((e) => ({ source: e.p, target: e.c, child: e.c }));
    this.sim?.stop();
    this.layoutDecay = dense ? 0.65 : busy ? 0.4 : 0.06;
    this.sim = forceSimulation<MapNode, MapLink>(this.order)
      .force('x', forceX<MapNode>((n) => target(n).x).strength(pull))
      .force('y', forceY<MapNode>((n) => target(n).y).strength(pull))
      .force(
        'collide',
        forceCollide<MapNode>((n) => this.collisionRadius(n, dense))
          .strength(1)
          .iterations(busy ? 1 : 2),
      )
      .force(
        'link',
        forceLink<MapNode, MapLink>(links)
          .distance(({ child }) => {
            const anchor = child.kind === 'dimension' ? anchors.get(child.id) : undefined;
            const root = anchors.get('root')!;
            if (anchor) return Math.hypot(anchor.x - root.x, anchor.y - root.y);
            return child.kind === 'attribute' ? 110 : child.kind === 'subtopic' ? 150 : 220;
          })
          .strength(0.75),
      )
      .force(
        'charge',
        forceManyBody<MapNode>()
          .strength((n) => (n.kind === 'root' ? -260 : n.kind === 'dimension' ? -200 : n.kind === 'topic' ? -90 : -40))
          .distanceMax(700),
      )
      .alphaDecay(this.layoutDecay)
      .stop();
  }

  private startDrag(n: MapNode) {
    const sim = this.sim;
    if (!sim) return;
    this.drag = n;
    n.fx = n.x;
    n.fy = n.y;
    sim.alphaDecay(DRAG_ALPHA_DECAY).alphaTarget(0.3);
    if (sim.alpha() < 0.3) sim.alpha(0.3);
  }

  // A dropped root or dimension stays where it was dragged to: its anchor moves there, and the
  // dimension's cluster follows. Other nodes keep their anchors.
  private endDrag() {
    const n = this.drag!;
    this.drag = null;
    n.fx = null;
    n.fy = null;
    this.sim?.alphaTarget(0);
    const anchor = this.anchors.get(n.id);
    if (!anchor) return;
    const shift = this.anchorShifts.get(n.id) ?? { x: 0, y: 0 };
    this.anchorShifts.set(n.id, { x: shift.x + n.x - anchor.x, y: shift.y + n.y - anchor.y });
    this.anchors.set(n.id, { x: n.x, y: n.y });
    this.sim?.nodes(this.order);
  }

  private subtree(id: string): string[] {
    const n = this.nodes.get(id);
    if (!n) return [];
    const ids = [n.id];
    for (let i = n.at + 1; i < this.order.length && this.order[i].depth > n.depth; i++) ids.push(this.order[i].id);
    return ids;
  }

  private pathTo(id: string): Set<string> {
    const path = new Set<string>();
    for (let n = this.nodes.get(id); n; n = n.node.parentId ? this.nodes.get(n.node.parentId) : undefined) {
      path.add(n.id);
    }
    return path;
  }

  private recomputeSets() {
    const hovered = this.hoverId;
    this.hotRoot = hovered === 'root';
    if (hovered && hovered !== 'root' && this.nodes.has(hovered)) {
      this.hoverPath = this.pathTo(hovered);
      this.hoverSet = new Set([...this.hoverPath, ...this.subtree(hovered)]);
    } else {
      this.hoverPath = null;
      this.hoverSet = null;
    }
    const selected = this.selectedId;
    if (selected && selected !== 'root' && this.nodes.has(selected)) {
      this.routeSet = this.pathTo(selected);
      this.focusSet = new Set([...this.routeSet, ...this.subtree(selected)]);
    } else {
      this.routeSet = null;
      this.focusSet = null;
    }
    let depth = 1;
    for (const id of this.hoverSet ?? []) depth = Math.max(depth, this.nodes.get(id)!.depth);
    this.localDepth = depth;
  }

  /** Lines on the path from the hovered or selected node to the centre, or all lines into the hovered centre. */
  private isHot(e: MapEdge): boolean {
    return (
      (this.hoverPath?.has(e.c.id) ?? false) ||
      (this.routeSet?.has(e.c.id) ?? false) ||
      (this.hotRoot && e.p.kind === 'root')
    );
  }

  /** Rings a node, brings its branch into view and sends a wave soon if none is running. */
  private pulse(id: string) {
    const n = this.nodes.get(id);
    if (!n) return;
    if (!this.reduced) this.rings.push({ n, start: this.time, soft: false });
    // Further clicks keep the view from before the first one.
    if (!this.overview) {
      const view = (this.goal && this.goalView(this.goal)) ?? this.cam;
      this.overview = { view, fitted: !this.userMoved || (this.goal !== null && 'fit' in this.goal) };
    }
    this.goal = { follow: id, until: this.time + 1900 };
    this.userMoved = true;
    this.localStart = this.time;
    if (!this.reduced && !waveRunning(WAVE, this.waveDepth, this.waveClock)) {
      this.waveStart = Math.max(this.waveStart, this.time + 380);
    }
    this.requestDraw();
  }

  // ───────────────────────── camera ─────────────────────────

  private toWorld(p: Point): Point {
    return { x: (p.x - this.W / 2) / this.cam.k + this.cam.x, y: (p.y - this.H / 2) / this.cam.k + this.cam.y };
  }

  /** Part of the canvas the map is framed in: clear of the panels, the top bar and the toolbar. */
  private frame(): Box {
    const right = this.W - this.insets.right - (this.insets.right ? 32 : PADDING.right);
    const bottom = this.H - this.insets.bottom - (this.insets.bottom ? 24 : PADDING.bottom);
    return {
      x0: PADDING.left,
      y0: PADDING.top,
      x1: Math.max(PADDING.left + 160, right),
      y1: Math.max(PADDING.top + 160, bottom),
    };
  }

  /** Camera that puts map point (x, y) in the middle of the frame at zoom `k`. */
  private centered(x: number, y: number, k: number): View {
    const frame = this.frame();
    return {
      x: x - ((frame.x0 + frame.x1) / 2 - this.W / 2) / k,
      y: y - ((frame.y0 + frame.y1) / 2 - this.H / 2) / k,
      k,
    };
  }

  /** Camera that fits `box` into `fill` of the frame, zoomed in at most to `maxZoom`. */
  private framing(box: Box, maxZoom: number, fill = 1): View {
    const frame = this.frame();
    const k = clamp(
      fill *
        Math.min(
          (frame.x1 - frame.x0) / Math.max(1, box.x1 - box.x0),
          (frame.y1 - frame.y0) / Math.max(1, box.y1 - box.y0),
        ),
      ZOOM_MIN,
      maxZoom,
    );
    return this.centered((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, k);
  }

  /** Box around the nodes where the layout is taking them, with their labels. */
  private box(nodes: MapNode[]): Box {
    const attributeLabels = this.attributeLabelsShown();
    const box: Box = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
    for (const n of nodes) {
      const x = n.x + n.ox;
      const y = n.y + n.oy;
      const label = n.kind !== 'attribute' || attributeLabels;
      const half = label ? Math.max(n.r, n.labelWidth / 2 + 4) : n.r;
      box.x0 = Math.min(box.x0, x - half);
      box.x1 = Math.max(box.x1, x + half);
      box.y0 = Math.min(box.y0, y - n.r);
      box.y1 = Math.max(box.y1, y + n.r + (label ? LABEL_OFFSET_Y + LABEL_LINE_HEIGHT : 0));
    }
    return box;
  }

  /** The whole map in the frame. */
  private fitView(): View {
    return this.order.length ? this.framing(this.box(this.order), ZOOM_MAX) : this.cam;
  }

  private goalView(goal: CameraGoal): View | null {
    if ('fixed' in goal) return goal.fixed;
    if ('fit' in goal) return this.fitView();
    const n = this.nodes.get(goal.follow);
    if (!n) return null;
    const ids = this.subtree(n.id);
    if (ids.length <= 1 || n.kind === 'attribute') {
      return this.centered(n.x + n.ox, n.y + n.oy, clamp(Math.max(this.cam.k, NODE_ZOOM[n.kind]), ZOOM_MIN, ZOOM_MAX));
    }
    return this.framing(this.box(ids.map((id) => this.nodes.get(id)!)), BRANCH_ZOOM[n.kind], BRANCH_FILL);
  }

  // ───────────────────────── frame loop ─────────────────────────

  private requestDraw() {
    this.dirty = true;
  }

  private start() {
    if (this.running || this.destroyed || this.hidden || document.hidden) return;
    this.running = true;
    this.last = performance.now();
    const loop = (now: number) => {
      if (!this.running) return;
      const dt = Math.min(0.05, Math.max(0, (now - this.last) / 1000));
      this.last = now;
      this.time += dt * 1000;
      // Nothing moves while a wave rests and nobody interacts: skip those frames.
      if (this.update(dt)) {
        this.dirty = false;
        this.draw();
      }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  private stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  /** Advances everything by `dt` seconds; returns whether the picture changed. */
  private update(dt: number): boolean {
    const now = this.time;
    let changed = this.dirty;
    const sim = this.sim;
    if (sim && (sim.alpha() > sim.alphaMin() || this.drag)) {
      sim.tick();
      changed = true;
    } else if (sim && sim.alphaDecay() !== this.layoutDecay) sim.alphaDecay(this.layoutDecay);

    const decay = Math.exp(-dt * 7);
    const ease = this.reduced ? 1 : 1 - Math.exp(-dt * 11);
    const hover = this.hoverSet;
    const focus = this.focusSet;
    for (const n of this.order) {
      if (n.ox || n.oy) {
        n.ox *= decay;
        n.oy *= decay;
        if (Math.abs(n.ox) < 0.01 && Math.abs(n.oy) < 0.01) {
          n.ox = 0;
          n.oy = 0;
        }
        changed = true;
      }
      const target = hover ? (hover.has(n.id) ? 1 : HOVER_DIM) : focus ? (focus.has(n.id) ? 1 : FOCUS_DIM) : 1;
      if (n.emph !== target) {
        n.emph += (target - n.emph) * ease;
        if (Math.abs(target - n.emph) < 0.002) n.emph = target;
        changed = true;
      }
      const x = n.x + n.ox;
      const y = n.y + n.oy;
      if (n.enterT < 1) {
        // Grows out of its parent, which comes earlier in `order` and is placed already.
        n.enterT = clamp((now - n.born - n.delay) / ENTER_MS);
        const parent = n.node.parentId ? this.nodes.get(n.node.parentId) : undefined;
        const e = easeOutCubic(n.enterT);
        n.rx = parent ? lerp(parent.rx, x, e) : x;
        n.ry = parent ? lerp(parent.ry, y, e) : y;
        n.scale = n.enterT <= 0 ? 0 : 0.3 + 0.7 * easeOutBack(n.enterT);
        n.vis = clamp(n.enterT * 2.2);
        changed = true;
      } else {
        n.rx = x;
        n.ry = y;
        n.scale = 1;
        n.vis = 1;
      }
    }

    if (this.ghosts.length) {
      this.ghosts = this.ghosts.filter((g) => now - g.start < EXIT_MS);
      for (const g of this.ghosts) {
        const e = easeInCubic(clamp((now - g.start) / EXIT_MS));
        g.x = g.into ? lerp(g.n.rx, g.into.rx, e) : g.n.rx;
        g.y = g.into ? lerp(g.n.ry, g.into.ry, e) : g.n.ry;
        g.a = (1 - e) * g.n.vis * g.n.emph;
        g.scale = 1 - 0.7 * e;
      }
      changed = true;
    }

    if (this.goal) {
      const goal = this.goalView(this.goal);
      if (!goal) this.goal = null;
      else {
        const f = this.reduced ? 1 : 1 - Math.exp(-dt * CAMERA_RATE);
        const { x, y, k } = this.cam;
        this.cam = {
          x: lerp(x, goal.x, f),
          y: lerp(y, goal.y, f),
          k: Math.exp(lerp(Math.log(k), Math.log(goal.k), f)),
        };
        if (now > this.goal.until) this.goal = null;
      }
      changed = true;
    }

    if (this.settle < 1) {
      this.settle = clamp((now - this.settleStart) / SETTLE_MS);
      changed = true;
    }

    if (!this.reduced) {
      const previous = this.waveClock;
      const raw = now - this.waveStart;
      this.waveClock = raw < 0 ? -1 : raw % wavePeriod(WAVE, this.waveDepth);
      const arrival = arrivalTime(WAVE, this.waveDepth);
      if (previous >= 0 && previous < arrival && this.waveClock >= arrival) {
        const root = this.nodes.get('root');
        if (root && root.vis > 0.5) this.rings.push({ n: root, start: now, soft: true });
      }
      // Swells run until the arrival, the centre's halo swells a little longer; the hovered branch never rests.
      if ((this.waveClock >= 0 && this.waveClock < arrival + 700) || this.hoverSet) changed = true;
    }

    if (this.rings.length) {
      this.rings = this.rings.filter((ring) => now - ring.start < (ring.soft ? 1300 : 900));
      changed = true;
    }
    return changed;
  }

  // ───────────────────────── drawing ─────────────────────────

  private inView(x: number, y: number, r: number): boolean {
    const b = this.bounds;
    return x + r >= b.x0 && x - r <= b.x1 && y + r >= b.y0 && y - r <= b.y1;
  }

  private segmentInView(x0: number, y0: number, x1: number, y1: number): boolean {
    const b = this.bounds;
    return !(
      (x0 < b.x0 && x1 < b.x0) ||
      (x0 > b.x1 && x1 > b.x1) ||
      (y0 < b.y0 && y1 < b.y0) ||
      (y0 > b.y1 && y1 > b.y1)
    );
  }

  private attributeLabelsShown(): boolean {
    return this.order.length <= DENSE_NODE_COUNT || this.cam.k >= ATTRIBUTE_LABEL_MIN_ZOOM;
  }

  private labelOf(n: MapNode, attributeLabels: boolean): string {
    return n.kind !== 'attribute' || attributeLabels || n.selected || n.id === this.hoverId ? n.node.name : '';
  }

  /** The centre's slices: each dimension's colour in the direction its node lies. */
  private updateSlices() {
    const model = this.model;
    const root = this.nodes.get('root');
    if (!model || !root) return;
    const count = model.dimensionIds.length;
    root.slices = model.dimensionIds.map((id, index) => {
      const d = this.nodes.get(id);
      const angle = d
        ? Math.atan2(d.y + d.oy - (root.y + root.oy), d.x + d.ox - (root.x + root.ox))
        : dimensionAngle(index, count);
      return { angle, rgb: hexToRgb(dimensionColor(model.nodes.get(id)!.dim)) };
    });
  }

  /** A soft radial blot of `rgb`: a node's halo, or a cluster's wash. */
  private sprite(kind: 'halo' | 'wash', rgb: Rgb): HTMLCanvasElement {
    const key = `${kind}${rgb.join(',')}`;
    let sprite = this.sprites.get(key);
    if (!sprite) {
      const size = kind === 'halo' ? 128 : 256;
      sprite = document.createElement('canvas');
      sprite.width = size;
      sprite.height = size;
      const g = sprite.getContext('2d')!;
      const gradient = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      const stops =
        kind === 'halo'
          ? [
              [0, 0.5],
              [0.32, 0.26],
              [0.68, 0.07],
              [1, 0],
            ]
          : [
              [0, 0.9],
              [0.45, 0.42],
              [1, 0],
            ];
      for (const [at, alpha] of stops) gradient.addColorStop(at, rgbaOf(rgb, alpha));
      g.fillStyle = gradient;
      g.fillRect(0, 0, size, size);
      this.sprites.set(key, sprite);
    }
    return sprite;
  }

  /** 0…1 swell of a node's halo while the ink from its children drains into it. */
  private hubBump(n: MapNode): number {
    if (!n.hasKids || this.reduced) return 0;
    let bump = hubSwell(WAVE, this.waveDepth, n.depth, this.waveClock, 230);
    if (this.hoverSet?.has(n.id))
      bump = Math.max(bump, hubSwell(LOCAL_WAVE, this.localDepth, n.depth, this.localClock(), 200));
    return bump;
  }

  private localClock(): number {
    return (this.time - this.localStart) % wavePeriod(LOCAL_WAVE, this.localDepth);
  }

  private draw() {
    if (this.hidden || this.destroyed || this.W < 2) return;
    const { ctx, dpr, cam } = this;
    const k = cam.k;
    const tx = this.W / 2 - cam.x * k;
    const ty = this.H / 2 - cam.y * k;
    const tf: Transform = { k: dpr * k, ox: dpr * tx, oy: dpr * ty };
    const margin = 80 / k;
    this.bounds = {
      x0: -tx / k - margin,
      y0: -ty / k - margin,
      x1: (this.W - tx) / k + margin,
      y1: (this.H - ty) / k + margin,
    };
    const settle = easeInOutCubic(this.settle);
    this.updateSlices();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawWash(tx, ty, k, settle);
    if (settle > 0) this.drawHalos(tf, settle);
    this.drawEdges(tf, settle);
    if (settle > 0 && !this.reduced) this.drawWaves(tf, settle);
    // The entrance's glow fades as the halos take over.
    this.drawNodes(tf, 1 - settle);
    this.drawRings(tf);
    this.drawLabels(tf);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
  }

  // Each dimension washes the paper under its cluster with three overlapping blots. The wash
  // canvas is a quarter of the size and multiplies onto the page through CSS.
  private drawWash(tx: number, ty: number, k: number, settle: number) {
    const g = this.washCtx;
    const scale = this.wash.width / this.W;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.globalAlpha = 1;
    g.clearRect(0, 0, this.wash.width, this.wash.height);
    if (settle <= 0) return;
    const stats = new Map<number, { x: number; y: number; n: number; w: number; d: number }>();
    for (const n of this.order) {
      if (n.dim === null) continue;
      const stat = stats.get(n.dim) ?? { x: 0, y: 0, n: 0, w: 0, d: 0 };
      stat.x += n.rx;
      stat.y += n.ry;
      stat.n += 1;
      stat.w += n.vis * n.emph;
      stats.set(n.dim, stat);
    }
    for (const stat of stats.values()) {
      stat.x /= stat.n;
      stat.y /= stat.n;
      stat.w /= stat.n;
    }
    for (const n of this.order) {
      if (n.dim === null) continue;
      const stat = stats.get(n.dim)!;
      stat.d += (n.rx - stat.x) ** 2 + (n.ry - stat.y) ** 2;
    }
    g.setTransform(k * scale, 0, 0, k * scale, tx * scale, ty * scale);
    for (const [dim, stat] of stats) {
      // Domains are not in DIMENSIONS; their keys are distinct too.
      const index = DIMENSION_INDEX.get(dim) ?? dim;
      const sprite = this.sprite('wash', hexToRgb(dimensionColor(dim)));
      const R = Math.sqrt(stat.d / stat.n) * 1.3 + 100;
      const base = 0.028 + 0.055 * (this.metrics.get(`dimension:${dim}`)?.weight ?? 0.25);
      for (let j = 0; j < 3; j++) {
        const angle = hash01(`blob${index}${j}`) * TAU;
        const offset = R * (0.18 + 0.12 * j);
        const radius = R * (1 - 0.18 * j);
        if (!this.inView(stat.x, stat.y, radius + offset)) continue;
        g.globalAlpha = base * stat.w * (j === 0 ? 1 : 0.7) * settle;
        g.drawImage(
          sprite,
          stat.x + Math.cos(angle) * offset - radius,
          stat.y + Math.sin(angle) * offset - radius,
          2 * radius,
          2 * radius,
        );
      }
    }
    g.globalAlpha = 1;
  }

  // Each node's colour spreads around it, wider and stronger the more important its facts are.
  // The pearl's halo is made of all five colours, each shifted towards its dimension.
  private drawHalos(tf: Transform, settle: number) {
    const ctx = this.ctx;
    const k = this.cam.k;
    ctx.setTransform(tf.k, 0, 0, tf.k, tf.ox, tf.oy);
    ctx.globalCompositeOperation = 'multiply';
    for (const n of this.order) {
      if (n.vis <= 0.01) continue;
      const root = n.kind === 'root';
      let R: number;
      let a: number;
      if (root) {
        R = n.r * 1.6 * n.scale;
        a = 0.1;
      } else {
        const t = n.metrics.weight ** 1.3;
        R = n.r * n.scale * (1.25 + 2.5 * t);
        a = 0.04 + 0.5 * t;
      }
      a *= n.vis * n.emph;
      const bump = this.hubBump(n);
      if (bump > 0.01) {
        R *= 1 + 0.18 * bump;
        a = Math.min(1, a * (1 + 0.6 * bump) + 0.04 * bump);
      }
      a *= settle;
      if (a < 0.01 || !this.inView(n.rx, n.ry, R)) continue;
      if (R * k < 2 && n.id !== this.hoverId) continue;
      if (root) {
        const H = R * 1.25;
        ctx.globalAlpha = Math.min(1, a * 3.2);
        for (const slice of n.slices ?? []) {
          const x = n.rx + Math.cos(slice.angle) * H * 0.3;
          const y = n.ry + Math.sin(slice.angle) * H * 0.3;
          ctx.drawImage(this.sprite('halo', slice.rgb), x - H, y - H, 2 * H, 2 * H);
        }
      } else {
        ctx.globalAlpha = Math.min(1, a);
        ctx.drawImage(this.sprite('halo', n.rgb), n.rx - R, n.ry - R, 2 * R, 2 * R);
      }
    }
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
  }

  // Lines start as the entrance drew them and widen to their chat support as the map settles.
  // Lines on the hovered or selected path take their dimension's colour.
  private drawEdges(tf: Transform, settle: number) {
    const ctx = this.ctx;
    ctx.lineCap = 'butt';
    const opacity = lerp(EDGE_OPACITY, 0.8, settle);
    const from = { x: 0, y: 0, r: 0 };
    const to = { x: 0, y: 0, r: 0 };
    for (const e of this.edges) {
      const { c, p } = e;
      const a = Math.min(c.vis, p.vis) * Math.min(c.emph, p.emph);
      if (a <= 0.01 || !this.segmentInView(c.rx, c.ry, p.rx, p.ry)) continue;
      from.x = p.rx;
      from.y = p.ry;
      from.r = p.r * p.scale;
      to.x = c.rx;
      to.y = c.ry;
      to.r = c.r * c.scale;
      const hot = this.isHot(e) ? settle : 0;
      if (hot < 1) {
        strokeEdge(
          ctx,
          tf,
          from,
          to,
          edgeStroke(c.kind),
          lerp(edgeWidth(c.kind), e.w, settle),
          opacity * a * (1 - hot),
        );
      }
      if (hot > 0) strokeEdge(ctx, tf, from, to, c.color, Math.max(2, e.w + 0.25), 0.5 * a * hot);
    }
    // A removed node keeps its line to its parent, which may be on its way out as well.
    const leaving = new Map(this.ghosts.map((g) => [g.n.id, g]));
    for (const g of this.ghosts) {
      if (g.a <= 0.01 || !g.n.node.parentId) continue;
      const parentId = g.n.node.parentId;
      const p = this.nodes.get(parentId);
      const q = p ? null : leaving.get(parentId);
      if (p) {
        from.x = p.rx;
        from.y = p.ry;
        from.r = p.r * p.scale;
      } else if (q) {
        from.x = q.x;
        from.y = q.y;
        from.r = q.n.r * q.scale;
      } else continue;
      to.x = g.x;
      to.y = g.y;
      to.r = g.n.r * g.scale;
      const w = Math.round(supportWidth(g.n.metrics.support) * 4) / 4;
      strokeEdge(ctx, tf, from, to, edgeStroke(g.n.kind), lerp(edgeWidth(g.n.kind), w, settle), opacity * g.a);
    }
  }

  // A soft swell of ink enters each line at the child, runs along it and drains into the parent.
  // The lines of the hovered branch carry their own, faster and stronger waves.
  private drawWaves(tf: Transform, settle: number) {
    const ctx = this.ctx;
    const k = this.cam.k;
    ctx.setTransform(tf.k, 0, 0, tf.k, tf.ox, tf.oy);
    ctx.lineCap = 'round';
    ctx.globalAlpha = 1;
    const local = this.hoverSet;
    const localClock = local ? this.localClock() : -1;
    for (const e of this.edges) {
      const { c, p } = e;
      const isLocal = local !== null && local.has(c.id) && local.has(p.id);
      const progress = isLocal
        ? swellProgress(LOCAL_WAVE, this.localDepth, c.depth, localClock)
        : swellProgress(WAVE, this.waveDepth, c.depth, this.waveClock);
      if (progress < 0) continue;
      const vis = Math.min(c.vis, p.vis);
      if (vis < 0.1 || !this.segmentInView(c.rx, c.ry, p.rx, p.ry)) continue;
      const dx = p.rx - c.rx;
      const dy = p.ry - c.ry;
      const length = Math.hypot(dx, dy);
      if (length < 2) continue;
      const ux = dx / length;
      const uy = dy / length;
      const start = c.r * c.scale + 1;
      const end = length - (p.r * p.scale + 1);
      if ((end - start) * k < 6) continue;
      const span = swellSpan(progress, start, end);
      if (!span) continue;
      const hot = this.isHot(e);
      const emph = isLocal ? 1 : Math.min(c.emph, p.emph);
      const peak = Math.min(1, (0.26 + 0.46 * e.s) * (isLocal ? 1.6 : 1)) * emph * vis * settle;
      if (peak < 0.02) continue;
      const color = mix(c.rgb, BLACK, hot ? 0.22 : 0.06);
      const gradient = ctx.createLinearGradient(
        c.rx + ux * (span.head - span.tail),
        c.ry + uy * (span.head - span.tail),
        c.rx + ux * span.head,
        c.ry + uy * span.head,
      );
      gradient.addColorStop(0, rgbaOf(color, 0));
      gradient.addColorStop(0.55, rgbaOf(color, peak * 0.45));
      gradient.addColorStop(0.88, rgbaOf(color, peak));
      gradient.addColorStop(1, rgbaOf(color, 0));
      ctx.strokeStyle = gradient;
      ctx.lineWidth = Math.max(1.6 / k, e.w + 1.1 + 1.3 * e.s + (hot ? 1.2 : 0));
      ctx.beginPath();
      ctx.moveTo(c.rx + ux * span.from, c.ry + uy * span.from);
      ctx.lineTo(c.rx + ux * span.to, c.ry + uy * span.to);
      ctx.stroke();
    }
    ctx.lineCap = 'butt';
  }

  private drawNodes(tf: Transform, glow: number) {
    const ctx = this.ctx;
    for (const g of this.ghosts) {
      if (g.a <= 0.01 || !this.inView(g.x, g.y, g.n.r + 30)) continue;
      drawNode(ctx, tf, g.paint, g.x, g.y, g.scale, g.a, glow);
    }
    for (const n of this.order) {
      if (n.scale <= 0 || !this.inView(n.rx, n.ry, n.r * n.scale + 30)) continue;
      drawNode(ctx, tf, n, n.rx, n.ry, n.scale, n.vis * n.emph, glow);
    }
  }

  // Rings spread from a clicked node, and softly from the centre when a wave arrives. The hovered
  // node gets a thin ring of its own.
  private drawRings(tf: Transform) {
    const ctx = this.ctx;
    const k = this.cam.k;
    ctx.setTransform(tf.k, 0, 0, tf.k, tf.ox, tf.oy);
    for (const ring of this.rings) {
      const e = clamp((this.time - ring.start) / (ring.soft ? 1300 : 900));
      const n = ring.n;
      ctx.strokeStyle = n.color;
      for (const lag of ring.soft ? [0] : [0, 0.18]) {
        const ee = clamp(e - lag);
        if (ee <= 0 || ee >= 1) continue;
        ctx.globalAlpha = (1 - ee) ** 2 * 0.55 * (ring.soft ? 0.5 : 1);
        ctx.lineWidth = Math.max(1, ((ring.soft ? 1.6 : 2.4) * (1 - ee)) / Math.min(1, k));
        ctx.beginPath();
        ctx.arc(n.rx, n.ry, n.r + 4 + easeOutCubic(ee) * (ring.soft ? 34 + n.r * 0.4 : 70 + n.r), 0, TAU);
        ctx.stroke();
      }
    }
    const hovered = this.hoverId ? this.nodes.get(this.hoverId) : undefined;
    if (hovered && hovered.vis > 0.5) {
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = Math.max(1, 1.2 / k);
      ctx.strokeStyle = hovered.color;
      ctx.beginPath();
      ctx.arc(hovered.rx, hovered.ry, hovered.r * hovered.scale + 4.5 / Math.min(1.4, k), 0, TAU);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // The hovered and the selected node's labels go on top of the others.
  private drawLabels(tf: Transform) {
    const ctx = this.ctx;
    const attributeLabels = this.attributeLabelsShown();
    const last: MapNode[] = [];
    for (const n of this.order) {
      n.label = this.labelOf(n, attributeLabels);
      if (!n.label || n.vis < 0.2 || !this.inView(n.rx, n.ry + n.r + 20, n.labelWidth / 2 + 20)) continue;
      if (n.selected || n.id === this.hoverId) last.push(n);
      else drawLabel(ctx, tf, n, n.rx, n.ry, n.vis * n.emph);
    }
    for (const n of last) drawLabel(ctx, tf, n, n.rx, n.ry, n.vis * n.emph);
  }

  // ───────────────────────── pointer ─────────────────────────

  private local(event: MouseEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  /** The node under `p`: the nearest whose disc, or else whose label, contains it. Tiny discs count as 7 px wide. */
  private pick(p: Point): MapNode | null {
    const w = this.toWorld(p);
    const minR = 7 / this.cam.k;
    let best: MapNode | null = null;
    let bestScore = Infinity;
    for (const n of this.order) {
      if (n.enterT < 0.5) continue;
      const r = Math.max(n.r * n.scale, minR);
      const d = Math.hypot(w.x - n.rx, w.y - n.ry);
      let score = d <= r ? d / r : Infinity;
      if (score === Infinity && n.label) {
        const top = n.ry + n.r + LABEL_OFFSET_Y;
        if (Math.abs(w.x - n.rx) <= n.labelWidth / 2 + 2 && w.y >= top && w.y <= top + LABEL_LINE_HEIGHT) score = 1;
      }
      if (score < bestScore) {
        bestScore = score;
        best = n;
      }
    }
    return best;
  }

  private setHover(n: MapNode | null) {
    const id = n?.id ?? null;
    if (id === this.hoverId) return;
    this.hoverId = id;
    this.localStart = this.time;
    this.recomputeSets();
    this.hideTip();
    if (id) this.tipTimer = window.setTimeout(this.showTip, TOOLTIP_DELAY);
    this.requestDraw();
  }

  private readonly showTip = () => {
    const n = this.hoverId ? this.nodes.get(this.hoverId) : undefined;
    if (!n || this.press || this.destroyed) return;
    this.tipShown = true;
    const { x, y, k } = this.cam;
    this.callbacks.onHover({
      id: n.id,
      x: (n.rx - x) * k + this.W / 2,
      y: (n.ry - y) * k + this.H / 2,
      width: this.W - this.insets.right,
      height: this.H - this.insets.bottom,
    });
  };

  private hideTip() {
    window.clearTimeout(this.tipTimer);
    if (!this.tipShown) return;
    this.tipShown = false;
    this.callbacks.onHover(null);
  }

  private click(n: MapNode) {
    this.hideTip();
    // A newly selected node is ringed when the selection arrives; the selected one is ringed here.
    if (n.id === this.selectedId) this.pulse(n.id);
    this.callbacks.onNodeClick(n.id);
  }

  private readonly onDown = (event: PointerEvent) => {
    if (this.hidden || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const p = this.local(event);
    this.pointers.set(event.pointerId, p);
    try {
      this.canvas.setPointerCapture(event.pointerId);
    } catch {
      // The pointer is already gone.
    }
    if (this.pointers.size === 2) this.startPinch();
    else if (this.pointers.size === 1) {
      this.press = { id: event.pointerId, x: p.x, y: p.y, lx: p.x, ly: p.y, n: this.pick(p), moved: false };
    }
  };

  private readonly onMove = (event: PointerEvent) => {
    if (this.hidden) return;
    const p = this.local(event);
    if (this.pointers.has(event.pointerId)) this.pointers.set(event.pointerId, p);
    if (this.pinch) {
      this.movePinch();
      return;
    }
    const press = this.press;
    if (press && press.id === event.pointerId) {
      if (!press.moved && Math.hypot(p.x - press.x, p.y - press.y) > 4) {
        press.moved = true;
        this.userMoved = true;
        this.goal = null;
        this.setHover(null);
        if (press.n && this.nodes.get(press.n.id) === press.n) this.startDrag(press.n);
        this.canvas.style.cursor = 'grabbing';
      }
      if (press.moved) {
        // A dragged node keeps its place under the pointer: it moves by the pointer's steps.
        const dx = (p.x - press.lx) / this.cam.k;
        const dy = (p.y - press.ly) / this.cam.k;
        if (this.drag) {
          this.drag.fx = (this.drag.fx ?? this.drag.x) + dx;
          this.drag.fy = (this.drag.fy ?? this.drag.y) + dy;
        } else this.cam = { ...this.cam, x: this.cam.x - dx, y: this.cam.y - dy };
        press.lx = p.x;
        press.ly = p.y;
        this.requestDraw();
      }
      return;
    }
    // Touch has no hover.
    if (event.pointerType === 'touch' || this.pointers.size) return;
    const hit = this.pick(p);
    this.setHover(hit);
    this.canvas.style.cursor = hit ? 'pointer' : 'grab';
  };

  private readonly onUp = (event: PointerEvent) => this.release(event, true);
  private readonly onCancel = (event: PointerEvent) => this.release(event, false);

  private release(event: PointerEvent, click: boolean) {
    this.pointers.delete(event.pointerId);
    if (this.pinch) {
      // The remaining finger does nothing until it lifts too.
      if (this.pointers.size < 2) this.pinch = null;
      return;
    }
    const press = this.press;
    if (!press || press.id !== event.pointerId) return;
    this.press = null;
    this.canvas.style.cursor = 'grab';
    if (this.drag) this.endDrag();
    if (!click || press.moved) return;
    if (press.n && this.nodes.get(press.n.id) === press.n) this.click(press.n);
    else {
      this.restoreOverview();
      this.callbacks.onBackgroundClick();
    }
  }

  /** Goes back to the view from before nodes were brought into view: refitted if it was fitted. */
  private restoreOverview() {
    const overview = this.overview;
    if (!overview) return;
    this.overview = null;
    this.goal = overview.fitted
      ? { fit: true, until: this.time + 1400 }
      : { fixed: overview.view, until: this.time + 1400 };
    this.userMoved = !overview.fitted;
    this.requestDraw();
  }

  private readonly onLeave = () => {
    if (!this.press) this.setHover(null);
  };

  private startPinch() {
    if (this.drag) this.endDrag();
    this.press = null;
    this.setHover(null);
    const [a, b] = [...this.pointers.values()];
    this.pinch = {
      distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
      k: this.cam.k,
      anchor: this.toWorld({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }),
    };
    this.goal = null;
    this.userMoved = true;
  }

  // The map point first between the fingers stays between them while they spread or pinch.
  private movePinch() {
    const pinch = this.pinch!;
    const [a, b] = [...this.pointers.values()];
    if (!a || !b) return;
    const k = clamp((pinch.k * Math.hypot(a.x - b.x, a.y - b.y)) / pinch.distance, ZOOM_MIN, ZOOM_MAX);
    this.cam = {
      x: pinch.anchor.x - ((a.x + b.x) / 2 - this.W / 2) / k,
      y: pinch.anchor.y - ((a.y + b.y) / 2 - this.H / 2) / k,
      k,
    };
    this.requestDraw();
  }

  private readonly onWheel = (event: WheelEvent) => {
    event.preventDefault();
    if (this.hidden) return;
    const p = this.local(event);
    const before = this.toWorld(p);
    const k = clamp(this.cam.k * Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0016)), ZOOM_MIN, ZOOM_MAX);
    this.cam = { x: before.x - (p.x - this.W / 2) / k, y: before.y - (p.y - this.H / 2) / k, k };
    this.goal = null;
    this.userMoved = true;
    this.hideTip();
    this.requestDraw();
  };

  private readonly onVisibility = () => {
    if (document.hidden) this.stop();
    else this.start();
  };
}
