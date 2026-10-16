import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CanvasEvent,
  DragElementForce,
  ExtensionCategory,
  Graph,
  NodeEvent,
  register,
  type BaseEdge,
  type BaseNode,
  type EdgeData,
  type IElementDragEvent,
  type NodeData,
} from '@antv/g6';
import type { ProfileModel, ProfileNode, ReviewView } from '../data';
import { ROOT_COLOR, dimensionColor, visibleNodes } from '../data';
import { useProfileStore } from '../store';
import { NodePreview } from './NodePreview';
import {
  DENSE_NODE_COUNT,
  buildGraphData,
  clusterCenters,
  computeSeeds,
  edgeSelection,
  expansionPositions,
  nodeSelection,
  overviewPositions,
  sameTopology,
  shiftClusters,
  type EdgeDatumData,
  type NodeDatumData,
  type Point,
  type Positions,
  type SelectionPath,
} from './graphLayout';

interface Props {
  graphRef: React.MutableRefObject<Graph | null>;
}

const size = (kind: ProfileNode['kind']) => ({ root: 76, dimension: 54, topic: 31, subtopic: 24, attribute: 12 })[kind];
const labelFontSize = (kind: ProfileNode['kind']) =>
  kind === 'root' ? 16 : kind === 'dimension' ? 14 : kind === 'attribute' ? 11 : 12;

// Every graph datum carries its own display data, so style callbacks never look anything up.
const nodeData = (datum: NodeData) => (datum.data ?? {}) as Partial<NodeDatumData>;
const edgeData = (datum: EdgeData) => (datum.data ?? {}) as Partial<EdgeDatumData>;
const accentColor = (data: Partial<NodeDatumData | EdgeDatumData>, rootColor: string) =>
  data.kind === 'root' ? rootColor : dimensionColor(data.dim);

// G6's force drag lets the simulation cool at the layout's own rate, which above 200 nodes is
// fast enough to stop the simulation while nodes are still moving. This variant cools it slowly
// during and after the drag so the graph eases to rest after release.
const DRAG_ALPHA_DECAY = 0.06;
type ForceLayoutAdapter = { instance?: { simulation?: { alphaDecay(decay: number): unknown } } };
class SettlingDragElementForce extends DragElementForce {
  protected onDragStart(event: IElementDragEvent) {
    super.onDragStart(event);
    if (!this.enable) return;
    for (const layout of this.context.layout?.getLayoutInstance() ?? []) {
      (layout as ForceLayoutAdapter | undefined)?.instance?.simulation?.alphaDecay(DRAG_ALPHA_DECAY);
    }
  }
}
register(ExtensionCategory.BEHAVIOR, 'drag-element-settle', SettlingDragElementForce);

// How strongly each node is pulled to its target: the root stays put, dimension nodes hold
// their position from `clusterCenters`, and everything else is kept near its dimension node.
const CLUSTER_PULL = { root: 1, dimension: 0.2, topic: 0.05, subtopic: 0.03, attribute: 0.03 };

function forceLayout(model: ProfileModel, nodeCount: number, root: Point, centers: Positions) {
  const dimensionIdOf = new Map(model.dimensionIds.map((id) => [model.nodes.get(id)!.dim, id]));
  const targets = new Map(
    [...model.nodes.values()].map((node) => [
      node.id,
      (node.kind === 'root' ? root : centers.get(dimensionIdOf.get(node.dim) ?? '')) ?? root,
    ]),
  );
  const target = (datum: NodeData) => targets.get(String(datum.id)) ?? root;
  const pull = (datum: NodeData) => CLUSTER_PULL[model.nodes.get(String(datum.id))?.kind ?? 'attribute'];
  const dense = nodeCount > DENSE_NODE_COUNT;
  const busy = nodeCount > 200;
  const measure = document.createElement('canvas').getContext('2d');
  const collisionRadii = new Map(
    [...model.nodes.values()].map((node) => {
      const nodeRadius = size(node.kind) / 2;
      if (dense && node.kind === 'attribute') return [node.id, nodeRadius + 8];
      const fontSize = labelFontSize(node.kind);
      if (measure) measure.font = `600 ${fontSize}px "Noto Sans", "Noto Sans CJK SC", sans-serif`;
      const labelWidth = measure?.measureText(node.name).width ?? node.name.length * fontSize;
      return [node.id, Math.max(nodeRadius + 8, Math.hypot(labelWidth / 2 + 4, nodeRadius + 11 + fontSize))];
    }),
  );
  return {
    type: 'd3-force' as const,
    // The root is held by the x/y forces below, so the graph needs no centering force.
    center: false as const,
    x: { x: (datum: NodeData) => target(datum).x, strength: pull },
    y: { y: (datum: NodeData) => target(datum).y, strength: pull },
    preventOverlap: true,
    collide: {
      radius: (datum: NodeData) => collisionRadii.get(String(datum.id)) ?? 20,
      strength: 1,
      iterations: busy ? 1 : 2,
    },
    // d3 resolves link ends to node objects before it reads the distances.
    linkDistance: (datum: { target: { id: unknown } }) => {
      const child = model.nodes.get(String(datum.target.id));
      const center = child?.kind === 'dimension' ? centers.get(child.id) : undefined;
      return center
        ? Math.hypot(center.x - root.x, center.y - root.y)
        : child?.kind === 'attribute'
          ? 110
          : child?.kind === 'subtopic'
            ? 150
            : 220;
    },
    nodeStrength: (datum: NodeData) => {
      const kind = model.nodes.get(String(datum.id))?.kind;
      return kind === 'root' ? -260 : kind === 'dimension' ? -200 : kind === 'topic' ? -90 : -40;
    },
    edgeStrength: 0.75,
    distanceMax: 700,
    alphaDecay: dense ? 0.65 : busy ? 0.4 : 0.06,
  };
}

function selectionPath(model: ProfileModel): SelectionPath {
  const selectedId = useProfileStore.getState().selectedId;
  const path = new Set<string>();
  let node = selectedId && selectedId !== 'root' ? model.nodes.get(selectedId) : undefined;
  while (node) {
    path.add(node.id);
    node = node.parentId ? model.nodes.get(node.parentId) : undefined;
  }
  return { selectedId, path };
}

function syncSelection(graph: Graph, model: ProfileModel): Promise<void> {
  const selection = selectionPath(model);
  const states = Object.fromEntries([
    ...graph.getNodeData().map((node) => [String(node.id), nodeSelection(String(node.id), selection)]),
    ...graph
      .getEdgeData()
      .map((edge) => [String(edge.id), edgeSelection(String(edge.source), String(edge.target), selection)]),
  ]);
  return graph.setElementState(states, false);
}

function playEntrance(graph: Graph, model: ProfileModel): () => void {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return () => {};
  const nodes = graph.getNodeData();
  const byId = new Map(nodes.map((node) => [String(node.id), node]));
  const depth = (id: string): number => {
    const parentId = model.nodes.get(id)?.parentId;
    return parentId && byId.has(parentId) ? depth(parentId) + 1 : 0;
  };
  nodes.sort((a, b) => depth(String(a.id)) - depth(String(b.id)));
  const stagger = Math.min(85, 1380 / Math.max(1, nodes.length - 1));
  const scene = graph.getCanvas().document;
  const cleanups: Array<() => void> = [];
  const track = (animation: ReturnType<BaseNode['animate']>, restoreStyle: () => void) => {
    if (!animation) return;
    let finished = false;
    const restore = () => {
      if (finished) return;
      finished = true;
      animation.finish();
      animation.cancel();
      if (!graph.destroyed) restoreStyle();
    };
    cleanups.push(restore);
    void animation.finished.then(restore);
  };
  nodes.forEach((node, index) => {
    const element = scene.getElementById(String(node.id)) as BaseNode | null;
    if (!element) return;
    const x = Number(node.style?.x ?? 0);
    const y = Number(node.style?.y ?? 0);
    const parentId = model.nodes.get(String(node.id))?.parentId;
    const parent = parentId ? byId.get(parentId) : undefined;
    const fromX = x + (Number(parent?.style?.x ?? x) - x) * 0.16;
    const fromY = y + (Number(parent?.style?.y ?? y) - y) * 0.16;
    const original = {
      opacity: Number(element.attributes.opacity ?? 1),
      transform: element.attributes.transform,
    };
    const animation = element.animate(
      [
        { opacity: 0, transform: `translate(${fromX}px, ${fromY}px) scale(0.35)` },
        { opacity: original.opacity, transform: `translate(${x}px, ${y}px) scale(1.08)`, offset: 0.72 },
        { opacity: original.opacity, transform: `translate(${x}px, ${y}px) scale(1)` },
      ],
      { duration: 620, delay: index * stagger, easing: 'ease-out', fill: 'both' },
    );
    track(animation, () => element.update(original));
    const edge = scene.getElementById(`edge:${node.id}`) as BaseEdge | null;
    if (edge) {
      const originalEdge = { opacity: Number(edge.attributes.opacity ?? 0.75), transform: edge.attributes.transform };
      const edgeAnimation = edge.animate([{ opacity: 0 }, { opacity: originalEdge.opacity }], {
        duration: 450,
        delay: Math.max(0, index * stagger - 100),
        easing: 'ease-out',
        fill: 'both',
      });
      track(edgeAnimation, () => edge.update(originalEdge));
    }
  });
  return () => cleanups.forEach((restore) => restore());
}

export function GraphCanvas({ graphRef }: Props) {
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const view = useProfileStore((state) => state.view);
  const containerRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [hovered, setHovered] = useState<{ id: string; x: number; y: number; width: number; height: number } | null>(
    null,
  );
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearHover = () => {
    if (hoverTimerRef.current !== null) clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = null;
    setHovered(null);
  };
  const expandedIds = useProfileStore((state) => state.expandedIds);
  const selectedId = useProfileStore((state) => state.selectedId);
  const visible = useMemo(() => visibleNodes(viewModel, expandedIds), [viewModel, expandedIds]);
  const initialPositions = useMemo(() => overviewPositions(viewModel), [viewModel]);
  const renderQueueRef = useRef<Promise<void>>(Promise.resolve());
  const renderVersionRef = useRef(0);
  const readyRef = useRef(false);
  const fittedRef = useRef(false);
  const stopEntranceRef = useRef<(() => void) | null>(null);
  const displayedCountRef = useRef(visible.length);

  const layoutViewRef = useRef(view);
  const viewPositionsRef = useRef(new Map<ReviewView, Map<string, { x: number; y: number }>>());
  const viewModelRef = useRef(viewModel);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    fittedRef.current = false;
    setReady(false);
    const graph = new Graph({
      container,
      width: container.clientWidth,
      height: container.clientHeight,
      data: { nodes: [], edges: [] },
      zoomRange: [0.28, 2.2],
      animation: { duration: 420, easing: 'ease-in-out' },
      layout: forceLayout(viewModel, visible.length, initialPositions.get('root')!, new Map()),
      behaviors: ['drag-canvas', 'zoom-canvas', { type: 'drag-element-settle', state: 'multi-selected' }],
      node: {
        type: 'circle',
        style: {
          opacity: 1,
          size: (datum) => size(nodeData(datum).kind ?? 'attribute'),
          fill: (datum) => {
            const { kind, dim } = nodeData(datum);
            if (!kind || kind === 'root') return ROOT_COLOR;
            if (kind === 'attribute') return '#ffffff';
            return dimensionColor(dim);
          },
          stroke: (datum) => accentColor(nodeData(datum), '#707986'),
          lineWidth: (datum) => (nodeData(datum).kind === 'attribute' ? 2.5 : 1.5),
          shadowColor: (datum) => accentColor(nodeData(datum), '#a3aab4'),
          haloStroke: (datum) => accentColor(nodeData(datum), '#a3aab4'),
          shadowBlur: (datum) => {
            const { kind } = nodeData(datum);
            return kind === 'root' ? 16 : kind === 'dimension' ? 24 : kind === 'attribute' ? 0 : 14;
          },
          labelText: (datum) => {
            const { kind, name } = nodeData(datum);
            return displayedCountRef.current > DENSE_NODE_COUNT && kind === 'attribute' ? '' : (name ?? '');
          },
          labelPlacement: 'bottom',
          iconText: (datum) => (nodeData(datum).kind === 'root' ? '✦' : ''),
          iconFill: '#f3f1ec',
          iconFontSize: 28,
          labelOffsetY: 11,
          labelFill: (datum) => (nodeData(datum).kind === 'root' ? '#4c5561' : '#34474b'),
          labelFontSize: (datum) => labelFontSize(nodeData(datum).kind ?? 'attribute'),
          labelFontWeight: (datum) => {
            const { kind } = nodeData(datum);
            return kind === 'root' || kind === 'dimension' ? 650 : 520;
          },
        },
        state: {
          selected: {
            opacity: 1,
            lineWidth: 2,
            stroke: '#ffffff',
            halo: true,
            haloLineWidth: 14,
            haloStrokeOpacity: 0.18,
            shadowBlur: 20,
            labelFontWeight: 700,
          },
        },
      },
      edge: {
        type: 'line',
        style: {
          stroke: (datum) => (edgeData(datum).kind === 'dimension' ? '#b8c8c6' : '#cad4d2'),
          lineWidth: (datum) => (edgeData(datum).kind === 'dimension' ? 2 : 1.2),
          opacity: 0.75,
        },
        state: {
          route: {
            stroke: (datum: EdgeData) => dimensionColor(edgeData(datum).dim),
            lineWidth: 2,
            opacity: 0.95,
          },
        },
      },
    });
    graphRef.current = graph;
    const stopEntrance = () => {
      clearHover();
      stopEntranceRef.current?.();
      stopEntranceRef.current = null;
    };
    container.addEventListener('pointerdown', stopEntrance, true);
    container.addEventListener('wheel', stopEntrance, { passive: true });

    graph.on(NodeEvent.CLICK, (event) => {
      if (!('target' in event) || !event.target || !('id' in event.target)) return;
      const id = String(event.target.id);
      const node = viewModelRef.current.nodes.get(id);
      if (!node) return;
      useProfileStore.getState().toggleExpanded(id);
    });

    graph.on(CanvasEvent.CLICK, () => useProfileStore.getState().clearSelection());

    graph.on(NodeEvent.POINTER_ENTER, (event) => {
      clearHover();
      if (!('target' in event) || !event.target || !('id' in event.target)) return;
      const id = String(event.target.id);
      const node = viewModelRef.current.nodes.get(id);
      if (!node) return;
      hoverTimerRef.current = setTimeout(() => {
        hoverTimerRef.current = null;
        const [clientX, clientY] = graph.getClientByCanvas(graph.getElementPosition(id));
        const rect = container.getBoundingClientRect();
        setHovered({ id, x: clientX - rect.left, y: clientY - rect.top, width: rect.width, height: rect.height });
      }, 250);
    });
    graph.on(NodeEvent.POINTER_LEAVE, clearHover);

    const resize = new ResizeObserver(() => {
      clearHover();
      if (container.clientWidth && container.clientHeight) graph.setSize(container.clientWidth, container.clientHeight);
    });
    resize.observe(container);
    return () => {
      container.removeEventListener('pointerdown', stopEntrance, true);
      container.removeEventListener('wheel', stopEntrance);
      stopEntrance();
      resize.disconnect();
      readyRef.current = false;
      renderVersionRef.current += 1;
      graphRef.current = null;
      graph.destroy();
    };
    // Intentional: the component is remounted via key, so the initial model/visible closure is never stale.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphRef]);

  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    const version = ++renderVersionRef.current;
    clearHover();
    stopEntranceRef.current?.();
    stopEntranceRef.current = null;
    const showProgress = fittedRef.current;
    if (showProgress) setUpdating(true);
    renderQueueRef.current = renderQueueRef.current
      .then(async () => {
        if (graph.destroyed || version !== renderVersionRef.current) return;
        if (showProgress)
          await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
        if (graph.destroyed || version !== renderVersionRef.current) return;
        if (fittedRef.current) graph.stopLayout();
        const previous = new Map<string, { x: number; y: number }>();
        for (const node of graph.getNodeData()) {
          if (typeof node.style?.x === 'number' && typeof node.style?.y === 'number') {
            previous.set(String(node.id), { x: node.style.x, y: node.style.y });
          }
        }
        const switchingView = layoutViewRef.current !== view;
        if (fittedRef.current) viewPositionsRef.current.set(layoutViewRef.current, previous);
        const restored = switchingView ? viewPositionsRef.current.get(view) : undefined;
        const root = previous.get('root') ?? initialPositions.get('root')!;
        const centers = clusterCenters(viewModel, visible, root);
        const seeds = shiftClusters(
          viewModel,
          computeSeeds(switchingView, previous, initialPositions, restored),
          centers,
        );
        const anchors = fittedRef.current ? expansionPositions(viewModel, visible, seeds) : undefined;
        const initial = anchors ?? shiftClusters(viewModel, initialPositions, centers);
        const { nodes, edges } = buildGraphData(visible, seeds, initial, selectionPath(viewModel));
        setHovered(null);
        displayedCountRef.current = visible.length;
        const keepLayout = sameTopology(switchingView, fittedRef.current, previous, nodes, graph.getEdgeData(), edges);
        if (!keepLayout) graph.setLayout(forceLayout(viewModel, visible.length, root, centers));
        viewModelRef.current = viewModel;
        layoutViewRef.current = view;
        graph.setData({ nodes, edges });
        if (keepLayout) await graph.draw();
        else await graph.render();
        if (graph.destroyed || version !== renderVersionRef.current) return;
        const firstRender = !fittedRef.current;
        if (firstRender) {
          await graph.fitView(undefined, false);
          if (graph.destroyed || version !== renderVersionRef.current) return;
          fittedRef.current = true;
        }
        readyRef.current = true;
        await syncSelection(graph, viewModel);
        if (graph.destroyed || version !== renderVersionRef.current) return;
        if (firstRender) {
          stopEntranceRef.current = playEntrance(graph, viewModel);
          setReady(true);
        }
        if (version === renderVersionRef.current) setUpdating(false);
      })
      .catch((error: unknown) => {
        if (!graph.destroyed) {
          setUpdating(false);
          console.error('Profile graph render failed', error);
        }
      });
  }, [visible, graphRef, initialPositions, view, viewModel]);

  useEffect(() => {
    const graph = graphRef.current;
    if (!graph || !readyRef.current || viewModelRef.current !== viewModel) return;
    stopEntranceRef.current?.();
    stopEntranceRef.current = null;
    void syncSelection(graph, viewModel).catch((error: unknown) => {
      if (!graph.destroyed) console.error('Profile graph selection failed', error);
    });
  }, [selectedId, graphRef, viewModel]);

  return (
    <>
      <div
        ref={containerRef}
        className={`graph-canvas${ready ? ' is-ready' : ''}`}
        role="img"
        aria-label="可交互的画像知识图谱"
      />
      {hovered && viewModel.nodes.has(hovered.id) && (
        <NodePreview
          viewModel={viewModel}
          node={viewModel.nodes.get(hovered.id)!}
          x={hovered.x}
          y={hovered.y}
          width={hovered.width}
          height={hovered.height}
        />
      )}
      {!ready && (
        <div className="canvas-loading" role="status">
          <span className="loading-mark">✳</span>
          <p>正在展开画像地图</p>
        </div>
      )}
      {ready && updating && (
        <div className="canvas-updating" role="status">
          <span>✳</span>正在整理图谱
        </div>
      )}
    </>
  );
}
