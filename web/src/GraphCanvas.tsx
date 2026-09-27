import { useEffect, useMemo, useRef, useState } from 'react';
import { CanvasEvent, Graph, NodeEvent, type BaseEdge, type BaseNode, type EdgeData, type NodeData } from '@antv/g6';
import type { ProfileModel, ProfileNode, ReviewView } from './data';
import { DIMENSION_COLORS, visibleNodes } from './data';
import { useGraphStore } from './store';
import { NodePreview } from './NodePreview';

interface Props {
  model: ProfileModel;
  view: ReviewView;
  graphRef: React.MutableRefObject<Graph | null>;
}

const size = (kind: ProfileNode['kind']) => ({ root: 76, dimension: 54, topic: 31, subtopic: 24, attribute: 12 })[kind];
const labelFontSize = (kind: ProfileNode['kind']) => kind === 'root' ? 16 : kind === 'dimension' ? 14 : kind === 'attribute' ? 11 : 12;

function overviewPositions(model: ProfileModel): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>([['root', { x: 0, y: 0 }]]);
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
  const dimensionCount = model.dimensionIds.length;
  model.dimensionIds.forEach((id, dimensionIndex) => {
    const angle = Math.PI - (dimensionIndex * 2 * Math.PI) / dimensionCount;
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

function expansionPositions(model: ProfileModel, visible: ProfileNode[], previous: Map<string, { x: number; y: number }>) {
  const positions = new Map(previous);
  const root = previous.get('root')!;
  const halfWidth = Math.PI / model.dimensionIds.length * 0.9;
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

function forceLayout(model: ProfileModel, nodeCount: number, anchors?: Map<string, { x: number; y: number }>) {
  const dense = nodeCount > 600;
  const busy = nodeCount > 200;
  const measure = document.createElement('canvas').getContext('2d');
  const collisionRadii = new Map([...model.nodes.values()].map((node) => {
    const nodeRadius = size(node.kind) / 2;
    if (dense && node.kind === 'attribute') return [node.id, nodeRadius + 8];
    const fontSize = labelFontSize(node.kind);
    if (measure) measure.font = `600 ${fontSize}px "Noto Sans", "Noto Sans CJK SC", sans-serif`;
    const labelWidth = measure?.measureText(node.name).width ?? node.name.length * fontSize;
    return [node.id, Math.max(nodeRadius + 8, Math.hypot(labelWidth / 2 + 4, nodeRadius + 11 + fontSize))];
  }));
  return {
    type: 'd3-force' as const,
    ...(anchors ? { center: false as const } : {}),
    preventOverlap: true,
    collide: {
      radius: (datum: NodeData) => collisionRadii.get(String(datum.id)) ?? 20,
      strength: 1,
      iterations: busy ? 1 : 2,
    },
    linkDistance: (datum: { target: unknown }) => {
      const child = model.nodes.get(String(datum.target));
      return child?.kind === 'dimension' ? 300 : child?.kind === 'attribute' ? 110 : child?.kind === 'subtopic' ? 150 : 220;
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

function selectionPath(model: ProfileModel) {
  const selectedId = useGraphStore.getState().selectedId;
  const path = new Set<string>();
  let node = selectedId && selectedId !== 'root' ? model.nodes.get(selectedId) : undefined;
  while (node) {
    path.add(node.id);
    node = node.parentId ? model.nodes.get(node.parentId) : undefined;
  }
  return { selectedId, path };
}

function nodeSelection(id: string, { selectedId }: ReturnType<typeof selectionPath>): string[] {
  return id === selectedId ? ['selected'] : [];
}

function edgeSelection(source: string, target: string, { path }: ReturnType<typeof selectionPath>): string[] {
  return path.has(source) && path.has(target) ? ['route'] : [];
}

function syncSelection(graph: Graph, model: ProfileModel): Promise<void> {
  const selection = selectionPath(model);
  const states = Object.fromEntries([
    ...graph.getNodeData().map((node) => [String(node.id), nodeSelection(String(node.id), selection)]),
    ...graph.getEdgeData().map((edge) => [String(edge.id), edgeSelection(String(edge.source), String(edge.target), selection)]),
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
    const animation = element.animate([
      { opacity: 0, transform: `translate(${fromX}px, ${fromY}px) scale(0.35)` },
      { opacity: original.opacity, transform: `translate(${x}px, ${y}px) scale(1.08)`, offset: 0.72 },
      { opacity: original.opacity, transform: `translate(${x}px, ${y}px) scale(1)` },
    ], { duration: 620, delay: index * stagger, easing: 'ease-out', fill: 'both' });
    track(animation, () => element.update(original));
    const edge = scene.getElementById(`edge:${node.id}`) as BaseEdge | null;
    if (edge) {
      const originalEdge = { opacity: Number(edge.attributes.opacity ?? 0.75), transform: edge.attributes.transform };
      const edgeAnimation = edge.animate([{ opacity: 0 }, { opacity: originalEdge.opacity }], {
        duration: 450, delay: Math.max(0, index * stagger - 100), easing: 'ease-out', fill: 'both',
      });
      track(edgeAnimation, () => edge.update(originalEdge));
    }
  });
  return () => cleanups.forEach((restore) => restore());
}

export function GraphCanvas({ model, view, graphRef }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [hovered, setHovered] = useState<{ id: string; x: number; y: number; width: number; height: number } | null>(null);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearHover = () => {
    if (hoverTimerRef.current !== null) clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = null;
    setHovered(null);
  };
  const expandedIds = useGraphStore((state) => state.expandedIds);
  const selectedId = useGraphStore((state) => state.selectedId);
  const visible = useMemo(() => visibleNodes(model, expandedIds), [model, expandedIds]);
  const initialPositions = useMemo(() => overviewPositions(model), [model]);
  const renderQueueRef = useRef<Promise<void>>(Promise.resolve());
  const renderVersionRef = useRef(0);
  const readyRef = useRef(false);
  const fittedRef = useRef(false);
  const stopEntranceRef = useRef<(() => void) | null>(null);
  const displayedCountRef = useRef(visible.length);

  const layoutViewRef = useRef(view);
  const viewPositionsRef = useRef(new Map<ReviewView, Map<string, { x: number; y: number }>>());
  const modelRef = useRef(model);
  const styledNodesRef = useRef(model.nodes);

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
      layout: forceLayout(model, visible.length),
      behaviors: ['drag-canvas', 'zoom-canvas', { type: 'drag-element-force', state: 'multi-selected' }],
      node: {
        type: 'circle',
        style: {
          opacity: 1,
          size: (datum) => size(styledNodesRef.current.get(String(datum.id))?.kind ?? 'attribute'),
          fill: (datum) => {
            const node = styledNodesRef.current.get(String(datum.id));
            if (!node || node.kind === 'root') return '#59616d';
            if (node.kind === 'attribute') return '#ffffff';
            return DIMENSION_COLORS[node.dim ?? 1];
          },
          stroke: (datum) => {
            const node = styledNodesRef.current.get(String(datum.id));
            return node?.kind === 'root' ? '#707986' : DIMENSION_COLORS[node?.dim ?? 1];
          },
          lineWidth: (datum) => styledNodesRef.current.get(String(datum.id))?.kind === 'attribute' ? 2.5 : 1.5,
          shadowColor: (datum) => {
            const node = styledNodesRef.current.get(String(datum.id));
            return node?.kind === 'root' ? '#a3aab4' : DIMENSION_COLORS[node?.dim ?? 1];
          },
          haloStroke: (datum) => {
            const node = styledNodesRef.current.get(String(datum.id));
            return node?.kind === 'root' ? '#a3aab4' : DIMENSION_COLORS[node?.dim ?? 1];
          },
          shadowBlur: (datum) => {
            const kind = styledNodesRef.current.get(String(datum.id))?.kind;
            return kind === 'root' ? 16 : kind === 'dimension' ? 24 : kind === 'attribute' ? 0 : 14;
          },
          labelText: (datum) => {
            const node = styledNodesRef.current.get(String(datum.id));
            return displayedCountRef.current > 600 && node?.kind === 'attribute' ? '' : node?.name ?? '';
          },
          labelPlacement: 'bottom',
          iconText: (datum) => styledNodesRef.current.get(String(datum.id))?.kind === 'root' ? '✦' : '',
          iconFill: '#f3f1ec',
          iconFontSize: 28,
          labelOffsetY: 11,
          labelFill: (datum) => styledNodesRef.current.get(String(datum.id))?.kind === 'root' ? '#4c5561' : '#34474b',
          labelFontSize: (datum) => labelFontSize(styledNodesRef.current.get(String(datum.id))?.kind ?? 'attribute'),
          labelFontWeight: (datum) => {
            const kind = styledNodesRef.current.get(String(datum.id))?.kind;
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
          stroke: (datum) => {
            const child = styledNodesRef.current.get(String(datum.target));
            return child?.kind === 'dimension' ? '#b8c8c6' : '#cad4d2';
          },
          lineWidth: (datum) => styledNodesRef.current.get(String(datum.target))?.kind === 'dimension' ? 2 : 1.2,
          opacity: 0.75,
        },
        state: {
          route: {
            stroke: (datum: EdgeData) => DIMENSION_COLORS[styledNodesRef.current.get(String(datum.target))?.dim ?? 1],
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
      const node = modelRef.current.nodes.get(id);
      if (!node) return;
      const state = useGraphStore.getState();
      state.toggleExpanded(modelRef.current, id);
    });

    graph.on(CanvasEvent.CLICK, () => useGraphStore.getState().clearSelection());

    graph.on(NodeEvent.POINTER_ENTER, (event) => {
      clearHover();
      if (!('target' in event) || !event.target || !('id' in event.target)) return;
      const id = String(event.target.id);
      const node = modelRef.current.nodes.get(id);
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
    renderQueueRef.current = renderQueueRef.current.then(async () => {
      if (graph.destroyed || version !== renderVersionRef.current) return;
      if (showProgress) await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
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
      const seeds = switchingView ? new Map([...initialPositions, ...(restored ?? [])]) : previous;
      if (switchingView) {
        const currentRoot = previous.get('root')!;
        const targetRoot = seeds.get('root')!;
        const offset = { x: currentRoot.x - targetRoot.x, y: currentRoot.y - targetRoot.y };
        for (const [id, position] of seeds) {
          seeds.set(id, { x: position.x + offset.x, y: position.y + offset.y });
        }
      }
      const anchors = fittedRef.current ? expansionPositions(model, visible, seeds) : undefined;
      const initial = anchors ?? initialPositions;
      const nextPositions = new Map<string, { x: number; y: number }>();
      const displayed = new Set(visible.map((node) => node.id));
      const selection = selectionPath(model);
      const nodes = visible.map((node, index) => {
        const prior = seeds.get(node.id);
        const parent = node.parentId ? nextPositions.get(node.parentId) : undefined;
        const initialPosition = initial.get(node.id);
        const angle = index * 2.39996;
        const distance = node.kind === 'dimension' ? 210 : node.kind === 'attribute' ? 90 : 135;
        const position = prior ?? initialPosition ?? (parent ? {
          x: parent.x + Math.cos(angle) * distance,
          y: parent.y + Math.sin(angle) * distance,
        } : { x: 0, y: 0 });
        nextPositions.set(node.id, position);
        return {
          id: node.id,
          states: nodeSelection(node.id, selection),
          data: { kind: node.kind, dim: node.dim },
          style: position,
        };
      });
      const edges = visible.filter((node) => node.parentId && displayed.has(node.parentId)).map((node) => ({
        id: `edge:${node.id}`,
        source: node.parentId!,
        target: node.id,
        states: edgeSelection(node.parentId!, node.id, selection),
      }));
      setHovered(null);
      displayedCountRef.current = visible.length;
      const sameTopology = !switchingView && fittedRef.current && previous.size === nodes.length
        && nodes.every((node) => previous.has(node.id))
        && graph.getEdgeData().length === edges.length
        && graph.getEdgeData().every((edge) => edges.some((next) => next.id === edge.id && next.source === edge.source));
      if (!sameTopology) graph.setLayout(forceLayout(model, visible.length, anchors));
      styledNodesRef.current = new Map([...styledNodesRef.current, ...model.nodes]);
      modelRef.current = model;
      layoutViewRef.current = view;
      graph.setData({ nodes, edges });
      if (sameTopology) await graph.draw();
      else await graph.render();
      styledNodesRef.current = model.nodes;
      if (graph.destroyed || version !== renderVersionRef.current) return;
      const firstRender = !fittedRef.current;
      if (firstRender) {
        await graph.fitView(undefined, false);
        if (graph.destroyed || version !== renderVersionRef.current) return;
        fittedRef.current = true;
      }
      readyRef.current = true;
      await syncSelection(graph, model);
      if (graph.destroyed || version !== renderVersionRef.current) return;
      if (firstRender) {
        stopEntranceRef.current = playEntrance(graph, model);
        setReady(true);
      }
      if (version === renderVersionRef.current) setUpdating(false);
    }).catch((error: unknown) => {
      if (!graph.destroyed) {
        setUpdating(false);
        console.error('Profile graph render failed', error);
      }
    });
  }, [visible, graphRef, initialPositions, view]);

  useEffect(() => {
    const graph = graphRef.current;
    if (!graph || !readyRef.current || modelRef.current !== model) return;
    stopEntranceRef.current?.();
    stopEntranceRef.current = null;
    void syncSelection(graph, model).catch((error: unknown) => {
      if (!graph.destroyed) console.error('Profile graph selection failed', error);
    });
  }, [selectedId, graphRef, model]);

  return <>
    <div ref={containerRef} className={`graph-canvas${ready ? ' is-ready' : ''}`} role="img" aria-label="可交互的画像知识图谱" />
    {hovered && model.nodes.has(hovered.id) && <NodePreview model={model} node={model.nodes.get(hovered.id)!} x={hovered.x} y={hovered.y} width={hovered.width} height={hovered.height} />}
    {!ready && <div className="canvas-loading" role="status"><span>✳</span>正在展开画像地图</div>}
    {ready && updating && <div className="canvas-updating" role="status"><span>✳</span>正在整理图谱</div>}
  </>;
}
