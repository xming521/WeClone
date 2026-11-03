import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { useTranslation } from 'react-i18next';
import { visibleNodes } from '../data';
import { useProfileStore } from '../store';
import { useTheme } from '../theme';
import { AtlasEngine, type AtlasControls, type HoverInfo, type Insets } from './atlasEngine';
import { nodeMetrics } from './atlasMetrics';
import { playEntrance, type EntranceConfig } from './entrance';
import type { Point } from './graphLayout';
import { openNightIntro, type NightIntro } from './nightIntro';
import { NodePreview } from './NodePreview';
import { nodeLooks } from './nodeLook';

interface Props {
  atlasRef: RefObject<AtlasControls | null>;
}

/** Room the open detail panel takes from the map: a column on the right, or a sheet at the bottom on phones. */
function panelInsets(host: HTMLElement, open: boolean): Insets {
  const panel = open ? host.closest('.app-shell')?.querySelector<HTMLElement>('.detail-panel') : null;
  if (!panel) return { right: 0, bottom: 0 };
  // The panel slides in by a transform, which the offsets leave out: they give its place once open.
  if (panel.offsetWidth >= host.clientWidth * 0.6) {
    return { right: 0, bottom: Math.max(0, host.clientHeight - panel.offsetTop) };
  }
  return { right: Math.max(0, host.clientWidth - panel.offsetLeft), bottom: 0 };
}

/** Ends the entrance if it still plays; the map shows at once. */
function stopEntrance(ref: RefObject<(() => void) | null>) {
  ref.current?.();
  ref.current = null;
}

/**
 * A reset or the other grouping mounts a new map. The old one's last picture stays a moment and sinks into its root
 * under the new map, whose entrance grows out of the same root, instead of leaving a blank page.
 */
function fadeOutMap(main: HTMLElement | null, canvases: HTMLCanvasElement[], root: Point | null) {
  // Canvases still in the page belong to a map that stays; a main area out of the page is a view that closed.
  if (!main?.isConnected || canvases.some((canvas) => canvas.isConnected)) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  // First in the main area, so under the new map's canvases.
  main.prepend(...canvases);
  for (const canvas of canvases) {
    Object.assign(canvas.style, {
      position: 'absolute',
      inset: '0',
      width: '100%',
      height: '100%',
      zIndex: '1',
      pointerEvents: 'none',
      transformOrigin: root ? `${root.x}px ${root.y}px` : 'center',
    });
    const remove = () => canvas.remove();
    canvas
      .animate(
        { opacity: [1, 0], transform: ['none', 'scale(0.95)'] },
        { duration: 340, easing: 'cubic-bezier(0.4, 0, 0.6, 1)', fill: 'forwards' },
      )
      .finished.then(remove, remove);
  }
}

export function GraphCanvas({ atlasRef }: Props) {
  const { t } = useTranslation();
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const view = useProfileStore((state) => state.view);
  const expandedIds = useProfileStore((state) => state.expandedIds);
  const selectedId = useProfileStore((state) => state.selectedId);
  const detailOpen = useProfileStore((state) => state.detailOpen);
  const visible = useMemo(() => visibleNodes(viewModel, expandedIds), [viewModel, expandedIds]);
  const looks = useMemo(() => nodeLooks(viewModel), [viewModel]);
  const metrics = useMemo(() => nodeMetrics(viewModel), [viewModel]);
  const washRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const entranceRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<AtlasEngine | null>(null);
  const introRef = useRef<NightIntro | null>(null);
  const stopEntranceRef = useRef<(() => void) | null>(null);
  // Selections made before the map shows, or carried over to a new model, are not animated.
  const mountedRef = useRef(false);
  const selectionModelRef = useRef(viewModel);
  const [hovered, setHovered] = useState<HoverInfo | null>(null);

  // The first visit opens on a night sky, which has to cover the page before its first paint.
  useLayoutEffect(() => {
    introRef.current = openNightIntro();
    return () => {
      introRef.current?.close();
      introRef.current = null;
    };
  }, []);

  useEffect(() => {
    const host = hostRef.current!;
    const main = host.parentElement;
    const canvas = canvasRef.current!;
    const wash = washRef.current!;
    const engine = new AtlasEngine(canvas, wash, {
      onNodeClick: (id) => useProfileStore.getState().clickNode(id),
      onBackgroundClick: () => useProfileStore.getState().clearSelection(),
      onHover: setHovered,
    });
    engineRef.current = engine;
    atlasRef.current = engine;
    // The overlay lets pointer events through to the map, so a click, drag or wheel ends the
    // entrance and acts at once.
    const interrupt = () => stopEntrance(stopEntranceRef);
    host.addEventListener('pointerdown', interrupt, true);
    host.addEventListener('wheel', interrupt, { capture: true, passive: true });
    let width = host.clientWidth;
    let height = host.clientHeight;
    const resize = new ResizeObserver(() => {
      if (host.clientWidth === width && host.clientHeight === height) return;
      width = host.clientWidth;
      height = host.clientHeight;
      // The entrance is drawn for the old size.
      interrupt();
      engine.resize();
      engine.setInsets(panelInsets(host, useProfileStore.getState().detailOpen));
    });
    resize.observe(host);
    return () => {
      host.removeEventListener('pointerdown', interrupt, true);
      host.removeEventListener('wheel', interrupt, true);
      resize.disconnect();
      interrupt();
      const root = engine.rootPoint();
      engine.destroy();
      engineRef.current = null;
      atlasRef.current = null;
      mountedRef.current = false;
      fadeOutMap(main, [wash, canvas], root);
    };
  }, [atlasRef]);

  useEffect(() => {
    engineRef.current?.setModel(viewModel, view, looks, metrics);
  }, [viewModel, view, looks, metrics]);

  // The canvas reads the tokens when it draws: another colour scheme needs a new picture.
  const theme = useTheme((state) => state.theme);
  useEffect(() => {
    engineRef.current?.repaint();
  }, [theme]);

  // A change during the entrance ends it: the entrance shows the map as first laid out.
  useEffect(() => {
    engineRef.current?.setTree(visible);
    stopEntrance(stopEntranceRef);
  }, [visible]);

  useEffect(() => {
    engineRef.current?.setSelected(selectedId, mountedRef.current && selectionModelRef.current === viewModel);
    selectionModelRef.current = viewModel;
    stopEntrance(stopEntranceRef);
  }, [selectedId, viewModel]);

  useEffect(() => {
    engineRef.current?.setInsets(panelInsets(hostRef.current!, detailOpen));
  }, [detailOpen]);

  // Once laid out, the map builds itself up under the entrance, or on the first visit the night intro.
  useEffect(() => {
    const engine = engineRef.current!;
    mountedRef.current = true;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      engine.reveal(false);
      return;
    }
    const config: EntranceConfig = {
      host: entranceRef.current!,
      ...engine.entranceScene(),
      onReveal: (finished) => engine.reveal(finished),
    };
    const model = useProfileStore.getState().viewModel!;
    stopEntranceRef.current = introRef.current?.play({ ...config, model }) ?? playEntrance(config);
  }, []);

  const hoveredNode = hovered ? viewModel.nodes.get(hovered.id) : undefined;
  const hoveredMetrics = hovered ? metrics.get(hovered.id) : undefined;
  return (
    <>
      <canvas ref={washRef} className="graph-wash" aria-hidden="true" />
      <div ref={hostRef} className="graph-canvas" role="img" aria-label={t(($) => $.atlas.canvas)}>
        <canvas ref={canvasRef} />
      </div>
      <div ref={entranceRef} className="graph-entrance" aria-hidden="true" />
      {hovered && hoveredNode && hoveredMetrics && (
        <NodePreview
          viewModel={viewModel}
          node={hoveredNode}
          metrics={hoveredMetrics}
          x={hovered.x}
          y={hovered.y}
          width={hovered.width}
          height={hovered.height}
        />
      )}
    </>
  );
}
