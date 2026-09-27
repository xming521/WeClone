import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Graph } from '@antv/g6';
import * as Tooltip from '@radix-ui/react-tooltip';
import {
  ArrowUpRight, ChevronLeft, ChevronRight, Compass, Fingerprint, Heart, Layers3, Maximize2,
  Menu, Minus, Move, Plus, RotateCcw, Search, Sparkles, Target, X,
} from 'lucide-react';
import { GraphCanvas } from './GraphCanvas';
import { DIMENSION_COLORS, ancestors } from './data';
import type { ProfileModel, ProfileNode, ReviewView } from './data';
import { ReviewActions, ReviewBar, ReviewOverlays, ReviewPanel, ReviewProvider, useReview } from './Review';
import { levelIsExpanded, useGraphStore } from './store';

const DIMENSION_ICONS = { 1: Fingerprint, 3: Target, 4: Layers3, 5: Compass, 8: Heart };

function collectFactIndices(model: ProfileModel, node: ProfileNode): number[] {
  if (node.kind === 'attribute') return node.factIndices;
  return node.childIds.flatMap((id) => collectFactIndices(model, model.nodes.get(id)!));
}

function IconButton({ label, children, onClick }: { label: string; children: React.ReactNode; onClick: () => void }) {
  return (
    <Tooltip.Root delayDuration={250}>
      <Tooltip.Trigger asChild>
        <button className="icon-button" type="button" aria-label={label} onClick={onClick}>{children}</button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="tooltip" sideOffset={8}>{label}<Tooltip.Arrow className="tooltip-arrow" /></Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

function SearchBox({ model }: { model: ProfileModel }) {
  const search = useGraphStore((state) => state.search);
  const setSearch = useGraphStore((state) => state.setSearch);
  const reveal = useGraphStore((state) => state.reveal);
  const normalized = search.trim().toLocaleLowerCase();
  const results = useMemo(() => normalized
    ? [...model.nodes.values()].filter((node) => node.kind !== 'root' && (node.name.toLocaleLowerCase().includes(normalized) || node.factIndices.some((index) => model.facts[index].value.toLocaleLowerCase().includes(normalized)))).slice(0, 8)
    : [], [model, normalized]);

  return (
    <div className="search-wrap">
      <Search size={18} strokeWidth={1.8} />
      <input
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && results[0]) reveal(model, results[0].id);
          if (event.key === 'Escape') setSearch('');
        }}
        placeholder="搜索主题、属性或内容"
        aria-label="搜索主题、属性或内容"
      />
      {search && <button className="search-clear" type="button" aria-label="清除搜索" onClick={() => setSearch('')}><X size={16} /></button>}
      {normalized && (
        <div className="search-results" role="listbox">
          {results.length ? results.map((node) => (
            <button key={node.id} type="button" role="option" aria-selected="false" onClick={() => reveal(model, node.id)}>
              <span className="search-result-dot" style={{ backgroundColor: DIMENSION_COLORS[node.dim ?? 1] }} />
              <span className="search-result-main"><strong>{node.name}</strong><small>{ancestors(model, node.id).slice(1, -1).map((part) => part.name).join(' / ')}</small></span>
              <ArrowUpRight size={15} />
            </button>
          )) : <div className="search-empty">没有找到相关画像</div>}
        </div>
      )}
    </div>
  );
}

function LeftPanel({ model, open, onClose, onInteract, onSelect }: { model: ProfileModel; open: boolean; onClose: () => void; onInteract: () => void; onSelect: () => void }) {
  const focusedDimension = useGraphStore((state) => state.focusedDimension);
  const reveal = useGraphStore((state) => state.reveal);
  return (
    <aside className={`left-panel${open ? '' : ' is-collapsed'}`} aria-hidden={!open} onPointerDownCapture={onInteract} onFocusCapture={onInteract}>
      <div className="brand-row"><span className="brand-symbol">✳</span><span>WeClone<span className="brand-separator">/</span>Profile Atlas</span><button className="sidebar-close" type="button" aria-label="收起左栏" onClick={onClose}><ChevronLeft size={17} /></button></div>
      <div className="intro-block">
        <div className="eyebrow"><span className="eyebrow-line" /> A MAP OF YOU</div>
        <h1>每一条线索，<br /><em>都更了解你。</em></h1>
        <p>沿着五条生活脉络，探索你的画像。</p>
      </div>
      <div className="nav-heading">画像维度</div>
      <nav className="dimension-list" aria-label="画像维度">
        {model.dimensionIds.map((id) => {
          const node = model.nodes.get(id)!;
          const color = DIMENSION_COLORS[node.dim!];
          const Icon = DIMENSION_ICONS[node.dim! as keyof typeof DIMENSION_ICONS];
          return (
            <button
              key={id}
              type="button"
              className={`dimension-item ${focusedDimension === node.dim ? 'active' : ''}`}
              onClick={() => { reveal(model, id); onSelect(); }}
              style={{ '--dimension-color': color } as CSSProperties}
            >
              <span className="dimension-icon"><Icon size={19} strokeWidth={1.7} /></span>
              <span className="dimension-copy"><strong>{node.name}</strong><small>{node.attributeCount} 项画像属性</small></span>
            </button>
          );
        })}
      </nav>
      <div className="left-bottom">
        <div className="data-footnote"><span className="live-dot" /> {model.facts.length} 条画像</div>
      </div>
    </aside>
  );
}

function DetailPanel({ model, open, onClose }: { model: ProfileModel; open: boolean; onClose: () => void }) {
  const selectedId = useGraphStore((state) => state.selectedId);
  const reveal = useGraphStore((state) => state.reveal);
  const selected = model.nodes.get(selectedId ?? 'root') ?? model.nodes.get('root')!;
  const path = ancestors(model, selected.id).slice(1);
  const color = DIMENSION_COLORS[selected.dim ?? 1];
  const indices = collectFactIndices(model, selected);
  const uniqueIndices = [...new Set(indices)];
  const children = selected.childIds.map((id) => model.nodes.get(id)!);

  return (
      <aside className={`detail-panel${open ? ' is-open' : ''}`} aria-hidden={!open} style={{ '--detail-accent': color } as CSSProperties}>
        <button className="detail-close" type="button" aria-label="关闭右侧栏" onClick={onClose}><X size={17} /></button>
        <div className="detail-topline"><span className="detail-kicker"><Sparkles size={14} /> 画像档案</span><span className="detail-serial">WECLONE / 2026</span></div>
        <div className="detail-content">
          <div className="breadcrumb">
            {path.length ? path.map((part, index) => <span key={part.id}>{index > 0 && <ChevronRight size={12} />}{part.name}</span>) : <span>总览</span>}
          </div>
          <div className="detail-title-wrap">
            <span className="detail-symbol" aria-hidden="true">✦</span>
            <h2>{selected.name}</h2>
            <p>{selected.kind === 'attribute' ? '画像属性' : selected.kind === 'dimension' ? '维度概览' : selected.kind === 'root' ? '完整画像' : '主题脉络'}</p>
          </div>
          <div className="detail-metrics">
            <div><strong>{selected.attributeCount}</strong><span>画像属性</span></div>
            <div><strong>{uniqueIndices.length}</strong><span>画像事实</span></div>
          </div>

          {selected.kind !== 'attribute' && (
            <section className="detail-section">
              <div className="section-heading"><h3>继续探索</h3><span>{children.length.toString().padStart(2, '0')}</span></div>
              <div className="detail-children">
                {children.map((child) => <button type="button" key={child.id} onClick={() => reveal(model, child.id)}>
                  <span className="child-dot" /><span>{child.name}</span><small>{child.attributeCount}</small><ChevronRight size={15} />
                </button>)}
              </div>
              {selected.kind === 'root' && <p className="detail-hint">选择左侧维度，逐层展开画像地图。</p>}
            </section>
          )}
          <ReviewPanel />
        </div>
        <div className="detail-footer"><span>PERSONAL KNOWLEDGE MAP</span><span>✳</span></div>
      </aside>
  );
}

function Atlas({ model, view }: { model: ProfileModel; view: ReviewView }) {
  const graphRef = useRef<Graph | null>(null);
  const [leftOpen, setLeftOpen] = useState(true);
  const [leftTouched, setLeftTouched] = useState(false);
  const focusedDimension = useGraphStore((state) => state.focusedDimension);
  const resetVersion = useGraphStore((state) => state.resetVersion);
  const reset = useGraphStore((state) => state.reset);
  const detailOpen = useGraphStore((state) => state.detailOpen);
  const closeDetail = useGraphStore((state) => state.closeDetail);
  const toggleLevel = useGraphStore((state) => state.toggleLevel);
  const active = model.nodes.get(`dimension:${focusedDimension}`);
  const expandedIds = useGraphStore((state) => state.expandedIds);

  useEffect(() => {
    if (leftTouched) return;
    const timer = window.setTimeout(() => setLeftOpen(false), 5000);
    return () => window.clearTimeout(timer);
  }, [leftTouched]);

  return (
    <Tooltip.Provider>
      <div className={`app-shell${leftOpen ? ' left-open' : ''}${detailOpen ? ' detail-open' : ''}`}>
        <LeftPanel model={model} open={leftOpen} onClose={() => { setLeftTouched(true); setLeftOpen(false); }} onInteract={() => setLeftTouched(true)} onSelect={() => { setLeftTouched(true); setLeftOpen(false); }} />
        <main className="atlas-main">
          <div className="canvas-atmosphere" />
          <div className="topbar">
            <div className="topbar-leading">
              {!leftOpen && <button className="sidebar-open" type="button" aria-label="展开左栏" onClick={() => { setLeftTouched(true); setLeftOpen(true); }}><Menu size={17} /><span>画像维度</span></button>}
              <div className="topbar-heading"><span className="topbar-dot" /> 画像地图 <span className="topbar-divider">/</span> <strong>{active?.name ?? '五维总览'}</strong></div>
            </div>
            <div className="topbar-actions"><SearchBox model={model} /><ReviewActions /></div>
          </div>
          <ReviewBar />
          {!model.facts.length && <div className="graph-empty">当前视图没有画像记录</div>}
          <GraphCanvas key={resetVersion} model={model} view={view} graphRef={graphRef} />
          <div className="canvas-toolbar">
            <div className="level-controls">
              {([1, 2, 3] as const).map((level) => {
                const expanded = levelIsExpanded(model, expandedIds, level, focusedDimension);
                const action = focusedDimension === null ? '先点击一个维度' : `${expanded ? '收起' : '展开'}当前维度的${level}级节点`;
                return <Tooltip.Root key={level} delayDuration={250}>
                  <Tooltip.Trigger asChild><button className={`level-button${expanded ? ' is-active' : ''}${focusedDimension === null ? ' is-disabled' : ''}`} type="button" aria-label={action} aria-disabled={focusedDimension === null} aria-pressed={expanded} onClick={() => toggleLevel(model, level)}>{level}级</button></Tooltip.Trigger>
                  <Tooltip.Portal><Tooltip.Content className="tooltip" sideOffset={8}>{action}<Tooltip.Arrow className="tooltip-arrow" /></Tooltip.Content></Tooltip.Portal>
                </Tooltip.Root>;
              })}
            </div>
            <span className="toolbar-separator" />
            <div className="view-controls">
              <IconButton label="放大" onClick={() => void graphRef.current?.zoomBy(1.2, { duration: 250 })}><Plus size={17} /></IconButton>
              <IconButton label="缩小" onClick={() => void graphRef.current?.zoomBy(0.83, { duration: 250 })}><Minus size={17} /></IconButton>
              <span className="toolbar-separator" />
              <IconButton label="适应画布" onClick={() => void graphRef.current?.fitView()}><Maximize2 size={16} /></IconButton>
              <span className="toolbar-separator" />
              <button className="reset-button" type="button" onClick={() => reset(model)}><RotateCcw size={15} />复原</button>
            </div>
          </div>
          <div className="canvas-help"><Move size={15} /> 拖动节点牵动图谱 · 点击探索</div>
        </main>
        <ReviewOverlays />
        <DetailPanel model={model} open={detailOpen} onClose={closeDetail} />
      </div>
    </Tooltip.Provider>
  );
}

export default function App() {
  const review = useReview();
  if (!review.model && review.error) return <div className="load-state"><span>✳</span><h1>画像地图暂时无法打开</h1><p>{review.error}</p><button onClick={() => void review.refresh().catch((reason: Error) => review.setError(reason.message))}>重试</button></div>;
  if (!review.model) return <div className="load-state"><span className="loading-mark">✳</span><h1>正在展开画像地图</h1></div>;
  return <ReviewProvider review={review}><Atlas model={review.model} view={review.view} /></ReviewProvider>;
}
