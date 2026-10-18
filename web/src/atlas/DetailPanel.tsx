import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { ChevronRight, Sparkles, X } from 'lucide-react';
import { ancestors, collectFacts, dimensionColor } from '../data';
import { useProfileStore } from '../store';
import { ReviewPanel } from '../review/ReviewPanel';

export function DetailPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const fullModel = useProfileStore((state) => state.fullModel)!;
  const selectedId = useProfileStore((state) => state.selectedId);
  const reveal = useProfileStore((state) => state.reveal);
  const tab = useProfileStore((state) => state.detailTab);
  const setTab = useProfileStore((state) => state.setDetailTab);
  const contentRef = useRef<HTMLDivElement>(null);
  const selected = fullModel.nodes.get(selectedId ?? 'root') ?? fullModel.nodes.get('root')!;
  const path = ancestors(fullModel, selected.id).slice(1);
  const color = dimensionColor(selected.dim);
  const facts = collectFacts(fullModel, selected.id);
  const pending = facts.filter((fact) => fact.status === 'pending').length;
  const approved = facts.filter((fact) => fact.status === 'approved').length;
  const reviewed = facts.length - pending;
  const children = (viewModel.nodes.get(selected.id)?.childIds ?? []).map((id) => viewModel.nodes.get(id)!);
  useEffect(() => {
    contentRef.current?.scrollTo(0, 0);
  }, [selected.id, tab]);

  return (
    <Tabs.Root asChild value={tab} onValueChange={(value) => setTab(value as typeof tab)}>
      <aside
        className={`detail-panel${open ? ' is-open' : ''}`}
        aria-hidden={!open}
        style={{ '--detail-accent': color } as CSSProperties}
      >
        <button className="detail-close" type="button" aria-label="关闭右侧栏" onClick={onClose}>
          <X size={17} />
        </button>
        <div className="detail-topline">
          <span className="detail-kicker">
            <Sparkles size={14} /> 画像档案
          </span>
          <span className="detail-serial">WECLONE / 2026</span>
        </div>
        <Tabs.List className="detail-tabbar" aria-label="画像档案视图">
          <Tabs.Trigger value="overview">总览</Tabs.Trigger>
          <Tabs.Trigger value="review">
            画像审核
            <span className="detail-pending-count" aria-label={`${pending} 条待审核`}>
              {pending}
            </span>
          </Tabs.Trigger>
        </Tabs.List>
        <div className="detail-content" ref={contentRef}>
          <div className="detail-scope-heading">
            {path.length > 1 && (
              <div className="breadcrumb">
                {path.slice(0, -1).map((part, index) => (
                  <span key={part.id}>
                    {index > 0 && <ChevronRight size={12} />}
                    {part.name}
                  </span>
                ))}
              </div>
            )}
            <h2>{selected.kind === 'root' ? '全部画像' : selected.name}</h2>
          </div>
          <Tabs.Content value="overview" forceMount hidden={tab !== 'overview'}>
            <div className="detail-metrics">
              <div>
                <strong>{selected.attributeCount}</strong>
                <span>画像属性</span>
              </div>
              <div>
                <strong>{facts.length}</strong>
                <span>画像事实</span>
              </div>
            </div>

            <section className="overview-review-progress" aria-label="审核进度">
              <div>
                <span>审核进度</span>
                <span>
                  {reviewed} / {facts.length}
                </span>
              </div>
              <div
                className="overview-progress-track"
                role="progressbar"
                aria-label="已审核画像"
                aria-valuemin={0}
                aria-valuemax={facts.length || 1}
                aria-valuenow={reviewed}
              >
                <span style={{ width: `${facts.length ? (reviewed / facts.length) * 100 : 0}%` }} />
              </div>
              <p>
                待审核 {pending} · 已采纳 {approved} · 未采纳 {reviewed - approved}
              </p>
            </section>

            {selected.kind !== 'attribute' && (
              <section className="detail-section">
                <div className="section-heading">
                  <h3>继续探索</h3>
                  <span>{children.length.toString().padStart(2, '0')}</span>
                </div>
                <div className="detail-children">
                  {children.map((child) => (
                    <button type="button" key={child.id} onClick={() => reveal(child.id)}>
                      <span className="child-dot" />
                      <span>{child.name}</span>
                      <small>{child.attributeCount}</small>
                      <ChevronRight size={15} />
                    </button>
                  ))}
                </div>
                {selected.kind === 'root' && <p className="detail-hint">选择左侧维度，逐层展开画像地图。</p>}
              </section>
            )}
          </Tabs.Content>
          <Tabs.Content value="review" forceMount hidden={tab !== 'review'}>
            <ReviewPanel />
          </Tabs.Content>
        </div>
        <div className="detail-footer">
          <span>PERSONAL KNOWLEDGE MAP</span>
          <span>✳</span>
        </div>
      </aside>
    </Tabs.Root>
  );
}
