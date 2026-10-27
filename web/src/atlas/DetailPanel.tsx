import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { ChevronRight, Sparkles, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ancestors, collectFacts, dimensionColor, displayProfileText } from '../data';
import { useProfileStore } from '../store';
import { PrivacyOverview } from '../review/privacy';
import { ReviewPanel } from '../review/ReviewPanel';

export function DetailPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useTranslation();
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
        <button className="detail-close" type="button" aria-label={t(($) => $.atlas.detail.close)} onClick={onClose}>
          <X size={17} />
        </button>
        <div className="detail-topline">
          <span className="detail-kicker">
            <Sparkles size={14} /> {t(($) => $.atlas.detail.kicker)}
          </span>
          <span className="detail-serial">WECLONE / 2026</span>
        </div>
        <Tabs.List className="detail-tabbar" aria-label={t(($) => $.atlas.detail.tabs)}>
          <Tabs.Trigger value="overview">{t(($) => $.atlas.detail.overview)}</Tabs.Trigger>
          <Tabs.Trigger value="review">
            {t(($) => $.atlas.detail.review)}
            <span className="detail-pending-count" aria-label={t(($) => $.atlas.detail.pending, { count: pending })}>
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
            <h2>{selected.kind === 'root' ? t(($) => $.atlas.detail.all) : selected.name}</h2>
            {selected.description && <p className="detail-description">{displayProfileText(selected.description)}</p>}
          </div>
          <Tabs.Content value="overview" forceMount hidden={tab !== 'overview'}>
            <div className="detail-metrics">
              <div>
                <strong>{selected.attributeCount}</strong>
                <span>{t(($) => $.atlas.detail.attributes)}</span>
              </div>
              <div>
                <strong>{facts.length}</strong>
                <span>{t(($) => $.atlas.detail.facts)}</span>
              </div>
            </div>

            <section className="overview-review-progress" aria-label={t(($) => $.atlas.detail.progress)}>
              <div>
                <span>{t(($) => $.atlas.detail.progress)}</span>
                <span>
                  {reviewed} / {facts.length}
                </span>
              </div>
              <div
                className="overview-progress-track"
                role="progressbar"
                aria-label={t(($) => $.atlas.detail.reviewed)}
                aria-valuemin={0}
                aria-valuemax={facts.length || 1}
                aria-valuenow={reviewed}
              >
                <span style={{ width: `${facts.length ? (reviewed / facts.length) * 100 : 0}%` }} />
              </div>
              <p>
                {t(($) => $.atlas.detail.progressCounts, {
                  pending,
                  approved,
                  rejected: reviewed - approved,
                })}
              </p>
            </section>

            <PrivacyOverview facts={facts} />

            {selected.kind !== 'attribute' && (
              <section className="detail-section">
                <div className="section-heading">
                  <h3>{t(($) => $.atlas.detail.explore)}</h3>
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
                {selected.kind === 'root' && (
                  <p className="detail-hint">
                    {fullModel.grouping === 'domain'
                      ? t(($) => $.atlas.detail.domainHint)
                      : t(($) => $.atlas.detail.dimensionHint)}
                  </p>
                )}
              </section>
            )}
          </Tabs.Content>
          <Tabs.Content value="review" forceMount hidden={tab !== 'review'}>
            <ReviewPanel />
          </Tabs.Content>
        </div>
        <div className="detail-footer">
          <span>{t(($) => $.atlas.detail.footer)}</span>
          <span>✳</span>
        </div>
      </aside>
    </Tabs.Root>
  );
}
