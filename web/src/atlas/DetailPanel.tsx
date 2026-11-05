import { useEffect, useRef } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { ChevronRight, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { BrandMark } from '../BrandMark';
import { ancestors, collectFacts, displayProfileText, SENSITIVITY_LEVELS } from '../data';
import type { ProfileFact, Sensitivity } from '../data';
import { useProfileStore } from '../store';
import { ReviewPanel } from '../review/ReviewPanel';
import { scopeColor } from './scopeColor';

function isLevel(level: string | undefined): level is Sensitivity {
  return SENSITIVITY_LEVELS.includes(level as Sensitivity);
}

/** The facts in scope by review state, as one segmented bar and its legend. */
function ReviewDistribution({ facts }: { facts: ProfileFact[] }) {
  const { t } = useTranslation();
  const pending = facts.filter((fact) => fact.status === 'pending').length;
  const approved = facts.filter((fact) => fact.status === 'approved').length;
  const rejected = facts.length - pending - approved;
  const reviewed = facts.length - pending;
  // `mark` is the design system's name for the state, `status` the review state's own.
  const parts = [
    { status: 'approved', mark: 'adopted', count: approved },
    { status: 'pending', mark: 'pending', count: pending },
    { status: 'rejected', mark: 'rejected', count: rejected },
  ] as const;
  return (
    <section className="detail-block" aria-label={t(($) => $.atlas.detail.progress)}>
      <div className="detail-block-head">
        <h3>{t(($) => $.atlas.detail.progress)}</h3>
        <span className="detail-block-count">
          {reviewed} / {facts.length}
        </span>
      </div>
      <div
        className="wc-progress"
        role="img"
        aria-label={t(($) => $.atlas.detail.progressCounts, { pending, approved, rejected })}
      >
        {parts.map(({ status, mark, count }) =>
          count ? <i key={status} data-k={mark} style={{ flex: count }} /> : null,
        )}
      </div>
      <ul className="wc-legend">
        {parts.map(({ status, mark, count }) => (
          <li key={status}>
            <span className="wc-glyph" data-status={mark} aria-hidden="true" />
            {t(($) => $.review.status[status])} <b>{count}</b>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** How the facts in scope spread over the privacy levels; nothing when none of them has a level. */
function PrivacyDistribution({ facts }: { facts: ProfileFact[] }) {
  const { t } = useTranslation();
  const counts = new Map<Sensitivity, number>();
  for (const fact of facts) {
    if (isLevel(fact.sensitivity)) counts.set(fact.sensitivity, (counts.get(fact.sensitivity) ?? 0) + 1);
  }
  const levelled = [...counts.values()].reduce((sum, count) => sum + count, 0);
  if (!levelled) return null;
  const present = SENSITIVITY_LEVELS.filter((level) => counts.get(level));
  const unrated = facts.length - levelled;
  const label = (level: Sensitivity) => t(($) => $.privacy.levels[level].label);
  const summary = [
    ...present.map((level) => `${label(level)} ${counts.get(level)}`),
    ...(unrated ? [`${t(($) => $.privacy.levels.unrated.label)} ${unrated}`] : []),
  ].join(t(($) => $.common.listSeparator));
  return (
    <section className="detail-block" aria-label={t(($) => $.privacy.overview)}>
      <div className="detail-block-head">
        <h3>{t(($) => $.privacy.title)}</h3>
        <span className="detail-block-note">
          {t(($) => $.privacy.privateCount, { n: (counts.get('private') ?? 0) + (counts.get('sensitive') ?? 0) })}
        </span>
      </div>
      <div className="wc-progress" role="img" aria-label={summary}>
        {present.map((level) => (
          <i key={level} data-privacy={level} style={{ flex: counts.get(level) }} />
        ))}
      </div>
      <ul className="wc-legend">
        {present.map((level) => (
          <li key={level} data-privacy={level}>
            <span className="wc-phase" aria-hidden="true" />
            {label(level)} <b>{counts.get(level)}</b>
          </li>
        ))}
        {unrated > 0 && (
          <li className="detail-legend-unrated">
            <span className="wc-phase" aria-hidden="true" />
            {t(($) => $.privacy.levels.unrated.label)} <b>{unrated}</b>
          </li>
        )}
      </ul>
    </section>
  );
}

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
  const trail = path.slice(0, -1);
  const facts = collectFacts(fullModel, selected.id);
  const pending = facts.filter((fact) => fact.status === 'pending').length;
  const children = (viewModel.nodes.get(selected.id)?.childIds ?? []).map((id) => viewModel.nodes.get(id)!);
  useEffect(() => {
    contentRef.current?.scrollTo(0, 0);
  }, [selected.id, tab]);

  return (
    <Tabs.Root asChild value={tab} onValueChange={(value) => setTab(value as typeof tab)}>
      <aside className={`detail-panel${open ? ' is-open' : ''}`} aria-hidden={!open} {...scopeColor(selected.dim)}>
        <div className="wc-panelhead">
          <span className="wc-kicker">{t(($) => $.atlas.detail.kicker)}</span>
          <span className="wc-serial">WECLONE / 2026</span>
          <button
            className="wc-iconbtn wc-iconbtn--sm"
            type="button"
            aria-label={t(($) => $.atlas.detail.close)}
            onClick={onClose}
          >
            <X size={16} strokeWidth={1.8} />
          </button>
        </div>
        <Tabs.List className="wc-tabs detail-tabs" aria-label={t(($) => $.atlas.detail.tabs)}>
          <Tabs.Trigger className="wc-tab" value="overview">
            {t(($) => $.atlas.detail.overview)}
          </Tabs.Trigger>
          <Tabs.Trigger className="wc-tab" value="review">
            {t(($) => $.atlas.detail.review)}
            {/* Nothing left to review: the tail goes, as it does on a graph node. */}
            <span
              className={pending ? 'wc-count wc-count--pending' : 'wc-count'}
              aria-label={t(($) => $.atlas.detail.pending, { count: pending })}
            >
              {pending}
            </span>
          </Tabs.Trigger>
        </Tabs.List>
        <div className="detail-content detail-scroll" ref={contentRef}>
          <div className="detail-scope-heading">
            {trail.length > 0 && (
              <ol className="wc-crumb detail-crumb" aria-label={t(($) => $.editor.location)}>
                <li>
                  <span className="wc-dot" aria-hidden="true" />
                </li>
                {trail.map((part, index) => (
                  <li key={part.id} className="detail-crumb-step">
                    {index > 0 && <ChevronRight size={12} strokeWidth={1.8} aria-hidden="true" />}
                    {viewModel.nodes.has(part.id) ? (
                      <button type="button" onClick={() => reveal(part.id)}>
                        {part.name}
                      </button>
                    ) : (
                      <span>{part.name}</span>
                    )}
                  </li>
                ))}
              </ol>
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

            <ReviewDistribution facts={facts} />
            <PrivacyDistribution facts={facts} />

            {selected.kind !== 'attribute' && (
              <section className="detail-block">
                <div className="detail-block-head">
                  <h3>{t(($) => $.atlas.detail.explore)}</h3>
                  <span className="detail-block-serial">{children.length.toString().padStart(2, '0')}</span>
                </div>
                <div className="detail-children">
                  {children.map((child) => (
                    <button
                      type="button"
                      className="detail-child"
                      key={child.id}
                      onClick={() => reveal(child.id)}
                      {...scopeColor(child.dim)}
                    >
                      <span className="wc-dot" aria-hidden="true" />
                      <span className="detail-child-name">{child.name}</span>
                      <span className="wc-count">{child.attributeCount}</span>
                      <ChevronRight size={14} strokeWidth={1.8} aria-hidden="true" />
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
          <span className="wc-serial">{t(($) => $.atlas.detail.footer)}</span>
          <BrandMark variant="favicon" />
        </div>
      </aside>
    </Tabs.Root>
  );
}
