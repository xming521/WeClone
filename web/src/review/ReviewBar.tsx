import { Layers, ListChecks } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { Tip } from '../atlas/IconButton';
import { useProfileStore } from '../store';
import { MARK_STATUS, REVIEW_VIEWS } from './status';

export function ReviewBar() {
  const { t } = useTranslation();
  const input = useProfileStore((state) => state.input)!;
  const currentView = useProfileStore((state) => state.view);
  const setView = useProfileStore((state) => state.setView);
  const startReview = useProfileStore((state) => state.startReview);
  const counts = { all: input.facts.length, pending: 0, approved: 0, rejected: 0 };
  input.facts.forEach((fact) => counts[fact.status]++);
  return (
    <div className="review-bar">
      <nav className="wc-rail" aria-label={t(($) => $.review.statusLabel)}>
        <div
          className="wc-progress wc-rail__bar"
          role="img"
          aria-label={t(($) => $.atlas.detail.progressCounts, {
            pending: counts.pending,
            approved: counts.approved,
            rejected: counts.rejected,
          })}
        >
          {counts.approved > 0 && <i data-k="adopted" style={{ flex: counts.approved }} />}
          {counts.pending > 0 && <i data-k="pending" style={{ flex: counts.pending }} />}
          {counts.rejected > 0 && <i data-k="rejected" style={{ flex: counts.rejected }} />}
        </div>
        {REVIEW_VIEWS.map((view, index, views) => {
          const label = t(($) => $.review.status[view]);
          // The adopted tip names the clone's portrait, which carries the clone dot.
          const tip =
            view === 'approved' ? (
              <Trans
                i18nKey={($) => $.review.bookmarkCloneTip}
                values={{ label, count: counts[view] }}
                components={{ clone: <span className="wc-clone" /> }}
              />
            ) : (
              t(($) => $.review.bookmarkTip, { label, count: counts[view] })
            );
          return (
            <Tip key={view} side="right" align="center" content={tip}>
              <button
                type="button"
                aria-label={t(($) => $.review.bookmark, { label, count: counts[view] })}
                aria-pressed={currentView === view}
                className="wc-rail__item"
                onClick={() => setView(view)}
                onKeyDown={(event) => {
                  if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
                  event.preventDefault();
                  const next = (index + (event.key === 'ArrowDown' ? 1 : views.length - 1)) % views.length;
                  const items = event.currentTarget
                    .closest('nav')
                    ?.querySelectorAll<HTMLButtonElement>('.wc-rail__item');
                  items?.[next]?.focus();
                  setView(views[next]);
                }}
              >
                <span className="wc-rail__n" aria-hidden="true">
                  {view === 'all' ? (
                    <Layers size={13} strokeWidth={1.8} />
                  ) : (
                    <span className="wc-glyph" data-status={MARK_STATUS[view]} />
                  )}
                  {counts[view]}
                </span>
                <span className="wc-rail__label" aria-hidden="true">
                  {label}
                </span>
              </button>
            </Tip>
          );
        })}
        <span className="review-start-sep" aria-hidden="true" />
        <Tip
          side="right"
          align="center"
          content={
            counts.pending ? t(($) => $.review.startTip, { count: counts.pending }) : t(($) => $.review.startDone)
          }
        >
          <button
            type="button"
            className="wc-rail__item review-start"
            aria-disabled={!counts.pending}
            onClick={() => counts.pending && startReview()}
          >
            <span className="wc-rail__n" aria-hidden="true">
              <ListChecks size={15} strokeWidth={1.8} />
            </span>
            <span className="wc-rail__label">{t(($) => $.review.start)}</span>
          </button>
        </Tip>
      </nav>
    </div>
  );
}
