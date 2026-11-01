import { Check } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { collectFacts, filterPrivacy } from '../data';
import type { ProfileFact, ReviewStatus } from '../data';
import { useProfileStore } from '../store';
import { FactCard } from './FactCard';
import { PrivacyFilter } from './privacy';
import { ReviewFilter } from './ReviewFilter';

export function ReviewPanel() {
  const { t } = useTranslation();
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const fullModel = useProfileStore((state) => state.fullModel)!;
  const view = useProfileStore((state) => state.view);
  const setView = useProfileStore((state) => state.setView);
  const selectedId = useProfileStore((state) => state.selectedId) ?? 'root';
  const privacy = useProfileStore((state) => state.privacy);
  const togglePrivacy = useProfileStore((state) => state.togglePrivacy);
  const scoped = collectFacts(viewModel, selectedId);
  const facts = filterPrivacy(scoped, privacy);
  const all = collectFacts(fullModel, selectedId);
  const counts = { pending: 0, approved: 0, rejected: 0 };
  all.forEach((fact) => counts[fact.status]++);
  return (
    <section className="detail-section review-list">
      <div className="review-list-toolbar">
        <ReviewFilter value={view} counts={{ all: all.length, ...counts }} onChange={setView} />
        <span>{t(($) => $.review.list.count, { count: facts.length })}</span>
      </div>
      <PrivacyFilter facts={scoped} value={privacy} onToggle={togglePrivacy} />
      <ReviewList key={`${selectedId}:${view}:${privacy.join()}`} facts={facts} />
    </section>
  );
}

function ReviewList({ facts }: { facts: ProfileFact[] }) {
  const { t } = useTranslation();
  const busy = useProfileStore((state) => state.busy);
  const review = useProfileStore((state) => state.review);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [limit, setLimit] = useState(30);
  const page = facts.slice(0, limit);
  // Selecting all takes every fact of the list, also those further down than shown.
  const selected = facts.filter((fact) => checked.has(fact.id));
  const selectAllRef = useRef<HTMLInputElement>(null);
  const mixed = selected.length > 0 && selected.length < facts.length;
  useEffect(() => {
    const box = selectAllRef.current;
    if (box) box.indeterminate = mixed;
  }, [mixed]);
  async function batch(status: ReviewStatus) {
    if (!selected.length) return;
    const hidden = selected.length - page.filter((fact) => checked.has(fact.id)).length;
    const question =
      status === 'approved'
        ? t(($) => $.review.list.confirmApprove, { count: selected.length, hidden })
        : t(($) => $.review.list.confirmReject, { count: selected.length, hidden });
    if (hidden && !window.confirm(question)) return;
    if (await review(selected, status)) setChecked(new Set());
  }
  return (
    <>
      {!!facts.length && (
        <div className={`wc-batch${selected.length ? ' has-selection' : ''}`}>
          <label className={`wc-check${mixed ? ' is-mixed' : ''}`}>
            <input
              ref={selectAllRef}
              type="checkbox"
              aria-checked={mixed ? 'mixed' : undefined}
              checked={selected.length === facts.length}
              onChange={(event) => setChecked(new Set(event.target.checked ? facts.map((fact) => fact.id) : []))}
            />
            <span className="wc-check__box" aria-hidden="true" />
            {selected.length
              ? t(($) => $.review.list.selected, { count: selected.length })
              : t(($) => $.review.list.selectAll, { count: facts.length })}
          </label>
          <button
            type="button"
            className={`wc-btn wc-btn--sm${selected.length ? ' wc-btn--adopt' : ''}`}
            disabled={busy || !selected.length}
            onClick={() => void batch('approved')}
          >
            {!!selected.length && <Check size={13} strokeWidth={1.8} aria-hidden="true" />}
            {t(($) => $.review.approve)}
          </button>
          <button
            type="button"
            className="wc-btn wc-btn--sm"
            disabled={busy || !selected.length}
            onClick={() => void batch('rejected')}
          >
            {t(($) => $.review.reject)}
          </button>
        </div>
      )}
      {!facts.length && <p className="detail-hint">{t(($) => $.review.list.empty)}</p>}
      {page.map((fact) => (
        <FactCard
          key={fact.id}
          fact={fact}
          checked={checked.has(fact.id)}
          onCheckedChange={(isChecked) => {
            const next = new Set(checked);
            if (isChecked) next.add(fact.id);
            else next.delete(fact.id);
            setChecked(next);
          }}
        />
      ))}
      {facts.length > limit && (
        <button type="button" className="wc-btn wc-btn--block load-more" onClick={() => setLimit(limit + 30)}>
          {t(($) => $.review.list.more, { count: facts.length - limit })}
        </button>
      )}
    </>
  );
}
