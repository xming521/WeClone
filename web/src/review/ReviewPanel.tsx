import { Check } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { collectFacts, filterPrivacy, sortFacts } from '../data';
import type { ProfileFact, ReviewStatus } from '../data';
import { isTyping } from '../feedback/keys';
import { useProfileStore } from '../store';
import { FactCard } from './FactCard';
import { PrivacyFilter } from './privacy';
import { ReviewFilter } from './ReviewFilter';
import { SortSelect } from './SortSelect';

export function ReviewPanel() {
  const { t } = useTranslation();
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const fullModel = useProfileStore((state) => state.fullModel)!;
  const view = useProfileStore((state) => state.view);
  const setView = useProfileStore((state) => state.setView);
  const selectedId = useProfileStore((state) => state.selectedId) ?? 'root';
  const privacy = useProfileStore((state) => state.privacy);
  const togglePrivacy = useProfileStore((state) => state.togglePrivacy);
  const sort = useProfileStore((state) => state.sort);
  const setSort = useProfileStore((state) => state.setSort);
  const sources = useProfileStore((state) => state.input?.sources);
  const scoped = collectFacts(viewModel, selectedId);
  const facts = sortFacts(filterPrivacy(scoped, privacy), sort, sources);
  const all = collectFacts(fullModel, selectedId);
  const counts = { pending: 0, approved: 0, rejected: 0 };
  all.forEach((fact) => counts[fact.status]++);
  return (
    <section className="detail-section review-list">
      <div className="review-list-toolbar">
        <ReviewFilter value={view} counts={{ all: all.length, ...counts }} onChange={setView} />
        <SortSelect value={sort} onChange={setSort} />
        <span>{t(($) => $.review.list.count, { count: facts.length })}</span>
      </div>
      <PrivacyFilter facts={scoped} value={privacy} onToggle={togglePrivacy} />
      <ReviewList key={`${selectedId}:${view}:${privacy.join()}:${sort}`} facts={facts} />
    </section>
  );
}

/** Next and previous card, then the actions on the marked card; Z (undo) is the notice's own. */
const KEYS = [
  ['J', 'next'],
  ['K', 'previous'],
  ['A', 'approve'],
  ['R', 'reject'],
  ['E', 'edit'],
  ['X', 'select'],
  ['Z', 'undo'],
] as const;
// The keys the list handles itself, lower case.
const LIST_KEYS = new Set<string>(KEYS.filter(([, action]) => action !== 'undo').map(([key]) => key.toLowerCase()));

function ReviewList({ facts }: { facts: ProfileFact[] }) {
  const { t } = useTranslation();
  const review = useProfileStore((state) => state.review);
  const edit = useProfileStore((state) => state.edit);
  const reviewKeys = useProfileStore((state) => state.reviewKeys);
  const showReviewKeys = useProfileStore((state) => state.showReviewKeys);
  // The shortcuts work while the review tab shows.
  const listening = useProfileStore((state) => state.detailOpen && state.detailTab === 'review');
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [limit, setLimit] = useState(30);
  // The card the shortcuts act on, and where it was. When it leaves the list, as a reviewed card leaves the
  // pending view, the card that takes its place is marked.
  const [cursor, setCursor] = useState<{ id: string; index: number } | null>(null);
  const found = cursor ? facts.findIndex((fact) => fact.id === cursor.id) : -1;
  const active = !facts.length ? -1 : found >= 0 ? found : Math.min(cursor?.index ?? 0, facts.length - 1);
  const activeId = facts[active]?.id;
  const listRef = useRef<HTMLDivElement>(null);
  const shown = Math.max(limit, active + 1);
  const page = facts.slice(0, shown);
  // Selecting all takes every fact of the list, also those further down than shown.
  const selected = facts.filter((fact) => checked.has(fact.id));
  const selectAllRef = useRef<HTMLInputElement>(null);
  const mixed = selected.length > 0 && selected.length < facts.length;
  useEffect(() => {
    const box = selectAllRef.current;
    if (box) box.indeterminate = mixed;
  }, [mixed]);
  function move(index: number) {
    const fact = facts[index];
    if (!fact) return;
    // Pressing next on the last card keeps it.
    setCursor({ id: fact.id, index });
    if (index >= limit) setLimit(limit + 30);
    // After the render that shows it.
    window.requestAnimationFrame(() =>
      listRef.current
        ?.querySelector(`[data-fact-id="${CSS.escape(fact.id)}"]`)
        ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }),
    );
  }
  const keyHandler = useRef<(event: KeyboardEvent) => void>(() => undefined);
  useEffect(() => {
    keyHandler.current = (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey || isTyping(event)) return;
      const key = event.key.toLowerCase();
      const fact = facts[active];
      if (!fact || !LIST_KEYS.has(key)) return;
      event.preventDefault();
      showReviewKeys();
      if (key === 'j') move(active + 1);
      else if (key === 'k') move(active - 1);
      else if (key === 'e') edit(fact);
      else if (key === 'x') toggle(fact.id, !checked.has(fact.id));
      else {
        const status = key === 'a' ? 'approved' : 'rejected';
        if (fact.status !== status) void review([fact], status);
        // On to the next card. On the last one the mark stays, and goes to the one before if this one leaves.
        move(active + 1);
      }
    };
  });
  useEffect(() => {
    if (!listening) return;
    const onKey = (event: KeyboardEvent) => keyHandler.current(event);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [listening]);
  function toggle(id: string, isChecked: boolean) {
    const next = new Set(checked);
    if (isChecked) next.add(id);
    else next.delete(id);
    setChecked(next);
  }
  async function batch(status: ReviewStatus) {
    if (!selected.length) return;
    const hidden = selected.length - page.filter((fact) => checked.has(fact.id)).length;
    const question =
      status === 'approved'
        ? t(($) => $.review.list.confirmApprove, { count: selected.length, hidden })
        : t(($) => $.review.list.confirmReject, { count: selected.length, hidden });
    if (hidden && !window.confirm(question)) return;
    void review(selected, status);
    setChecked(new Set());
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
            disabled={!selected.length}
            onClick={() => void batch('approved')}
          >
            {!!selected.length && <Check size={13} strokeWidth={1.8} aria-hidden="true" />}
            {t(($) => $.review.approve)}
          </button>
          <button
            type="button"
            className="wc-btn wc-btn--sm"
            disabled={!selected.length}
            onClick={() => void batch('rejected')}
          >
            {t(($) => $.review.reject)}
          </button>
        </div>
      )}
      {!facts.length && <p className="detail-hint">{t(($) => $.review.list.empty)}</p>}
      {!!facts.length && (
        <p className="review-keys" aria-label={t(($) => $.review.keys.label)}>
          {KEYS.map(([key, action]) => (
            <span key={key}>
              <kbd className="wc-kbd">{key}</kbd>
              {t(($) => $.review.keys[action])}
            </span>
          ))}
        </p>
      )}
      <div className="review-cards" ref={listRef}>
        {page.map((fact) => (
          <FactCard
            key={fact.id}
            fact={fact}
            marked={reviewKeys && fact.id === activeId}
            checked={checked.has(fact.id)}
            onCheckedChange={(isChecked) => toggle(fact.id, isChecked)}
          />
        ))}
      </div>
      {facts.length > shown && (
        <button type="button" className="wc-btn wc-btn--block load-more" onClick={() => setLimit(shown + 30)}>
          {t(($) => $.review.list.more, { count: facts.length - shown })}
        </button>
      )}
    </>
  );
}
