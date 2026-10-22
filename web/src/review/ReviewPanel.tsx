import { useEffect, useRef, useState } from 'react';
import { collectFacts, filterPrivacy } from '../data';
import type { ProfileFact, ReviewStatus } from '../data';
import { useProfileStore } from '../store';
import { FactCard } from './FactCard';
import { PrivacyFilter } from './privacy';
import { ReviewFilter } from './ReviewFilter';

export function ReviewPanel() {
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
        <span>{facts.length} 条画像</span>
      </div>
      <PrivacyFilter facts={scoped} value={privacy} onToggle={togglePrivacy} />
      <ReviewList key={`${selectedId}:${view}:${privacy.join()}`} facts={facts} />
    </section>
  );
}

function ReviewList({ facts }: { facts: ProfileFact[] }) {
  const busy = useProfileStore((state) => state.busy);
  const review = useProfileStore((state) => state.review);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [limit, setLimit] = useState(30);
  const page = facts.slice(0, limit);
  // Selecting all takes every fact of the list, also those further down than shown.
  const selected = facts.filter((fact) => checked.has(fact.id));
  const selectAllRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const box = selectAllRef.current;
    if (box) box.indeterminate = selected.length > 0 && selected.length < facts.length;
  }, [selected.length, facts.length]);
  async function batch(status: ReviewStatus) {
    if (!selected.length) return;
    const hidden = selected.length - page.filter((fact) => checked.has(fact.id)).length;
    const action = status === 'approved' ? '采纳' : '不采纳';
    if (hidden && !window.confirm(`${action}选中的 ${selected.length} 条画像？其中 ${hidden} 条还没有显示在列表里。`)) {
      return;
    }
    if (await review(selected, status)) setChecked(new Set());
  }
  return (
    <>
      {!!facts.length && (
        <div className="batch-toolbar">
          <label>
            <input
              ref={selectAllRef}
              className="review-checkbox"
              type="checkbox"
              checked={selected.length === facts.length}
              onChange={(event) => setChecked(new Set(event.target.checked ? facts.map((fact) => fact.id) : []))}
            />
            全选 {facts.length} 条
          </label>
          <button disabled={busy || !selected.length} onClick={() => void batch('approved')}>
            采纳 {selected.length || ''}
          </button>
          <button disabled={busy || !selected.length} onClick={() => void batch('rejected')}>
            不采纳
          </button>
        </div>
      )}
      {!facts.length && <p className="detail-hint">当前视图没有画像记录。</p>}
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
        <button className="load-more" onClick={() => setLimit(limit + 30)}>
          继续显示（剩余 {facts.length - limit} 条）
        </button>
      )}
    </>
  );
}
