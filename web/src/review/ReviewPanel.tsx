import { useEffect, useRef, useState } from 'react';
import { collectFacts } from '../data';
import type { ProfileFact, ReviewStatus } from '../data';
import { useProfileStore } from '../store';
import { FactCard } from './FactCard';
import { ReviewFilter } from './ReviewFilter';

export function ReviewPanel() {
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const fullModel = useProfileStore((state) => state.fullModel)!;
  const view = useProfileStore((state) => state.view);
  const setView = useProfileStore((state) => state.setView);
  const selectedId = useProfileStore((state) => state.selectedId) ?? 'root';
  const facts = collectFacts(viewModel, selectedId);
  const all = collectFacts(fullModel, selectedId);
  const counts = { pending: 0, approved: 0, rejected: 0 };
  all.forEach((fact) => counts[fact.status]++);
  return (
    <section className="detail-section review-list">
      <div className="review-list-toolbar">
        <ReviewFilter value={view} counts={{ all: all.length, ...counts }} onChange={setView} />
        <span>{facts.length} 条画像</span>
      </div>
      <ReviewList key={`${selectedId}:${view}`} facts={facts} />
    </section>
  );
}

function ReviewList({ facts }: { facts: ProfileFact[] }) {
  const busy = useProfileStore((state) => state.busy);
  const review = useProfileStore((state) => state.review);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [limit, setLimit] = useState(30);
  const page = facts.slice(0, limit);
  const selected = page.filter((fact) => checked.has(fact.id));
  const selectAllRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (selectAllRef.current) selectAllRef.current.indeterminate = selected.length > 0 && selected.length < page.length;
  }, [selected.length, page.length]);
  async function batch(status: ReviewStatus) {
    if (!selected.length) return;
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
              checked={page.length > 0 && selected.length === page.length}
              onChange={(event) => setChecked(new Set(event.target.checked ? page.map((fact) => fact.id) : []))}
            />
            选中已展示 {page.length} 条
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
