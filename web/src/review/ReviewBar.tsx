import * as Tooltip from '@radix-ui/react-tooltip';
import type { ReviewView } from '../data';
import { useProfileStore } from '../store';
import { STATUS, VIEW_ICONS } from './status';

export function ReviewBar() {
  const input = useProfileStore((state) => state.input)!;
  const currentView = useProfileStore((state) => state.view);
  const setView = useProfileStore((state) => state.setView);
  const counts = { all: input.facts.length, pending: 0, approved: 0, rejected: 0 };
  input.facts.forEach((fact) => counts[fact.status]++);
  return (
    <div className="review-bar">
      <nav className="review-tabs" aria-label="审核状态">
        {(Object.keys(STATUS) as ReviewView[]).map((view, index, views) => {
          const Icon = VIEW_ICONS[view];
          const label = STATUS[view];
          return (
            <Tooltip.Root key={view} delayDuration={160}>
              <Tooltip.Trigger asChild>
                <button
                  type="button"
                  aria-label={`${label} ${counts[view]} 条画像`}
                  aria-pressed={currentView === view}
                  className={`review-bookmark${currentView === view ? ' active' : ''}`}
                  onClick={() => setView(view)}
                  onKeyDown={(event) => {
                    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
                    event.preventDefault();
                    const next = (index + (event.key === 'ArrowDown' ? 1 : views.length - 1)) % views.length;
                    const bookmarks = event.currentTarget
                      .closest('nav')
                      ?.querySelectorAll<HTMLButtonElement>('.review-bookmark');
                    bookmarks?.[next]?.focus();
                    setView(views[next]);
                  }}
                >
                  <Icon size={32} strokeWidth={1.5} aria-hidden="true" />
                  <span className="review-bookmark-copy" aria-hidden="true">
                    <span>{label}</span>
                    <span className="review-bookmark-count">{counts[view]}</span>
                  </span>
                </button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content className="tooltip" side="right" sideOffset={12}>
                  {label} · {counts[view]} 条画像{view === 'approved' && ' · 分身画像'}
                  <Tooltip.Arrow className="tooltip-arrow" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          );
        })}
      </nav>
    </div>
  );
}
