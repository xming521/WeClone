import * as Select from '@radix-ui/react-select';
import { ChevronDown, Layers } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { ReviewView } from '../data';
import { MARK_STATUS, REVIEW_VIEWS } from './status';

/** The icon of "all", else the status mark of the view. */
function ViewMark({ view }: { view: ReviewView }) {
  if (view === 'all') return <Layers size={15} strokeWidth={1.8} aria-hidden="true" />;
  return <span className="wc-glyph" data-status={MARK_STATUS[view]} aria-hidden="true" />;
}

const countClass = (view: ReviewView) => (view === 'pending' ? 'wc-count wc-count--pending' : 'wc-count');

export function ReviewFilter({
  value,
  counts,
  onChange,
}: {
  value: ReviewView;
  counts: Record<ReviewView, number>;
  onChange: (view: ReviewView) => void;
}) {
  const { t } = useTranslation();
  return (
    <Select.Root value={value} onValueChange={(view) => onChange(view as ReviewView)}>
      <Select.Trigger
        className="wc-select"
        aria-label={t(($) => $.review.filterLabel, { label: t(($) => $.review.status[value]), count: counts[value] })}
      >
        <ViewMark view={value} />
        <span>{t(($) => $.review.status[value])}</span>
        <span className={countClass(value)}>{counts[value]}</span>
        <ChevronDown size={14} strokeWidth={1.8} aria-hidden="true" />
      </Select.Trigger>
      <Select.Portal>
        <Select.Content
          className="wc-menu review-filter-menu"
          position="popper"
          side="bottom"
          align="start"
          sideOffset={6}
          aria-label={t(($) => $.review.statusLabel)}
          onEscapeKeyDown={(event) => event.stopPropagation()}
        >
          <Select.Viewport>
            <Select.Group>
              <Select.Label className="wc-menu__head">{t(($) => $.review.statusLabel)}</Select.Label>
              {REVIEW_VIEWS.map((view) => (
                <Select.Item key={view} value={view} className="wc-menu__item" aria-checked={view === value}>
                  <ViewMark view={view} />
                  <Select.ItemText>{t(($) => $.review.status[view])}</Select.ItemText>
                  <span className={countClass(view)}>{counts[view]}</span>
                </Select.Item>
              ))}
            </Select.Group>
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}
