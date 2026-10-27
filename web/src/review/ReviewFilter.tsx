import * as Select from '@radix-ui/react-select';
import { Check, ChevronDown } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { ReviewView } from '../data';
import { REVIEW_VIEWS, VIEW_ICONS } from './status';

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
  const Icon = VIEW_ICONS[value];
  return (
    <Select.Root value={value} onValueChange={(view) => onChange(view as ReviewView)}>
      <Select.Trigger
        className="review-filter-trigger"
        aria-label={t(($) => $.review.filterLabel, { label: t(($) => $.review.status[value]), count: counts[value] })}
      >
        <Icon size={15} aria-hidden="true" />
        <span>{t(($) => $.review.status[value])}</span>
        <span className="review-filter-count">{counts[value]}</span>
        <ChevronDown className="review-filter-chevron" size={14} aria-hidden="true" />
      </Select.Trigger>
      <Select.Portal>
        <Select.Content
          className="review-filter-menu"
          position="popper"
          side="bottom"
          align="start"
          sideOffset={7}
          aria-label={t(($) => $.review.statusLabel)}
          onEscapeKeyDown={(event) => event.stopPropagation()}
        >
          <Select.Viewport>
            {REVIEW_VIEWS.map((view) => {
              const OptionIcon = VIEW_ICONS[view];
              return (
                <Select.Item key={view} value={view} className="review-filter-item">
                  <OptionIcon size={15} aria-hidden="true" />
                  <Select.ItemText asChild>
                    <span className="review-filter-item-text">
                      <span className="review-filter-name">{t(($) => $.review.status[view])}</span>
                      <span className="review-filter-count">{counts[view]}</span>
                    </span>
                  </Select.ItemText>
                  <span className="review-filter-check">
                    <Select.ItemIndicator>
                      <Check size={14} aria-hidden="true" />
                    </Select.ItemIndicator>
                  </span>
                </Select.Item>
              );
            })}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}
