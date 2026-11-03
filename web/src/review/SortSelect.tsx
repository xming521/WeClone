import * as Select from '@radix-ui/react-select';
import { ArrowDownUp, ChevronDown } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { REVIEW_SORTS } from '../data';
import type { ReviewSort } from '../data';

export function SortSelect({ value, onChange }: { value: ReviewSort; onChange: (sort: ReviewSort) => void }) {
  const { t } = useTranslation();
  return (
    <Select.Root value={value} onValueChange={(sort) => onChange(sort as ReviewSort)}>
      <Select.Trigger
        className="wc-select review-sort"
        aria-label={t(($) => $.review.sort.label, { order: t(($) => $.review.sort.orders[value]) })}
        title={t(($) => $.review.sort.orders[value])}
      >
        <ArrowDownUp size={14} strokeWidth={1.8} aria-hidden="true" />
        <span>{t(($) => $.review.sort.short[value])}</span>
        <ChevronDown size={14} strokeWidth={1.8} aria-hidden="true" />
      </Select.Trigger>
      <Select.Portal>
        <Select.Content
          className="wc-menu review-filter-menu"
          position="popper"
          side="bottom"
          align="start"
          sideOffset={6}
          onEscapeKeyDown={(event) => event.stopPropagation()}
        >
          <Select.Viewport>
            <Select.Group>
              <Select.Label className="wc-menu__head">{t(($) => $.review.sort.title)}</Select.Label>
              {REVIEW_SORTS.map((sort) => (
                <Select.Item key={sort} value={sort} className="wc-menu__item" aria-checked={sort === value}>
                  <Select.ItemText>{t(($) => $.review.sort.orders[sort])}</Select.ItemText>
                </Select.Item>
              ))}
            </Select.Group>
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}
