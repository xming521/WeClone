import { ArrowUpRight, Check, ChevronRight, MessageSquare, MessagesSquare, Pencil, Undo2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { displayProfileText, factTags, isDimensionId, latestSourceTime } from '../data';
import type { ProfileFact } from '../data';
import { useProfileStore } from '../store';
import { FactHistory } from './FactHistory';
import { PrivacyBadge, TagList } from './privacy';
import { FactScores, SourceScores } from './ProfileScores';
import { MARK_STATUS } from './status';

export function FactCard({
  fact,
  marked = false,
  checked,
  onCheckedChange,
}: {
  fact: ProfileFact;
  /** The card the review shortcuts act on. */
  marked?: boolean;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const { t } = useTranslation();
  const sources = useProfileStore((state) => state.input?.sources);
  const review = useProfileStore((state) => state.review);
  const edit = useProfileStore((state) => state.edit);
  const openChat = useProfileStore((state) => state.openChat);
  const chatId = useProfileStore((state) => state.chat?.id);
  // The domain view mixes dimensions under one topic, so each card names its own.
  const byDomain = useProfileStore((state) => state.viewModel?.grouping === 'domain');
  const dim = fact.dim;
  const dimensionName = byDomain && isDimensionId(dim) ? t(($) => $.model.dimensions[dim]) : undefined;
  const meta = [dimensionName, fact.origin === 'manual' && t(($) => $.review.fact.manual)].filter(Boolean).join(' · ');
  const status = MARK_STATUS[fact.status];
  const latest = latestSourceTime(fact, sources)?.slice(0, 10);
  return (
    <article
      className="wc-memory fact-card"
      data-status={status}
      data-fact-id={fact.id}
      data-marked={marked || undefined}
      aria-current={marked || undefined}
      aria-label={fact.attr}
    >
      <div className="wc-memory__head">
        <label className="wc-check">
          <input
            type="checkbox"
            aria-label={t(($) => $.review.fact.select, { attr: fact.attr })}
            checked={checked}
            onChange={(event) => onCheckedChange(event.target.checked)}
          />
          <span className="wc-check__box" aria-hidden="true" />
          {fact.attr}
        </label>
        <PrivacyBadge level={fact.sensitivity} />
        <span className="wc-status" data-status={status}>
          {t(($) => $.review.status[fact.status])}
        </span>
      </div>
      <p className="wc-memory__value">{displayProfileText(fact.value)}</p>
      {meta && <small className="fact-meta">{meta}</small>}
      <TagList tags={factTags(fact, sources)} limit={5} />
      <FactScores fact={fact} />
      <div className="wc-memory__actions">
        {fact.status === 'pending' && (
          <button
            type="button"
            className="wc-btn wc-btn--adopt wc-btn--sm"
            onClick={() => void review([fact], 'approved')}
          >
            <Check size={13} strokeWidth={1.8} aria-hidden="true" />
            {t(($) => $.review.approve)}
          </button>
        )}
        {fact.status !== 'pending' && (
          <button type="button" className="wc-btn wc-btn--sm" onClick={() => void review([fact], 'pending')}>
            <Undo2 size={13} strokeWidth={1.8} aria-hidden="true" />
            {t(($) => $.review.fact.unreview)}
          </button>
        )}
        {fact.status === 'rejected' && (
          <button type="button" className="wc-btn wc-btn--sm" onClick={() => void review([fact], 'approved')}>
            {t(($) => $.review.approve)}
          </button>
        )}
        {fact.status !== 'rejected' && (
          <button type="button" className="wc-btn wc-btn--sm" onClick={() => void review([fact], 'rejected')}>
            {t(($) => $.review.reject)}
          </button>
        )}
        <button type="button" className="wc-btn wc-btn--ghost wc-btn--sm" onClick={() => edit(fact)}>
          <Pencil size={13} strokeWidth={1.8} aria-hidden="true" />
          {t(($) => $.review.fact.edit)}
        </button>
      </div>
      <details className="evidence-details">
        <summary className="wc-memory__evidence">
          <MessagesSquare size={14} strokeWidth={1.8} aria-hidden="true" />
          <span className="evidence-label">
            {fact.origin === 'manual'
              ? t(($) => $.review.fact.initial)
              : t(($) => $.review.fact.evidence, { count: fact.source_ids.length })}
          </span>
          {latest && <time dateTime={latest}>{t(($) => $.review.fact.latest, { date: latest })}</time>}
          <ChevronRight className="evidence-chevron" size={14} strokeWidth={1.8} aria-hidden="true" />
        </summary>
        <div className="evidence-items">
          <div className="evidence-item">
            <strong>{fact.original.attr}</strong>
            <p>{fact.original.value}</p>
          </div>
          {fact.source_ids.map((id, index) => {
            const source = sources?.[id];
            const label = t(($) => $.review.fact.source, { n: index + 1 });
            return (
              <div className="evidence-item" key={id}>
                <div className="evidence-source-header">
                  <span>
                    <MessageSquare size={13} strokeWidth={1.8} aria-hidden="true" />
                    {label}
                  </span>
                  <button
                    type="button"
                    className="wc-btn wc-btn--sm view-source-chat"
                    disabled={!source}
                    aria-label={t(($) => $.review.fact.viewChatLabel, { label })}
                    aria-pressed={chatId === id}
                    onClick={(event) => openChat({ id, label }, event.currentTarget)}
                  >
                    {t(($) => $.review.fact.viewChat)}
                    <ArrowUpRight size={13} strokeWidth={1.8} aria-hidden="true" />
                  </button>
                </div>
                <div className="evidence-meta">
                  <time className="evidence-time">
                    {source?.sample_time?.slice(0, 10) ?? t(($) => $.review.fact.noTime)}
                  </time>
                  <PrivacyBadge level={source?.sensitivity} />
                </div>
                <p>{source?.content ?? t(($) => $.review.fact.noContent)}</p>
                <TagList tags={source?.tags ?? []} />
                <SourceScores source={source} />
              </div>
            );
          })}
        </div>
      </details>
      <FactHistory key={`${fact.id}:${fact.version}`} fact={fact} />
    </article>
  );
}
