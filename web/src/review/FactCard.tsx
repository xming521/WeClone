import { ArrowUpRight, Check, MessageSquare, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { displayProfileText, factTags, isDimensionId } from '../data';
import type { ProfileFact } from '../data';
import { useProfileStore } from '../store';
import { FactHistory } from './FactHistory';
import { PrivacyBadge, TagList } from './privacy';
import { FactScores, SourceScores } from './ProfileScores';

export function FactCard({
  fact,
  checked,
  onCheckedChange,
}: {
  fact: ProfileFact;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const { t } = useTranslation();
  const sources = useProfileStore((state) => state.input?.sources);
  const busy = useProfileStore((state) => state.busy);
  const review = useProfileStore((state) => state.review);
  const edit = useProfileStore((state) => state.edit);
  const openChat = useProfileStore((state) => state.openChat);
  const chatId = useProfileStore((state) => state.chat?.id);
  // The domain view mixes dimensions under one topic, so each card names its own.
  const byDomain = useProfileStore((state) => state.viewModel?.grouping === 'domain');
  const dim = fact.dim;
  const dimensionName = byDomain && isDimensionId(dim) ? t(($) => $.model.dimensions[dim]) : undefined;
  const meta = [dimensionName, fact.origin === 'manual' && t(($) => $.review.fact.manual)].filter(Boolean).join(' · ');
  return (
    <article className="fact-card">
      <div className="fact-heading">
        <label>
          <input
            className="review-checkbox"
            aria-label={t(($) => $.review.fact.select, { attr: fact.attr })}
            type="checkbox"
            checked={checked}
            onChange={(event) => onCheckedChange(event.target.checked)}
          />{' '}
          {fact.attr}
        </label>
        <span className="fact-badges">
          <PrivacyBadge level={fact.sensitivity} />
          <span className={`status-badge ${fact.status}`}>{t(($) => $.review.status[fact.status])}</span>
        </span>
      </div>
      <p>{displayProfileText(fact.value)}</p>
      {meta && <small>{meta}</small>}
      <TagList tags={factTags(fact, sources)} limit={5} />
      <FactScores fact={fact} />
      <div className="fact-actions">
        {fact.status !== 'approved' && (
          <button className="adopt" disabled={busy} onClick={() => void review([fact], 'approved')}>
            <Check size={14} aria-hidden="true" />
            {t(($) => $.review.approve)}
          </button>
        )}
        {fact.status !== 'rejected' && (
          <button disabled={busy} onClick={() => void review([fact], 'rejected')}>
            <X size={14} aria-hidden="true" />
            {t(($) => $.review.reject)}
          </button>
        )}
        {fact.status !== 'pending' && (
          <button disabled={busy} onClick={() => void review([fact], 'pending')}>
            {t(($) => $.review.fact.unreview)}
          </button>
        )}
        <button disabled={busy} onClick={() => edit(fact)}>
          {t(($) => $.review.fact.edit)}
        </button>
      </div>
      <details className="evidence-details">
        <summary>
          {fact.origin === 'manual'
            ? t(($) => $.review.fact.initial)
            : t(($) => $.review.fact.evidence, { count: fact.source_ids.length })}
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
                    <MessageSquare size={13} aria-hidden="true" />
                    {label}
                  </span>
                  <button
                    type="button"
                    className="view-source-chat"
                    disabled={!source}
                    aria-label={t(($) => $.review.fact.viewChatLabel, { label })}
                    aria-pressed={chatId === id}
                    onClick={(event) => openChat({ id, label }, event.currentTarget)}
                  >
                    {t(($) => $.review.fact.viewChat)}
                    <ArrowUpRight size={13} aria-hidden="true" />
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
