import { ArrowUpRight, Check, MessageSquare, X } from 'lucide-react';
import { displayProfileText } from '../data';
import type { ProfileFact } from '../data';
import { useProfileStore } from '../store';
import { FactHistory } from './FactHistory';
import { FactScores, SourceScores } from './ProfileScores';
import { STATUS } from './status';

export function FactCard({
  fact,
  checked,
  onCheckedChange,
}: {
  fact: ProfileFact;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const sources = useProfileStore((state) => state.input?.sources);
  const busy = useProfileStore((state) => state.busy);
  const review = useProfileStore((state) => state.review);
  const edit = useProfileStore((state) => state.edit);
  const openChat = useProfileStore((state) => state.openChat);
  const chatId = useProfileStore((state) => state.chat?.id);
  // The domain view mixes dimensions under one topic, so each card names its own.
  const dimensionName = useProfileStore((state) =>
    state.viewModel?.grouping === 'domain'
      ? state.input?.dimensions.find((dimension) => dimension.dim === fact.dim)?.name
      : undefined,
  );
  const meta = [dimensionName, fact.origin === 'manual' && '手动添加'].filter(Boolean).join(' · ');
  return (
    <article className="fact-card">
      <div className="fact-heading">
        <label>
          <input
            className="review-checkbox"
            aria-label={`选择 ${fact.attr}`}
            type="checkbox"
            checked={checked}
            onChange={(event) => onCheckedChange(event.target.checked)}
          />{' '}
          {fact.attr}
        </label>
        <span className={`status-badge ${fact.status}`}>{STATUS[fact.status]}</span>
      </div>
      <p>{displayProfileText(fact.value)}</p>
      {meta && <small>{meta}</small>}
      <FactScores fact={fact} />
      <div className="fact-actions">
        {fact.status !== 'approved' && (
          <button className="adopt" disabled={busy} onClick={() => void review([fact], 'approved')}>
            <Check size={14} aria-hidden="true" />
            采纳
          </button>
        )}
        {fact.status !== 'rejected' && (
          <button disabled={busy} onClick={() => void review([fact], 'rejected')}>
            <X size={14} aria-hidden="true" />
            不采纳
          </button>
        )}
        {fact.status !== 'pending' && (
          <button disabled={busy} onClick={() => void review([fact], 'pending')}>
            移回待审核
          </button>
        )}
        <button disabled={busy} onClick={() => edit(fact)}>
          编辑
        </button>
      </div>
      <details className="evidence-details">
        <summary>{fact.origin === 'manual' ? '初始内容' : `原始抽取与 ${fact.source_ids.length} 条证据`}</summary>
        <div className="evidence-items">
          <div className="evidence-item">
            <strong>{fact.original.attr}</strong>
            <p>{fact.original.value}</p>
          </div>
          {fact.source_ids.map((id, index) => {
            const source = sources?.[id];
            const label = `来源 ${index + 1}`;
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
                    aria-label={`查看${label}的原始聊天记录`}
                    aria-pressed={chatId === id}
                    onClick={(event) => openChat({ id, label }, event.currentTarget)}
                  >
                    查看原文
                    <ArrowUpRight size={13} aria-hidden="true" />
                  </button>
                </div>
                <time className="evidence-time">{source?.sample_time?.slice(0, 10) ?? '未标注时间'}</time>
                <p>{source?.content ?? '来源正文不可用'}</p>
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
