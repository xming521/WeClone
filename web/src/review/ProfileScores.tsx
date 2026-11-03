import { Flag, ShieldCheck } from 'lucide-react';
import type { CSSProperties } from 'react';
import { useTranslation } from 'react-i18next';
import type { ProfileFact, ProfileSource } from '../data';

// 量表来源：weclone/prompts/distill_memory.py（importance 1-4、confidence 2-4）及 weclone/data/agent/distill_windows.py 的 validate_memory。
// 可信度从 2 起：抽取量表没有 1 档，证据不足以支持的记忆不会被抽取，因此 2 是最低有效分。
const SCORE_RANGES = {
  confidence: { min: 2, max: 4 },
  importance: { min: 1, max: 4 },
} as const;

/** A score on its scale. Compact scores leave out the icon and show whole numbers as such, averages to one decimal. */
export function Score({
  kind,
  value,
  compact = false,
}: {
  kind: 'confidence' | 'importance';
  value?: number;
  compact?: boolean;
}) {
  const { t } = useTranslation();
  const label = kind === 'confidence' ? t(($) => $.scores.confidence) : t(($) => $.scores.importance);
  const Icon = kind === 'confidence' ? ShieldCheck : Flag;
  const { min, max } = SCORE_RANGES[kind];
  const valid = typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
  const score = valid ? value : null;
  const shown = score === null ? t(($) => $.scores.unrated) : score.toFixed(compact && Number.isInteger(score) ? 0 : 1);
  return (
    <div
      className="wc-score"
      data-kind={kind}
      style={{ '--v': score ?? 0 } as CSSProperties}
      aria-label={t(($) => $.scores.label, { label, score: score === null ? shown : `${shown} / ${max}` })}
    >
      <span className="wc-score__label">
        {!compact && <Icon size={13} strokeWidth={1.8} aria-hidden="true" />}
        {label}
      </span>
      <span className="wc-pips" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </span>
      <span className="wc-score__val">
        {score === null ? shown : <b>{shown}</b>}
        {score !== null && `/${max}`}
      </span>
    </div>
  );
}

export function FactScores({ fact }: { fact: ProfileFact }) {
  const { t } = useTranslation();
  return (
    <div className="wc-memory__scores" aria-label={t(($) => $.scores.factLabel)}>
      <Score kind="confidence" value={fact.origin === 'manual' ? undefined : fact.source_confidence_mean} />
      <Score kind="importance" value={fact.origin === 'manual' ? undefined : fact.source_importance_mean} />
      {typeof fact.support_count === 'number' && fact.support_count > 0 && (
        <div className="score-support">{t(($) => $.scores.support, { count: fact.support_count })}</div>
      )}
    </div>
  );
}

export function SourceScores({ source }: { source?: ProfileSource }) {
  return (
    <div className="source-scores">
      <Score kind="confidence" value={source?.confidence} compact />
      <Score kind="importance" value={source?.importance} compact />
    </div>
  );
}
