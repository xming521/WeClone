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

/** A score on its scale. Compact scores show whole numbers as such and averages to one decimal. */
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
      className={`profile-score ${kind}${compact ? ' compact' : ''}`}
      aria-label={t(($) => $.scores.label, { label, score: score === null ? shown : `${shown} / ${max}` })}
    >
      <span className="score-label">
        {!compact && <Icon size={14} aria-hidden="true" />}
        {label}
      </span>
      {!compact && (
        <div className="score-value">
          {score === null ? (
            <span className="score-missing">{t(($) => $.scores.unrated)}</span>
          ) : (
            <>
              <strong>{score.toFixed(1)}</strong>
              <span>{`/ ${max}`}</span>
            </>
          )}
        </div>
      )}
      <span className="score-scale" aria-hidden="true">
        {Array.from({ length: max }, (_, index) => (
          <span className="score-segment" key={index}>
            <span
              style={{ '--score-fill': `${Math.max(0, Math.min(1, (score ?? 0) - index)) * 100}%` } as CSSProperties}
            />
          </span>
        ))}
      </span>
      {compact && <span className="source-score-value">{shown}</span>}
    </div>
  );
}

export function FactScores({ fact }: { fact: ProfileFact }) {
  const { t } = useTranslation();
  return (
    <div className="fact-scores" aria-label={t(($) => $.scores.factLabel)}>
      <div className="fact-score-grid">
        <Score kind="confidence" value={fact.origin === 'manual' ? undefined : fact.source_confidence_mean} />
        <Score kind="importance" value={fact.origin === 'manual' ? undefined : fact.source_importance_mean} />
      </div>
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
