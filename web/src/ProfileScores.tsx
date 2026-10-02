import { Flag, ShieldCheck } from 'lucide-react';
import type { CSSProperties } from 'react';
import type { ProfileFact, ProfileSource } from './data';

function Score({ kind, value, compact = false }: { kind: 'confidence' | 'importance'; value?: number; compact?: boolean }) {
  const label = kind === 'confidence' ? '可信度' : '重要度';
  const Icon = kind === 'confidence' ? ShieldCheck : Flag;
  const valid = typeof value === 'number' && Number.isFinite(value) && value >= (kind === 'confidence' ? 2 : 1) && value <= 4;
  const score = valid ? value : null;
  return <div className={`profile-score ${kind}${compact ? ' compact' : ''}`} aria-label={`${label}：${score === null ? '未评分' : `${score.toFixed(compact ? 0 : 1)} / 4`}`}>
    <span className="score-label">{!compact && <Icon size={14} aria-hidden="true" />}{label}</span>
    {!compact && <div className="score-value">{score === null ? <span className="score-missing">未评分</span> : <><strong>{score.toFixed(1)}</strong><span>/ 4</span></>}</div>}
    <span className="score-scale" aria-hidden="true">{[0, 1, 2, 3].map(index => <span className="score-segment" key={index}>
      <span style={{ '--score-fill': `${Math.max(0, Math.min(1, (score ?? 0) - index)) * 100}%` } as CSSProperties} />
    </span>)}</span>
    {compact && <span className="source-score-value">{score === null ? '未评分' : score.toFixed(0)}</span>}
  </div>;
}

export function FactScores({ fact }: { fact: ProfileFact }) {
  return <div className="fact-scores" aria-label="原始来源的平均评分，编辑画像后不会重新评分">
    {typeof fact.support_count === 'number' && fact.support_count > 0 && <div className="score-support">{fact.support_count} 段聊天支持</div>}
    <div className="fact-score-grid">
      <Score kind="confidence" value={fact.origin === 'manual' ? undefined : fact.source_confidence_mean} />
      <Score kind="importance" value={fact.origin === 'manual' ? undefined : fact.source_importance_mean} />
    </div>
  </div>;
}

export function SourceScores({ source }: { source?: ProfileSource }) {
  return <div className="source-scores"><Score kind="confidence" value={source?.confidence} compact /><Score kind="importance" value={source?.importance} compact /></div>;
}
