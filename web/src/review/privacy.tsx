import { Lock, LockOpen } from 'lucide-react';
import { privacyLevel, SENSITIVITY_LEVELS } from '../data';
import type { PrivacyLevel, ProfileFact, Sensitivity } from '../data';

// Labels for the four levels of SENSITIVITY_RULES in weclone/prompts/memory_organization.py, least private first.
export const SENSITIVITY: Record<Sensitivity, { label: string; hint: string }> = {
  public: { label: '公开', hint: '对陌生人也会说' },
  normal: { label: '普通', hint: '熟人之间会聊' },
  private: { label: '私密', hint: '只对亲近的人说' },
  sensitive: { label: '敏感', hint: '只留在原来的聊天里' },
};

/** The privacy level of a fact or a source; nothing for profiles built before levels existed. */
export function PrivacyBadge({ level }: { level?: Sensitivity }) {
  if (!level || !(level in SENSITIVITY)) return null;
  const { label, hint } = SENSITIVITY[level];
  const Icon = level === 'public' ? LockOpen : Lock;
  return (
    <span className={`privacy-badge ${level}`} title={`隐私度：${hint}`} aria-label={`隐私度：${label}，${hint}`}>
      <Icon size={10} aria-hidden="true" />
      {label}
    </span>
  );
}

/** Tag chips, at most `limit` of them and a count of the rest. */
export function TagList({ tags, limit = Infinity }: { tags: string[]; limit?: number }) {
  if (!tags.length) return null;
  const rest = tags.length - limit;
  return (
    <ul className="tag-list" aria-label="检索标签">
      {tags.slice(0, limit).map((tag) => (
        <li key={tag}>{tag}</li>
      ))}
      {rest > 0 && (
        <li className="tag-more" title={tags.slice(limit).join('、')}>
          +{rest}
        </li>
      )}
    </ul>
  );
}

/** How the facts in scope spread over the levels; nothing when none of them has a level. */
export function PrivacyOverview({ facts }: { facts: ProfileFact[] }) {
  const counts = new Map<Sensitivity, number>();
  for (const fact of facts) {
    if (fact.sensitivity && fact.sensitivity in SENSITIVITY) {
      counts.set(fact.sensitivity, (counts.get(fact.sensitivity) ?? 0) + 1);
    }
  }
  const levelled = [...counts.values()].reduce((sum, count) => sum + count, 0);
  if (!levelled) return null;
  const parts = SENSITIVITY_LEVELS.filter((level) => counts.get(level)).map(
    (level) => `${SENSITIVITY[level].label} ${counts.get(level)}`,
  );
  if (facts.length > levelled) parts.push(`未分级 ${facts.length - levelled}`);
  return (
    <section className="overview-privacy" aria-label="隐私度分布">
      <div>
        <span>隐私度</span>
        <span>私密及敏感 {(counts.get('private') ?? 0) + (counts.get('sensitive') ?? 0)}</span>
      </div>
      <div className="privacy-track" aria-hidden="true">
        {SENSITIVITY_LEVELS.map((level) =>
          counts.get(level) ? <span key={level} className={level} style={{ flexGrow: counts.get(level) }} /> : null,
        )}
      </div>
      <p>{parts.join(' · ')}</p>
    </section>
  );
}

/**
 * Toggles that narrow the review list to some privacy levels, each with its count among `facts`.
 * Only levels present show, plus chosen ones so they can be turned off; nothing shows without levels.
 */
export function PrivacyFilter({
  facts,
  value,
  onToggle,
}: {
  facts: ProfileFact[];
  value: PrivacyLevel[];
  onToggle: (level: PrivacyLevel) => void;
}) {
  const counts = new Map<PrivacyLevel, number>();
  for (const fact of facts) counts.set(privacyLevel(fact), (counts.get(privacyLevel(fact)) ?? 0) + 1);
  const levels = [...SENSITIVITY_LEVELS, 'unrated' as const].filter(
    (level) => counts.get(level) || value.includes(level),
  );
  if (!value.length && levels.every((level) => level === 'unrated')) return null;
  return (
    <div className="privacy-filter" role="group" aria-label="按隐私度筛选，不选显示全部">
      <span>隐私度</span>
      {levels.map((level) => {
        const pressed = value.includes(level);
        const { label, hint } = level === 'unrated' ? { label: '未分级', hint: '没有隐私度' } : SENSITIVITY[level];
        return (
          <button
            key={level}
            type="button"
            className={`privacy-chip ${level}`}
            aria-pressed={pressed}
            title={hint}
            onClick={() => onToggle(level)}
          >
            {label}
            <span>{counts.get(level) ?? 0}</span>
          </button>
        );
      })}
    </div>
  );
}
