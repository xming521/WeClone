import { Lock, LockOpen } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { privacyLevel, SENSITIVITY_LEVELS } from '../data';
import type { PrivacyLevel, ProfileFact, Sensitivity } from '../data';

// The catalog labels the four levels of SENSITIVITY_RULES in weclone/prompts/memory_organization.py.
function isLevel(level: string | undefined): level is Sensitivity {
  return SENSITIVITY_LEVELS.includes(level as Sensitivity);
}

/** The privacy level of a fact or a source; nothing for profiles built before levels existed. */
export function PrivacyBadge({ level }: { level?: Sensitivity }) {
  const { t } = useTranslation();
  if (!isLevel(level)) return null;
  const label = t(($) => $.privacy.levels[level].label);
  const hint = t(($) => $.privacy.levels[level].hint);
  const Icon = level === 'public' ? LockOpen : Lock;
  return (
    <span
      className={`privacy-badge ${level}`}
      title={t(($) => $.privacy.badgeTitle, { hint })}
      aria-label={t(($) => $.privacy.badgeLabel, { label, hint })}
    >
      <Icon size={10} aria-hidden="true" />
      {label}
    </span>
  );
}

/** Tag chips, at most `limit` of them and a count of the rest. */
export function TagList({ tags, limit = Infinity }: { tags: string[]; limit?: number }) {
  const { t } = useTranslation();
  if (!tags.length) return null;
  const rest = tags.length - limit;
  return (
    <ul className="tag-list" aria-label={t(($) => $.privacy.tags)}>
      {tags.slice(0, limit).map((tag) => (
        <li key={tag}>{tag}</li>
      ))}
      {rest > 0 && (
        <li className="tag-more" title={tags.slice(limit).join(t(($) => $.common.listSeparator))}>
          +{rest}
        </li>
      )}
    </ul>
  );
}

/** How the facts in scope spread over the levels; nothing when none of them has a level. */
export function PrivacyOverview({ facts }: { facts: ProfileFact[] }) {
  const { t } = useTranslation();
  const counts = new Map<Sensitivity, number>();
  for (const fact of facts) {
    if (isLevel(fact.sensitivity)) {
      counts.set(fact.sensitivity, (counts.get(fact.sensitivity) ?? 0) + 1);
    }
  }
  const levelled = [...counts.values()].reduce((sum, count) => sum + count, 0);
  if (!levelled) return null;
  const parts = SENSITIVITY_LEVELS.filter((level) => counts.get(level)).map((level) =>
    t(($) => $.privacy.levelCount, { label: t(($) => $.privacy.levels[level].label), n: counts.get(level) ?? 0 }),
  );
  if (facts.length > levelled)
    parts.push(
      t(($) => $.privacy.levelCount, { label: t(($) => $.privacy.levels.unrated.label), n: facts.length - levelled }),
    );
  return (
    <section className="overview-privacy" aria-label={t(($) => $.privacy.overview)}>
      <div>
        <span>{t(($) => $.privacy.title)}</span>
        <span>
          {t(($) => $.privacy.privateCount, {
            n: (counts.get('private') ?? 0) + (counts.get('sensitive') ?? 0),
          })}
        </span>
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
  const { t } = useTranslation();
  const counts = new Map<PrivacyLevel, number>();
  for (const fact of facts) counts.set(privacyLevel(fact), (counts.get(privacyLevel(fact)) ?? 0) + 1);
  const levels = [...SENSITIVITY_LEVELS, 'unrated' as const].filter(
    (level) => counts.get(level) || value.includes(level),
  );
  if (!value.length && levels.every((level) => level === 'unrated')) return null;
  return (
    <div className="privacy-filter" role="group" aria-label={t(($) => $.privacy.filter)}>
      <span>{t(($) => $.privacy.title)}</span>
      {levels.map((level) => {
        const pressed = value.includes(level);
        return (
          <button
            key={level}
            type="button"
            className={`privacy-chip ${level}`}
            aria-pressed={pressed}
            title={t(($) => $.privacy.levels[level].hint)}
            onClick={() => onToggle(level)}
          >
            {t(($) => $.privacy.levels[level].label)}
            <span>{counts.get(level) ?? 0}</span>
          </button>
        );
      })}
    </div>
  );
}
