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
  return (
    <span
      className="wc-privacy"
      data-privacy={level}
      title={t(($) => $.privacy.badgeTitle, { hint })}
      aria-label={t(($) => $.privacy.badgeLabel, { label, hint })}
    >
      <span className="wc-phase" aria-hidden="true" />
      {label}
    </span>
  );
}

/** Tags written as #text, at most `limit` of them and a count of the rest. */
export function TagList({ tags, limit = Infinity }: { tags: string[]; limit?: number }) {
  const { t } = useTranslation();
  if (!tags.length) return null;
  const rest = tags.length - limit;
  return (
    <ul className="wc-tags" aria-label={t(($) => $.privacy.tags)}>
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
      <span className="privacy-filter-title">{t(($) => $.privacy.title)}</span>
      {levels.map((level) => {
        const pressed = value.includes(level);
        return (
          <button
            key={level}
            type="button"
            className="wc-chip"
            // Unrated has no level: it takes the plain chip, without a water-level circle.
            data-privacy={level === 'unrated' ? undefined : level}
            aria-pressed={pressed}
            title={t(($) => $.privacy.levels[level].hint)}
            onClick={() => onToggle(level)}
          >
            {level !== 'unrated' && <span className="wc-phase" aria-hidden="true" />}
            {t(($) => $.privacy.levels[level].label)}
            <span className="wc-chip__n">{counts.get(level) ?? 0}</span>
          </button>
        );
      })}
    </div>
  );
}
