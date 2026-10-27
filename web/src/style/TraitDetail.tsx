import { useState, type CSSProperties } from 'react';
import { Sparkles, X } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { TRAIT_GROUPS, formatPercent, tierOf, type Trait } from './traits';
import type { VariationClass } from './types';

export function TraitDetail({ trait: open, onClose }: { trait: Trait | null; onClose: () => void }) {
  const { t } = useTranslation();
  // Keep the last trait on screen while the panel slides out.
  const [shown, setShown] = useState(open);
  if (open && open !== shown) setShown(open);
  const trait = open ?? shown;
  const group = TRAIT_GROUPS.find((entry) => entry.id === trait?.group);
  const tier = trait ? tierOf(trait.rate) : null;
  const amount = trait ? t(($) => $.style.units[trait.unit ?? 'messages'], { count: trait.messages }) : '';
  return (
    <aside
      className={`detail-panel trait-detail${open ? ' is-open' : ''}`}
      aria-hidden={!open}
      style={{ '--bubble-color': group?.color } as CSSProperties}
    >
      <button className="detail-close" type="button" aria-label={t(($) => $.atlas.detail.close)} onClick={onClose}>
        <X size={17} />
      </button>
      <div className="detail-topline">
        <span className="detail-kicker">
          <Sparkles size={14} /> {t(($) => $.style.detail.kicker)}
        </span>
        <span className="detail-serial">{group && t(($) => $.style.groups[group.id].name)}</span>
      </div>
      {trait && (
        <div className="detail-content">
          <div className="trait-hero">
            {/* The opened bubble again, popping in each time another one is chosen. */}
            <div key={trait.key} className={`trait-emblem tier-${tier?.id}`}>
              <span className="trait-bubble-body">
                <strong>{formatPercent(trait.rate)}</strong>
              </span>
            </div>
            <div>
              <h2 className="trait-title">{t(($) => $.style.traits[trait.id].title)}</h2>
              <p className="trait-note">{t(($) => $.style.traits[trait.id].note)}</p>
            </div>
          </div>
          <p className="trait-frequency">
            <Trans
              i18nKey={trait.when ? ($) => $.style.detail.frequencyWhen : ($) => $.style.detail.frequency}
              values={{
                tier: tier ? t(($) => $.style.tiers[tier.id]) : '',
                when: trait.when ? t(($) => $.style.when[trait.when!]) : '',
                rate:
                  trait.rate >= 0.95
                    ? t(($) => $.style.detail.almostAll)
                    : trait.rate >= 0.3
                      ? t(($) => $.style.detail.outOfTen, { count: Math.round(trait.rate * 10) })
                      : t(($) => $.style.detail.onceEvery, { every: Math.round(1 / trait.rate) }),
              }}
              components={{ b: <b /> }}
            />
          </p>

          {trait.examples.length > 0 && (
            <section className="trait-section">
              <h3>{t(($) => $.style.detail.examples)}</h3>
              <div className="trait-examples">
                {trait.examples.map((example) => (
                  <span key={example}>{example}</span>
                ))}
              </div>
            </section>
          )}

          {trait.contacts.length > 1 && (
            <section className="trait-section">
              <h3>{t(($) => $.style.detail.contacts)}</h3>
              {trait.variation && trait.variation !== 'insufficient' && (
                <p className="trait-variation">
                  {t(($) => $.style.variation[trait.variation as Exclude<VariationClass, 'insufficient'>])}
                </p>
              )}
              <div className="trait-contacts">
                {trait.contacts.map(({ contact, rate }) => (
                  <div key={contact.id} className="trait-contact">
                    <span className="trait-contact-name" title={contact.label}>
                      {contact.label}
                    </span>
                    <span className="trait-contact-bar">
                      <i style={{ width: `${rate * 100}%` }} />
                      <b style={{ left: `${trait.rate * 100}%` }} />
                    </span>
                    <span className="trait-contact-value">{formatPercent(rate)}</span>
                  </div>
                ))}
              </div>
              <p className="trait-source">{t(($) => $.style.detail.contactsNote)}</p>
            </section>
          )}

          <p className="trait-source">
            {trait.sampled
              ? t(($) => $.style.detail.sampled, { amount })
              : t(($) => $.style.detail.counted, { amount })}
          </p>
        </div>
      )}
    </aside>
  );
}
