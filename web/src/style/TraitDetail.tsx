import { useState } from 'react';
import { X } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { BrandMark } from '../BrandMark';
import { TRAIT_GROUPS, formatPercent, groupVars, tierOf, type Trait } from './traits';
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
      data-sec=""
      style={group ? groupVars(group.id) : undefined}
    >
      <div className="wc-panelhead">
        <span className="wc-kicker">{t(($) => $.style.detail.kicker)}</span>
        <span className="wc-serial">WECLONE / 2026</span>
        <button
          className="wc-iconbtn wc-iconbtn--sm"
          type="button"
          aria-label={t(($) => $.atlas.detail.close)}
          onClick={onClose}
        >
          <X size={16} strokeWidth={1.8} />
        </button>
      </div>
      {trait && (
        <div className="detail-content detail-scroll">
          <div className="detail-scope-heading">
            {group && (
              <ol className="wc-crumb detail-crumb">
                <li>
                  <span className="wc-dot" aria-hidden="true" />
                </li>
                <li className="detail-crumb-step">
                  <span>{t(($) => $.style.groups[group.id].name)}</span>
                </li>
              </ol>
            )}
            <div className="trait-hero">
              {/* The opened bubble again, popping in each time another one is chosen. */}
              <div key={trait.key} className={`trait-emblem tier-${tier?.id}`}>
                <span className="trait-bubble-body">
                  <strong>{formatPercent(trait.rate)}</strong>
                </span>
              </div>
              <h2>{t(($) => $.style.traits[trait.id].title)}</h2>
            </div>
            <p className="detail-description">{t(($) => $.style.traits[trait.id].note)}</p>
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
            <section className="detail-block">
              <div className="detail-block-head">
                <h3>{t(($) => $.style.detail.examples)}</h3>
              </div>
              <div className="trait-examples">
                {trait.examples.map((example) => (
                  <span key={example} className="wc-bubble" data-from="you">
                    {example}
                  </span>
                ))}
              </div>
            </section>
          )}

          {trait.contacts.length > 1 && (
            <section className="detail-block">
              <div className="detail-block-head">
                <h3>{t(($) => $.style.detail.contacts)}</h3>
              </div>
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
      <div className="detail-footer">
        <span className="wc-serial">{t(($) => $.atlas.detail.footer)}</span>
        <BrandMark variant="favicon" />
      </div>
    </aside>
  );
}
