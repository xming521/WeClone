import { useEffect, useMemo, useState } from 'react';
import type { LucideIcon } from 'lucide-react';
import { Info, Keyboard, MessagesSquare, Smile, Type } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { api, errorCode } from '../api';
import { LogoutButton } from '../Auth';
import { AppBrand, BrandMark } from '../BrandMark';
import { BuildIndicator } from '../build/BuildIndicator';
import { LanguageSwitch } from '../LanguageSwitch';
import { SettingsLink } from '../settings/SettingsLink';
import { ModeSwitch } from '../ModeSwitch';
import { BubbleField } from './BubbleField';
import { TraitDetail } from './TraitDetail';
import { MIN_RATE, TIERS, TRAIT_GROUPS, groupVars, selectTraits, type TraitGroupId } from './traits';
import type { StyleInput } from './types';
import './styleView.css';

const GROUP_ICONS: Record<TraitGroupId, LucideIcon> = {
  typing: Keyboard,
  words: Type,
  emotion: Smile,
  manner: MessagesSquare,
};

export function StyleApp() {
  const { t } = useTranslation();
  const [data, setData] = useState<StyleInput | null>(null);
  const [error, setError] = useState('');
  // A run without a style, or no run at all, is fixed by a build.
  const [buildable, setBuildable] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [focus, setFocus] = useState<TraitGroupId | null>(null);

  useEffect(() => {
    let live = true;
    api
      .get<StyleInput>('style')
      .then((value) => {
        if (live) setData(value);
      })
      .catch((reason: Error) => {
        if (!live) return;
        setError(reason.message);
        setBuildable(['style_missing', 'profile_missing'].includes(errorCode(reason) ?? ''));
      });
    return () => {
      live = false;
    };
  }, []);

  const traits = useMemo(() => (data ? selectTraits(data) : []), [data]);

  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelected(null);
    };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, []);

  if (error)
    return (
      <div className="load-state">
        <BrandMark />
        <h1>{t(($) => $.style.loadFailed)}</h1>
        <p>{error}</p>
        {buildable ? (
          <a className="wc-btn wc-btn--primary" href="#/build">
            {t(($) => $.app.empty.build)}
          </a>
        ) : (
          <a className="wc-btn wc-btn--primary" href="#/">
            {t(($) => $.style.back)}
          </a>
        )}
      </div>
    );
  if (!data)
    return (
      <div className="load-state">
        <BrandMark className="loading-mark" />
        <h1>{t(($) => $.style.loading)}</h1>
      </div>
    );

  const trait = traits.find((entry) => entry.key === selected) ?? null;
  const focusGroup = TRAIT_GROUPS.find((group) => group.id === focus);
  const { start, end } = data.run.window;
  const messages = data.run.scope.surface?.messages;

  return (
    <div className={`app-shell left-open style-shell${trait ? ' detail-open' : ''}`}>
      <AppBrand />
      <aside className="left-panel">
        <div className="left-panel__head" />
        <div className="intro-block">
          <div className="wc-eyebrow eyebrow">{t(($) => $.style.eyebrow)}</div>
          <h1 className="wc-hero">
            <Trans i18nKey={($) => $.style.tagline} components={{ br: <br />, em: <em /> }} />
          </h1>
          {traits.length > 0 && (
            <p>
              {t(($) => $.style.highlights, {
                traits: traits
                  .slice(0, 3)
                  .map((entry) => t(($) => $.style.quoted, { title: t(($) => $.style.traits[entry.id].title) }))
                  .join(t(($) => $.style.quotedSeparator)),
              })}
            </p>
          )}
        </div>
        <div className="style-groups-title">{t(($) => $.style.groupsTitle)}</div>
        <nav className="wc-dims" aria-label={t(($) => $.style.groupsTitle)}>
          {TRAIT_GROUPS.map((group) => {
            const Icon = GROUP_ICONS[group.id];
            const count = traits.filter((entry) => entry.group === group.id).length;
            return (
              <button
                key={group.id}
                type="button"
                className="wc-dim"
                aria-current={focus === group.id ? 'true' : undefined}
                data-sec=""
                onClick={() => setFocus(focus === group.id ? null : group.id)}
                style={groupVars(group.id)}
              >
                <span className="wc-disc" aria-hidden="true">
                  <Icon size={17} strokeWidth={1.8} />
                </span>
                <span className="wc-dim__copy">
                  <strong>{t(($) => $.style.groups[group.id].name)}</strong>
                  <small>
                    {t(($) => $.style.groupCount, { count, hint: t(($) => $.style.groups[group.id].hint) })}
                  </small>
                </span>
              </button>
            );
          })}
        </nav>
        <div className="left-bottom">
          <div className="data-footnote">
            {t(($) => $.style.footnote, {
              start: start?.slice(0, 7) ?? '',
              end: end?.slice(0, 7) ?? '',
              count: messages ?? 0,
            })}
          </div>
        </div>
      </aside>

      <main className="atlas-main">
        <div className="canvas-atmosphere" />
        <div className="topbar">
          <div className="topbar-leading">
            <ModeSwitch mode="style" />
            <div className="topbar-heading">
              <span className="topbar-divider">/</span>{' '}
              <strong>{focusGroup ? t(($) => $.style.groups[focusGroup.id].name) : t(($) => $.style.main)}</strong>
            </div>
          </div>
          <div className="topbar-actions">
            <BuildIndicator />
            <LanguageSwitch />
            <SettingsLink />
            <LogoutButton />
          </div>
        </div>
        {traits.length ? (
          <BubbleField traits={traits} selected={selected} focus={focus} onSelect={setSelected} />
        ) : (
          <div className="graph-empty">{t(($) => $.style.empty)}</div>
        )}
        <div className="bubble-legend">
          <ul className="wc-legend">
            {TIERS.map((tier) => (
              <li key={tier.id}>
                <i className={`legend-swatch tier-${tier.id}`} aria-hidden="true" />
                {t(($) => $.style.tiers[tier.id])}
              </li>
            ))}
          </ul>
          <p className="wc-hint">
            <Info size={15} strokeWidth={1.8} aria-hidden="true" />
            <span>{t(($) => $.style.legend, { every: Math.round(1 / MIN_RATE) })}</span>
          </p>
        </div>
      </main>

      <TraitDetail trait={trait} onClose={() => setSelected(null)} />
    </div>
  );
}
