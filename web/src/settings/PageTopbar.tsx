import { useTranslation } from 'react-i18next';
import { LogoutButton } from '../Auth';
import { AppBrand } from '../BrandMark';
import { BuildIndicator } from '../build/BuildIndicator';
import { LanguageSwitch } from '../LanguageSwitch';
import { ModeSwitch, type AppMode } from '../ModeSwitch';
import { SettingsLink } from './SettingsLink';

/** The topbar of the document-like pages: brand, page switch and heading, then language, settings and sign-out. */
export function PageTopbar({ mode, heading }: { mode: AppMode; heading: string }) {
  const { t } = useTranslation();
  return (
    <header className="summary-topbar">
      <div className="topbar-leading">
        <AppBrand />
        <ModeSwitch mode={mode} />
        <div className="topbar-heading">
          <span className="topbar-divider">/</span> <strong>{heading}</strong>
        </div>
      </div>
      <div className="topbar-actions" aria-label={t(($) => $.settings.actions)}>
        <BuildIndicator />
        <LanguageSwitch />
        <SettingsLink />
        <LogoutButton />
      </div>
    </header>
  );
}
