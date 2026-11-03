import { Settings } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** The gear beside the sign-out button: settings, and from there the about page. */
export function SettingsLink() {
  const { t } = useTranslation();
  return (
    <a
      className="wc-iconbtn wc-iconbtn--outline"
      href="#/settings"
      title={t(($) => $.settings.title)}
      aria-label={t(($) => $.settings.title)}
    >
      <Settings size={16} strokeWidth={1.8} />
    </a>
  );
}
