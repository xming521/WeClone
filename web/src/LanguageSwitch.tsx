import { useTranslation } from 'react-i18next';
import { LANGUAGES, setLanguage } from './i18n';

// Each language is named in itself, the same in every catalog.
const NAMES = {
  // eslint-disable-next-line no-restricted-syntax -- the language's own name
  'zh-CN': { short: '中', full: '中文' },
  en: { short: 'EN', full: 'English' },
} as const;

export function LanguageSwitch() {
  const { t, i18n } = useTranslation();
  return (
    <div className="language-switch" role="group" aria-label={t(($) => $.common.language)}>
      {LANGUAGES.map((language) => (
        <button
          key={language}
          type="button"
          lang={language}
          title={NAMES[language].full}
          aria-label={NAMES[language].full}
          aria-pressed={i18n.language === language}
          onClick={() => setLanguage(language)}
        >
          {NAMES[language].short}
        </button>
      ))}
    </div>
  );
}
