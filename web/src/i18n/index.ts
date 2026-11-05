// UI copy lives in the catalogs under locales/. Profile content (facts, summaries, chat) is never
// translated: it is shown in the language it was written in.
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { chineseNumeral } from './format';
import en from './locales/en/translation.json';
import zhCN from './locales/zh-CN/translation.json';

export const LANGUAGES = ['zh-CN', 'en'] as const;
export type Language = (typeof LANGUAGES)[number];

const LANGUAGE_KEY = 'weclone.locale';

/** The language chosen on the page before, else the browser's: Chinese for any variant of it, English otherwise. */
function initialLanguage(): Language {
  try {
    const saved = window.localStorage.getItem(LANGUAGE_KEY);
    const language = LANGUAGES.find((item) => item === saved);
    if (language) return language;
  } catch {
    // Without storage the browser decides.
  }
  return navigator.language.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en';
}

export function setLanguage(language: Language) {
  try {
    window.localStorage.setItem(LANGUAGE_KEY, language);
  } catch {
    // Without storage the choice lasts for this page only.
  }
  void i18n.changeLanguage(language);
}

void i18n.use(initReactI18next).init({
  resources: { 'zh-CN': { translation: zhCN }, en: { translation: en } },
  lng: initialLanguage(),
  fallbackLng: 'zh-CN',
  initAsync: false,
  enableSelector: true,
  // React escapes what it renders.
  interpolation: { escapeValue: false },
});
i18n.services.formatter?.add('zhNumeral', (value: number) => chineseNumeral(value));

function applyDocument() {
  document.documentElement.lang = i18n.language;
  document.title = i18n.t(($) => $.app.title);
}
applyDocument();
i18n.on('languageChanged', applyDocument);

export default i18n;
