import { defineConfig } from 'i18next-cli';

// The catalogs are written by hand. Do not run `extract`: it cannot follow selectors indexed by
// variables, such as `$.review.status[view]`, and drops their keys. `types` regenerates
// src/i18n/resources.d.ts after a catalog changes. zh-CN is the source: the types and the other
// catalogs follow its keys, which tests/catalogs.test.ts checks.
export default defineConfig({
  locales: ['zh-CN', 'en'],
  extract: {
    input: ['src/**/*.{ts,tsx}'],
    output: 'src/i18n/locales/{{language}}/{{namespace}}.json',
    defaultNS: 'translation',
    nsSeparator: false,
    primaryLanguage: 'zh-CN',
  },
  types: {
    input: ['src/i18n/locales/zh-CN/*.json'],
    output: 'src/i18n/i18next.d.ts',
    resourcesFile: 'src/i18n/resources.d.ts',
    enableSelector: true,
  },
});
