import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['node_modules', 'dev-data', 'dist'] },
  {
    files: ['src/**/*.{ts,tsx}', 'tests/**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
    plugins: { 'react-hooks': reactHooks },
    rules: {
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
    },
  },
  {
    // UI copy lives in src/i18n/locales. Chinese text in code is content logic and says so with a disable comment.
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/i18n/**'],
    rules: {
      'no-restricted-syntax': [
        'error',
        ...['Literal[value=/CJK/]', 'TemplateElement[value.raw=/CJK/]', 'JSXText[value=/CJK/]'].map((selector) => ({
          selector: selector.replace('CJK', '[\\u3000-\\u303f\\u4e00-\\u9fff\\uff00-\\uffef]'),
          message: 'Put UI copy in src/i18n/locales and read it with t().',
        })),
      ],
    },
  },
);
