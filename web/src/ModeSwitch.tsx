import { useTranslation } from 'react-i18next';

export type AppMode = 'profile' | 'style' | 'summary' | 'assistant' | 'bots';

export function modeFromHash(hash: string): AppMode {
  if (hash.startsWith('#/style')) return 'style';
  if (hash.startsWith('#/assistant')) return 'assistant';
  if (hash.startsWith('#/bots')) return 'bots';
  return hash.startsWith('#/summary') ? 'summary' : 'profile';
}

export function ModeSwitch({ mode }: { mode: AppMode }) {
  const { t } = useTranslation();
  return (
    <nav className="mode-switch" aria-label={t(($) => $.modes.label)}>
      <a href="#/" aria-current={mode === 'profile' ? 'page' : undefined}>
        {t(($) => $.modes.profile)}
      </a>
      <a href="#/style" aria-current={mode === 'style' ? 'page' : undefined}>
        {t(($) => $.modes.style)}
      </a>
      <a href="#/summary" aria-current={mode === 'summary' ? 'page' : undefined}>
        {t(($) => $.modes.summary)}
      </a>
      <a href="#/assistant" aria-current={mode === 'assistant' ? 'page' : undefined}>
        {t(($) => $.modes.assistant)}
      </a>
      <a href="#/bots" aria-current={mode === 'bots' ? 'page' : undefined}>
        {t(($) => $.modes.bots)}
      </a>
    </nav>
  );
}
