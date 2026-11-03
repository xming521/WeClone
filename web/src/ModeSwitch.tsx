import { useEffect, useRef } from 'react';
import { Bot, LayoutList, MessagesSquare, Orbit, Plug } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** The pages: the five of the page switch, then settings and about, which the gear in the topbar opens. */
export type AppMode = 'profile' | 'style' | 'summary' | 'assistant' | 'bots' | 'settings' | 'about';

export function modeFromHash(hash: string): AppMode {
  if (hash.startsWith('#/style')) return 'style';
  if (hash.startsWith('#/assistant')) return 'assistant';
  if (hash.startsWith('#/bots')) return 'bots';
  if (hash.startsWith('#/settings')) return 'settings';
  if (hash.startsWith('#/about')) return 'about';
  return hash.startsWith('#/summary') ? 'summary' : 'profile';
}

/** The DS page switch (ModeNav). The icons are the DS's fixed pairing; a narrow bar keeps only the current page's. */
export function ModeSwitch({ mode }: { mode: AppMode }) {
  const { t } = useTranslation();
  const nav = useRef<HTMLElement>(null);
  // On a phone the bar scrolls sideways: bring the current page into view.
  useEffect(() => {
    const bar = nav.current;
    const current = bar?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!bar || !current) return;
    const left = current.getBoundingClientRect().left - bar.getBoundingClientRect().left + bar.scrollLeft;
    bar.scrollLeft = left - (bar.clientWidth - current.offsetWidth) / 2;
  }, [mode]);
  return (
    <nav ref={nav} className="wc-modes" aria-label={t(($) => $.modes.label)}>
      <a href="#/" aria-current={mode === 'profile' ? 'page' : undefined}>
        <Orbit size={15} strokeWidth={1.8} aria-hidden="true" />
        {t(($) => $.modes.profile)}
      </a>
      <a href="#/style" aria-current={mode === 'style' ? 'page' : undefined}>
        <MessagesSquare size={15} strokeWidth={1.8} aria-hidden="true" />
        {t(($) => $.modes.style)}
      </a>
      <a href="#/summary" aria-current={mode === 'summary' ? 'page' : undefined}>
        <LayoutList size={15} strokeWidth={1.8} aria-hidden="true" />
        {t(($) => $.modes.summary)}
      </a>
      <a href="#/assistant" aria-current={mode === 'assistant' ? 'page' : undefined}>
        <Plug size={15} strokeWidth={1.8} aria-hidden="true" />
        {t(($) => $.modes.assistant)}
      </a>
      <a href="#/bots" aria-current={mode === 'bots' ? 'page' : undefined}>
        <Bot size={15} strokeWidth={1.8} aria-hidden="true" />
        {t(($) => $.modes.bots)}
      </a>
    </nav>
  );
}
