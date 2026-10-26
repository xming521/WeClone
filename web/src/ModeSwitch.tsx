export type AppMode = 'profile' | 'style' | 'summary';

export function modeFromHash(hash: string): AppMode {
  if (hash.startsWith('#/style')) return 'style';
  return hash.startsWith('#/summary') ? 'summary' : 'profile';
}

export function ModeSwitch({ mode }: { mode: AppMode }) {
  return (
    <nav className="mode-switch" aria-label="页面">
      <a href="#/" aria-current={mode === 'profile' ? 'page' : undefined}>
        人生画像
      </a>
      <a href="#/style" aria-current={mode === 'style' ? 'page' : undefined}>
        说话风格
      </a>
      <a href="#/summary" aria-current={mode === 'summary' ? 'page' : undefined}>
        领域总结
      </a>
    </nav>
  );
}
