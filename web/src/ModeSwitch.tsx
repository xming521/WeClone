export type AppMode = 'profile' | 'style';

export function modeFromHash(hash: string): AppMode {
  return hash.startsWith('#/style') ? 'style' : 'profile';
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
    </nav>
  );
}
