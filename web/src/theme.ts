import { create } from 'zustand';

/** The colour scheme the person chose; `system` follows the device. */
export type ThemeChoice = 'system' | 'light' | 'dark';
export const THEME_CHOICES: ThemeChoice[] = ['system', 'light', 'dark'];

const THEME_KEY = 'weclone.theme';
const query = window.matchMedia('(prefers-color-scheme: dark)');

function savedChoice(): ThemeChoice {
  try {
    const saved = window.localStorage.getItem(THEME_KEY);
    return THEME_CHOICES.find((choice) => choice === saved) ?? 'system';
  } catch {
    return 'system';
  }
}

const resolve = (choice: ThemeChoice) => (choice === 'system' ? (query.matches ? 'dark' : 'light') : choice);

interface ThemeState {
  choice: ThemeChoice;
  /** The scheme on screen. */
  theme: 'light' | 'dark';
  choose: (choice: ThemeChoice) => void;
}

// The design system's tokens switch on `data-theme` of the root element; canvases read the tokens when they draw.
function apply(theme: 'light' | 'dark') {
  document.documentElement.setAttribute('data-theme', theme);
}

export const useTheme = create<ThemeState>((set) => {
  const choice = savedChoice();
  apply(resolve(choice));
  return {
    choice,
    theme: resolve(choice),
    choose: (next) => {
      try {
        window.localStorage.setItem(THEME_KEY, next);
      } catch {
        // Without storage the choice lasts for this page only.
      }
      apply(resolve(next));
      set({ choice: next, theme: resolve(next) });
    },
  };
});

query.addEventListener('change', () => {
  const { choice } = useTheme.getState();
  if (choice !== 'system') return;
  apply(resolve(choice));
  useTheme.setState({ theme: resolve(choice) });
});
