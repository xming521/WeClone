import { useEffect, useState } from 'react';
import { Undo2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { isTyping } from './keys';
import { useToasts } from './toast';

// Long enough to read the notice and reach for undo; the timer waits while the pointer or focus is on it.
const VISIBLE_MS = 5000;

export function Toaster() {
  const { t } = useTranslation();
  const current = useToasts((state) => state.current);
  const dismiss = useToasts((state) => state.dismiss);
  const undo = useToasts((state) => state.undo);
  const [held, setHeld] = useState(false);
  useEffect(() => {
    if (!current || held) return;
    const timer = window.setTimeout(() => dismiss(current.id), VISIBLE_MS);
    return () => window.clearTimeout(timer);
  }, [current, held, dismiss]);
  // Z, or Ctrl/⌘+Z, takes back the last action while its notice shows, unless the key goes into a field or dialog.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== 'z' || event.altKey || event.shiftKey || isTyping(event)) return;
      if (event.ctrlKey && event.metaKey) return;
      if (undo()) event.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [undo]);
  return (
    <div className="toast-region" role="status" aria-live="polite">
      {current && (
        <div
          key={current.id}
          className="wc-toast"
          data-tone={current.tone}
          onPointerEnter={() => setHeld(true)}
          onPointerLeave={() => setHeld(false)}
          onFocus={() => setHeld(true)}
          onBlur={() => setHeld(false)}
        >
          <span className="wc-toast__text">{current.message}</span>
          {current.undo && (
            <button type="button" className="wc-toast__action" onClick={() => undo()}>
              <Undo2 size={13} strokeWidth={1.8} aria-hidden="true" />
              {t(($) => $.toast.undo)}
              <kbd className="wc-kbd">Z</kbd>
            </button>
          )}
          <button
            type="button"
            className="wc-toast__close"
            aria-label={t(($) => $.common.dismiss)}
            onClick={() => dismiss(current.id)}
          >
            <X size={14} strokeWidth={1.8} aria-hidden="true" />
          </button>
        </div>
      )}
    </div>
  );
}
