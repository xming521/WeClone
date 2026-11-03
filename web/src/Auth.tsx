import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { LockOpen, LogOut } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api } from './api';
import { BrandMark } from './BrandMark';
import { LanguageSwitch } from './LanguageSwitch';
import { formatDuration } from './settings/duration';
import { useProfileStore } from './store';

const AuthContext = createContext<() => Promise<void>>(async () => {});

export function LogoutButton() {
  const { t } = useTranslation();
  const logout = useContext(AuthContext);
  return (
    <button
      type="button"
      className="wc-iconbtn wc-iconbtn--outline"
      onClick={() => void logout()}
      title={t(($) => $.auth.logout)}
      aria-label={t(($) => $.auth.logout)}
    >
      <LogOut size={16} strokeWidth={1.8} />
    </button>
  );
}

export function AuthGate({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  const [expires, setExpires] = useState<number | null>(null);
  const [checking, setChecking] = useState(true);
  const [initialized, setInitialized] = useState<boolean | null>(null);
  const [mode, setMode] = useState<'encrypted' | 'plaintext'>('plaintext');
  const [demoPassword, setDemoPassword] = useState<string | null>(null);
  const [sessionSeconds, setSessionSeconds] = useState(2 * 3600);
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  function clear() {
    setExpires(null);
    setPassword('');
    setConfirmation('');
    useProfileStore.getState().resetSession();
  }

  useEffect(() => {
    let active = true;
    const unauthorized = () => {
      if (active) clear();
    };
    window.addEventListener('weclone-unauthorized', unauthorized);
    async function check() {
      try {
        const [status, session] = await Promise.all([
          api.get<{
            initialized: boolean;
            storage_mode: 'encrypted' | 'plaintext';
            demo_password: string | null;
            session_seconds: number;
          }>('auth/status'),
          api.get<{ expires_at: number | null }>('auth/session'),
        ]);
        if (active) {
          setInitialized(status.initialized);
          setMode(status.storage_mode);
          setDemoPassword(status.demo_password);
          setSessionSeconds(status.session_seconds);
          if (session.expires_at === null || session.expires_at * 1000 <= Date.now()) clear();
          else setExpires(session.expires_at);
          setError('');
        }
      } catch (reason) {
        if (active) setError((reason as Error).message);
      } finally {
        if (active) setChecking(false);
      }
    }
    void check();
    const onFocus = () => {
      void check();
    };
    window.addEventListener('focus', onFocus);
    return () => {
      active = false;
      window.removeEventListener('weclone-unauthorized', unauthorized);
      window.removeEventListener('focus', onFocus);
    };
  }, []);

  useEffect(() => {
    if (expires === null) return;
    // Check absolute expiry even while the page stays open.
    const timer = window.setInterval(() => {
      if (Date.now() >= expires * 1000) clear();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [expires]);

  async function logout() {
    try {
      await api.post('auth/logout');
      clear();
      setError('');
    } catch (reason) {
      setError((reason as Error).message);
    }
  }

  if (checking)
    return (
      <div className="load-state">
        <BrandMark className="loading-mark" />
        <h1>{t(($) => $.auth.checking)}</h1>
      </div>
    );
  if (expires !== null)
    return (
      <AuthContext.Provider value={logout}>
        {children}
        {error && (
          <div className="review-error" role="alert">
            {error}
          </div>
        )}
      </AuthContext.Provider>
    );
  const setup = initialized === false;
  // The error lands on the last field, like the DS: its outline turns danger and the hint names the problem.
  const errorId = 'auth-error';
  const lastField = setup ? 'confirm' : 'password';
  return (
    <main className="auth-page">
      <LanguageSwitch />
      <form
        className="wc-login"
        aria-labelledby="auth-title"
        onSubmit={async (event) => {
          event.preventDefault();
          if (setup && password !== confirmation) {
            setError(t(($) => $.auth.mismatch));
            return;
          }
          setBusy(true);
          setError('');
          try {
            const session = await api.post<{ expires_at: number }>(
              setup ? 'auth/setup' : 'auth/login',
              setup ? { password, confirmation } : { password },
            );
            setPassword('');
            setConfirmation('');
            setInitialized(true);
            setExpires(session.expires_at);
          } catch (reason) {
            setError((reason as Error).message);
          } finally {
            setBusy(false);
          }
        }}
      >
        <BrandMark className="auth-mark" />
        <h1 id="auth-title" className="auth-title">
          {setup ? t(($) => $.auth.setupTitle) : t(($) => $.auth.loginTitle)}
        </h1>
        <p className="auth-hint">
          {setup
            ? mode === 'encrypted'
              ? t(($) => $.auth.setupEncrypted)
              : t(($) => $.auth.setupPlaintext)
            : demoPassword
              ? t(($) => $.auth.demoHint, { password: demoPassword })
              : t(($) => $.auth.loginHint)}
        </p>
        <div className="wc-field">
          <label className="wc-field__label" htmlFor="access-password">
            {setup ? t(($) => $.auth.newPassword) : t(($) => $.auth.password)}
          </label>
          <div className="wc-input" aria-invalid={error && lastField === 'password' ? true : undefined}>
            <input
              id="access-password"
              name="password"
              type="password"
              autoComplete={setup ? 'new-password' : 'current-password'}
              required
              autoFocus
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              disabled={busy || initialized === null}
              aria-describedby={error && lastField === 'password' ? errorId : undefined}
            />
          </div>
          {error && lastField === 'password' && (
            <p className="wc-field__hint" data-tone="danger" id={errorId} role="alert">
              {error}
            </p>
          )}
        </div>
        {setup && (
          <div className="wc-field auth-next-field">
            <label className="wc-field__label" htmlFor="confirm-password">
              {t(($) => $.auth.confirmPassword)}
            </label>
            <div className="wc-input" aria-invalid={error && lastField === 'confirm' ? true : undefined}>
              <input
                id="confirm-password"
                name="confirmation"
                type="password"
                autoComplete="new-password"
                required
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                disabled={busy}
                aria-describedby={error && lastField === 'confirm' ? errorId : undefined}
              />
            </div>
            {error && lastField === 'confirm' && (
              <p className="wc-field__hint" data-tone="danger" id={errorId} role="alert">
                {error}
              </p>
            )}
          </div>
        )}
        <button
          className="wc-btn wc-btn--primary wc-btn--lg wc-btn--block auth-submit"
          type="submit"
          disabled={busy || initialized === null || !password || (setup && !confirmation)}
        >
          <LockOpen size={16} strokeWidth={1.8} />
          {busy ? t(($) => $.auth.verifying) : setup ? t(($) => $.auth.setupSubmit) : t(($) => $.auth.loginSubmit)}
        </button>
        <p className="auth-note">{t(($) => $.auth.sessionNote, { duration: formatDuration(sessionSeconds) })}</p>
        <footer className="auth-foot">
          <b className="auth-wordmark">WeClone</b>
          <span className="wc-serial">PROFILE ATLAS</span>
        </footer>
      </form>
    </main>
  );
}
