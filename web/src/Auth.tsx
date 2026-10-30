import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { LogOut } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api } from './api';
import { LanguageSwitch } from './LanguageSwitch';
import { useProfileStore } from './store';

const AuthContext = createContext<() => Promise<void>>(async () => {});

export function LogoutButton() {
  const { t } = useTranslation();
  const logout = useContext(AuthContext);
  return (
    <button
      type="button"
      className="logout-button"
      onClick={() => void logout()}
      title={t(($) => $.auth.logout)}
      aria-label={t(($) => $.auth.logout)}
    >
      <LogOut size={17} />
    </button>
  );
}

export function AuthGate({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  const [expires, setExpires] = useState<number | null>(null);
  const [checking, setChecking] = useState(true);
  const [initialized, setInitialized] = useState<boolean | null>(null);
  const [mode, setMode] = useState<'encrypted' | 'plaintext'>('plaintext');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  // Leaves the fields alone: signing in already empties them, and the check on window focus runs while one may
  // be typing the password.
  function clear() {
    setExpires(null);
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
          api.get<{ initialized: boolean; storage_mode: 'encrypted' | 'plaintext' }>('auth/status'),
          api.get<{ expires_at: number | null }>('auth/session'),
        ]);
        if (active) {
          setInitialized(status.initialized);
          setMode(status.storage_mode);
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
        <span className="loading-mark">✳</span>
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
  return (
    <main className="auth-page">
      <LanguageSwitch />
      <form
        className="auth-card"
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
        <span className="brand-symbol">✳</span>
        <p className="auth-brand">WeClone</p>
        <h1>{setup ? t(($) => $.auth.setupTitle) : t(($) => $.auth.loginTitle)}</h1>
        <p>
          {setup
            ? mode === 'encrypted'
              ? t(($) => $.auth.setupEncrypted)
              : t(($) => $.auth.setupPlaintext)
            : t(($) => $.auth.loginHint)}
        </p>
        <label htmlFor="access-password">{setup ? t(($) => $.auth.newPassword) : t(($) => $.auth.password)}</label>
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
        />
        {setup && (
          <>
            <label htmlFor="confirm-password">{t(($) => $.auth.confirmPassword)}</label>
            <input
              id="confirm-password"
              name="confirmation"
              type="password"
              autoComplete="new-password"
              required
              value={confirmation}
              onChange={(event) => setConfirmation(event.target.value)}
              disabled={busy}
            />
          </>
        )}
        {error && (
          <p className="auth-error" role="alert">
            {error}
          </p>
        )}
        <button type="submit" disabled={busy || initialized === null || !password || (setup && !confirmation)}>
          {busy ? t(($) => $.auth.verifying) : setup ? t(($) => $.auth.setupSubmit) : t(($) => $.auth.loginSubmit)}
        </button>
        <small>{t(($) => $.auth.sessionNote)}</small>
      </form>
    </main>
  );
}
