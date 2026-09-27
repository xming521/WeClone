import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { LogOut } from 'lucide-react';
import { useGraphStore } from './store';

const AuthContext = createContext<() => Promise<void>>(async () => {});

export async function api<T>(path: string, body?: unknown, method = 'POST'): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    cache: 'no-store',
    ...(body === undefined ? {} : {
      method, headers: { 'Content-Type': 'application/json', 'X-WeClone-Request': '1' },
      body: JSON.stringify(body),
    }),
  });
  if (!response.ok) {
    if (response.status === 401 && path !== 'auth/login') window.dispatchEvent(new Event('weclone-unauthorized'));
    const data = await response.json().catch(() => null);
    throw new Error(typeof data?.detail === 'string' ? data.detail : `请求失败：HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function LogoutButton() {
  const logout = useContext(AuthContext);
  return <button type="button" className="logout-button" onClick={() => void logout()} title="退出登录" aria-label="退出登录"><LogOut size={17} /></button>;
}

export function AuthGate({ children }: { children: ReactNode }) {
  const [expires, setExpires] = useState<number | null>(null);
  const [checking, setChecking] = useState(true);
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  function clear() {
    setExpires(null);
    useGraphStore.setState({ search: '', selectedId: 'root', focusedDimension: null, expandedIds: new Set(), detailOpen: false });
  }

  useEffect(() => {
    let active = true;
    const unauthorized = () => { if (active) clear(); };
    window.addEventListener('weclone-unauthorized', unauthorized);
    async function check() {
      try {
        const session = await api<{ expires_at: number }>('auth/session');
        if (active) { setExpires(session.expires_at); setError(''); }
      } catch (reason) {
        if (active) setError((reason as Error).message);
      } finally {
        if (active) setChecking(false);
      }
    }
    void check();
    const onFocus = () => { void check(); };
    window.addEventListener('focus', onFocus);
    return () => {
      active = false;
      window.removeEventListener('weclone-unauthorized', unauthorized);
      window.removeEventListener('focus', onFocus);
    };
  }, []);

  useEffect(() => {
    if (expires === null) return;
    // Browser timers are limited to a signed 32-bit delay; 30 days exceeds it.
    const timer = window.setInterval(() => { if (Date.now() >= expires * 1000) clear(); }, 1000);
    return () => window.clearInterval(timer);
  }, [expires]);

  async function logout() {
    try {
      await api('auth/logout', {});
      clear();
      setError('');
    } catch (reason) { setError((reason as Error).message); }
  }

  if (checking) return <div className="load-state"><span className="loading-mark">✳</span><h1>正在验证登录状态</h1></div>;
  if (expires !== null) return <AuthContext.Provider value={logout}>{children}{error && <div className="review-error" role="alert">{error}</div>}</AuthContext.Provider>;
  return <main className="auth-page"><form className="auth-card" onSubmit={async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const session = await api<{ expires_at: number }>('auth/login', { password });
      setPassword('');
      setExpires(session.expires_at);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  }}>
    <span className="brand-symbol">✳</span><p className="auth-brand">WeClone</p>
    <h1>解锁个人记忆</h1><p>输入启动终端显示的访问密码。</p>
    <label htmlFor="access-password">访问密码</label>
    <input id="access-password" name="password" type="password" autoComplete="current-password" required autoFocus maxLength={256} value={password} onChange={(event) => setPassword(event.target.value)} disabled={busy} />
    {error && <p className="auth-error" role="alert">{error}</p>}
    <button type="submit" disabled={busy || !password}>{busy ? '正在验证…' : '解锁'}</button>
    <small>登录后，此浏览器 30 天内无需再次输入密码。请在自己的设备上使用。</small>
  </form></main>;
}
