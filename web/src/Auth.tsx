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
  const [initialized, setInitialized] = useState<boolean | null>(null);
  const [mode, setMode] = useState<'encrypted' | 'plaintext'>('plaintext');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  function clear() {
    setExpires(null);
    setPassword('');
    setConfirmation('');
    useGraphStore.setState({ search: '', selectedId: 'root', focusedDimension: null, expandedIds: new Set(), detailOpen: false });
  }

  useEffect(() => {
    let active = true;
    const unauthorized = () => { if (active) clear(); };
    window.addEventListener('weclone-unauthorized', unauthorized);
    async function check() {
      try {
        const [status, session] = await Promise.all([
          api<{ initialized: boolean; storage_mode: 'encrypted' | 'plaintext' }>('auth/status'),
          api<{ expires_at: number | null }>('auth/session'),
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
    // Check absolute expiry even while the page stays open.
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
  const setup = initialized === false;
  return <main className="auth-page"><form className="auth-card" onSubmit={async (event) => {
    event.preventDefault();
    if (setup && password !== confirmation) { setError('两次输入的密码不一致'); return; }
    setBusy(true);
    setError('');
    try {
      const session = await api<{ expires_at: number }>(setup ? 'auth/setup' : 'auth/login', setup ? { password, confirmation } : { password });
      setPassword('');
      setConfirmation('');
      setInitialized(true);
      setExpires(session.expires_at);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  }}>
    <span className="brand-symbol">✳</span><p className="auth-brand">WeClone</p>
    <h1>{setup ? '设置个人密码' : '解锁个人记忆'}</h1>
    <p>{setup ? (mode === 'encrypted' ? '同一个密码用于网页访问和数据加密。忘记密码后需要重新抽取全部数据。' : '设置用于网页访问的密码。当前数据采用明文存储。') : '输入你设置的密码。'}</p>
    <label htmlFor="access-password">{setup ? '设置密码' : '访问密码'}</label>
    <input id="access-password" name="password" type="password" autoComplete={setup ? 'new-password' : 'current-password'} required autoFocus value={password} onChange={(event) => setPassword(event.target.value)} disabled={busy || initialized === null} />
    {setup && <>
      <label htmlFor="confirm-password">再次输入密码</label>
      <input id="confirm-password" name="confirmation" type="password" autoComplete="new-password" required value={confirmation} onChange={(event) => setConfirmation(event.target.value)} disabled={busy} />
    </>}
    {error && <p className="auth-error" role="alert">{error}</p>}
    <button type="submit" disabled={busy || initialized === null || !password || (setup && !confirmation)}>{busy ? '正在验证…' : setup ? '设置并解锁' : '解锁'}</button>
    <small>登录有效期为 2 小时，到期后需重新输入密码。请在自己的设备上使用。</small>
  </form></main>;
}
