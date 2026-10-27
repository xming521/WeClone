import { useCallback, useEffect, useState } from 'react';
import { Lock, LockOpen } from 'lucide-react';
import { api } from '../api';
import { LogoutButton } from '../Auth';
import { ModeSwitch } from '../ModeSwitch';
import '../summary/summary.css';
import { CopyField, TokenCard, moment, type TokenState } from './accessParts';
import './assistant.css';

// Shapes of GET /api/assistant, served by weclone/server/assistant.py.
type Level = 'full' | 'normal' | 'public';

interface CatalogDomain {
  name: string;
  /** Facts in the domain at the chosen level, rejected ones left out. */
  facts: number;
  /** Whether it has an approved, current summary at the chosen level. */
  summary: boolean;
  shared: boolean;
}

interface AssistantView {
  encrypted: boolean;
  /** Unix seconds when assistants stop reading encrypted data, null while they may not. */
  allowed_until: number | null;
  hours: number[];
  mcp_url: string;
  /** The summaries weclone_profile hands out now. */
  summaries: number;
  /** What every assistant reads, from its next call on. */
  level: Level;
  /** Domains assistants read, null for every domain including ones that come later. */
  domains: string[] | null;
  catalog: CatalogDomain[];
  /** The one token assistants connect with, null until one is generated. */
  token: TokenState | null;
}

const LEVELS: { level: Level; label: string; hint: string }[] = [
  { level: 'full', label: '全部', hint: '所有条目，包括私密和敏感的。这些内容会交给助手背后的云端模型' },
  { level: 'normal', label: '普通', hint: '公开和普通条目' },
  { level: 'public', label: '公开', hint: '只有公开条目' },
];
const CLIENTS = ['Claude Code', 'Codex'] as const;
type Client = (typeof CLIENTS)[number];

function hoursLabel(hours: number) {
  return hours % 24 ? `${hours} 小时` : `${hours / 24} 天`;
}

/** Commands that give a client the token. They replace an earlier WeClone entry, so they can run again. */
function setupCommands(client: Client, url: string, token: string) {
  if (client === 'Claude Code')
    return [
      'claude mcp remove -s user weclone >/dev/null 2>&1',
      `claude mcp add -s user --transport http weclone ${url} --header "Authorization: Bearer ${token}"`,
    ].join('\n');
  return [
    'codex mcp remove weclone >/dev/null 2>&1',
    "cat >> ~/.codex/config.toml <<'EOF'",
    '',
    '[mcp_servers.weclone]',
    `url = "${url}"`,
    `http_headers = { Authorization = "Bearer ${token}" }`,
    'EOF',
  ].join('\n');
}

export function AssistantApp() {
  const [view, setView] = useState<AssistantView | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [created, setCreated] = useState('');
  const [client, setClient] = useState<Client>('Claude Code');

  const load = useCallback(async () => {
    try {
      setView(await api.get<AssistantView>('assistant'));
    } catch (reason) {
      setError((reason as Error).message);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // The server ends access on its own. The page catches up once the time is up.
  useEffect(() => {
    if (view?.allowed_until == null) return;
    const timer = window.setTimeout(() => void load(), Math.max(view.allowed_until * 1000 - Date.now(), 0) + 1000);
    return () => window.clearTimeout(timer);
  }, [view, load]);

  async function run(request: () => Promise<unknown>) {
    setBusy(true);
    setError('');
    try {
      await request();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
      await load();
    }
  }

  function share(domains: string[] | null) {
    void run(() => api.put('assistant/domains', { domains }));
  }

  function toggle(domain: CatalogDomain) {
    const names = view!.catalog
      .filter((item) => (item.name === domain.name ? !item.shared : item.shared))
      .map((item) => item.name);
    share(names);
  }

  function issue() {
    if (view?.token && !window.confirm('重新生成后旧令牌立即失效，已经配置好的助手都要用新的命令再配置一次。继续吗？'))
      return;
    void run(async () => setCreated((await api.post<{ token: string }>('assistant/token')).token));
  }

  function withdraw() {
    if (!window.confirm('停用后所有助手都读不到 WeClone，之后可以重新生成令牌。继续吗？')) return;
    void run(async () => {
      await api.delete('assistant/token');
      setCreated('');
    });
  }

  if (!view && error)
    return (
      <div className="load-state">
        <span>✳</span>
        <h1>助手接入页暂时无法打开</h1>
        <p>{error}</p>
        <button onClick={() => void load()}>重试</button>
      </div>
    );
  if (!view)
    return (
      <div className="load-state">
        <span className="loading-mark">✳</span>
        <h1>正在读取助手接入设置</h1>
      </div>
    );

  const allowed = view.allowed_until != null;
  const chosen = LEVELS.find((item) => item.level === view.level)!;
  const sharedCount = view.catalog.filter((item) => item.shared).length;

  return (
    <div className="summary-page">
      <header className="summary-topbar">
        <div className="topbar-leading">
          <ModeSwitch mode="assistant" />
          <div className="topbar-heading">
            <span className="topbar-divider">/</span>{' '}
            <strong>{view.token ? `已接入 · ${chosen.label}` : '还没有令牌'}</strong>
          </div>
        </div>
        <div className="topbar-actions">
          <LogoutButton />
        </div>
      </header>
      <main className="summary-main assistant-main">
        <section className="summary-intro">
          <h1>让个人助手读取画像</h1>
          <p>
            Claude Code、Codex 等支持 MCP 的助手可以连接这里的 WeClone：weclone_profile
            返回你确认过的领域总结，weclone_search
            按关键词或领域查画像条目。所有助手共用一个令牌，读到哪些内容由下面选的级别和领域决定，每次调用都会记下返回了什么。
          </p>
          <p className="assistant-address">
            MCP 地址 <code>{view.mcp_url}</code>
            {!view.encrypted && '。数据是明文存储的，令牌有效时助手随时可以读取'}
          </p>
          {error && (
            <div className="summary-error" role="alert">
              <span>{error}</span>
              <button type="button" onClick={() => setError('')}>
                知道了
              </button>
            </div>
          )}
        </section>

        <section className="assistant-card" aria-label="提供哪些信息">
          <div className="assistant-card-head">
            <h2>提供哪些信息</h2>
          </div>
          <div className="summary-levels" role="radiogroup" aria-label="隐私级别">
            <span>隐私级别</span>
            {LEVELS.map((item) => (
              <button
                key={item.level}
                type="button"
                role="radio"
                aria-checked={item.level === view.level}
                disabled={busy}
                onClick={() =>
                  item.level !== view.level && void run(() => api.patch('assistant', { level: item.level }))
                }
              >
                {item.label}
              </button>
            ))}
          </div>
          <p className={`summary-hint${view.level === 'full' ? ' summary-warning' : ''}`}>{chosen.hint}。</p>
          <div className="assistant-domains-head">
            <h3>领域</h3>
            <span>
              {view.domains === null
                ? '全部提供，以后新出现的领域也提供'
                : `提供 ${sharedCount}/${view.catalog.length} 个，以后新出现或改名的领域不提供`}
            </span>
            {view.domains !== null && (
              <button type="button" disabled={busy} onClick={() => share(null)}>
                改回全部领域
              </button>
            )}
          </div>
          {view.catalog.length ? (
            <ul className="assistant-domains" aria-label="提供的领域">
              {view.catalog.map((item) => (
                <li key={item.name}>
                  <label className={item.shared ? 'shared' : undefined}>
                    <input type="checkbox" checked={item.shared} disabled={busy} onChange={() => toggle(item)} />
                    <span>{item.name}</span>
                    <small>
                      {item.facts} 条{item.summary ? ' · 有总结' : ''}
                    </small>
                  </label>
                </li>
              ))}
            </ul>
          ) : (
            <p className="summary-placeholder">还没有画像，运行 build-profile 之后这里会列出各个领域。</p>
          )}
          <p className="summary-hint">
            勾掉的领域，总结和条目都不给助手。助手现在能读到 {view.summaries} 段领域总结：这个级别、勾选的领域里、在
            <a href="#/summary">领域总结页</a>确认过的。改了从助手的下一次调用起生效，不用重新配置。
          </p>
        </section>

        {view.encrypted && (
          <section className="assistant-card" aria-label="允许助手读取">
            <div className="assistant-card-head">
              <h2>{allowed ? <LockOpen size={15} /> : <Lock size={15} />} 允许助手读取</h2>
              <span className={`assistant-state${allowed ? ' on' : ''}`}>
                {allowed ? `到 ${moment(view.allowed_until!)} 自动关闭` : '已关闭'}
              </span>
            </div>
            <p className="summary-hint">
              数据是加密的，助手只能在你允许的时间里读取。允许期间服务在内存里保留一份密钥，到时间或服务重启就清除。
            </p>
            <div className="assistant-actions">
              {view.hours.map((hours) => (
                <button
                  key={hours}
                  type="button"
                  disabled={busy}
                  onClick={() => void run(() => api.post('assistant/access', { hours }))}
                >
                  {allowed ? '改为' : '允许'} {hoursLabel(hours)}
                </button>
              ))}
              {allowed && (
                <button type="button" disabled={busy} onClick={() => void run(() => api.delete('assistant/access'))}>
                  立即关闭
                </button>
              )}
            </div>
          </section>
        )}

        <TokenCard
          token={view.token}
          created={Boolean(created)}
          busy={busy}
          unused="还没有助手用过"
          empty="生成令牌后，在运行助手的机器上执行配置命令。"
          onIssue={issue}
          onWithdraw={withdraw}
        >
          {created && (
            <div className="assistant-setup">
              <p className="summary-hint">
                令牌只显示这一次。在运行助手的机器上执行对应的命令，之后新开的会话就能调用 WeClone 的工具。
              </p>
              <div className="summary-levels" role="radiogroup" aria-label="助手">
                {CLIENTS.map((item) => (
                  <button
                    key={item}
                    type="button"
                    role="radio"
                    aria-checked={item === client}
                    onClick={() => setClient(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
              <CopyField label={`${client} 配置命令`} text={setupCommands(client, view.mcp_url, created)} />
              <p className="summary-hint">
                其他助手：地址填上面的 MCP 地址，请求头加 <code>Authorization: Bearer 令牌</code>。
              </p>
              <CopyField label="令牌" text={created} />
              <button className="assistant-done" type="button" onClick={() => setCreated('')}>
                都配置好了，收起
              </button>
            </div>
          )}
        </TokenCard>
      </main>
    </div>
  );
}
