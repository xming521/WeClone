import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import { LogoutButton } from '../Auth';
import { ModeSwitch } from '../ModeSwitch';
import { CopyField, TokenCard, type TokenState } from '../assistant/accessParts';
import '../assistant/assistant.css';
import '../summary/summary.css';

// Shape of GET /api/bots, served by weclone/server/bots.py.
interface BotsView {
  encrypted: boolean;
  /** Whether the server loads the fine-tuned clone (started with --inference), so /v1 answers. */
  inference: boolean;
  api_url: string;
  /** The one token bots call with, null until one is generated. */
  token: TokenState | null;
}

export function BotsApp() {
  const [view, setView] = useState<BotsView | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [created, setCreated] = useState('');

  const load = useCallback(async () => {
    try {
      setView(await api.get<BotsView>('bots'));
    } catch (reason) {
      setError((reason as Error).message);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

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

  function issue() {
    if (view?.token && !window.confirm('重新生成后旧令牌立即失效，所有机器人都要换成新令牌。继续吗？')) return;
    void run(async () => setCreated((await api.post<{ token: string }>('bots/token')).token));
  }

  function withdraw() {
    if (!window.confirm('停用后所有机器人都调用不了分身，之后可以重新生成令牌。继续吗？')) return;
    void run(async () => {
      await api.delete('bots/token');
      setCreated('');
    });
  }

  if (!view && error)
    return (
      <div className="load-state">
        <span>✳</span>
        <h1>机器人接入页暂时无法打开</h1>
        <p>{error}</p>
        <button onClick={() => void load()}>重试</button>
      </div>
    );
  if (!view)
    return (
      <div className="load-state">
        <span className="loading-mark">✳</span>
        <h1>正在读取机器人接入设置</h1>
      </div>
    );

  return (
    <div className="summary-page">
      <header className="summary-topbar">
        <div className="topbar-leading">
          <ModeSwitch mode="bots" />
          <div className="topbar-heading">
            <span className="topbar-divider">/</span> <strong>{view.token ? '已接入' : '还没有令牌'}</strong>
          </div>
        </div>
        <div className="topbar-actions">
          <LogoutButton />
        </div>
      </header>
      <main className="summary-main assistant-main">
        <section className="summary-intro">
          <h1>让聊天机器人调用分身</h1>
          <p>
            AstrBot、LangBot 等机器人平台通过 OpenAI 兼容接口调用微调后的分身模型。所有机器人共用一个令牌，填在平台的
            API Key 里。
          </p>
          <p className="assistant-address">
            接口地址 <code>{view.api_url}</code>
          </p>
          <p className="summary-hint">
            令牌生成后一直有效，服务重启也不受影响，重新生成或停用后才失效。
            {view.encrypted &&
              '数据是加密的：生成令牌时，数据密钥用令牌加密后存进本机的 bots.sqlite3，机器人调用时凭令牌解开，不需要你登录。令牌和这个文件同时泄露就能解密全部数据，请像密码一样保管令牌。改密码不影响令牌，怀疑泄露时请重新生成。'}
          </p>
          {!view.inference && (
            <p className="summary-hint summary-warning">
              服务现在没有加载分身模型，用 <code>weclone-cli server --inference</code> 启动后机器人才能调用。
            </p>
          )}
          {error && (
            <div className="summary-error" role="alert">
              <span>{error}</span>
              <button type="button" onClick={() => setError('')}>
                知道了
              </button>
            </div>
          )}
        </section>

        <TokenCard
          token={view.token}
          created={Boolean(created)}
          busy={busy}
          unused="还没有机器人用过"
          empty="生成令牌后，把它填到机器人平台的 API Key 里。"
          onIssue={issue}
          onWithdraw={withdraw}
        >
          {created && (
            <div className="assistant-setup">
              <p className="summary-hint">
                令牌只显示这一次。在机器人平台新增一个 OpenAI 类型的模型服务，API Base URL 填：
              </p>
              <CopyField label="API Base URL" text={view.api_url} />
              <p className="summary-hint">API Key 填：</p>
              <CopyField label="API Key" text={created} />
              <p className="summary-hint">
                模型名称随意填写，例如 gpt-3.5-turbo。微调后的模型不会调用工具，请在平台上关掉工具和插件。机器人平台跑在
                Docker 里时，地址里的 127.0.0.1 要换成宿主机在 Docker 网络里的地址（例如 172.17.0.1），并在启动服务时用
                --host 指定这个地址。
              </p>
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
