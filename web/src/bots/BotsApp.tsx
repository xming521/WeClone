import { useCallback, useEffect, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { api } from '../api';
import { LogoutButton } from '../Auth';
import { LanguageSwitch } from '../LanguageSwitch';
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
  const { t } = useTranslation();
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
    if (view?.token && !window.confirm(t(($) => $.bots.confirmReissue))) return;
    void run(async () => setCreated((await api.post<{ token: string }>('bots/token')).token));
  }

  function withdraw() {
    if (!window.confirm(t(($) => $.bots.confirmWithdraw))) return;
    void run(async () => {
      await api.delete('bots/token');
      setCreated('');
    });
  }

  if (!view && error)
    return (
      <div className="load-state">
        <span>✳</span>
        <h1>{t(($) => $.bots.loadFailed)}</h1>
        <p>{error}</p>
        <button onClick={() => void load()}>{t(($) => $.common.retry)}</button>
      </div>
    );
  if (!view)
    return (
      <div className="load-state">
        <span className="loading-mark">✳</span>
        <h1>{t(($) => $.bots.loading)}</h1>
      </div>
    );

  return (
    <div className="summary-page">
      <header className="summary-topbar">
        <div className="topbar-leading">
          <ModeSwitch mode="bots" />
          <div className="topbar-heading">
            <span className="topbar-divider">/</span>{' '}
            <strong>{view.token ? t(($) => $.bots.connected) : t(($) => $.assistant.noToken)}</strong>
          </div>
        </div>
        <div className="topbar-actions">
          <LanguageSwitch />
          <LogoutButton />
        </div>
      </header>
      <main className="summary-main assistant-main">
        <section className="summary-intro">
          <h1>{t(($) => $.bots.title)}</h1>
          <p>{t(($) => $.bots.intro)}</p>
          <p className="assistant-address">
            <Trans i18nKey={($) => $.bots.address} values={{ url: view.api_url }} components={{ code: <code /> }} />
          </p>
          <p className="summary-hint">
            {t(($) => $.bots.tokenLife)}
            {view.encrypted && t(($) => $.bots.encrypted)}
          </p>
          {!view.inference && (
            <p className="summary-hint summary-warning">
              <Trans i18nKey={($) => $.bots.noInference} components={{ code: <code /> }} />
            </p>
          )}
          {error && (
            <div className="summary-error" role="alert">
              <span>{error}</span>
              <button type="button" onClick={() => setError('')}>
                {t(($) => $.common.dismiss)}
              </button>
            </div>
          )}
        </section>

        <TokenCard
          token={view.token}
          created={Boolean(created)}
          busy={busy}
          unused={t(($) => $.bots.unused)}
          empty={t(($) => $.bots.noTokenHint)}
          onIssue={issue}
          onWithdraw={withdraw}
        >
          {created && (
            <div className="assistant-setup">
              <p className="summary-hint">{t(($) => $.bots.setup.once)}</p>
              <CopyField label="API Base URL" text={view.api_url} />
              <p className="summary-hint">{t(($) => $.bots.setup.key)}</p>
              <CopyField label="API Key" text={created} />
              <p className="summary-hint">{t(($) => $.bots.setup.model)}</p>
              <button className="assistant-done" type="button" onClick={() => setCreated('')}>
                {t(($) => $.assistant.setup.done)}
              </button>
            </div>
          )}
        </TokenCard>
      </main>
    </div>
  );
}
