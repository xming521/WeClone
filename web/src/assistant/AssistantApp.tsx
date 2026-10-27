import { useCallback, useEffect, useState } from 'react';
import { Lock, LockOpen } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { api } from '../api';
import { LogoutButton } from '../Auth';
import { LanguageSwitch } from '../LanguageSwitch';
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

const LEVELS: Level[] = ['full', 'normal', 'public'];
const CLIENTS = ['Claude Code', 'Codex'] as const;
type Client = (typeof CLIENTS)[number];

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
  const { t } = useTranslation();
  const duration = (hours: number) =>
    hours % 24 ? t(($) => $.assistant.hours, { count: hours }) : t(($) => $.assistant.days, { count: hours / 24 });
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
    if (view?.token && !window.confirm(t(($) => $.assistant.confirmReissue))) return;
    void run(async () => setCreated((await api.post<{ token: string }>('assistant/token')).token));
  }

  function withdraw() {
    if (!window.confirm(t(($) => $.assistant.confirmWithdraw))) return;
    void run(async () => {
      await api.delete('assistant/token');
      setCreated('');
    });
  }

  if (!view && error)
    return (
      <div className="load-state">
        <span>✳</span>
        <h1>{t(($) => $.assistant.loadFailed)}</h1>
        <p>{error}</p>
        <button onClick={() => void load()}>{t(($) => $.common.retry)}</button>
      </div>
    );
  if (!view)
    return (
      <div className="load-state">
        <span className="loading-mark">✳</span>
        <h1>{t(($) => $.assistant.loading)}</h1>
      </div>
    );

  const allowed = view.allowed_until != null;
  const sharedCount = view.catalog.filter((item) => item.shared).length;

  return (
    <div className="summary-page">
      <header className="summary-topbar">
        <div className="topbar-leading">
          <ModeSwitch mode="assistant" />
          <div className="topbar-heading">
            <span className="topbar-divider">/</span>{' '}
            <strong>
              {view.token
                ? t(($) => $.assistant.connected, { level: t(($) => $.assistant.levels[view.level].label) })
                : t(($) => $.assistant.noToken)}
            </strong>
          </div>
        </div>
        <div className="topbar-actions">
          <LanguageSwitch />
          <LogoutButton />
        </div>
      </header>
      <main className="summary-main assistant-main">
        <section className="summary-intro">
          <h1>{t(($) => $.assistant.title)}</h1>
          <p>{t(($) => $.assistant.intro)}</p>
          <p className="assistant-address">
            <Trans
              i18nKey={view.encrypted ? ($) => $.assistant.address : ($) => $.assistant.addressPlaintext}
              values={{ url: view.mcp_url }}
              components={{ code: <code /> }}
            />
          </p>
          {error && (
            <div className="summary-error" role="alert">
              <span>{error}</span>
              <button type="button" onClick={() => setError('')}>
                {t(($) => $.common.dismiss)}
              </button>
            </div>
          )}
        </section>

        <section className="assistant-card" aria-label={t(($) => $.assistant.scope)}>
          <div className="assistant-card-head">
            <h2>{t(($) => $.assistant.scope)}</h2>
          </div>
          <div className="summary-levels" role="radiogroup" aria-label={t(($) => $.assistant.level)}>
            <span>{t(($) => $.assistant.level)}</span>
            {LEVELS.map((level) => (
              <button
                key={level}
                type="button"
                role="radio"
                aria-checked={level === view.level}
                disabled={busy}
                onClick={() => level !== view.level && void run(() => api.patch('assistant', { level }))}
              >
                {t(($) => $.assistant.levels[level].label)}
              </button>
            ))}
          </div>
          <p className={`summary-hint${view.level === 'full' ? ' summary-warning' : ''}`}>
            {t(($) => $.assistant.levels[view.level].hint)}
          </p>
          <div className="assistant-domains-head">
            <h3>{t(($) => $.assistant.domains)}</h3>
            <span>
              {view.domains === null
                ? t(($) => $.assistant.allDomains)
                : t(($) => $.assistant.someDomains, { shared: sharedCount, total: view.catalog.length })}
            </span>
            {view.domains !== null && (
              <button type="button" disabled={busy} onClick={() => share(null)}>
                {t(($) => $.assistant.resetDomains)}
              </button>
            )}
          </div>
          {view.catalog.length ? (
            <ul className="assistant-domains" aria-label={t(($) => $.assistant.sharedDomains)}>
              {view.catalog.map((item) => (
                <li key={item.name}>
                  <label className={item.shared ? 'shared' : undefined}>
                    <input type="checkbox" checked={item.shared} disabled={busy} onChange={() => toggle(item)} />
                    <span>{item.name}</span>
                    <small>
                      {item.summary
                        ? t(($) => $.assistant.domainFactsSummary, { count: item.facts })
                        : t(($) => $.assistant.domainFacts, { count: item.facts })}
                    </small>
                  </label>
                </li>
              ))}
            </ul>
          ) : (
            <p className="summary-placeholder">{t(($) => $.assistant.noDomains)}</p>
          )}
          <p className="summary-hint">
            <Trans
              i18nKey={($) => $.assistant.domainsNote}
              count={view.summaries}
              components={{ link: <a href="#/summary" /> }}
            />
          </p>
        </section>

        {view.encrypted && (
          <section className="assistant-card" aria-label={t(($) => $.assistant.access.title)}>
            <div className="assistant-card-head">
              <h2>
                {allowed ? <LockOpen size={15} /> : <Lock size={15} />} {t(($) => $.assistant.access.title)}
              </h2>
              <span className={`assistant-state${allowed ? ' on' : ''}`}>
                {allowed
                  ? t(($) => $.assistant.access.until, { time: moment(view.allowed_until!) })
                  : t(($) => $.assistant.access.closed)}
              </span>
            </div>
            <p className="summary-hint">{t(($) => $.assistant.access.hint)}</p>
            <div className="assistant-actions">
              {view.hours.map((hours) => (
                <button
                  key={hours}
                  type="button"
                  disabled={busy}
                  onClick={() => void run(() => api.post('assistant/access', { hours }))}
                >
                  {allowed
                    ? t(($) => $.assistant.access.change, { duration: duration(hours) })
                    : t(($) => $.assistant.access.allow, { duration: duration(hours) })}
                </button>
              ))}
              {allowed && (
                <button type="button" disabled={busy} onClick={() => void run(() => api.delete('assistant/access'))}>
                  {t(($) => $.assistant.access.close)}
                </button>
              )}
            </div>
          </section>
        )}

        <TokenCard
          token={view.token}
          created={Boolean(created)}
          busy={busy}
          unused={t(($) => $.assistant.unused)}
          empty={t(($) => $.assistant.noTokenHint)}
          onIssue={issue}
          onWithdraw={withdraw}
        >
          {created && (
            <div className="assistant-setup">
              <p className="summary-hint">{t(($) => $.assistant.setup.once)}</p>
              <div className="summary-levels" role="radiogroup" aria-label={t(($) => $.assistant.setup.clients)}>
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
              <CopyField
                label={t(($) => $.assistant.setup.commands, { client })}
                text={setupCommands(client, view.mcp_url, created)}
              />
              <p className="summary-hint">
                <Trans i18nKey={($) => $.assistant.setup.other} components={{ code: <code /> }} />
              </p>
              <CopyField label={t(($) => $.access.token.title)} text={created} />
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
