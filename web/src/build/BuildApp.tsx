import { useCallback, useEffect, useId, useState, type FormEvent } from 'react';
import { CircleAlert, CircleCheck, CircleDashed, CircleMinus, LoaderCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api, ApiError } from '../api';
import i18n from '../i18n';
import { BrandMark } from '../BrandMark';
import { PageTopbar } from '../settings/PageTopbar';
import { useRun } from '../settings/useRun';
import { useProfileStore } from '../store';
import { useBuildStatus, useBuildStore } from './buildState';
import {
  clock,
  type BuildCheck,
  type BuildStage,
  type BuildStatus,
  type Problem,
  type Purpose,
  type StageState,
} from './progress';
import '../summary/summary.css';
import '../assistant/assistant.css';
import '../settings/settings.css';
import './build.css';

const PURPOSES: Purpose[] = ['assistant', 'clone'];
// The clone's own requirement, shown under its choice instead of in the list.
const STYLE_PROBLEM = 'style_deps_missing';
const STAGE_ICONS = {
  pending: CircleDashed,
  running: LoaderCircle,
  succeeded: CircleCheck,
  failed: CircleAlert,
  cancelled: CircleMinus,
} satisfies Record<StageState, unknown>;

/** A check problem in the page's language when it knows the code, else the server's text. */
function problemText(problem: Problem): string {
  const key = `build.problems.${problem.code}`;
  if (!i18n.exists(key)) return problem.message;
  const params = Object.fromEntries(
    Object.entries(problem.params).map(([name, value]) => [
      name,
      Array.isArray(value) ? value.join(i18n.t(($) => $.common.listSeparator)) : value,
    ]),
  );
  return i18n.t(key, params);
}

function since(start: string | null, end: string | null, now: number): string | null {
  if (!start) return null;
  return clock(((end ? Date.parse(end) : now) - Date.parse(start)) / 1000);
}

/** The time now, a second at a time while `live`. */
function useNow(live: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [live]);
  return now;
}

function StageList({ status }: { status: BuildStatus }) {
  const { t } = useTranslation();
  const now = useNow(status.state === 'running');
  const steps = status.stages.at(-1)?.step ?? 0;
  return (
    <ol className="build-stages">
      {status.stages.map((stage: BuildStage) => {
        const Icon = STAGE_ICONS[stage.state];
        const elapsed = since(stage.started_at, stage.finished_at, now);
        return (
          <li key={stage.key} className="build-stage" data-state={stage.state}>
            <Icon className="build-stage__icon" size={16} strokeWidth={1.9} aria-hidden="true" />
            <span className="build-stage__step">
              {stage.step}/{steps}
            </span>
            <span className="build-stage__name">
              {t(($) => $.build.stages[stage.key])}
              <span className="wc-sr">{t(($) => $.build.stageStates[stage.state])}</span>
            </span>
            <span className="build-stage__count">
              {stage.total != null && t(($) => $.build.files, { done: stage.done ?? 0, total: stage.total })}
            </span>
            <span className="build-stage__time">{elapsed}</span>
          </li>
        );
      })}
    </ol>
  );
}

// The limit typed for the last build started here, offered again for the next one.
const TOKENS_KEY = 'weclone.build.maxContextTokens';

function savedTokens(): string {
  try {
    return window.localStorage.getItem(TOKENS_KEY) ?? '';
  } catch {
    return '';
  }
}

function tokenLimit(text: string): number | null {
  return /^\d+$/.test(text) && Number(text) > 0 ? Number(text) : null;
}

function Prepare({ last }: { last: BuildStatus | null }) {
  const { t } = useTranslation();
  const tokensId = useId();
  const [purpose, setPurpose] = useState<Purpose>(last?.purpose ?? 'assistant');
  const [tokens, setTokens] = useState(savedTokens);
  // The limit the check last ran with, and a counter that runs the check again.
  const [checked, setChecked] = useState(() => tokenLimit(tokens));
  const [round, setRound] = useState(0);
  const [check, setCheck] = useState<BuildCheck | null>(null);
  const [checking, setChecking] = useState(true);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const limit = tokenLimit(tokens);

  const runCheck = useCallback(async (purpose: Purpose, limit: number | null) => {
    setChecking(true);
    try {
      const query = new URLSearchParams({ purpose });
      if (limit) query.set('max_context_tokens', String(limit));
      setCheck(await api.get<BuildCheck>(`build/check?${query}`));
      setError('');
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void runCheck(purpose, checked);
  }, [purpose, checked, round, runCheck]);
  const recheck = () => (limit === checked ? setRound(round + 1) : setChecked(limit));

  async function start(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const body = { purpose, consent, max_context_tokens: limit };
      useBuildStore.setState({ status: await api.post<BuildStatus>('build', body) });
      try {
        if (limit) window.localStorage.setItem(TOKENS_KEY, String(limit));
      } catch {
        // Without storage the limit is typed again next time.
      }
      void useBuildStore.getState().refresh();
    } catch (reason) {
      setError((reason as Error).message);
      // Problems found since the last check replace it.
      const problems = reason instanceof ApiError ? reason.data.problems : undefined;
      if (check && Array.isArray(problems)) setCheck({ ...check, problems: problems as Problem[] });
    } finally {
      setBusy(false);
    }
  }

  const problems = check?.problems ?? [];
  const listed = problems.filter((problem) => problem.code !== STYLE_PROBLEM);
  const styleProblems = problems.filter((problem) => problem.code === STYLE_PROBLEM);
  const backend = check?.backend;
  const askTokens =
    backend?.provider === 'api' || problems.some((problem) => problem.code === 'context_unknown') || tokens !== '';
  const ready = Boolean(check) && !problems.length && !checking;

  return (
    <form className="assistant-card build-card" onSubmit={(event) => void start(event)}>
      {last?.state === 'succeeded' && last.current && (
        <p className="summary-hint build-last">
          {t(($) => $.build.prepare.last, {
            time: new Date(last.finished_at ?? last.started_at).toLocaleString(i18n.language, {
              dateStyle: 'medium',
              timeStyle: 'short',
            }),
          })}
        </p>
      )}
      <div className="settings-row">
        <div className="settings-row__text">
          <span className="settings-row__label">{t(($) => $.build.prepare.purpose)}</span>
          <span className="settings-row__hint">{t(($) => $.build.purposes[purpose].hint)}</span>
        </div>
        <div className="settings-row__control">
          <div className="wc-seg" role="radiogroup" aria-label={t(($) => $.build.prepare.purpose)}>
            {PURPOSES.map((item) => (
              <button
                key={item}
                type="button"
                role="radio"
                aria-checked={purpose === item}
                disabled={busy}
                onClick={() => setPurpose(item)}
              >
                {t(($) => $.build.purposes[item].label)}
              </button>
            ))}
          </div>
        </div>
        {styleProblems.map((problem) => (
          <p key={problem.code} className="wc-field__hint build-problem" data-tone="danger">
            {problemText(problem)}
          </p>
        ))}
      </div>

      <div className="settings-row">
        <div className="settings-row__text">
          <span className="settings-row__label">{t(($) => $.build.prepare.backend)}</span>
          <span className="settings-row__hint">{t(($) => $.build.prepare.backendHint)}</span>
        </div>
        <div className="settings-row__control build-facts">
          {backend?.provider ? (
            <>
              <code>{backend.provider}</code>
              {backend.model && <code>{backend.model}</code>}
              {backend.max_context_tokens != null && (
                <span>{t(($) => $.build.prepare.limit, { count: backend.max_context_tokens })}</span>
              )}
            </>
          ) : (
            <span>{checking ? t(($) => $.build.prepare.checking) : t(($) => $.settings.run.none)}</span>
          )}
        </div>
        {askTokens && (
          <div className="wc-field build-tokens">
            <label className="wc-field__label" htmlFor={tokensId}>
              {t(($) => $.build.prepare.tokens)}
            </label>
            <div className="wc-input">
              <input
                id={tokensId}
                inputMode="numeric"
                placeholder="128000"
                value={tokens}
                disabled={busy}
                onChange={(event) => setTokens(event.target.value.trim())}
                onBlur={() => limit !== checked && setChecked(limit)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    recheck();
                  }
                }}
              />
            </div>
            <span className="wc-field__hint">{t(($) => $.build.prepare.tokensHint)}</span>
          </div>
        )}
      </div>

      <div className="settings-row">
        <div className="settings-row__text">
          <span className="settings-row__label">{t(($) => $.build.prepare.chats)}</span>
          <span className="settings-row__hint">{t(($) => $.build.prepare.chatsHint)}</span>
        </div>
        <div className="settings-row__control build-facts">
          {check ? (
            <>
              {check.chats.platform && <code>{check.chats.platform}</code>}
              <span>{t(($) => $.build.prepare.chatCount, { count: check.chats.count })}</span>
            </>
          ) : (
            <span>{t(($) => $.build.prepare.checking)}</span>
          )}
        </div>
      </div>

      {listed.length > 0 && (
        <div className="build-problems" role="alert">
          <strong>{t(($) => $.build.prepare.problems)}</strong>
          <ul>
            {listed.map((problem, index) => (
              <li key={`${problem.code}-${index}`}>{problemText(problem)}</li>
            ))}
          </ul>
          <button className="wc-btn wc-btn--sm" type="button" disabled={checking} onClick={recheck}>
            {t(($) => $.build.prepare.recheck)}
          </button>
        </div>
      )}

      <label className="wc-check build-consent">
        <input type="checkbox" checked={consent} disabled={busy} onChange={() => setConsent(!consent)} />
        <span className="wc-check__box" aria-hidden="true" />
        <span>{t(($) => $.build.prepare.consent, { backend: backend?.provider ?? '' })}</span>
      </label>

      {error && (
        <div className="summary-error" role="alert">
          <span>{error}</span>
          <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={() => setError('')}>
            {t(($) => $.common.dismiss)}
          </button>
        </div>
      )}

      <div className="build-actions">
        <button className="wc-btn wc-btn--primary" type="submit" disabled={!ready || !consent || busy}>
          {t(($) => $.build.prepare.start)}
        </button>
      </div>
    </form>
  );
}

function Running({ status }: { status: BuildStatus }) {
  const { t } = useTranslation();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function cancel() {
    if (!window.confirm(t(($) => $.build.running.confirmCancel))) return;
    setBusy(true);
    try {
      await api.delete('build');
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
      await useBuildStore.getState().refresh();
    }
  }
  return (
    <section className="assistant-card build-card">
      <div className="assistant-card-head">
        <h2>{t(($) => $.build.purposes[status.purpose].label)}</h2>
        <span className="assistant-state on">{t(($) => $.build.running.state)}</span>
      </div>
      <StageList status={status} />
      <p className="summary-hint">{t(($) => $.build.running.leave, { log: status.log })}</p>
      {error && (
        <div className="summary-error" role="alert">
          <span>{error}</span>
          <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={() => setError('')}>
            {t(($) => $.common.dismiss)}
          </button>
        </div>
      )}
      <div className="build-actions">
        <button className="wc-btn wc-btn--danger" type="button" disabled={busy} onClick={() => void cancel()}>
          {t(($) => $.build.running.cancel)}
        </button>
      </div>
    </section>
  );
}

function Finished({ status, onAgain }: { status: BuildStatus; onAgain: () => void }) {
  const { t } = useTranslation();
  const run = useRun();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const succeeded = status.state === 'succeeded';
  const stage = status.stages.find((item) => item.key === status.error?.key);

  async function open() {
    setBusy(true);
    try {
      await api.put('run', { name: status.run });
      useProfileStore.getState().resetSession();
      await useBuildStore.getState().refresh();
      window.location.hash = '#/';
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="assistant-card build-card">
      <div className="assistant-card-head">
        <h2>{t(($) => $.build.finished[status.state as Exclude<BuildStatus['state'], 'running'>].title)}</h2>
        <span className={`assistant-state${succeeded ? ' on' : ''}`}>
          {t(($) => $.build.purposes[status.purpose].label)}
        </span>
      </div>
      <p className="summary-hint">
        {t(($) => $.build.finished[status.state as Exclude<BuildStatus['state'], 'running'>].hint)}
      </p>
      {status.error && (
        <div className="build-problems" role="alert">
          <strong>
            {stage
              ? t(($) => $.build.finished.stoppedAt, { stage: t(($) => $.build.stages[stage.key]) })
              : t(($) => $.build.finished.stoppedBefore)}
          </strong>
          <p>
            <code>{status.error.type}</code>
            {status.error.message ? ` ${status.error.message}` : ''}
          </p>
        </div>
      )}
      <StageList status={status} />
      {succeeded && run?.reviewed_at && <p className="summary-hint">{t(($) => $.build.finished.reviewsStay)}</p>}
      {!succeeded && <p className="summary-hint">{t(($) => $.build.finished.log, { log: status.log })}</p>}
      {error && (
        <div className="summary-error" role="alert">
          <span>{error}</span>
          <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={() => setError('')}>
            {t(($) => $.common.dismiss)}
          </button>
        </div>
      )}
      <div className="build-actions">
        {succeeded ? (
          <>
            <button className="wc-btn" type="button" onClick={onAgain}>
              {t(($) => $.build.finished.another)}
            </button>
            <button className="wc-btn wc-btn--primary" type="button" disabled={busy} onClick={() => void open()}>
              {t(($) => $.build.finished.open)}
            </button>
          </>
        ) : (
          <button className="wc-btn wc-btn--primary" type="button" onClick={onAgain}>
            {t(($) => $.build.finished.again)}
          </button>
        )}
      </div>
    </section>
  );
}

export function BuildApp() {
  const { t } = useTranslation();
  const status = useBuildStatus();
  // The finished build stays on the page until the owner asks for another.
  const [fresh, setFresh] = useState(false);
  useEffect(() => {
    if (status?.state === 'running') setFresh(false);
  }, [status?.state]);

  if (status === undefined)
    return (
      <div className="load-state">
        <BrandMark className="loading-mark" />
        <h1>{t(($) => $.build.loading)}</h1>
      </div>
    );
  // A succeeded build the server already shows needs nothing more: the page offers the next one.
  const settled = status === null || (status.state === 'succeeded' && status.current);
  let body;
  if (status?.state === 'running') body = <Running status={status} />;
  else if (status && !settled && !fresh) body = <Finished status={status} onAgain={() => setFresh(true)} />;
  else body = <Prepare last={status} />;
  return (
    <div className="summary-page">
      <PageTopbar mode="build" heading={t(($) => $.build.title)} />
      <main className="summary-main assistant-main build-main">
        <section className="summary-intro">
          <h1>{t(($) => $.build.title)}</h1>
          <p>{t(($) => $.build.intro)}</p>
        </section>
        {body}
      </main>
    </div>
  );
}
