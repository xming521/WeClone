import { useCallback, useEffect, useId, useState, type FormEvent, type ReactNode } from 'react';
import { ArrowRight, CircleAlert, CircleCheck, CircleDashed, CircleMinus, LoaderCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api, ApiError } from '../api';
import { replayNightIntro } from '../atlas/nightIntro';
import i18n from '../i18n';
import { BrandMark } from '../BrandMark';
import { confirm } from '../feedback/confirm';
import { PageTopbar } from '../settings/PageTopbar';
import { useRun } from '../settings/useRun';
import { useProfileStore } from '../store';
import { useBuildStatus, useBuildStore } from './buildState';
import { ChatUpload } from './ChatUpload';
import { Party } from './Party';
import { PurposeArt } from './PurposeArt';
import { BuildBar } from './BuildBar';
import {
  buildStep,
  clock,
  type BuildCheck,
  type BuildStage,
  type BuildStatus,
  type ChatSource,
  type Problem,
  type Purpose,
  type StageState,
} from './progress';
import '../summary/summary.css';
import '../assistant/assistant.css';
import '../settings/settings.css';
import './build.css';

const PURPOSES: Purpose[] = ['assistant', 'clone'];
// What each build makes, named by the pages that show it: the clone's build adds the speaking style (stage_list in
// weclone/data/agent/profile_pipeline.py).
const OUTPUTS = {
  assistant: ['profile', 'summary'],
  clone: ['profile', 'summary', 'style'],
} as const satisfies Record<Purpose, readonly ('profile' | 'summary' | 'style')[]>;
// The clone's own requirement, shown on its card instead of in the list.
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
  return (
    <ol className="build-stages">
      {status.stages.map((stage: BuildStage, index) => {
        const Icon = STAGE_ICONS[stage.state];
        const elapsed = since(stage.started_at, stage.finished_at, now);
        return (
          <li key={stage.key} className="build-stage" data-state={stage.state}>
            <Icon className="build-stage__icon" size={16} strokeWidth={1.9} aria-hidden="true" />
            <span className="build-stage__step">
              {index + 1}/{status.stages.length}
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

/** A numbered step of the form, its number filled in once the step is done. */
function Step({
  n,
  id,
  title,
  done,
  aside,
  children,
}: {
  n: number;
  id: string;
  title: string;
  done: boolean;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="build-step" aria-labelledby={id}>
      <div className="build-step__head">
        <span className="build-step__n" data-done={done || undefined} aria-hidden="true">
          {n}
        </span>
        <h2 id={id}>{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

/** One purpose as a card to pick: its picture, what it is for and the pages the build fills. */
function PurposeCard({
  purpose,
  checked,
  disabled,
  onPick,
  children,
}: {
  purpose: Purpose;
  checked: boolean;
  disabled: boolean;
  onPick: () => void;
  children?: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <label className="build-purpose">
      <input type="radio" name="purpose" className="wc-sr" checked={checked} disabled={disabled} onChange={onPick} />
      <PurposeArt purpose={purpose} />
      <span className="build-purpose__body">
        <span className="build-purpose__head">
          <span className="build-purpose__name">{t(($) => $.build.purposes[purpose].label)}</span>
          <span className="build-purpose__check" aria-hidden="true" />
        </span>
        <span className="build-purpose__hint">{t(($) => $.build.purposes[purpose].hint)}</span>
        <span className="build-purpose__outputs">
          {OUTPUTS[purpose].map((page) => (
            <span key={page} data-extra={!(OUTPUTS.assistant as readonly string[]).includes(page) || undefined}>
              {t(($) => $.modes[page])}
            </span>
          ))}
        </span>
        {children}
      </span>
    </label>
  );
}

function Prepare({ last }: { last: BuildStatus | null }) {
  const { t } = useTranslation();
  const purposeId = useId();
  const chatsId = useId();
  const confirmId = useId();
  const tokensId = useId();
  const [purpose, setPurpose] = useState<Purpose>(last?.purpose ?? 'assistant');
  const [tokens, setTokens] = useState(savedTokens);
  // The limit the check last ran with, and a counter that runs the check again.
  const [checked, setChecked] = useState(() => tokenLimit(tokens));
  const [round, setRound] = useState(0);
  const [check, setCheck] = useState<BuildCheck | null>(null);
  const [checking, setChecking] = useState(true);
  // The chats asked for, null for the server's choice: the uploaded ones when there are any.
  const [source, setSource] = useState<ChatSource | null>(null);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const limit = tokenLimit(tokens);

  const runCheck = useCallback(async (purpose: Purpose, limit: number | null, source: ChatSource | null) => {
    setChecking(true);
    try {
      const query = new URLSearchParams({ purpose });
      if (limit) query.set('max_context_tokens', String(limit));
      if (source) query.set('chats', source);
      setCheck(await api.get<BuildCheck>(`build/check?${query}`));
      setError('');
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void runCheck(purpose, checked, source);
  }, [purpose, checked, source, round, runCheck]);
  const recheck = () => (limit === checked ? setRound((value) => value + 1) : setChecked(limit));

  async function start(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      // The chats the check showed.
      const body = { purpose, consent, max_context_tokens: limit, chats: check?.chats.source };
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
  const chats = check?.chats;
  const chosen = chats?.[chats.source];
  const askTokens =
    backend?.provider === 'api' || problems.some((problem) => problem.code === 'context_unknown') || tokens !== '';
  const ready = Boolean(check) && !problems.length && !checking;

  return (
    <form className="build-form" onSubmit={(event) => void start(event)}>
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

      <Step
        n={1}
        id={purposeId}
        title={t(($) => $.build.prepare.purpose)}
        done
        aside={<span className="build-step__aside">{t(($) => $.build.prepare.purposeHint)}</span>}
      >
        <div className="build-purposes" role="radiogroup" aria-labelledby={purposeId}>
          {PURPOSES.map((item) => (
            <PurposeCard
              key={item}
              purpose={item}
              checked={purpose === item}
              disabled={busy}
              onPick={() => setPurpose(item)}
            >
              {item === purpose &&
                styleProblems.map((problem) => (
                  <span key={problem.code} className="build-purpose__problem" role="alert">
                    {problemText(problem)}
                  </span>
                ))}
            </PurposeCard>
          ))}
        </div>
      </Step>

      <Step
        n={2}
        id={chatsId}
        title={t(($) => $.build.prepare.chats)}
        done={Boolean(chosen?.count)}
        aside={
          <span className="build-step__aside build-count" data-ok={Boolean(chosen?.count) || undefined}>
            {chats && chosen ? (
              <>
                {chosen.count > 0 && <CircleCheck size={15} strokeWidth={2} aria-hidden="true" />}
                {chats.source === 'config' && chats.config.platform && chats.config.platform !== 'chat' && (
                  <code>{chats.config.platform}</code>
                )}
                {t(($) => $.build.prepare.chatCount, { count: chosen.count })}
              </>
            ) : (
              t(($) => $.build.prepare.checking)
            )}
          </span>
        }
      >
        {chats && chats.upload.count > 0 && chats.config.count > 0 && (
          <div className="wc-seg" role="radiogroup" aria-labelledby={chatsId}>
            {(['upload', 'config'] as const).map((item) => (
              <button
                key={item}
                type="button"
                role="radio"
                aria-checked={chats.source === item}
                disabled={busy}
                onClick={() => setSource(item)}
              >
                {t(($) => $.build.prepare.sources[item])} · {chats[item].count}
              </button>
            ))}
          </div>
        )}
        {chats?.source === 'config' && <p className="build-step__note">{t(($) => $.build.prepare.chatsHint)}</p>}
        <ChatUpload
          disabled={busy}
          empty={!chats?.upload.count && !chats?.config.count}
          onUploaded={() => {
            setSource('upload');
            setRound((value) => value + 1);
          }}
        />
      </Step>

      <Step n={3} id={confirmId} title={t(($) => $.build.prepare.confirm)} done={ready && consent}>
        <div className="assistant-card build-panel">
          <div className="build-backend">
            <div className="settings-row__text">
              <span className="settings-row__label">{t(($) => $.build.prepare.backend)}</span>
              <span className="settings-row__hint">{t(($) => $.build.prepare.backendHint)}</span>
            </div>
            <div className="build-facts">
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

          <p className="build-notice">{t(($) => $.build.prepare.notice)}</p>

          {error && (
            <div className="summary-error" role="alert">
              <span>{error}</span>
              <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={() => setError('')}>
                {t(($) => $.common.dismiss)}
              </button>
            </div>
          )}

          <div className="build-actions build-start">
            <label className="wc-check build-consent">
              <input type="checkbox" checked={consent} disabled={busy} onChange={() => setConsent(!consent)} />
              <span className="wc-check__box" aria-hidden="true" />
              <span>{t(($) => $.build.prepare.consent, { backend: backend?.provider ?? '' })}</span>
            </label>
            <button className="wc-btn wc-btn--primary wc-btn--lg" type="submit" disabled={!ready || !consent || busy}>
              {t(($) => $.build.prepare.start)}
            </button>
          </div>
        </div>
      </Step>
    </form>
  );
}

/** The stage a running build is on, big, above the bar of its progress. */
function Now({ status }: { status: BuildStatus }) {
  const { t } = useTranslation();
  const now = useNow(true);
  const { step, steps } = buildStep(status);
  const stage = status.stages[step - 1];
  return (
    <div className="build-now">
      <div className="build-now__text">
        <span className="build-now__step">{t(($) => $.build.running.step, { step, steps })}</span>
        <h3 className="build-now__stage">{t(($) => $.build.stages[stage.key])}</h3>
        <span className="build-now__meta">
          {stage.total != null && <span>{t(($) => $.build.files, { done: stage.done ?? 0, total: stage.total })}</span>}
          <span>
            {t(($) => $.build.running.elapsed, { time: clock((now - Date.parse(status.started_at)) / 1000) })}
          </span>
        </span>
      </div>
      <BuildBar status={status} stage={stage.key} now={now} />
    </div>
  );
}

function Running({ status }: { status: BuildStatus }) {
  const { t } = useTranslation();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function cancel() {
    const ok = await confirm({
      title: t(($) => $.build.running.confirmCancelTitle),
      body: t(($) => $.build.running.confirmCancel),
      action: t(($) => $.build.running.cancel),
      dismiss: t(($) => $.build.running.keep),
      tone: 'danger',
    });
    if (!ok) return;
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
      <Now status={status} />
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

/** A succeeded build, celebrated until the owner goes to see the portrait or builds another. */
function Celebration({ status }: { status: BuildStatus }) {
  const { t } = useTranslation();
  const run = useRun();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  // Counts the celebrations, so the picture can start again.
  const [round, setRound] = useState(0);
  const took = clock((Date.parse(status.finished_at ?? status.started_at) - Date.parse(status.started_at)) / 1000);
  const see = () => useBuildStore.getState().see(status.run);

  async function open() {
    setBusy(true);
    try {
      // The server opens a build on finishing unless the portrait shown has reviews.
      if (!status.current) {
        await api.put('run', { name: status.run });
        useProfileStore.getState().resetSession();
        await useBuildStore.getState().refresh();
      }
      see();
      // The new portrait opens as on a first visit, with the night intro.
      replayNightIntro();
      window.location.hash = '#/';
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="assistant-card build-card build-done">
      <button
        className="build-done__art"
        type="button"
        title={t(($) => $.build.finished.succeeded.replay)}
        aria-label={t(($) => $.build.finished.succeeded.replay)}
        onClick={() => setRound((value) => value + 1)}
      >
        <Party key={round} />
      </button>
      <span className="assistant-state on">{t(($) => $.build.purposes[status.purpose].label)}</span>
      <h2 className="build-done__title">{t(($) => $.build.finished.succeeded.title)}</h2>
      <p className="build-done__facts">
        {status.chats != null && <span>{t(($) => $.build.finished.succeeded.read, { count: status.chats })}</span>}
        <span>{t(($) => $.build.finished.succeeded.took, { time: took })}</span>
      </p>
      <p className="summary-hint">
        {status.current ? t(($) => $.build.finished.succeeded.opened) : t(($) => $.build.finished.succeeded.hint)}
      </p>
      {!status.current && run?.reviewed_at && <p className="summary-hint">{t(($) => $.build.finished.reviewsStay)}</p>}
      {error && (
        <div className="summary-error" role="alert">
          <span>{error}</span>
          <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={() => setError('')}>
            {t(($) => $.common.dismiss)}
          </button>
        </div>
      )}
      <div className="build-actions build-done__actions">
        <button className="wc-btn wc-btn--lg" type="button" onClick={see}>
          {t(($) => $.build.finished.another)}
        </button>
        <button className="wc-btn wc-btn--primary wc-btn--lg" type="button" disabled={busy} onClick={() => void open()}>
          {status.current ? t(($) => $.build.finished.succeeded.view) : t(($) => $.build.finished.open)}
          <ArrowRight size={16} strokeWidth={2} aria-hidden="true" />
        </button>
      </div>
      <details className="build-done__details">
        <summary>{t(($) => $.build.finished.succeeded.details)}</summary>
        <StageList status={status} />
      </details>
    </section>
  );
}

/** A build that failed, was cancelled or stopped without saying how. */
function Finished({ status, onAgain }: { status: BuildStatus; onAgain: () => void }) {
  const { t } = useTranslation();
  const stage = status.stages.find((item) => item.key === status.error?.key);
  const state = status.state as Exclude<BuildStatus['state'], 'running' | 'succeeded'>;

  return (
    <section className="assistant-card build-card">
      <div className="assistant-card-head">
        <h2>{t(($) => $.build.finished[state].title)}</h2>
        <span className="assistant-state">{t(($) => $.build.purposes[status.purpose].label)}</span>
      </div>
      <p className="summary-hint">{t(($) => $.build.finished[state].hint)}</p>
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
      <p className="summary-hint">{t(($) => $.build.finished.log, { log: status.log })}</p>
      <div className="build-actions">
        <button className="wc-btn wc-btn--primary" type="button" onClick={onAgain}>
          {t(($) => $.build.finished.again)}
        </button>
      </div>
    </section>
  );
}

export function BuildApp() {
  const { t } = useTranslation();
  const status = useBuildStatus();
  const seen = useBuildStore((state) => state.seen);
  // A build that did not succeed stays on the page until the owner asks for another.
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
  let body;
  if (status?.state === 'running') body = <Running status={status} />;
  else if (status?.state === 'succeeded' && status.run !== seen) body = <Celebration status={status} />;
  else if (status && status.state !== 'succeeded' && !fresh)
    body = <Finished status={status} onAgain={() => setFresh(true)} />;
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
