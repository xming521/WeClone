import { useCallback, useEffect, useId, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { ArrowRight, CircleAlert, CircleCheck, CircleDashed, CircleMinus, LoaderCircle } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { api, ApiError, errorCode } from '../api';
import { replayNightIntro } from '../atlas/nightIntro';
import i18n from '../i18n';
import { BrandMark } from '../BrandMark';
import { confirm } from '../feedback/confirm';
import { ErrorDetails } from '../feedback/ErrorDetails';
import { ModelSettings } from '../settings/ModelSettings';
import type { Provider } from '../settings/modelSettings';
import { PageTopbar } from '../settings/PageTopbar';
import { useRun } from '../settings/useRun';
import { useProfileStore } from '../store';
import { useBuildStatus, useBuildStore } from './buildState';
import { ChatUpload } from './ChatUpload';
import { EstimateRow, trialNeeded } from './EstimateRow';
import { ProblemText } from './problems';
import { Party } from './Party';
import { PurposeArt } from './PurposeArt';
import { BuildBar } from './BuildBar';
import {
  buildStep,
  clock,
  overspent,
  projectBuild,
  type BuildCheck,
  type BuildStage,
  type BuildStatus,
  type Estimate,
  type Problem,
  type Purpose,
  type StageState,
} from './progress';
import { amountText, estimatedTotal, spend, timeText } from './spend';
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
// No chats uploaded yet, which the reminder under the start button says.
const CHATS_PROBLEM = 'chats_missing';
const STAGE_ICONS = {
  pending: CircleDashed,
  running: LoaderCircle,
  succeeded: CircleCheck,
  failed: CircleAlert,
  cancelled: CircleMinus,
} satisfies Record<StageState, unknown>;

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

/** How far a stage is: requests when it counts them, else the chats it has read. */
function progressText(stage: BuildStage): string {
  const counts = { done: stage.done ?? 0, total: stage.total ?? 0 };
  return stage.unit === 'requests' ? i18n.t(($) => $.build.requestsDone, counts) : i18n.t(($) => $.build.files, counts);
}

/** What a stage has sent and cost. Its requests are left out when its progress already counts them. */
function stageSpend(stage: BuildStage): string | null {
  const usage = stage.usage;
  if (!usage || usage.requests <= 0) return null;
  const parts = [
    stage.unit !== 'requests' ? i18n.t(($) => $.build.requests, { count: usage.requests }) : null,
    usage.cost !== null ? amountText(usage.cost) : null,
  ].filter(Boolean);
  return parts.join(' · ') || null;
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
            <span className="build-stage__count">{stage.total != null && progressText(stage)}</span>
            <span className="build-stage__spend">{stageSpend(stage)}</span>
            <span className="build-stage__time">{elapsed}</span>
          </li>
        );
      })}
    </ol>
  );
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
  const needsId = useId();
  const [purpose, setPurpose] = useState<Purpose>(last?.purpose ?? 'assistant');
  // Runs the check again.
  const [round, setRound] = useState(0);
  const [check, setCheck] = useState<BuildCheck | null>(null);
  const [checking, setChecking] = useState(true);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  // The backend in the model form while it has changes not saved yet, and the form's save.
  const [pendingBackend, setPendingBackend] = useState<Provider | null>(null);
  const saveBackend = useRef<() => Promise<boolean>>(null);
  const [error, setError] = useState('');
  // The estimate of what the build sends and costs, for the saved settings, and the trial that makes it more exact.
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [estimateError, setEstimateError] = useState('');
  const [estimateRound, setEstimateRound] = useState(0);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState<{ message: string; detail: string } | null>(null);
  // When the last trial ended: an estimate asked for before that may not have it in, so it is not shown.
  const probedAt = useRef(0);

  const runCheck = useCallback(async (purpose: Purpose) => {
    setChecking(true);
    try {
      const query = new URLSearchParams({ purpose });
      setCheck(await api.get<BuildCheck>(`build/check?${query}`));
      setError('');
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setChecking(false);
    }
  }, []);

  // A new build starts from its own upload, so chats uploaded on an earlier visit go before the first check.
  const cleared = useRef<Promise<unknown>>(null);
  useEffect(() => {
    cleared.current ??= api.delete('data/uploaded').catch(() => undefined);
    void cleared.current.then(() => runCheck(purpose));
  }, [purpose, round, runCheck]);
  const recheck = () => setRound((value) => value + 1);

  // The estimate waits for a check that found nothing, and is made again after each check and after a retry, which is
  // also what changing the purpose or saving the model settings leads to.
  const estimable = Boolean(check) && !check?.problems.length && !checking;
  useEffect(() => {
    setEstimate(null);
    setEstimateError('');
    if (!estimable) return;
    let stale = false;
    const asked = Date.now();
    setEstimating(true);
    api
      .get<Estimate>(`build/estimate?${new URLSearchParams({ purpose })}`)
      .then((result) => {
        if (!stale && asked >= probedAt.current) setEstimate(result);
      })
      .catch((reason: Error) => {
        if (!stale) setEstimateError(reason.message);
      })
      .finally(() => {
        if (!stale) setEstimating(false);
      });
    return () => {
      stale = true;
      setEstimating(false);
    };
  }, [estimable, purpose, round, estimateRound]);

  // Sends the trial requests, which the estimate then comes from. Like starting, it reads the saved settings.
  async function probe() {
    setProbing(true);
    setProbeError(null);
    try {
      if (pendingBackend && !(await saveBackend.current?.())) return;
      setEstimate(await api.post<Estimate>('build/probe', { purpose }));
      probedAt.current = Date.now();
      setEstimateError('');
    } catch (reason) {
      const detail = reason instanceof ApiError ? reason.data.reason : null;
      setProbeError({
        message: (reason as Error).message,
        // The service's own words, which the page cannot put in its own language.
        detail:
          ['probe_failed', 'usage_limit'].includes(errorCode(reason) ?? '') && typeof detail === 'string' ? detail : '',
      });
    } finally {
      setProbing(false);
    }
  }

  async function start(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      // A build reads the saved settings, so changes still in the form are saved first. A trial of the settings before
      // does not price the new ones, which then wait for their own.
      if (pendingBackend) {
        if (!(await saveBackend.current?.())) return;
        const fresh = await api.get<Estimate>(`build/estimate?${new URLSearchParams({ purpose })}`);
        if (trialNeeded(fresh, false)) return;
      }
      const body = { purpose, consent };
      useBuildStore.setState({ status: await api.post<BuildStatus>('build', body) });
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

  // What the check found about the saved model service is out of date while the form has changes, which starting
  // saves and checks again.
  const problems = (check?.problems ?? []).filter((problem) => !(pendingBackend && problem.backend));
  const listed = problems.filter((problem) => problem.code !== STYLE_PROBLEM && problem.code !== CHATS_PROBLEM);
  const styleProblems = problems.filter((problem) => problem.code === STYLE_PROBLEM);
  const backend = check?.backend;
  const provider = pendingBackend ?? backend?.provider;
  const chats = check?.chats;
  const ready = Boolean(check) && !problems.length && !checking;
  // A build starts once a trial has priced it, unless it has nothing to try or the estimate could not be made.
  const trialMissing = trialNeeded(estimate, Boolean(pendingBackend));
  const startable = ready && consent && !trialMissing && Boolean(estimate || estimateError);
  // What starting waits for, said under the button; the check and the estimate in progress say so themselves.
  const needs = [
    ...(!checking && problems.some((problem) => problem.code === CHATS_PROBLEM)
      ? [t(($) => $.build.prepare.needs.chats)]
      : []),
    ...(listed.length && !checking ? [t(($) => $.build.prepare.needs.problems)] : []),
    ...(trialMissing && !probing
      ? [
          (pendingBackend ?? estimate?.backend) === 'codex_exec'
            ? t(($) => $.build.prepare.needs.trialCodex)
            : t(($) => $.build.prepare.needs.trial),
        ]
      : []),
    ...(consent ? [] : [t(($) => $.build.prepare.needs.consent)]),
  ];

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
              disabled={busy || probing}
              onPick={() => setPurpose(item)}
            >
              {item === purpose &&
                styleProblems.map((problem) => (
                  <span key={problem.code} className="build-purpose__problem" role="alert">
                    <ProblemText problem={problem} />
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
        done={Boolean(chats?.count)}
        aside={
          <span className="build-step__aside build-count" data-ok={Boolean(chats?.count) || undefined}>
            {chats ? (
              <>
                {chats.count > 0 && <CircleCheck size={15} strokeWidth={2} aria-hidden="true" />}
                {t(($) => $.build.prepare.chatCount, { count: chats.count })}
              </>
            ) : (
              t(($) => $.build.prepare.checking)
            )}
          </span>
        }
      >
        <ChatUpload disabled={busy} empty={!chats?.count} onUploaded={() => setRound((value) => value + 1)} />
      </Step>

      <Step n={3} id={confirmId} title={t(($) => $.build.prepare.confirm)} done={startable}>
        <div className="assistant-card build-panel">
          <div className="build-backend">
            <ModelSettings
              build={{
                // Codex is not there to use: the form starts on the API.
                preferApi: Boolean(check?.problems.some((problem) => problem.code === 'codex_missing')),
                onPending: setPendingBackend,
                saver: saveBackend,
                onSaved: () => setRound((value) => value + 1),
                facts: (
                  <>
                    <div className="settings-row__text">
                      <span className="settings-row__label">{t(($) => $.build.prepare.backend)}</span>
                      <span className="settings-row__hint">
                        <Trans
                          i18nKey={($) => $.build.prepare.backendHint}
                          components={{ a: <a className="wc-link" href="#/settings" /> }}
                        />
                      </span>
                    </div>
                    <div className="build-facts">
                      {backend?.provider ? (
                        <>
                          <span>{t(($) => $.settings.model.backends[backend.provider!])}</span>
                          {backend.model && <code>{backend.model}</code>}
                        </>
                      ) : (
                        <span>{checking ? t(($) => $.build.prepare.checking) : t(($) => $.settings.run.none)}</span>
                      )}
                    </div>
                  </>
                ),
              }}
            />
          </div>

          {listed.length > 0 && (
            <div className="build-problems" role="alert">
              <strong>{t(($) => $.build.prepare.problems)}</strong>
              <ul>
                {listed.map((problem, index) => (
                  <li key={`${problem.code}-${index}`}>
                    <ProblemText problem={problem} />
                    {typeof problem.params.detail === 'string' && <ErrorDetails text={problem.params.detail} />}
                  </li>
                ))}
              </ul>
              <button className="wc-btn wc-btn--sm" type="button" disabled={checking} onClick={recheck}>
                {t(($) => $.build.prepare.recheck)}
              </button>
            </div>
          )}

          {estimable && (
            <EstimateRow
              estimate={estimate}
              unsaved={pendingBackend}
              estimating={estimating}
              unavailable={estimateError}
              probing={probing}
              probeError={probeError}
              onProbe={() => void probe()}
              onRetry={() => setEstimateRound((value) => value + 1)}
              onDismissProbeError={() => setProbeError(null)}
            />
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
              <span>
                {t(($) => $.build.prepare.consent, {
                  backend: provider ? t(($) => $.settings.model.backends[provider]) : '',
                })}
              </span>
            </label>
            <button
              className="wc-btn wc-btn--primary wc-btn--lg"
              type="submit"
              disabled={!startable || busy}
              aria-describedby={needs.length ? needsId : undefined}
            >
              {t(($) => $.build.prepare.start)}
            </button>
          </div>
          {needs.length > 0 && !busy && (
            <p className="build-needs" id={needsId}>
              {t(($) => $.build.prepare.startNeeds, {
                needs: new Intl.ListFormat(i18n.language, { type: 'conjunction' }).format(needs),
              })}
            </p>
          )}
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
  const spent = spend(status.usage);
  const projection = projectBuild(status, now);
  // Before enough requests are in to project from, the estimate made before the build stands in.
  const forecast = projection
    ? t(($) => (projection.cost === null ? $.build.running.remaining : $.build.running.remainingCost), {
        time: timeText(projection.seconds, projection.seconds),
        amount: amountText(projection.cost ?? 0),
      })
    : estimatedTotal(status.estimate);
  return (
    <div className="build-now">
      <div className="build-now__text">
        <span className="build-now__step">{t(($) => $.build.running.step, { step, steps })}</span>
        <h3 className="build-now__stage">{t(($) => $.build.stages[stage.key])}</h3>
        <span className="build-now__meta">
          {stage.total != null && <span>{progressText(stage)}</span>}
          <span>
            {t(($) => $.build.running.elapsed, { time: clock((now - Date.parse(status.started_at)) / 1000) })}
          </span>
          {spent && <span>{spent.text}</span>}
          {forecast && <span>{forecast}</span>}
        </span>
        {overspent(status, projection) && (
          <p className="build-now__warn" role="status">
            <CircleAlert size={15} strokeWidth={2} aria-hidden="true" />
            {t(($) => $.build.running.overspent)}
          </p>
        )}
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
      <p className="summary-hint">{t(($) => $.build.running.leave)}</p>
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
  const spent = spend(status.usage);
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
        {spent && <span>{spent.text}</span>}
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
  const spent = spend(status.usage);
  // The exception and the log, for a bug report: a cancelled build has nothing to report.
  const details = [
    status.error && [status.error.type, status.error.message].filter(Boolean).join(': '),
    status.log && t(($) => $.build.finished.log, { log: status.log }),
  ]
    .filter(Boolean)
    .join('\n');

  return (
    <section className="assistant-card build-card">
      <div className="assistant-card-head">
        <h2>{t(($) => $.build.finished[state].title)}</h2>
        <span className="assistant-state">{t(($) => $.build.purposes[status.purpose].label)}</span>
      </div>
      <p className="summary-hint">{t(($) => $.build.finished[state].hint)}</p>
      {spent && (
        <p className="summary-hint">
          {spent.kind === 'cost'
            ? t(($) => $.build.finished.spentCost, { amount: spent.value })
            : t(($) => $.build.finished.spentTokens, { tokens: spent.value })}
        </p>
      )}
      {status.error && (
        <div className="build-problems" role="alert">
          <strong>
            {stage
              ? t(($) => $.build.finished.stoppedAt, { stage: t(($) => $.build.stages[stage.key]) })
              : t(($) => $.build.finished.stoppedBefore)}
          </strong>
          {status.error.type === 'UsageLimitReached' && <p>{t(($) => $.build.finished.usageLimit)}</p>}
        </div>
      )}
      {state !== 'cancelled' && details && <ErrorDetails text={details} />}
      <StageList status={status} />
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
