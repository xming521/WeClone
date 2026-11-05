import { useCallback, useEffect, useId, useState, type FormEvent, type ReactNode } from 'react';
import { ChevronRight, Info, Monitor, Moon, Sun } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api } from '../api';
import type { RunVersion } from '../build/progress';
import { PRIVACY_LEVELS } from '../data';
import { toast } from '../feedback/toast';
import { LanguageSwitch } from '../LanguageSwitch';
import { saveDefaultPrivacy, useProfileStore } from '../store';
import { THEME_CHOICES, useTheme } from '../theme';
import { formatDuration, formatTime } from './duration';
import { PageTopbar } from './PageTopbar';
import '../summary/summary.css';
import '../assistant/assistant.css';
import './settings.css';

// Shape of GET /api/auth/status, served by weclone/server/auth.py.
interface AuthStatus {
  storage_mode: 'encrypted' | 'plaintext';
  demo_password: string | null;
  session_seconds: number;
  /** Shortest and longest session the server accepts, in seconds. */
  session_range: [number, number];
}

// Shape of GET /api/run, served by weclone/server/profile_review.py.
export interface RunInfo {
  /** Null when no run is open, as before the first build. */
  name: string | null;
  /** Null as well in the demo, which keeps the server's directories to itself. */
  path: string | null;
  generated_at: string | null;
  reviewed_at: string | null;
  style: boolean;
  demo: boolean;
  version: string | null;
}

// Round lengths from a short sitting to a week; the server's range decides which of them are offered.
const SESSION_CHOICES = [30 * 60, 2 * 3600, 8 * 3600, 24 * 3600, 7 * 24 * 3600];
const THEME_ICONS = { system: Monitor, light: Sun, dark: Moon };

function Section({ title, children }: { title: string; children: ReactNode }) {
  const id = useId();
  return (
    <section className="assistant-card settings-section" aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      {children}
    </section>
  );
}

function Row({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="settings-row">
      <div className="settings-row__text">
        <span className="settings-row__label">{label}</span>
        {hint && <span className="settings-row__hint">{hint}</span>}
      </div>
      <div className="settings-row__control">{children}</div>
    </div>
  );
}

export function SettingsApp() {
  const { t } = useTranslation();
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [run, setRun] = useState<RunInfo | null>(null);
  const [versions, setVersions] = useState<RunVersion[]>([]);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState('');
  const theme = useTheme((state) => state.choice);
  const chooseTheme = useTheme((state) => state.choose);
  const privacy = useProfileStore((state) => state.privacy);

  const loadRuns = useCallback(() => {
    api.get<RunInfo>('run').then(setRun, (reason: Error) => setError(reason.message));
    api.get<RunVersion[]>('runs').then(setVersions, (reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    api.get<AuthStatus>('auth/status').then(setStatus, (reason: Error) => setError(reason.message));
    loadRuns();
  }, [loadRuns]);

  async function switchTo(version: RunVersion) {
    setSwitching(true);
    try {
      await api.put('run', { name: version.name });
      // The map, the review queue and their selection belonged to the run shown before.
      useProfileStore.getState().resetSession();
      toast(t(($) => $.settings.run.switched, { time: formatTime(version.generated_at) ?? version.name }));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setSwitching(false);
      loadRuns();
    }
  }

  async function setSession(seconds: number) {
    try {
      await api.put('auth/session-length', { seconds });
      setStatus((current) => current && { ...current, session_seconds: seconds });
      toast(t(($) => $.settings.session.saved, { duration: formatDuration(seconds) }));
    } catch (reason) {
      setError((reason as Error).message);
    }
  }

  const sessions = status
    ? SESSION_CHOICES.filter((seconds) => seconds >= status.session_range[0] && seconds <= status.session_range[1])
    : [];
  return (
    <div className="summary-page">
      <PageTopbar mode="settings" heading={t(($) => $.settings.title)} />
      <main className="summary-main assistant-main settings-main">
        <section className="summary-intro">
          <h1>{t(($) => $.settings.title)}</h1>
          <p>{t(($) => $.settings.intro)}</p>
          {error && (
            <div className="summary-error" role="alert">
              <span>{error}</span>
              <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={() => setError('')}>
                {t(($) => $.common.dismiss)}
              </button>
            </div>
          )}
        </section>

        <Section title={t(($) => $.settings.appearance.title)}>
          <Row label={t(($) => $.settings.appearance.theme)} hint={t(($) => $.settings.appearance.themeHint)}>
            <div className="wc-seg" role="radiogroup" aria-label={t(($) => $.settings.appearance.theme)}>
              {THEME_CHOICES.map((choice) => {
                const Icon = THEME_ICONS[choice];
                return (
                  <button
                    key={choice}
                    type="button"
                    role="radio"
                    aria-checked={theme === choice}
                    onClick={() => chooseTheme(choice)}
                  >
                    <Icon size={14} strokeWidth={1.8} aria-hidden="true" />
                    {t(($) => $.settings.appearance.themes[choice])}
                  </button>
                );
              })}
            </div>
          </Row>
          <Row label={t(($) => $.common.language)}>
            <LanguageSwitch />
          </Row>
        </Section>

        <Section title={t(($) => $.settings.review.title)}>
          <Row label={t(($) => $.settings.review.privacy)} hint={t(($) => $.settings.review.privacyHint)}>
            <div className="settings-chips" role="group" aria-label={t(($) => $.settings.review.privacy)}>
              {PRIVACY_LEVELS.map((level) => (
                <button
                  key={level}
                  type="button"
                  className="wc-chip"
                  data-privacy={level === 'unrated' ? undefined : level}
                  aria-pressed={privacy.includes(level)}
                  title={t(($) => $.privacy.levels[level].hint)}
                  onClick={() =>
                    saveDefaultPrivacy(
                      privacy.includes(level) ? privacy.filter((item) => item !== level) : [...privacy, level],
                    )
                  }
                >
                  {level !== 'unrated' && <span className="wc-phase" aria-hidden="true" />}
                  {t(($) => $.privacy.levels[level].label)}
                </button>
              ))}
            </div>
          </Row>
        </Section>

        <Section title={t(($) => $.settings.security.title)}>
          {status && (
            <>
              <Row
                label={t(($) => $.settings.security.storage)}
                hint={
                  status.storage_mode === 'encrypted'
                    ? t(($) => $.settings.security.encrypted)
                    : t(($) => $.settings.security.plaintext)
                }
              >
                <span className="assistant-state">
                  {status.storage_mode === 'encrypted'
                    ? t(($) => $.settings.security.encryptedState)
                    : t(($) => $.settings.security.plaintextState)}
                </span>
              </Row>
              <Row label={t(($) => $.settings.session.title)} hint={t(($) => $.settings.session.hint)}>
                <div className="wc-seg" role="radiogroup" aria-label={t(($) => $.settings.session.title)}>
                  {sessions.map((seconds) => (
                    <button
                      key={seconds}
                      type="button"
                      role="radio"
                      aria-checked={status.session_seconds === seconds}
                      disabled={status.demo_password !== null}
                      onClick={() => status.session_seconds !== seconds && void setSession(seconds)}
                    >
                      {formatDuration(seconds)}
                    </button>
                  ))}
                </div>
              </Row>
              <PasswordForm demo={status.demo_password !== null} />
            </>
          )}
        </Section>

        <Section title={t(($) => $.settings.run.title)}>
          {run && (
            <>
              {run.name ? (
                <RunDetails run={run} />
              ) : (
                <p className="summary-placeholder">{t(($) => $.settings.run.empty)}</p>
              )}
              {!run.demo && (
                <VersionList versions={versions} busy={switching} onSwitch={(version) => void switchTo(version)} />
              )}
            </>
          )}
        </Section>

        <a className="assistant-card settings-about-link" href="#/about">
          <Info size={18} strokeWidth={1.8} aria-hidden="true" />
          <span>
            <strong>{t(($) => $.about.title)}</strong>
            <span>{t(($) => $.settings.aboutHint)}</span>
          </span>
          <ChevronRight size={16} strokeWidth={1.8} aria-hidden="true" />
        </a>
      </main>
    </div>
  );
}

export function RunDetails({ run }: { run: RunInfo }) {
  const { t } = useTranslation();
  const none = t(($) => $.settings.run.none);
  return (
    <>
      <dl className="settings-facts">
        <div>
          <dt>{t(($) => $.settings.run.name)}</dt>
          <dd>{run.demo ? t(($) => $.settings.run.demo) : <code>{run.name}</code>}</dd>
        </div>
        {run.path && (
          <div>
            <dt>{t(($) => $.settings.run.path)}</dt>
            <dd>
              <code>{run.path}</code>
            </dd>
          </div>
        )}
        <div>
          <dt>{t(($) => $.settings.run.generated)}</dt>
          <dd>{formatTime(run.generated_at) ?? none}</dd>
        </div>
        <div>
          <dt>{t(($) => $.settings.run.reviewed)}</dt>
          <dd>{formatTime(run.reviewed_at) ?? t(($) => $.settings.run.notReviewed)}</dd>
        </div>
        <div>
          <dt>{t(($) => $.settings.run.style)}</dt>
          <dd>{run.style ? t(($) => $.settings.run.styleYes) : t(($) => $.settings.run.styleNo)}</dd>
        </div>
      </dl>
    </>
  );
}

/** Every run under the runs folder, newest first, with the one shown marked and the others a click away. */
function VersionList({
  versions,
  busy,
  onSwitch,
}: {
  versions: RunVersion[];
  busy: boolean;
  onSwitch: (version: RunVersion) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="settings-versions">
      <div className="settings-versions__head">
        <span className="settings-row__label">{t(($) => $.settings.run.versions)}</span>
        <a className="wc-btn wc-btn--sm" href="#/build">
          {t(($) => $.settings.run.build)}
        </a>
      </div>
      {versions.length > 0 && (
        <ul>
          {versions.map((version) => (
            <li key={version.name} aria-current={version.current || undefined}>
              <div className="settings-row__text">
                <span className="settings-versions__name">
                  {formatTime(version.generated_at ?? version.started_at) ?? <code>{version.name}</code>}
                  {version.current && <span className="assistant-state on">{t(($) => $.settings.run.current)}</span>}
                  {version.state !== 'succeeded' && (
                    <span className="assistant-state">{t(($) => $.build.states[version.state])}</span>
                  )}
                </span>
                <span className="settings-row__hint">
                  {[
                    version.purpose && t(($) => $.build.purposes[version.purpose!].label),
                    version.complete &&
                      (version.reviewed_at
                        ? t(($) => $.settings.run.reviewedAt, { time: formatTime(version.reviewed_at) ?? '' })
                        : t(($) => $.settings.run.notReviewed)),
                    version.style && t(($) => $.settings.run.withStyle),
                  ]
                    .filter(Boolean)
                    .join(t(($) => $.settings.run.separator))}
                </span>
              </div>
              {version.state === 'running' ? (
                <a className="wc-btn wc-btn--sm" href="#/build">
                  {t(($) => $.settings.run.progress)}
                </a>
              ) : (
                version.complete &&
                !version.current && (
                  <button className="wc-btn wc-btn--sm" type="button" disabled={busy} onClick={() => onSwitch(version)}>
                    {t(($) => $.settings.run.switch)}
                  </button>
                )
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PasswordForm({ demo }: { demo: boolean }) {
  const { t } = useTranslation();
  const [current, setCurrent] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (password !== confirmation) {
      setError(t(($) => $.auth.mismatch));
      return;
    }
    setBusy(true);
    setError('');
    try {
      await api.post('auth/password', { old_password: current, password, confirmation });
      setCurrent('');
      setPassword('');
      setConfirmation('');
      toast(t(($) => $.settings.password.done));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const field = (id: string, label: string, value: string, set: (value: string) => void, autoComplete: string) => (
    <div className="wc-field">
      <label className="wc-field__label" htmlFor={id}>
        {label}
      </label>
      <div className="wc-input">
        <input
          id={id}
          type="password"
          autoComplete={autoComplete}
          required
          disabled={busy || demo}
          value={value}
          onChange={(event) => set(event.target.value)}
        />
      </div>
    </div>
  );
  return (
    <form className="settings-password" onSubmit={(event) => void submit(event)}>
      <div className="settings-row__text">
        <span className="settings-row__label">{t(($) => $.settings.password.title)}</span>
        <span className="settings-row__hint">
          {demo ? t(($) => $.settings.password.demo) : t(($) => $.settings.password.hint)}
        </span>
      </div>
      <div className="settings-password__fields">
        {field(
          'current-password',
          t(($) => $.settings.password.current),
          current,
          setCurrent,
          'current-password',
        )}
        {field(
          'new-password',
          t(($) => $.settings.password.next),
          password,
          setPassword,
          'new-password',
        )}
        {field(
          'confirm-password',
          t(($) => $.auth.confirmPassword),
          confirmation,
          setConfirmation,
          'new-password',
        )}
      </div>
      {error && (
        <p className="wc-field__hint" data-tone="danger" role="alert">
          {error}
        </p>
      )}
      <div>
        <button className="wc-btn wc-btn--primary" type="submit" disabled={busy || demo}>
          {t(($) => $.settings.password.submit)}
        </button>
      </div>
    </form>
  );
}
