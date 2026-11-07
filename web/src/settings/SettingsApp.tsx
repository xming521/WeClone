import { useCallback, useEffect, useId, useState, type FormEvent, type ReactNode } from 'react';
import { ChevronRight, Info, Monitor, Moon, Sun, Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api } from '../api';
import type { RunVersion } from '../build/progress';
import { PRIVACY_LEVELS } from '../data';
import { confirm } from '../feedback/confirm';
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
  generated_at: string | null;
  reviewed_at: string | null;
  style: boolean;
  demo: boolean;
  version: string | null;
}

// Round lengths from a short sitting to a week; the server's range decides which of them are offered.
const SESSION_CHOICES = [30 * 60, 2 * 3600, 8 * 3600, 24 * 3600, 7 * 24 * 3600];
const THEME_ICONS = { system: Monitor, light: Sun, dark: Moon };

function Section({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  const id = useId();
  return (
    <section className="assistant-card settings-section" aria-labelledby={id}>
      <div className="settings-section__head">
        <h2 id={id}>{title}</h2>
        {action}
      </div>
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
  const [busy, setBusy] = useState(false);
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
    setBusy(true);
    try {
      await api.put('run', { name: version.name });
      // The map, the review queue and their selection belonged to the run shown before.
      useProfileStore.getState().resetSession();
      toast(t(($) => $.settings.run.switched, { time: formatTime(version.generated_at) ?? version.name }));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
      loadRuns();
    }
  }

  async function remove(version: RunVersion) {
    const time = formatTime(version.generated_at ?? version.started_at) ?? version.name;
    const ok = await confirm(
      version.complete
        ? {
            title: t(($) => $.settings.run.confirmDeleteTitle),
            body: t(($) => $.settings.run.confirmDelete, { time }),
            action: t(($) => $.common.delete),
            tone: 'danger',
          }
        : {
            title: t(($) => $.settings.run.confirmDeleteBuildTitle),
            body: t(($) => $.settings.run.confirmDeleteBuild, { time }),
            action: t(($) => $.common.delete),
            tone: 'danger',
          },
    );
    if (!ok) return;
    setBusy(true);
    try {
      await api.delete(`runs/${encodeURIComponent(version.name)}`);
      toast(
        version.complete ? t(($) => $.settings.run.deleted, { time }) : t(($) => $.settings.run.deletedBuild, { time }),
      );
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
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

        {/* First: which portrait every other page shows. */}
        <Section
          title={t(($) => $.settings.run.title)}
          action={
            run &&
            !run.demo && (
              <a className="wc-btn wc-btn--sm" href="#/build">
                {t(($) => $.settings.run.build)}
              </a>
            )
          }
        >
          {run?.demo && <p className="summary-placeholder">{t(($) => $.settings.run.demo)}</p>}
          {run && !run.demo && (
            <VersionList
              versions={versions}
              busy={busy}
              onSwitch={(version) => void switchTo(version)}
              onDelete={(version) => void remove(version)}
            />
          )}
        </Section>

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

/** Every run, newest first: the one shown marked, the others a click away or deleted. A build that ended without a
 * portrait can only be deleted, which clears the chat snapshot and partial results it left. */
function VersionList({
  versions,
  busy,
  onSwitch,
  onDelete,
}: {
  versions: RunVersion[];
  busy: boolean;
  onSwitch: (version: RunVersion) => void;
  onDelete: (version: RunVersion) => void;
}) {
  const { t } = useTranslation();
  const deleteLabel = (version: RunVersion) =>
    version.complete ? t(($) => $.settings.run.delete) : t(($) => $.settings.run.deleteBuild);
  if (!versions.length) return <p className="summary-placeholder">{t(($) => $.settings.run.empty)}</p>;
  return (
    <ul className="settings-versions">
      {versions.map((version) => (
        <li key={version.name} aria-current={version.current || undefined}>
          <div className="settings-row__text">
            {/* The run's folder name, for the command line. */}
            <span className="settings-versions__name" title={version.name}>
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
            !version.current && (
              <div className="settings-versions__actions">
                {version.complete && (
                  <button className="wc-btn wc-btn--sm" type="button" disabled={busy} onClick={() => onSwitch(version)}>
                    {t(($) => $.settings.run.switch)}
                  </button>
                )}
                <button
                  className="wc-iconbtn wc-iconbtn--sm settings-versions__delete"
                  type="button"
                  disabled={busy}
                  aria-label={deleteLabel(version)}
                  title={deleteLabel(version)}
                  onClick={() => onDelete(version)}
                >
                  <Trash2 size={15} strokeWidth={1.8} aria-hidden="true" />
                </button>
              </div>
            )
          )}
        </li>
      ))}
    </ul>
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
