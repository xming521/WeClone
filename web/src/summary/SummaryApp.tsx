import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, ClipboardCopy, EyeOff, PencilLine, Sparkles, Undo2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api } from '../api';
import { LevelSwitch } from '../assistant/accessParts';
import { LogoutButton } from '../Auth';
import { AppBrand, BrandMark } from '../BrandMark';
import { BuildIndicator } from '../build/BuildIndicator';
import { LanguageSwitch } from '../LanguageSwitch';
import { SettingsLink } from '../settings/SettingsLink';
import { ModeSwitch } from '../ModeSwitch';
import { confirm } from '../feedback/confirm';
import { SUMMARY_LEVELS, stateCounts, summaryState } from './summaryState';
import type {
  DomainSummary,
  SummaryDomain,
  SummaryLevel,
  SummaryState,
  SummaryStatus,
  SummaryView,
} from './summaryState';
import './summary.css';

function SummaryEditor({
  summary,
  limits,
  busy,
  onCancel,
  onSave,
}: {
  summary: DomainSummary;
  limits: SummaryView['limits'];
  busy: boolean;
  onCancel: () => void;
  onSave: (title: string, text: string, approve: boolean) => void;
}) {
  const { t } = useTranslation();
  const [title, setTitle] = useState(summary.title);
  const [text, setText] = useState(summary.summary);
  const valid =
    title.trim().length > 0 &&
    text.trim().length > 0 &&
    title.trim().length <= limits.title &&
    text.trim().length <= limits.summary;
  return (
    <form
      className="summary-editor"
      onSubmit={(event) => {
        event.preventDefault();
        if (valid) onSave(title, text, true);
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') onCancel();
      }}
    >
      <label className="wc-field">
        <span className="wc-field__label">{t(($) => $.summary.editor.title)}</span>
        <span className="wc-input">
          <input value={title} maxLength={limits.title} autoFocus onChange={(event) => setTitle(event.target.value)} />
        </span>
      </label>
      <label className="wc-field">
        <span className="wc-field__label">{t(($) => $.summary.editor.summary)}</span>
        <textarea value={text} rows={3} maxLength={limits.summary} onChange={(event) => setText(event.target.value)} />
        <span className="wc-field__hint summary-count">
          {text.trim().length}/{limits.summary}
        </span>
      </label>
      <div className="summary-actions">
        <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={onCancel}>
          {t(($) => $.common.cancel)}
        </button>
        <button
          className="wc-btn wc-btn--sm"
          type="button"
          disabled={!valid || busy}
          onClick={() => onSave(title, text, false)}
        >
          {t(($) => $.common.save)}
        </button>
        <button className="wc-btn wc-btn--sm" type="submit" disabled={!valid || busy}>
          {t(($) => $.summary.editor.saveApprove)}
        </button>
      </div>
    </form>
  );
}

/** The ReviewStatus glyph of each state that is a decision of yours. */
const REVIEW_STATUS = { pending: 'pending', approved: 'adopted', rejected: 'rejected' } as const;

/** A decision of yours reads as the review glyph and its word. A state of generation (nothing to write from, not
 * written yet, written but out of date) is only a word: nothing waits for your review. */
function StateLabel({ state, text }: { state: SummaryState; text: string }) {
  if (state === 'pending' || state === 'approved' || state === 'rejected')
    return (
      <span className="wc-status" data-status={REVIEW_STATUS[state]}>
        {text}
      </span>
    );
  return <span className={`summary-state ${state}`}>{text}</span>;
}

function SummaryCard({
  item,
  limits,
  empty,
  busy,
  editing,
  onEdit,
  onCancel,
  onSave,
  onStatus,
}: {
  item: SummaryDomain;
  limits: SummaryView['limits'];
  /** Why a domain has nothing to summarise at this level. */
  empty: string;
  busy: boolean;
  editing: boolean;
  onEdit: () => void;
  onCancel: () => void;
  onSave: (title: string, text: string, approve: boolean) => void;
  onStatus: (status: SummaryStatus) => void;
}) {
  const { t } = useTranslation();
  const state = summaryState(item);
  const summary = item.summary;
  return (
    <li className={`summary-card ${state}`}>
      <div className="summary-card-head">
        <span className="summary-domain">{item.domain}</span>
        <span className="summary-meta">
          {t(($) => $.summary.card.facts, { count: item.facts })}
          {item.as_of && (
            <>
              {' · '}
              <span className="wc-mono">{t(($) => $.summary.card.asOf, { date: item.as_of })}</span>
            </>
          )}
        </span>
        <StateLabel state={state} text={t(($) => $.summary.states[state])} />
      </div>
      {summary && editing ? (
        <SummaryEditor summary={summary} limits={limits} busy={busy} onCancel={onCancel} onSave={onSave} />
      ) : summary && state !== 'empty' ? (
        <div className="summary-body">
          <strong>{summary.title}</strong>
          <p>{summary.summary}</p>
          {state === 'stale' && <p className="summary-note">{t(($) => $.summary.card.stale)}</p>}
        </div>
      ) : (
        <p className="summary-placeholder">
          {state === 'empty' ? empty : state === 'missing' ? t(($) => $.summary.card.missing) : undefined}
        </p>
      )}
      {summary && !editing && state !== 'empty' && (
        <div className="summary-actions">
          {state === 'pending' && (
            <button className="wc-btn wc-btn--sm" type="button" disabled={busy} onClick={() => onStatus('approved')}>
              <Check size={14} strokeWidth={1.8} /> {t(($) => $.summary.card.approve)}
            </button>
          )}
          {(state === 'pending' || state === 'approved') && (
            <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" disabled={busy} onClick={onEdit}>
              <PencilLine size={14} strokeWidth={1.8} /> {t(($) => $.summary.card.edit)}
            </button>
          )}
          {state === 'approved' && (
            <button className="wc-btn wc-btn--sm" type="button" disabled={busy} onClick={() => onStatus('pending')}>
              <Undo2 size={14} strokeWidth={1.8} /> {t(($) => $.summary.card.unapprove)}
            </button>
          )}
          {state === 'rejected' ? (
            <button className="wc-btn wc-btn--sm" type="button" disabled={busy} onClick={() => onStatus('pending')}>
              <Undo2 size={14} strokeWidth={1.8} /> {t(($) => $.summary.card.restore)}
            </button>
          ) : (
            <button className="wc-btn wc-btn--sm" type="button" disabled={busy} onClick={() => onStatus('rejected')}>
              <EyeOff size={14} strokeWidth={1.8} /> {t(($) => $.summary.card.reject)}
            </button>
          )}
        </div>
      )}
    </li>
  );
}

export function SummaryApp() {
  const { t } = useTranslation();
  const [level, setLevel] = useState<SummaryLevel>('full');
  const [view, setView] = useState<SummaryView | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState<'' | 'generate' | 'save'>('');
  const [editing, setEditing] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const sequence = useRef(0);
  const exportRef = useRef<HTMLTextAreaElement>(null);

  const load = useCallback(async () => {
    const ticket = ++sequence.current;
    try {
      const next = await api.get<SummaryView>(`summaries?level=${level}`);
      if (ticket === sequence.current) setView(next);
    } catch (reason) {
      if (ticket === sequence.current) setError((reason as Error).message);
    }
  }, [level]);

  useEffect(() => {
    void load();
  }, [load]);

  // A generation started elsewhere (another tab, or before a reload) is polled until it ends.
  useEffect(() => {
    if (!view?.generating || busy) return;
    const timer = window.setTimeout(() => void load(), 3000);
    return () => window.clearTimeout(timer);
  }, [view, busy, load]);

  async function run(kind: 'generate' | 'save', request: () => Promise<unknown>) {
    setBusy(kind);
    setError('');
    try {
      await request();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy('');
      await load();
    }
  }

  function review(items: SummaryDomain[], status: SummaryStatus) {
    void run('save', () =>
      api.post('summaries/review', {
        level,
        status,
        items: items.map((item) => ({ domain: item.domain, version: item.summary!.version })),
      }),
    );
  }

  function save(item: SummaryDomain, title: string, summary: string, approve: boolean) {
    void run('save', async () => {
      await api.patch('summaries', {
        level,
        domain: item.domain,
        title,
        summary,
        approve,
        version: item.summary!.version,
      });
      setEditing(null);
    });
  }

  async function copy() {
    if (!view?.text) return;
    if (
      level === 'full' &&
      !(await confirm({
        title: t(($) => $.summary.export.confirmFullTitle),
        body: t(($) => $.summary.export.confirmFull),
        action: t(($) => $.common.copy),
      }))
    )
      return;
    try {
      await navigator.clipboard.writeText(view.text);
    } catch {
      // Pages served over plain http to another machine have no clipboard API.
      exportRef.current?.select();
      if (!document.execCommand('copy')) {
        setError(t(($) => $.common.copyBlocked));
        return;
      }
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  if (!view && error)
    return (
      <div className="load-state">
        <BrandMark />
        <h1>{t(($) => $.summary.loadFailed)}</h1>
        <p>{error}</p>
        <button className="wc-btn" type="button" onClick={() => void load()}>
          {t(($) => $.common.retry)}
        </button>
      </div>
    );
  if (!view)
    return (
      <div className="load-state">
        <BrandMark className="loading-mark" />
        <h1>{t(($) => $.summary.loading)}</h1>
      </div>
    );

  const counts = stateCounts(view.domains);
  const due = counts.missing + counts.stale;
  const pending = view.domains.filter((item) => summaryState(item) === 'pending');
  const generating = busy === 'generate' || view.generating;

  return (
    <div className="summary-page">
      <header className="summary-topbar">
        <div className="topbar-leading">
          <AppBrand />
          <ModeSwitch mode="summary" />
          <div className="topbar-heading">
            <span className="topbar-divider">/</span>{' '}
            <strong>{t(($) => $.summary.domains, { count: view.domains.length })}</strong>
          </div>
        </div>
        <div className="topbar-actions">
          <BuildIndicator />
          <LanguageSwitch />
          <SettingsLink />
          <LogoutButton />
        </div>
      </header>
      <main className="summary-main">
        <section className="summary-intro">
          <h1>{t(($) => $.summary.title)}</h1>
          <p>
            {level === 'full' ? t(($) => $.summary.intro.full) : t(($) => $.summary.intro.short)}
            {t(($) => $.common.sentenceSeparator)}
            {t(($) => $.summary.intro.review)}
          </p>
          <div className="summary-toolbar">
            <LevelSwitch
              label={t(($) => $.summary.level)}
              value={level}
              disabled={!!busy}
              levels={SUMMARY_LEVELS.map((item) => ({
                level: item,
                text: item === 'full' ? t(($) => $.summary.full.label) : t(($) => $.privacy.levels[item].label),
                title: item === 'full' ? t(($) => $.summary.full.hint) : t(($) => $.privacy.levels[item].hint),
              }))}
              onChange={(item) => {
                setEditing(null);
                setLevel(item);
              }}
            />
            <button
              className="wc-btn wc-btn--primary"
              type="button"
              disabled={!!busy || generating || !due}
              onClick={() => void run('generate', () => api.post('summaries/generate', { level }))}
            >
              <Sparkles size={14} strokeWidth={1.8} />
              {generating
                ? t(($) => $.summary.generating)
                : due
                  ? t(($) => $.summary.generate, { count: due })
                  : t(($) => $.summary.generated)}
            </button>
            <button
              className="wc-btn"
              type="button"
              disabled={!!busy || !pending.length}
              onClick={() => review(pending, 'approved')}
            >
              <Check size={14} strokeWidth={1.8} />{' '}
              {pending.length
                ? t(($) => $.summary.approveAllCount, { count: pending.length })
                : t(($) => $.summary.approveAll)}
            </button>
            <span className="summary-counts">
              {t(($) => $.summary.counts, {
                approved: counts.approved,
                pending: counts.pending,
                due,
                rejected: counts.rejected,
              })}
            </span>
          </div>
          {generating ? (
            <p className="summary-hint">{t(($) => $.summary.generatingHint)}</p>
          ) : (
            view.failure && (
              <div className="summary-error" role="status">
                <span>
                  {t(($) => $.summary.failed, { count: view.failure.failed })}
                  {view.failure.reason &&
                    t(($) => $.common.sentenceSeparator) +
                      t(($) => $.summary.failedReason, { reason: view.failure.reason })}
                </span>
              </div>
            )
          )}
          {error && (
            <div className="summary-error" role="alert">
              <span>{error}</span>
              <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={() => setError('')}>
                {t(($) => $.common.dismiss)}
              </button>
            </div>
          )}
        </section>
        <div className="summary-columns">
          <ol className="summary-list">
            {view.domains.map((item) => (
              <SummaryCard
                key={item.domain}
                item={item}
                limits={view.limits}
                empty={level === 'full' ? t(($) => $.summary.card.emptyFull) : t(($) => $.summary.card.empty)}
                busy={!!busy}
                editing={editing === item.domain}
                onEdit={() => setEditing(item.domain)}
                onCancel={() => setEditing(null)}
                onSave={(title, text, approve) => save(item, title, text, approve)}
                onStatus={(status) => review([item], status)}
              />
            ))}
          </ol>
          <aside className="summary-export" aria-label={t(($) => $.summary.export.title)}>
            <h2>{t(($) => $.summary.export.title)}</h2>
            {view.text ? (
              <>
                <textarea ref={exportRef} readOnly value={view.text} onFocus={(event) => event.target.select()} />
                <div className="summary-export-foot">
                  <span className="wc-count">{t(($) => $.summary.export.chars, { count: view.text.length })}</span>
                  <button className="wc-btn wc-btn--sm" type="button" onClick={() => void copy()}>
                    <ClipboardCopy size={14} strokeWidth={1.8} />{' '}
                    {copied ? t(($) => $.common.copied) : t(($) => $.common.copy)}
                  </button>
                </div>
                {level === 'full' ? (
                  <p className="summary-hint summary-warning">
                    <span className="wc-phase" data-privacy="sensitive" aria-hidden="true" />
                    {t(($) => $.summary.export.fullWarning)}
                  </p>
                ) : (
                  <p className="summary-hint">{t(($) => $.summary.export.hint)}</p>
                )}
              </>
            ) : (
              <p className="summary-placeholder">{t(($) => $.summary.export.empty)}</p>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}
