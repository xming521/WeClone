import { useId, useRef, useState, type ReactNode } from 'react';
import { Check, ClipboardCopy, KeyRound, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import i18n from '../i18n';
import type { SummaryLevel } from '../summary/summaryState';

// Parts shared by the pages that let other programs in: personal assistants (MCP) and chat bots (/v1).

/** The one token a kind of client connects with, as GET /api/assistant and /api/bots describe it. */
export interface TokenState {
  created: string;
  last_used: string | null;
}

/** The PrivacyBadge degree a level stands for. A level caps what is handed out at the most private degree it still
 * includes, so `public` and `normal` are their own degrees and `full` reaches the top one, sensitive. */
export const LEVEL_PRIVACY = { public: 'public', normal: 'normal', full: 'sensitive' } as const;

export function moment(value: string | number) {
  const date = new Date(typeof value === 'number' ? value * 1000 : value);
  return date.toLocaleString(i18n.language, { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/** Picks the privacy level a page works at: one exclusive choice about how much of the same portrait is in play. */
export function LevelSwitch({
  label,
  value,
  levels,
  disabled,
  onChange,
}: {
  label: string;
  value: SummaryLevel;
  levels: { level: SummaryLevel; text: string; title?: string }[];
  disabled?: boolean;
  onChange: (level: SummaryLevel) => void;
}) {
  const labelId = useId();
  return (
    <div className="summary-levels">
      <span id={labelId}>{label}</span>
      <div className="wc-seg" role="radiogroup" aria-labelledby={labelId}>
        {levels.map(({ level, text, title }) => (
          <button
            key={level}
            type="button"
            role="radio"
            aria-checked={level === value}
            disabled={disabled}
            title={title}
            onClick={() => onChange(level)}
          >
            <span className="wc-phase" data-privacy={LEVEL_PRIVACY[level]} aria-hidden="true" />
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}

export function CopyField({ label, text }: { label: string; text: string }) {
  const { t } = useTranslation();
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const [failed, setFailed] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Pages served over plain http to another machine have no clipboard API.
      const selection = window.getSelection();
      if (ref.current && selection) selection.selectAllChildren(ref.current);
      if (!document.execCommand('copy')) {
        setFailed(true);
        return;
      }
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <div className="assistant-copy">
      <pre ref={ref} aria-label={label}>
        {text}
      </pre>
      <button className="wc-btn wc-btn--sm" type="button" onClick={() => void copy()}>
        {copied ? <Check size={14} strokeWidth={1.8} /> : <ClipboardCopy size={14} strokeWidth={1.8} />}
        {copied ? t(($) => $.common.copied) : t(($) => $.common.copy)}
      </button>
      {failed && <small>{t(($) => $.access.copyBlocked)}</small>}
    </div>
  );
}

/** The token card: when it was made and last used, generate or withdraw it, and `children` once a new one shows.
 * `tone` says whose connection the token state describes: assistants read what you decided to share (`you`), bots
 * call the clone itself (`clone`). */
export function TokenCard(props: {
  token: TokenState | null;
  tone: 'you' | 'clone';
  /** Whether a token was just generated, so the setup below it is open. */
  created: boolean;
  busy: boolean;
  /** Said while nobody has used the token yet. */
  unused: string;
  /** Said while there is no token. */
  empty: string;
  onIssue: () => void;
  onWithdraw: () => void;
  children?: ReactNode;
}) {
  const { t } = useTranslation();
  const { token } = props;
  // The clone's own colour, with its dot, only where the state is the clone's.
  const state = !token ? '' : props.tone === 'clone' ? ' on clone wc-clone' : ' on';
  return (
    <section
      className={`assistant-card${props.created ? ' assistant-created' : ''}`}
      aria-label={t(($) => $.access.token.title)}
    >
      <div className="assistant-card-head">
        <h2>
          <KeyRound size={16} strokeWidth={1.8} /> {t(($) => $.access.token.title)}
        </h2>
        <span className={`assistant-state${state}`}>
          {token ? t(($) => $.access.token.active) : t(($) => $.access.token.none)}
        </span>
      </div>
      <p className="summary-hint">
        {token
          ? `${t(($) => $.access.token.created, { time: moment(token.created) })} · ${
              token.last_used ? t(($) => $.access.token.lastUsed, { time: moment(token.last_used) }) : props.unused
            }`
          : props.empty}
      </p>
      <div className="assistant-actions">
        {/* Without a token, generating one is the page's step. With one, it can only break what is set up. */}
        <button
          className={`wc-btn${token ? '' : ' wc-btn--primary'}`}
          type="button"
          disabled={props.busy}
          onClick={props.onIssue}
        >
          {token ? <RefreshCw size={14} strokeWidth={1.8} /> : <KeyRound size={14} strokeWidth={1.8} />}
          {token ? t(($) => $.access.token.reissue) : t(($) => $.access.token.issue)}
        </button>
        {token && (
          <button className="wc-btn wc-btn--danger" type="button" disabled={props.busy} onClick={props.onWithdraw}>
            {t(($) => $.access.token.withdraw)}
          </button>
        )}
      </div>
      {props.children}
    </section>
  );
}
