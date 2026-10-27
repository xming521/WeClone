import { useRef, useState, type ReactNode } from 'react';
import { Check, ClipboardCopy, KeyRound, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import i18n from '../i18n';

// Parts shared by the pages that let other programs in: personal assistants (MCP) and chat bots (/v1).

/** The one token a kind of client connects with, as GET /api/assistant and /api/bots describe it. */
export interface TokenState {
  created: string;
  last_used: string | null;
}

export function moment(value: string | number) {
  const date = new Date(typeof value === 'number' ? value * 1000 : value);
  return date.toLocaleString(i18n.language, { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
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
      <button type="button" onClick={() => void copy()}>
        {copied ? <Check size={14} /> : <ClipboardCopy size={14} />}
        {copied ? t(($) => $.common.copied) : t(($) => $.common.copy)}
      </button>
      {failed && <small>{t(($) => $.access.copyBlocked)}</small>}
    </div>
  );
}

/** The token card: when it was made and last used, generate or withdraw it, and `children` once a new one shows. */
export function TokenCard(props: {
  token: TokenState | null;
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
  return (
    <section
      className={`assistant-card${props.created ? ' assistant-created' : ''}`}
      aria-label={t(($) => $.access.token.title)}
    >
      <div className="assistant-card-head">
        <h2>
          <KeyRound size={15} /> {t(($) => $.access.token.title)}
        </h2>
        <span className={`assistant-state${token ? ' on' : ''}`}>
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
        <button className="primary" type="button" disabled={props.busy} onClick={props.onIssue}>
          {token ? <RefreshCw size={14} /> : <KeyRound size={14} />}
          {token ? t(($) => $.access.token.reissue) : t(($) => $.access.token.issue)}
        </button>
        {token && (
          <button type="button" disabled={props.busy} onClick={props.onWithdraw}>
            {t(($) => $.access.token.withdraw)}
          </button>
        )}
      </div>
      {props.children}
    </section>
  );
}
