import { useRef, useState, type ReactNode } from 'react';
import { Check, ClipboardCopy, KeyRound, RefreshCw } from 'lucide-react';

// Parts shared by the pages that let other programs in: personal assistants (MCP) and chat bots (/v1).

/** The one token a kind of client connects with, as GET /api/assistant and /api/bots describe it. */
export interface TokenState {
  created: string;
  last_used: string | null;
}

export function moment(value: string | number) {
  const date = new Date(typeof value === 'number' ? value * 1000 : value);
  return date.toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function CopyField({ label, text }: { label: string; text: string }) {
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
        {copied ? '已复制' : '复制'}
      </button>
      {failed && <small>浏览器不允许直接复制，请选中文字后手动复制</small>}
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
  const { token } = props;
  return (
    <section className={`assistant-card${props.created ? ' assistant-created' : ''}`} aria-label="令牌">
      <div className="assistant-card-head">
        <h2>
          <KeyRound size={15} /> 令牌
        </h2>
        <span className={`assistant-state${token ? ' on' : ''}`}>{token ? '有效' : '未生成'}</span>
      </div>
      <p className="summary-hint">
        {token
          ? `生成于 ${moment(token.created)} · ${token.last_used ? `最近使用 ${moment(token.last_used)}` : props.unused}`
          : props.empty}
      </p>
      <div className="assistant-actions">
        <button className="primary" type="button" disabled={props.busy} onClick={props.onIssue}>
          {token ? <RefreshCw size={14} /> : <KeyRound size={14} />}
          {token ? '重新生成' : '生成令牌'}
        </button>
        {token && (
          <button type="button" disabled={props.busy} onClick={props.onWithdraw}>
            停用
          </button>
        )}
      </div>
      {props.children}
    </section>
  );
}
