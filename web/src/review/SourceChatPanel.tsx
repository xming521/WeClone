import { useEffect, useRef, useState } from 'react';
import { LockKeyhole, MessagesSquare, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api } from '../api';
import type { ChatSelection, SourceChat } from '../data';

export function SourceChatPanel({ selection, onClose }: { selection: ChatSelection; onClose: () => void }) {
  const { t } = useTranslation();
  const [result, setResult] = useState<{ id: string; data?: SourceChat; error?: string } | null>(null);
  const [retry, setRetry] = useState(0);
  const closeRef = useRef<HTMLButtonElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    closeRef.current?.focus();
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !document.querySelector('dialog[open]')) {
        event.preventDefault();
        close.current();
      }
    };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, []);
  useEffect(() => {
    let active = true;
    setResult(null);
    bodyRef.current?.scrollTo(0, 0);
    void api
      .get<SourceChat>(`sources/${encodeURIComponent(selection.id)}/chat`)
      .then((data) => {
        if (active) setResult({ id: selection.id, data });
      })
      .catch((reason: Error) => {
        if (active) setResult({ id: selection.id, error: reason.message });
      });
    return () => {
      active = false;
    };
  }, [selection.id, retry]);
  const current = result?.id === selection.id ? result : null;
  const data = current?.data;
  const peer = data?.chat_with || t(($) => $.chat.peer);
  return (
    <aside className="source-chat-panel" role="dialog" aria-modal="false" aria-labelledby="source-chat-title">
      <header className="source-chat-header">
        <div>
          <h2 id="source-chat-title">
            <MessagesSquare size={19} aria-hidden="true" />
            {t(($) => $.chat.title)}
          </h2>
          <p>
            {selection.label}
            {data && <> · {peer}</>}
          </p>
        </div>
        <button type="button" ref={closeRef} onClick={onClose} aria-label={t(($) => $.chat.close)}>
          <X size={18} />
        </button>
      </header>
      <div className="source-chat-body" ref={bodyRef} aria-busy={!current}>
        {!current && (
          <p className="source-chat-notice" role="status">
            {t(($) => $.chat.loading)}
          </p>
        )}
        {current?.error && (
          <div className="source-chat-notice" role="alert">
            <p>{current.error}</p>
            <button type="button" onClick={() => setRetry((value) => value + 1)}>
              {t(($) => $.common.retry)}
            </button>
          </div>
        )}
        {data && (
          <>
            <div className="source-chat-date">
              {data.sample_time?.replace('T', ' ') ?? t(($) => $.review.fact.noTime)}
            </div>
            {data.messages.map((message) => (
              <div className={`source-chat-message${message.role === 'assistant' ? ' mine' : ''}`} key={message.id}>
                <span className="source-chat-avatar" aria-hidden="true">
                  {message.role === 'assistant' ? t(($) => $.chat.me) : peer.slice(0, 1)}
                </span>
                <div className="source-chat-message-main">
                  <div className="source-chat-meta">
                    <span>{message.role === 'assistant' ? t(($) => $.chat.self) : peer}</span>
                    {message.time && <time>{message.time.replace('T', ' ')}</time>}
                  </div>
                  <div className="source-chat-bubble">{message.content}</div>
                </div>
              </div>
            ))}
          </>
        )}
      </div>
      <footer className="source-chat-footer">
        <span>{data ? t(($) => $.chat.messages, { count: data.messages.length }) : t(($) => $.chat.footer)}</span>
        <span>
          <LockKeyhole size={12} aria-hidden="true" />
          {t(($) => $.chat.readOnly)}
        </span>
      </footer>
    </aside>
  );
}
