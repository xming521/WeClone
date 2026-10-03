import { useEffect, useRef, useState } from 'react';
import { LockKeyhole, MessagesSquare, X } from 'lucide-react';
import { api } from './Auth';
import type { SourceChat } from './data';

export interface ChatSelection { id: string; label: string }

export function SourceChatPanel({ selection, onClose }: { selection: ChatSelection; onClose: () => void }) {
  const [result, setResult] = useState<{ id: string; data?: SourceChat; error?: string } | null>(null);
  const [retry, setRetry] = useState(0);
  const closeRef = useRef<HTMLButtonElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    closeRef.current?.focus();
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !document.querySelector('dialog[open]')) { event.preventDefault(); close.current(); }
    };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, []);
  useEffect(() => {
    let active = true;
    setResult(null);
    bodyRef.current?.scrollTo(0, 0);
    void api<SourceChat>(`sources/${encodeURIComponent(selection.id)}/chat`).then(data => {
      if (active) setResult({ id: selection.id, data });
    }).catch((reason: Error) => {
      if (active) setResult({ id: selection.id, error: reason.message });
    });
    return () => { active = false; };
  }, [selection.id, retry]);
  const current = result?.id === selection.id ? result : null;
  const data = current?.data;
  return <aside className="source-chat-panel" role="dialog" aria-modal="false" aria-labelledby="source-chat-title">
    <header className="source-chat-header">
      <div><h2 id="source-chat-title"><MessagesSquare size={19} aria-hidden="true" />原始聊天记录</h2>
        <p>{selection.label}{data && <> · {data.chat_with}</>}</p></div>
      <button type="button" ref={closeRef} onClick={onClose} aria-label="关闭聊天记录"><X size={18} /></button>
    </header>
    <div className="source-chat-body" ref={bodyRef} aria-busy={!current}>
      {!current && <p className="source-chat-notice" role="status">正在加载聊天记录…</p>}
      {current?.error && <div className="source-chat-notice" role="alert"><p>{current.error}</p><button type="button" onClick={() => setRetry(value => value + 1)}>重试</button></div>}
      {data && <><div className="source-chat-date">{data.sample_time?.replace('T', ' ') ?? '未标注时间'}</div>
        {data.messages.map(message => <div className={`source-chat-message${message.role === 'assistant' ? ' mine' : ''}`} key={message.id}>
          <span className="source-chat-avatar" aria-hidden="true">{message.role === 'assistant' ? '我' : message.speaker.slice(0, 1)}</span>
          <div className="source-chat-message-main"><div className="source-chat-meta"><span>{message.speaker}</span>{message.time && <time>{message.time.replace('T', ' ')}</time>}</div>
            <div className="source-chat-bubble">{message.content}</div></div>
        </div>)}</>}
    </div>
    <footer className="source-chat-footer"><span>{data ? `${data.messages.length} 条消息 · 抽取时的聊天片段` : '聊天记录'}</span><span><LockKeyhole size={12} aria-hidden="true" />只读</span></footer>
  </aside>;
}
