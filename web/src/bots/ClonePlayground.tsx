import { useEffect, useRef, useState } from 'react';
import { MessageCircle, RotateCcw, Send, Square } from 'lucide-react';
import { useTranslation } from 'react-i18next';

type Message = { role: 'user' | 'assistant'; content: string };

/**
 * Streams the clone's answer to the conversation from POST /v1/chat/completions, the OpenAI-compatible route bots
 * call. The page calls it with the owner's session instead of a bot token, and sends no system prompt, so the clone
 * answers with the one the server is set up with, as it answers bots.
 */
async function streamReply(messages: Message[], signal: AbortSignal, onText: (text: string) => void) {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json', 'X-WeClone-Request': '1' },
    body: JSON.stringify({ model: 'weclone', messages, stream: true }),
  });
  if (!response.ok || !response.body) {
    if (response.status === 401) window.dispatchEvent(new Event('weclone-unauthorized'));
    const data = await response.json().catch(() => null);
    throw new Error(typeof data?.detail === 'string' ? data.detail : `HTTP ${response.status}`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  let text = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // Server-sent events: one `data:` line per chunk, the stream ends with `data: [DONE]`.
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      const data = line.replace(/^data:\s?/, '').trim();
      if (!line.startsWith('data:') || !data || data === '[DONE]') continue;
      const delta = JSON.parse(data)?.choices?.[0]?.delta?.content;
      if (typeof delta === 'string') onText((text += delta));
    }
  }
}

/** A conversation with the fine-tuned clone, to hear how it talks before bots reach it. */
export function ClonePlayground({ inference }: { inference: boolean }) {
  const { t } = useTranslation();
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState('');
  const [controller, setController] = useState<AbortController | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const streaming = controller !== null;

  useEffect(() => {
    const thread = threadRef.current;
    if (thread) thread.scrollTop = thread.scrollHeight;
  }, [messages]);
  useEffect(() => () => controller?.abort(), [controller]);

  async function send(event?: { preventDefault(): void }) {
    event?.preventDefault();
    const content = draft.trim();
    if (!content || streaming || !inference) return;
    const asked: Message[] = [...messages, { role: 'user', content }];
    const abort = new AbortController();
    setMessages([...asked, { role: 'assistant', content: '' }]);
    setDraft('');
    setError('');
    setController(abort);
    let answer = '';
    try {
      await streamReply(asked, abort.signal, (text) => {
        answer = text;
        setMessages([...asked, { role: 'assistant', content: text }]);
      });
    } catch (reason) {
      // A stopped answer stays as far as it got. Without one, the question goes back to the box: the model takes
      // turns, and asking again keeps them in order.
      if (abort.signal.aborted && answer) return;
      setMessages(messages);
      setDraft(content);
      if (!abort.signal.aborted) setError((reason as Error).message);
    } finally {
      setController(null);
    }
  }

  return (
    <section className="assistant-card playground" aria-labelledby="playground-title">
      <div className="assistant-card-head">
        <h2 id="playground-title">
          <MessageCircle size={18} strokeWidth={1.8} aria-hidden="true" />
          {t(($) => $.bots.playground.title)}
        </h2>
        <button
          type="button"
          className="wc-btn wc-btn--ghost wc-btn--sm"
          disabled={!messages.length || streaming}
          onClick={() => {
            setMessages([]);
            setError('');
          }}
        >
          <RotateCcw size={13} strokeWidth={1.8} aria-hidden="true" />
          {t(($) => $.bots.playground.restart)}
        </button>
      </div>
      <p className="summary-hint">
        {inference ? t(($) => $.bots.playground.hint) : t(($) => $.bots.playground.unavailable)}
      </p>
      <div className="wc-thread playground-thread" ref={threadRef} aria-live="polite">
        {!messages.length && <p className="playground-empty">{t(($) => $.bots.playground.empty)}</p>}
        {messages.map((message, index) => (
          <div
            key={index}
            className={message.role === 'assistant' ? 'wc-bubble playground-clone' : 'wc-bubble'}
            data-from={message.role === 'assistant' ? 'other' : 'you'}
          >
            {message.content ||
              (streaming && index === messages.length - 1 && (
                <span className="playground-typing">{t(($) => $.bots.playground.typing)}</span>
              ))}
          </div>
        ))}
      </div>
      {error && (
        <p className="wc-field__hint" data-tone="danger" role="alert">
          {error}
        </p>
      )}
      <form className="playground-form" onSubmit={(event) => void send(event)}>
        <div className="wc-input">
          <textarea
            rows={2}
            value={draft}
            disabled={!inference}
            placeholder={t(($) => $.bots.playground.placeholder)}
            aria-label={t(($) => $.bots.playground.placeholder)}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends, Shift+Enter starts a new line; Enter that confirms an input method's text does neither.
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) void send(event);
            }}
          />
        </div>
        {streaming ? (
          <button type="button" className="wc-btn" onClick={() => controller?.abort()}>
            <Square size={13} strokeWidth={1.8} aria-hidden="true" />
            {t(($) => $.bots.playground.stop)}
          </button>
        ) : (
          <button type="submit" className="wc-btn wc-btn--primary" disabled={!inference || !draft.trim()}>
            <Send size={13} strokeWidth={1.8} aria-hidden="true" />
            {t(($) => $.bots.playground.send)}
          </button>
        )}
      </form>
    </section>
  );
}
