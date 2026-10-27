import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, ClipboardCopy, EyeOff, PencilLine, Sparkles, Undo2 } from 'lucide-react';
import { api } from '../api';
import { LogoutButton } from '../Auth';
import { chineseNumeral } from '../data';
import { ModeSwitch } from '../ModeSwitch';
import { SENSITIVITY } from '../review/privacy';
import { STATE_LABELS, SUMMARY_LEVELS, stateCounts, summaryState } from './summaryState';
import type {
  DomainSummary,
  SummaryDomain,
  SummaryLevel,
  SummaryState,
  SummaryStatus,
  SummaryView,
} from './summaryState';
import './summary.css';

const LEVEL_LABELS: Record<SummaryLevel, { label: string; hint: string }> = {
  public: SENSITIVITY.public,
  normal: SENSITIVITY.normal,
  full: { label: '全部', hint: '所有还成立的条目，包括私密和敏感的' },
};
// What a summary of each level is written from, as the backend's in_scope picks it.
const SHORT_INTRO =
  '每个生活领域写一段较短的总结，只用隐私度不高于所选级别的身份、目标、条件和偏好，不用计划承诺和已过时的信息。';
const INTROS: Record<SummaryLevel, string> = {
  full: '每个生活领域写一段总结，用这个领域所有还成立的条目，包括私密和敏感的。build-profile 最后一步会生成这一级。',
  normal: SHORT_INTRO,
  public: SHORT_INTRO,
};
const PLACEHOLDERS: Partial<Record<SummaryState, string>> = {
  empty: '这个领域没有能写进总结的条目：要么高于所选隐私度，要么是计划承诺或已过时的信息。',
  missing: '还没有生成总结。',
};
const EMPTY_FULL = '这个领域没有还成立的条目。';
const FACT_CHOICES = [0, 3, 5, 10];

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
      <label>
        标题
        <input value={title} maxLength={limits.title} autoFocus onChange={(event) => setTitle(event.target.value)} />
      </label>
      <label>
        总结
        <textarea value={text} rows={3} maxLength={limits.summary} onChange={(event) => setText(event.target.value)} />
        <small>
          {text.trim().length}/{limits.summary}
        </small>
      </label>
      <div className="summary-actions">
        <button type="button" onClick={onCancel}>
          取消
        </button>
        <button type="button" disabled={!valid || busy} onClick={() => onSave(title, text, false)}>
          保存
        </button>
        <button className="primary" type="submit" disabled={!valid || busy}>
          保存并确认
        </button>
      </div>
    </form>
  );
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
  const state = summaryState(item);
  const summary = item.summary;
  const meta = [`${item.facts} 条可用`, item.as_of && `截至 ${item.as_of}`].filter(Boolean).join(' · ');
  return (
    <li className={`summary-card ${state}`}>
      <div className="summary-card-head">
        <span className="summary-domain">{item.domain}</span>
        <span className="summary-meta">{meta}</span>
        <span className={`summary-state ${state}`}>{STATE_LABELS[state]}</span>
      </div>
      {summary && editing ? (
        <SummaryEditor summary={summary} limits={limits} busy={busy} onCancel={onCancel} onSave={onSave} />
      ) : summary && state !== 'empty' ? (
        <div className="summary-body">
          <strong>{summary.title}</strong>
          <p>{summary.summary}</p>
          {state === 'stale' && <p className="summary-note">画像有变动，这段已不再导出，重新生成后再确认。</p>}
        </div>
      ) : (
        <p className="summary-placeholder">{state === 'empty' ? empty : PLACEHOLDERS[state]}</p>
      )}
      {summary && !editing && state !== 'empty' && (
        <div className="summary-actions">
          {state === 'pending' && (
            <button className="primary" type="button" disabled={busy} onClick={() => onStatus('approved')}>
              <Check size={14} /> 确认
            </button>
          )}
          {(state === 'pending' || state === 'approved') && (
            <button type="button" disabled={busy} onClick={onEdit}>
              <PencilLine size={14} /> 编辑
            </button>
          )}
          {state === 'approved' && (
            <button type="button" disabled={busy} onClick={() => onStatus('pending')}>
              <Undo2 size={14} /> 撤回确认
            </button>
          )}
          {state === 'rejected' ? (
            <button type="button" disabled={busy} onClick={() => onStatus('pending')}>
              <Undo2 size={14} /> 恢复
            </button>
          ) : (
            <button type="button" disabled={busy} onClick={() => onStatus('rejected')}>
              <EyeOff size={14} /> 不导出
            </button>
          )}
        </div>
      )}
    </li>
  );
}

export function SummaryApp() {
  const [level, setLevel] = useState<SummaryLevel>('full');
  const [facts, setFacts] = useState(0);
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
      const next = await api.get<SummaryView>(`summaries?level=${level}&facts=${facts}`);
      if (ticket === sequence.current) setView(next);
    } catch (reason) {
      if (ticket === sequence.current) setError((reason as Error).message);
    }
  }, [level, facts]);

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
    if (level === 'full' && !window.confirm('这份文本含私密和敏感的内容，贴出去就收不回来。确定复制吗？')) return;
    try {
      await navigator.clipboard.writeText(view.text);
    } catch {
      // Pages served over plain http to another machine have no clipboard API.
      exportRef.current?.select();
      if (!document.execCommand('copy')) {
        setError('浏览器不允许直接复制，请在文本框里全选后手动复制');
        return;
      }
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  if (!view && error)
    return (
      <div className="load-state">
        <span>✳</span>
        <h1>领域总结暂时无法打开</h1>
        <p>{error}</p>
        <button onClick={() => void load()}>重试</button>
      </div>
    );
  if (!view)
    return (
      <div className="load-state">
        <span className="loading-mark">✳</span>
        <h1>正在读取领域总结</h1>
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
          <ModeSwitch mode="summary" />
          <div className="topbar-heading">
            <span className="topbar-divider">/</span> <strong>{chineseNumeral(view.domains.length)}个领域</strong>
          </div>
        </div>
        <div className="topbar-actions">
          <LogoutButton />
        </div>
      </header>
      <main className="summary-main">
        <section className="summary-intro">
          <h1>给个人助理的背景资料</h1>
          <p>
            {INTROS[level]}
            确认过的总结才会进入导出文本和助手接入，所依据的画像一改动就停止导出，等重新生成并确认。
          </p>
          <div className="summary-toolbar">
            <div className="summary-levels" role="radiogroup" aria-label="隐私度上限">
              <span>隐私度上限</span>
              {SUMMARY_LEVELS.map((item) => (
                <button
                  key={item}
                  type="button"
                  role="radio"
                  aria-checked={item === level}
                  disabled={!!busy}
                  title={LEVEL_LABELS[item].hint}
                  onClick={() => {
                    setEditing(null);
                    setLevel(item);
                  }}
                >
                  {LEVEL_LABELS[item].label}
                </button>
              ))}
            </div>
            <button
              className="primary"
              type="button"
              disabled={!!busy || generating || !due}
              onClick={() => void run('generate', () => api.post('summaries/generate', { level }))}
            >
              <Sparkles size={14} />
              {generating ? '正在生成…' : due ? `生成 ${due} 个领域` : '都已生成'}
            </button>
            <button type="button" disabled={!!busy || !pending.length} onClick={() => review(pending, 'approved')}>
              <Check size={14} /> 全部确认{pending.length ? `（${pending.length}）` : ''}
            </button>
            <span className="summary-counts">
              已确认 {counts.approved} · 待确认 {counts.pending} · 需生成 {due} · 不导出 {counts.rejected}
            </span>
          </div>
          {generating ? (
            <p className="summary-hint">
              每个领域调用一次模型，通常一两分钟，模型繁忙时会更久。生成在后台进行，可以先离开这个页面。
            </p>
          ) : (
            view.error && (
              <div className="summary-error" role="status">
                <span>{view.error}</span>
              </div>
            )
          )}
          {error && (
            <div className="summary-error" role="alert">
              <span>{error}</span>
              <button type="button" onClick={() => setError('')}>
                知道了
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
                empty={level === 'full' ? EMPTY_FULL : PLACEHOLDERS.empty!}
                busy={!!busy}
                editing={editing === item.domain}
                onEdit={() => setEditing(item.domain)}
                onCancel={() => setEditing(null)}
                onSave={(title, text, approve) => save(item, title, text, approve)}
                onStatus={(status) => review([item], status)}
              />
            ))}
          </ol>
          <aside className="summary-export" aria-label="导出文本">
            <div className="summary-export-head">
              <h2>导出文本</h2>
              <label>
                每个领域附原始条目
                <select value={facts} disabled={!!busy} onChange={(event) => setFacts(Number(event.target.value))}>
                  {FACT_CHOICES.filter((count) => count <= view.limits.facts).map((count) => (
                    <option key={count} value={count}>
                      {count ? `${count} 条` : '不附'}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {view.text ? (
              <>
                <textarea ref={exportRef} readOnly value={view.text} onFocus={(event) => event.target.select()} />
                <div className="summary-export-foot">
                  <span>{view.text.length} 字</span>
                  <button className="primary" type="button" onClick={() => void copy()}>
                    <ClipboardCopy size={14} /> {copied ? '已复制' : '复制'}
                  </button>
                </div>
                {level === 'full' ? (
                  <p className="summary-hint summary-warning">
                    这一级含私密和敏感的内容，贴出去就收不回来。只想给助理少一些，可以切到普通或公开再生成一份。
                  </p>
                ) : (
                  <p className="summary-hint">
                    可以粘贴到个人助理的“关于我”、自定义指令或 USER.md。贴出去的内容收不回来，复制前请再看一遍。
                  </p>
                )}
              </>
            ) : (
              <p className="summary-placeholder">确认至少一个领域的总结后，这里会出现可以粘贴给个人助理的文本。</p>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}
