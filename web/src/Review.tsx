import { createContext, useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { ArrowUpRight, BadgeCheck, Check, ChevronDown, FileClock, CircleX, Download, Layers3, MessageSquare, Plus, X } from 'lucide-react';
import * as Tooltip from '@radix-ui/react-tooltip';
import { buildModel, collectFacts, displayProfileText, exportProfile, filterProfile, retainedSelection } from './data';
import type { ProfileFact, ProfileInput, ReviewStatus, ReviewView } from './data';
import { useGraphStore } from './store';
import { api } from './Auth';
import { FactScores, SourceScores } from './ProfileScores';
import { SourceChatPanel } from './SourceChatPanel';
import type { ChatSelection } from './SourceChatPanel';

export const STATUS: Record<ReviewView, string> = { all: '全部', pending: '待审核', approved: '已采纳', rejected: '未采纳' };
const VIEW_ICONS = { all: Layers3, pending: FileClock, approved: BadgeCheck, rejected: CircleX };

interface Editor {
  fact: ProfileFact | null;
  value: string;
  attr: string;
  group_id: string;
}

export function useReview() {
  const [input, setInput] = useState<ProfileInput | null>(null);
  const [view, setView] = useState<ReviewView>('all');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [editor, setEditor] = useState<Editor | null>(null);
  const [chat, setChat] = useState<ChatSelection | null>(null);
  const detailTab = useGraphStore(state => state.detailTab);
  const detailSelection = useGraphStore(state => state.selectedId);
  const detailOpen = useGraphStore(state => state.detailOpen);
  useEffect(() => { setChat(null); }, [detailTab, detailSelection, detailOpen]);
  const chatOpener = useRef<HTMLButtonElement | null>(null);
  function openChat(selection: ChatSelection, button: HTMLButtonElement) { chatOpener.current = button; setChat(selection); }
  function closeChat() { setChat(null); if (chatOpener.current?.isConnected) chatOpener.current.focus(); }
  const saving = useRef(false);
  const model = useMemo(() => input ? buildModel(filterProfile(input, view)) : null, [input, view]);
  const fullModel = useMemo(() => input ? buildModel(input) : null, [input]);
  const previousModel = useRef(model);

  async function refresh() {
    const next = await api<ProfileInput>('profile');
    setInput(next);
    return next;
  }

  useEffect(() => { void refresh().catch((reason: Error) => setError(reason.message)); }, []);
  useEffect(() => {
    if (!model) return;
    const previous = previousModel.current;
    previousModel.current = model;
    if (!previous) {
      useGraphStore.getState().initialize(model);
      return;
    }
    const state = useGraphStore.getState();
    const expandedIds = new Set([...state.expandedIds].filter((id) => model.nodes.has(id)));
    expandedIds.add('root');
    model.dimensionIds.forEach((id) => { if (!previous.nodes.has(id)) expandedIds.add(id); });
    const selectedId = retainedSelection(previous, model, state.selectedId);
    const focusedDimension = selectedId ? model.nodes.get(selectedId)?.dim ?? null : null;
    useGraphStore.setState({ expandedIds, selectedId, focusedDimension });
  }, [model]);

  useEffect(() => {
    if (!editor) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [editor]);

  async function mutate(path: string, body: unknown, method = 'POST', closeEditor = false) {
    if (saving.current) return false;
    saving.current = true;
    setBusy(true);
    setError('');
    try {
      await api(path, body, method);
      if (closeEditor) setEditor(null);
      try { await refresh(); } catch { setError('已保存，但刷新数据失败。请点击“刷新数据”后继续。'); }
      return true;
    } catch (reason) {
      setError((reason as Error).message);
      return false;
    } finally {
      saving.current = false;
      setBusy(false);
    }
  }

  function edit(fact: ProfileFact | null) {
    if (!input) return;
    setError('');
    const selected = model?.nodes.get(useGraphStore.getState().selectedId ?? 'root');
    const location = input.locations.find((item) => item.id === selected?.id)
      ?? input.locations.find((item) => item.id === selected?.parentId) ?? input.locations[0];
    setEditor({ fact, value: fact?.value ?? '', attr: fact?.attr ?? '', group_id: fact?.group_id ?? location?.id ?? '' });
  }

  function review(facts: ProfileFact[], status: ReviewStatus) {
    return mutate('reviews/batch', { items: facts.map(({ id, version }) => ({ id, version })), status });
  }

  function save(approve: boolean) {
    if (!editor) return;
    const { fact, value, attr, group_id } = editor;
    return mutate(fact ? `facts/${fact.id}` : 'facts', {
      value, attr, group_id, approve, ...(fact ? { version: fact.version } : {}),
    }, fact ? 'PATCH' : 'POST', true);
  }

  return { input, model, fullModel, view, setView, error, setError, busy, editor, setEditor, refresh, edit, review, save, chat, openChat, closeChat };
}

type Review = ReturnType<typeof useReview>;
const Context = createContext<Review | null>(null);
export function ReviewProvider({ review, children }: { review: Review; children: ReactNode }) {
  return <Context.Provider value={review}>{children}</Context.Provider>;
}
function useController() { return useContext(Context)!; }

export function ReviewBar() {
  const c = useController();
  const counts = { all: c.input!.facts.length, pending: 0, approved: 0, rejected: 0 };
  c.input!.facts.forEach((fact) => counts[fact.status]++);
  return <div className="review-bar">
    <nav className="review-tabs" aria-label="审核状态">
      {(Object.keys(STATUS) as ReviewView[]).map((view, index, views) => {
        const Icon = VIEW_ICONS[view];
        const label = STATUS[view];
        return <Tooltip.Root key={view} delayDuration={160}>
          <Tooltip.Trigger asChild>
            <button type="button" aria-label={`${label} ${counts[view]} 条画像`} aria-pressed={c.view === view}
              className={`review-bookmark${c.view === view ? ' active' : ''}`}
              onClick={() => c.setView(view)}
              onKeyDown={(event) => {
                if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
                event.preventDefault();
                const next = (index + (event.key === 'ArrowDown' ? 1 : views.length - 1)) % views.length;
                event.currentTarget.closest('nav')?.querySelectorAll<HTMLButtonElement>('.review-bookmark')[next]?.focus();
                c.setView(views[next]);
              }}>
              <Icon size={32} strokeWidth={1.5} aria-hidden="true" />
              <span className="review-bookmark-copy" aria-hidden="true">
                <span>{label}</span><span className="review-bookmark-count">{counts[view]}</span>
              </span>
            </button>
          </Tooltip.Trigger>
          <Tooltip.Portal><Tooltip.Content className="tooltip" side="right" sideOffset={12}>
            {label} · {counts[view]} 条画像{view === 'approved' && ' · 分身画像'}<Tooltip.Arrow className="tooltip-arrow" />
          </Tooltip.Content></Tooltip.Portal>
        </Tooltip.Root>;
      })}
    </nav>
  </div>;
}

export function ReviewActions() {
  const c = useController();
  const [selection, setSelection] = useState<{ view: ReviewView; profile: ReturnType<typeof exportProfile> } | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descriptionId = useId();
  useEffect(() => { if (selection) dialogRef.current?.showModal(); }, [selection]);
  function close() { dialogRef.current?.close(); setSelection(null); }
  function download() {
    if (!selection) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(selection.profile, null, 2)], { type: 'application/json;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `profile_${selection.view}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    close();
  }
  return <>
    <div className="header-review-actions">
      <button disabled={c.busy} onClick={() => c.edit(null)} aria-label="新增画像" title="新增画像"><Plus size={17} /></button>
      <button type="button" disabled={c.busy || !c.input} onClick={() => {
        if (c.input) setSelection({ view: c.view, profile: exportProfile(c.input, c.view) });
      }} aria-label="导出当前视图" title="导出当前视图"><Download size={17} /></button>
    </div>
    {selection && <dialog ref={dialogRef} className="editor-backdrop" aria-labelledby={titleId} aria-describedby={descriptionId}
      onCancel={(event) => { event.preventDefault(); close(); }}>
      <div className="fact-editor export-confirm">
        <h2 id={titleId}>导出画像</h2>
        <p id={descriptionId}>是否导出当前“<strong>{STATUS[selection.view]}</strong>”视图中的 <strong>{selection.profile.facts.length}</strong> 条画像？</p>
        <p className="editor-hint">导出为 JSON 文件，包含画像层级和关联来源。</p>
        <div className="editor-actions"><button type="button" autoFocus onClick={close}>取消</button>
          <button className="primary" type="button" onClick={download}>确认导出</button></div>
      </div>
    </dialog>}
  </>;
}

interface HistoryEntry { at: string; action: string; before: ProfileFact | null; after: ProfileFact }
function History({ fact }: { fact: ProfileFact }) {
  const [rows, setRows] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState('');
  const [open, setOpen] = useState(false);
  return <div className="fact-history">
    <button onClick={async () => {
      setOpen(!open);
      if (!rows) try { setRows(await api<HistoryEntry[]>(`facts/${fact.id}/history`)); } catch (reason) { setError((reason as Error).message); }
    }}>{open ? '收起修改历史' : '查看修改历史'}</button>
    {open && <div>{error || (rows?.length === 0 ? '暂无修改记录' : rows?.map((row, i) => <div className="history-row" key={i}>
      <time>{new Date(row.at).toLocaleString()}</time>
      <p>{row.action === 'review' ? '审核' : row.action === 'create' ? '新增' : '编辑'}：{row.before ? STATUS[row.before.status] : '新记录'} → {STATUS[row.after.status]}</p>
      {row.action !== 'review' && <><p>{row.before && `修改前：${row.before.attr} · ${displayProfileText(row.before.value)}`}</p><p>保存后：{row.after.attr} · {displayProfileText(row.after.value)}</p>
        {row.before?.group_id !== row.after.group_id && <p>归属位置已变更</p>}</>}
    </div>))}</div>}
  </div>;
}

function ReviewFilter({ value, counts, onChange }: { value: ReviewView; counts: Record<ReviewView, number>; onChange: (view: ReviewView) => void }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<ReviewView>(value);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const listId = useId();
  const views = Object.keys(STATUS) as ReviewView[];
  const Icon = VIEW_ICONS[value];
  useEffect(() => {
    if (!open) return;
    root.current?.querySelector<HTMLButtonElement>(`[data-view="${active}"]`)?.focus();
  }, [open, active]);
  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener('pointerdown', dismiss);
    return () => document.removeEventListener('pointerdown', dismiss);
  }, [open]);
  function choose(view: ReviewView) { onChange(view); setOpen(false); trigger.current?.focus(); }
  return <div className="review-filter" ref={root} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
  }} onKeyDown={event => {
    if (event.key === 'Escape' && open) { event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus(); }
  }}>
    <button className="review-filter-trigger" ref={trigger} type="button" aria-label={`筛选审核状态：${STATUS[value]}，${counts[value]} 条`}
      aria-haspopup="listbox" aria-expanded={open} aria-controls={open ? listId : undefined}
      onClick={() => { setActive(value); setOpen(!open); }} onKeyDown={event => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setActive(value); setOpen(true); }
      }}><Icon size={15} aria-hidden="true" /><span>{STATUS[value]}</span><span className="review-filter-count">{counts[value]}</span><ChevronDown className="review-filter-chevron" size={14} aria-hidden="true" /></button>
    {open && <div className="review-filter-menu" id={listId} role="listbox" aria-label="审核状态">
      {views.map((view, index) => { const OptionIcon = VIEW_ICONS[view]; return <button key={view} type="button" role="option" aria-selected={value === view}
        tabIndex={active === view ? 0 : -1} data-view={view} onClick={() => choose(view)} onKeyDown={event => {
          if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          setActive(views[event.key === 'Home' ? 0 : event.key === 'End' ? views.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : views.length - 1)) % views.length]);
        }}><OptionIcon size={15} aria-hidden="true" /><span className="review-filter-name">{STATUS[view]}</span><span className="review-filter-count">{counts[view]}</span><Check className="review-filter-check" size={14} aria-hidden="true" /></button>; })}
    </div>}
  </div>;
}

export function ReviewPanel() {
  const c = useController();
  const selectedId = useGraphStore((state) => state.selectedId) ?? 'root';
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [limit, setLimit] = useState(30);
  const facts = collectFacts(c.model!, selectedId);
  const all = collectFacts(c.fullModel!, selectedId);
  const page = facts.slice(0, limit);
  const selected = page.filter((fact) => checked.has(fact.id));
  const selectAllRef = useRef<HTMLInputElement>(null);
  useEffect(() => { if (selectAllRef.current) selectAllRef.current.indeterminate = selected.length > 0 && selected.length < page.length; }, [selected.length, page.length]);
  useEffect(() => { setChecked(new Set()); setLimit(30); }, [selectedId, c.view]);
  const counts = { pending: 0, approved: 0, rejected: 0 };
  all.forEach((fact) => counts[fact.status]++);
  async function batch(status: ReviewStatus) {
    if (!selected.length) return;
    if (await c.review(selected, status)) setChecked(new Set());
  }
  return <section className="detail-section review-list">
    <div className="review-list-toolbar">
      <ReviewFilter value={c.view} counts={{ all: all.length, ...counts }} onChange={c.setView} /><span>{facts.length} 条画像</span>
    </div>
    {!!facts.length && <div className="batch-toolbar">
      <label><input ref={selectAllRef} className="review-checkbox" type="checkbox" checked={page.length > 0 && selected.length === page.length} onChange={(event) => setChecked(new Set(event.target.checked ? page.map((fact) => fact.id) : []))} />选中已展示 {page.length} 条</label>
      <button disabled={c.busy || !selected.length} onClick={() => void batch('approved')}>采纳 {selected.length || ''}</button>
      <button disabled={c.busy || !selected.length} onClick={() => void batch('rejected')}>不采纳</button>
    </div>}
    {!facts.length && <p className="detail-hint">当前视图没有画像记录。</p>}
    {page.map((fact) => <article className="fact-card" key={fact.id}>
      <div className="fact-heading"><label><input className="review-checkbox" aria-label={`选择 ${fact.attr}`} type="checkbox" checked={checked.has(fact.id)} onChange={(event) => {
        const next = new Set(checked); if (event.target.checked) next.add(fact.id); else next.delete(fact.id); setChecked(next);
      }} /> {fact.attr}</label><span className={`status-badge ${fact.status}`}>{STATUS[fact.status]}</span></div>
      <p>{displayProfileText(fact.value)}</p>
      {fact.origin === 'manual' && <small>手动添加</small>}
      <FactScores fact={fact} />
      <div className="fact-actions">
        {fact.status !== 'approved' && <button className="adopt" disabled={c.busy} onClick={() => void c.review([fact], 'approved')}><Check size={14} aria-hidden="true" />采纳</button>}
        {fact.status !== 'rejected' && <button disabled={c.busy} onClick={() => void c.review([fact], 'rejected')}><X size={14} aria-hidden="true" />不采纳</button>}
        {fact.status !== 'pending' && <button disabled={c.busy} onClick={() => void c.review([fact], 'pending')}>移回待审核</button>}
        <button disabled={c.busy} onClick={() => c.edit(fact)}>编辑</button>
      </div>
      <details className="evidence-details"><summary>{fact.origin === 'manual' ? '初始内容' : `原始抽取与 ${fact.source_ids.length} 条证据`}</summary>
        <div className="evidence-items"><div className="evidence-item"><strong>{fact.original.attr}</strong><p>{fact.original.value}</p></div>
          {fact.source_ids.map((id, index) => { const source = c.input!.sources[id]; const label = `来源 ${index + 1}`; return <div className="evidence-item" key={id}>
            <div className="evidence-source-header"><span><MessageSquare size={13} aria-hidden="true" />{label}</span>
              <button type="button" className="view-source-chat" disabled={!source} aria-label={`查看${label}的原始聊天记录`} aria-pressed={c.chat?.id === id}
                onClick={event => c.openChat({ id, label }, event.currentTarget)}>查看原文<ArrowUpRight size={13} aria-hidden="true" /></button></div>
            <time className="evidence-time">{source?.sample_time?.slice(0, 10) ?? '未标注时间'}</time><p>{source?.content ?? '来源正文不可用'}</p>
            <SourceScores source={source} />
          </div>; })}
        </div>
      </details>
      <History key={`${fact.id}:${fact.version}`} fact={fact} />
    </article>)}
    {facts.length > limit && <button className="load-more" onClick={() => setLimit(limit + 30)}>继续显示（剩余 {facts.length - limit} 条）</button>}
  </section>;
}

export function ReviewOverlays() {
  const c = useController();
  const editor = c.editor;
  const dialogRef = useRef<HTMLDialogElement>(null);
  const editing = editor !== null;
  useEffect(() => { if (editing) dialogRef.current?.showModal(); }, [editing]);
  function close() {
    if (c.busy || !editor) return;
    const dirty = editor.value !== (editor.fact?.value ?? '') || editor.attr !== (editor.fact?.attr ?? '')
      || (!!editor.fact && editor.group_id !== editor.fact.group_id);
    if (!dirty || window.confirm('放弃未保存的修改？')) { c.setEditor(null); c.setError(''); }
  }
  function locationName(id: string): string {
    const location = c.input!.locations.find((item) => item.id === id);
    return location ? `${location.parent_id ? `${locationName(location.parent_id)} / ` : ''}${location.name}` : '';
  }
  return <>
    {c.chat && <SourceChatPanel selection={c.chat} onClose={c.closeChat} />}
    {c.error && !editor && <div className="review-error" role="alert">{c.error}<button disabled={c.busy} onClick={() => void c.refresh().then(() => c.setError('')).catch((reason: Error) => c.setError(reason.message))}>刷新数据</button></div>}
    {editor && <dialog ref={dialogRef} className="editor-backdrop" onCancel={(event) => { event.preventDefault(); close(); }}><form className="fact-editor" role="dialog" aria-modal="true" aria-labelledby="editor-title" onSubmit={(event) => { event.preventDefault(); void c.save(false); }}>
      <h2 id="editor-title">{editor.fact ? '编辑画像' : '新增画像'}</h2>
      <label>属性名称<input disabled={c.busy} autoFocus required value={editor.attr} onChange={(event) => c.setEditor({ ...editor, attr: event.target.value })} /></label>
      <label>归属位置<select disabled={c.busy} required value={editor.group_id} onChange={(event) => c.setEditor({ ...editor, group_id: event.target.value })}>
        {c.input!.locations.map((item) => <option value={item.id} key={item.id}>{locationName(item.id)}</option>)}
      </select></label>
      <label>画像内容<textarea disabled={c.busy} required rows={5} value={editor.value} onChange={(event) => c.setEditor({ ...editor, value: event.target.value })} /></label>
      {editor.fact && <details><summary>查看原始内容</summary><p>{editor.fact.original.attr} · {editor.fact.original.value}</p></details>}
      <p className="editor-hint">保存修改后移回待审核，暂不用于分身；“保存并采纳”后纳入分身画像。</p>
      {c.error && <div className="editor-error" role="alert">{c.error}<button type="button" disabled={c.busy} onClick={async () => {
        if (!window.confirm('重新载入服务器数据会放弃当前表单中的修改，继续？')) return;
        try { const input = await c.refresh(); const latest = input.facts.find((fact) => fact.id === editor.fact?.id); if (latest) c.edit(latest); c.setError(''); }
        catch (reason) { c.setError((reason as Error).message); }
      }}>重新载入记录</button></div>}
      <div className="editor-actions"><button type="button" disabled={c.busy} onClick={close}>取消</button><button disabled={c.busy}>保存</button>
        <button className="primary" type="button" disabled={c.busy || !editor.attr.trim() || !editor.value.trim() || !editor.group_id} onClick={() => void c.save(true)}>保存并采纳</button></div>
    </form></dialog>}
  </>;
}
