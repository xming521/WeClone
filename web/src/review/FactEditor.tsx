import { useEffect, useId, useRef, useState } from 'react';
import { useProfileStore } from '../store';
import type { EditTarget, FactDraft } from '../store';

function draftOf(target: EditTarget): FactDraft {
  return {
    value: target.fact?.value ?? '',
    attr: target.fact?.attr ?? '',
    group_id: target.fact?.group_id ?? target.groupId,
  };
}

// The dialog mounts when an edit target appears, so every opening starts from a fresh draft.
export function FactEditor() {
  const editing = useProfileStore((state) => state.editing);
  return editing ? <FactEditorDialog editing={editing} /> : null;
}

function FactEditorDialog({ editing }: { editing: EditTarget }) {
  const input = useProfileStore((state) => state.input)!;
  const error = useProfileStore((state) => state.error);
  const busy = useProfileStore((state) => state.busy);
  const setError = useProfileStore((state) => state.setError);
  const closeEditor = useProfileStore((state) => state.closeEditor);
  const refresh = useProfileStore((state) => state.refresh);
  const edit = useProfileStore((state) => state.edit);
  const save = useProfileStore((state) => state.save);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const [draft, setDraft] = useState(() => draftOf(editing));
  useEffect(() => {
    dialogRef.current?.showModal();
  }, []);
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, []);
  function close() {
    if (busy) return;
    const dirty =
      draft.value !== (editing.fact?.value ?? '') ||
      draft.attr !== (editing.fact?.attr ?? '') ||
      (!!editing.fact && draft.group_id !== editing.fact.group_id);
    if (!dirty || window.confirm('放弃未保存的修改？')) {
      closeEditor();
      setError('');
    }
  }
  function locationName(id: string): string {
    const location = input.locations.find((item) => item.id === id);
    return location ? `${location.parent_id ? `${locationName(location.parent_id)} / ` : ''}${location.name}` : '';
  }
  return (
    <dialog
      ref={dialogRef}
      className="editor-backdrop"
      aria-labelledby={titleId}
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
    >
      <form
        className="fact-editor"
        onSubmit={(event) => {
          event.preventDefault();
          void save(draft, false);
        }}
      >
        <h2 id={titleId}>{editing.fact ? '编辑画像' : '新增画像'}</h2>
        <label>
          属性名称
          <input
            disabled={busy}
            autoFocus
            required
            value={draft.attr}
            onChange={(event) => setDraft({ ...draft, attr: event.target.value })}
          />
        </label>
        <label>
          归属位置
          <select
            disabled={busy}
            required
            value={draft.group_id}
            onChange={(event) => setDraft({ ...draft, group_id: event.target.value })}
          >
            {input.locations.map((item) => (
              <option value={item.id} key={item.id}>
                {locationName(item.id)}
              </option>
            ))}
          </select>
        </label>
        <label>
          画像内容
          <textarea
            disabled={busy}
            required
            rows={5}
            value={draft.value}
            onChange={(event) => setDraft({ ...draft, value: event.target.value })}
          />
        </label>
        {editing.fact && (
          <details>
            <summary>查看原始内容</summary>
            <p>
              {editing.fact.original.attr} · {editing.fact.original.value}
            </p>
          </details>
        )}
        <p className="editor-hint">保存修改后移回待审核，暂不用于分身；“保存并采纳”后纳入分身画像。</p>
        {error && (
          <div className="editor-error" role="alert">
            {error}
            <button
              type="button"
              disabled={busy}
              onClick={async () => {
                if (!window.confirm('重新载入服务器数据会放弃当前表单中的修改，继续？')) return;
                try {
                  const next = await refresh();
                  const latest = next.facts.find((fact) => fact.id === editing.fact?.id);
                  if (latest) {
                    setDraft(draftOf({ fact: latest, groupId: editing.groupId }));
                    edit(latest);
                  }
                  setError('');
                } catch (reason) {
                  setError((reason as Error).message);
                }
              }}
            >
              重新载入记录
            </button>
          </div>
        )}
        <div className="editor-actions">
          <button type="button" disabled={busy} onClick={close}>
            取消
          </button>
          <button disabled={busy}>保存</button>
          <button
            className="primary"
            type="button"
            disabled={busy || !draft.attr.trim() || !draft.value.trim() || !draft.group_id}
            onClick={() => void save(draft, true)}
          >
            保存并采纳
          </button>
        </div>
      </form>
    </dialog>
  );
}
