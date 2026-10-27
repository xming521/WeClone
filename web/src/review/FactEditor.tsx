import { useEffect, useId, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { isDimensionId } from '../data';
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
  const { t } = useTranslation();
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
    if (!dirty || window.confirm(t(($) => $.editor.discard))) {
      closeEditor();
      setError('');
    }
  }
  function locationName(id: string): string {
    const location = input.locations.find((item) => item.id === id);
    if (!location) return '';
    const dim = location.dim;
    // Dimensions are named in the page's language, like on the canvas.
    const name =
      location.kind === 'dimension' && isDimensionId(dim) ? t(($) => $.model.dimensions[dim]) : location.name;
    return `${location.parent_id ? `${locationName(location.parent_id)} / ` : ''}${name}`;
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
        <h2 id={titleId}>{editing.fact ? t(($) => $.editor.editTitle) : t(($) => $.editor.addTitle)}</h2>
        <label>
          {t(($) => $.editor.attr)}
          <input
            disabled={busy}
            autoFocus
            required
            value={draft.attr}
            onChange={(event) => setDraft({ ...draft, attr: event.target.value })}
          />
        </label>
        <label>
          {t(($) => $.editor.location)}
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
          {t(($) => $.editor.value)}
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
            <summary>{t(($) => $.editor.original)}</summary>
            <p>
              {editing.fact.original.attr} · {editing.fact.original.value}
            </p>
          </details>
        )}
        <p className="editor-hint">{t(($) => $.editor.hint)}</p>
        {error && (
          <div className="editor-error" role="alert">
            {error}
            <button
              type="button"
              disabled={busy}
              onClick={async () => {
                if (!window.confirm(t(($) => $.editor.reloadConfirm))) return;
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
              {t(($) => $.editor.reload)}
            </button>
          </div>
        )}
        <div className="editor-actions">
          <button type="button" disabled={busy} onClick={close}>
            {t(($) => $.common.cancel)}
          </button>
          <button disabled={busy}>{t(($) => $.common.save)}</button>
          <button
            className="primary"
            type="button"
            disabled={busy || !draft.attr.trim() || !draft.value.trim() || !draft.group_id}
            onClick={() => void save(draft, true)}
          >
            {t(($) => $.editor.saveApprove)}
          </button>
        </div>
      </form>
    </dialog>
  );
}
