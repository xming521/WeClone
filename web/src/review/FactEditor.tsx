import { useEffect, useId, useRef, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { isDimensionId } from '../data';
import { useProfileStore } from '../store';
import type { EditTarget, FactDraft } from '../store';
import { confirm } from '../feedback/confirm';
import { SelectInput } from '../Dropdown';

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
  async function close() {
    if (busy) return;
    const dirty =
      draft.value !== (editing.fact?.value ?? '') ||
      draft.attr !== (editing.fact?.attr ?? '') ||
      (!!editing.fact && draft.group_id !== editing.fact.group_id);
    if (
      dirty &&
      !(await confirm({
        title: t(($) => $.editor.discard),
        action: t(($) => $.editor.discardSubmit),
        dismiss: t(($) => $.editor.keepEditing),
        tone: 'danger',
      }))
    )
      return;
    closeEditor();
    setError('');
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
        void close();
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
        <label className="wc-field">
          <span className="wc-field__label">{t(($) => $.editor.attr)}</span>
          <span className="wc-input">
            <input
              disabled={busy}
              autoFocus
              required
              value={draft.attr}
              onChange={(event) => setDraft({ ...draft, attr: event.target.value })}
            />
          </span>
        </label>
        <label className="wc-field">
          <span className="wc-field__label">{t(($) => $.editor.location)}</span>
          <SelectInput
            disabled={busy}
            required
            value={draft.group_id}
            onChange={(group_id) => setDraft({ ...draft, group_id })}
            options={input.locations.map((item) => ({ value: item.id, label: locationName(item.id) }))}
          />
        </label>
        <label className="wc-field">
          <span className="wc-field__label">{t(($) => $.editor.value)}</span>
          <span className="wc-input wc-input--area">
            <textarea
              disabled={busy}
              required
              rows={5}
              value={draft.value}
              onChange={(event) => setDraft({ ...draft, value: event.target.value })}
            />
          </span>
        </label>
        {editing.fact && (
          <details>
            <summary>{t(($) => $.editor.original)}</summary>
            <p>
              {editing.fact.original.attr} · {editing.fact.original.value}
            </p>
          </details>
        )}
        <p className="editor-hint">
          <Trans i18nKey={($) => $.editor.hint} components={{ clone: <span className="wc-clone" /> }} />
        </p>
        {error && (
          <div className="editor-error" role="alert">
            {error}
            <button
              type="button"
              className="wc-btn wc-btn--sm"
              disabled={busy}
              onClick={async () => {
                const ok = await confirm({
                  title: t(($) => $.editor.reloadConfirmTitle),
                  body: t(($) => $.editor.reloadConfirm),
                  action: t(($) => $.editor.reload),
                  tone: 'danger',
                });
                if (!ok) return;
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
          <button type="button" className="wc-btn" disabled={busy} onClick={() => void close()}>
            {t(($) => $.common.cancel)}
          </button>
          <button className="wc-btn" disabled={busy}>
            {t(($) => $.common.save)}
          </button>
          <button
            className="wc-btn wc-btn--adopt"
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
