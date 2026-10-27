import { useEffect, useId, useRef, useState } from 'react';
import { Download, Plus } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { exportProfile } from '../data';
import type { ReviewView } from '../data';
import { useProfileStore } from '../store';

export function ProfileActions() {
  const { t } = useTranslation();
  const input = useProfileStore((state) => state.input);
  const view = useProfileStore((state) => state.view);
  const busy = useProfileStore((state) => state.busy);
  const edit = useProfileStore((state) => state.edit);
  const [selection, setSelection] = useState<{ view: ReviewView; profile: ReturnType<typeof exportProfile> } | null>(
    null,
  );
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descriptionId = useId();
  useEffect(() => {
    if (selection) dialogRef.current?.showModal();
  }, [selection]);
  function close() {
    dialogRef.current?.close();
    setSelection(null);
  }
  function download() {
    if (!selection) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(selection.profile, null, 2)], { type: 'application/json;charset=utf-8' }),
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = `profile_${selection.view}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    close();
  }
  return (
    <>
      <div className="header-review-actions">
        <button
          disabled={busy}
          onClick={() => edit(null)}
          aria-label={t(($) => $.editor.addTitle)}
          title={t(($) => $.editor.addTitle)}
        >
          <Plus size={17} />
        </button>
        <button
          type="button"
          disabled={busy || !input}
          onClick={() => {
            if (input) setSelection({ view, profile: exportProfile(input, view) });
          }}
          aria-label={t(($) => $.export.action)}
          title={t(($) => $.export.action)}
        >
          <Download size={17} />
        </button>
      </div>
      {selection && (
        <dialog
          ref={dialogRef}
          className="editor-backdrop"
          aria-labelledby={titleId}
          aria-describedby={descriptionId}
          onCancel={(event) => {
            event.preventDefault();
            close();
          }}
        >
          <div className="fact-editor export-confirm">
            <h2 id={titleId}>{t(($) => $.export.title)}</h2>
            <p id={descriptionId}>
              <Trans
                i18nKey={($) => $.export.confirm}
                values={{ view: t(($) => $.review.status[selection.view]), count: selection.profile.facts.length }}
                components={{ strong: <strong /> }}
              />
            </p>
            <p className="editor-hint">{t(($) => $.export.hint)}</p>
            <div className="editor-actions">
              <button type="button" autoFocus onClick={close}>
                {t(($) => $.common.cancel)}
              </button>
              <button className="primary" type="button" onClick={download}>
                {t(($) => $.export.submit)}
              </button>
            </div>
          </div>
        </dialog>
      )}
    </>
  );
}
