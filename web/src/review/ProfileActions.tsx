import { useEffect, useId, useRef, useState } from 'react';
import { Download, Plus } from 'lucide-react';
import { exportProfile } from '../data';
import type { ReviewView } from '../data';
import { useProfileStore } from '../store';
import { STATUS } from './status';

export function ProfileActions() {
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
        <button disabled={busy} onClick={() => edit(null)} aria-label="新增画像" title="新增画像">
          <Plus size={17} />
        </button>
        <button
          type="button"
          disabled={busy || !input}
          onClick={() => {
            if (input) setSelection({ view, profile: exportProfile(input, view) });
          }}
          aria-label="导出当前视图"
          title="导出当前视图"
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
            <h2 id={titleId}>导出画像</h2>
            <p id={descriptionId}>
              是否导出当前“<strong>{STATUS[selection.view]}</strong>”视图中的{' '}
              <strong>{selection.profile.facts.length}</strong> 条画像？
            </p>
            <p className="editor-hint">导出为 JSON 文件，包含画像层级和关联来源。</p>
            <div className="editor-actions">
              <button type="button" autoFocus onClick={close}>
                取消
              </button>
              <button className="primary" type="button" onClick={download}>
                确认导出
              </button>
            </div>
          </div>
        </dialog>
      )}
    </>
  );
}
