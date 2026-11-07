import { useEffect, useId, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useConfirmations, type Confirmation } from './confirm';

/** The question of `confirm()`, on the editor's dialog sheet like the export question. Cancel has the focus, so Enter
 * never goes ahead by accident; Esc cancels. */
export function ConfirmDialog() {
  const current = useConfirmations((state) => state.current);
  return current ? <ConfirmSheet key={current.id} confirmation={current} /> : null;
}

function ConfirmSheet({ confirmation }: { confirmation: Confirmation }) {
  const { t } = useTranslation();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const bodyId = useId();
  useEffect(() => {
    dialogRef.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialogRef}
      className="editor-backdrop"
      role="alertdialog"
      aria-labelledby={titleId}
      aria-describedby={confirmation.body ? bodyId : undefined}
      onClose={(event) => confirmation.answer(event.currentTarget.returnValue === 'ok')}
    >
      <form method="dialog" className="fact-editor confirm-sheet">
        <h2 id={titleId}>{confirmation.title}</h2>
        {confirmation.body && <p id={bodyId}>{confirmation.body}</p>}
        <div className="editor-actions">
          <button className="wc-btn" value="" autoFocus>
            {confirmation.dismiss ?? t(($) => $.common.cancel)}
          </button>
          <button className={`wc-btn wc-btn--${confirmation.tone ?? 'primary'}`} value="ok">
            {confirmation.action}
          </button>
        </div>
      </form>
    </dialog>
  );
}
