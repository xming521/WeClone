import { useState, type DragEvent } from 'react';
import { FolderUp, LoaderCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { api } from '../api';
import i18n from '../i18n';
import { droppedFiles, pickedFiles, type Picked, type UploadResult } from './upload';

/** Sends the files one by one and places the chats they hold. `onProgress` hears the bytes sent so far. */
export async function uploadChats(files: Picked[], onProgress: (sent: number) => void): Promise<UploadResult> {
  const { id } = await api.post<{ id: string }>('data/uploads');
  let sent = 0;
  try {
    for (const { path, file } of files) {
      await api.put(`data/uploads/${id}/files?${new URLSearchParams({ path })}`, file);
      sent += file.size;
      onProgress(sent);
    }
    return await api.post<UploadResult>(`data/uploads/${id}/done`);
  } catch (reason) {
    void api.delete(`data/uploads/${id}`).catch(() => undefined);
    throw reason;
  }
}

function size(bytes: number): string {
  return new Intl.NumberFormat(i18n.language, {
    style: 'unit',
    unit: 'megabyte',
    maximumFractionDigits: 1,
  }).format(bytes / 1e6);
}

type Upload =
  | { state: 'idle' }
  | { state: 'sending'; sent: number; total: number }
  | { state: 'done'; result: UploadResult }
  | { state: 'failed'; message: string };

/** Chat folders and zips picked or dropped here go into csv_folder. `onUploaded` runs the check again. */
export function ChatUpload({ disabled, onUploaded }: { disabled: boolean; onUploaded: () => void }) {
  const { t } = useTranslation();
  const [upload, setUpload] = useState<Upload>({ state: 'idle' });
  const [over, setOver] = useState(false);
  const sending = upload.state === 'sending';

  async function send(files: Picked[]) {
    if (!files.length) {
      setUpload({ state: 'failed', message: t(($) => $.build.upload.nothing) });
      return;
    }
    const total = files.reduce((sum, picked) => sum + picked.file.size, 0);
    setUpload({ state: 'sending', sent: 0, total });
    try {
      const result = await uploadChats(files, (sent) => setUpload({ state: 'sending', sent, total }));
      setUpload({ state: 'done', result });
      onUploaded();
    } catch (reason) {
      setUpload({ state: 'failed', message: (reason as Error).message });
    }
  }

  function drop(event: DragEvent) {
    event.preventDefault();
    setOver(false);
    if (disabled || sending) return;
    droppedFiles(event.dataTransfer).then(send, (reason: Error) =>
      setUpload({ state: 'failed', message: reason.message }),
    );
  }

  const off = disabled || sending;
  return (
    <div
      className="build-upload"
      data-over={over || undefined}
      onDragOver={(event) => {
        event.preventDefault();
        if (!off) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
    >
      <FolderUp className="build-upload__icon" size={18} strokeWidth={1.8} aria-hidden="true" />
      <span className="build-upload__text">{t(($) => $.build.upload.drop)}</span>
      <span className="build-upload__actions">
        <label className="wc-btn wc-btn--sm" aria-disabled={off || undefined}>
          {t(($) => $.build.upload.folder)}
          <input
            type="file"
            className="wc-sr"
            multiple
            disabled={off}
            // React has no prop for folder picking.
            ref={(node) => node?.setAttribute('webkitdirectory', '')}
            onChange={(event) => {
              void send(pickedFiles(event.target.files ?? []));
              event.target.value = '';
            }}
          />
        </label>
        <label className="wc-btn wc-btn--sm" aria-disabled={off || undefined}>
          {t(($) => $.build.upload.zip)}
          <input
            type="file"
            className="wc-sr"
            accept=".zip,application/zip"
            multiple
            disabled={off}
            onChange={(event) => {
              void send(pickedFiles(event.target.files ?? []));
              event.target.value = '';
            }}
          />
        </label>
      </span>
      {upload.state !== 'idle' && (
        <p
          className="wc-field__hint build-upload__state"
          data-tone={upload.state === 'failed' ? 'danger' : undefined}
          role={upload.state === 'failed' ? 'alert' : 'status'}
        >
          {upload.state === 'sending' && (
            <>
              <LoaderCircle className="build-spin" size={13} aria-hidden="true" />{' '}
              {t(($) => $.build.upload.sending, { sent: size(upload.sent), total: size(upload.total) })}
            </>
          )}
          {upload.state === 'done' &&
            (upload.result.replaced
              ? t(($) => $.build.upload.doneReplaced, {
                  count: upload.result.chats,
                  replaced: upload.result.replaced,
                })
              : t(($) => $.build.upload.done, { count: upload.result.chats }))}
          {upload.state === 'failed' && upload.message}
        </p>
      )}
    </div>
  );
}
