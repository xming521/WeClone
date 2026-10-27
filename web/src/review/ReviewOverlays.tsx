import { useTranslation } from 'react-i18next';
import { useProfileStore } from '../store';
import { FactEditor } from './FactEditor';
import { SourceChatPanel } from './SourceChatPanel';

export function ReviewOverlays() {
  const { t } = useTranslation();
  const editing = useProfileStore((state) => state.editing !== null);
  const chat = useProfileStore((state) => state.chat);
  const error = useProfileStore((state) => state.error);
  const busy = useProfileStore((state) => state.busy);
  const reload = useProfileStore((state) => state.reload);
  const closeChat = useProfileStore((state) => state.closeChat);
  return (
    <>
      {chat && <SourceChatPanel selection={chat} onClose={closeChat} />}
      {error && !editing && (
        <div className="review-error" role="alert">
          {error}
          <button disabled={busy} onClick={() => void reload()}>
            {t(($) => $.review.refresh)}
          </button>
        </div>
      )}
      <FactEditor />
    </>
  );
}
