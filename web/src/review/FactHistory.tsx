import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api } from '../api';
import { displayProfileText } from '../data';
import type { ProfileFact } from '../data';

interface HistoryEntry {
  at: string;
  action: string;
  before: ProfileFact | null;
  after: ProfileFact;
}
export function FactHistory({ fact }: { fact: ProfileFact }) {
  const { t, i18n } = useTranslation();
  const [rows, setRows] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState('');
  const [open, setOpen] = useState(false);
  return (
    <div className="fact-history">
      <button
        onClick={async () => {
          setOpen(!open);
          if (!rows) {
            // Clear an earlier failure so a successful retry shows the rows.
            setError('');
            try {
              setRows(await api.get<HistoryEntry[]>(`facts/${fact.id}/history`));
            } catch (reason) {
              setError((reason as Error).message);
            }
          }
        }}
      >
        {open ? t(($) => $.history.hide) : t(($) => $.history.show)}
      </button>
      {open && (
        <div>
          {error ||
            (rows?.length === 0
              ? t(($) => $.history.empty)
              : rows?.map((row, i) => (
                  <div className="history-row" key={i}>
                    <time>{new Date(row.at).toLocaleString(i18n.language)}</time>
                    <p>
                      {t(($) => $.history.change, {
                        action:
                          row.action === 'review'
                            ? t(($) => $.history.actions.review)
                            : row.action === 'create'
                              ? t(($) => $.history.actions.create)
                              : t(($) => $.history.actions.edit),
                        before: row.before ? t(($) => $.review.status[row.before!.status]) : t(($) => $.history.new),
                        after: t(($) => $.review.status[row.after.status]),
                      })}
                    </p>
                    {row.action !== 'review' && (
                      <>
                        <p>
                          {row.before &&
                            t(($) => $.history.before, {
                              attr: row.before.attr,
                              value: displayProfileText(row.before.value),
                            })}
                        </p>
                        <p>
                          {t(($) => $.history.after, {
                            attr: row.after.attr,
                            value: displayProfileText(row.after.value),
                          })}
                        </p>
                        {row.before?.group_id !== row.after.group_id && <p>{t(($) => $.history.moved)}</p>}
                      </>
                    )}
                  </div>
                )))}
        </div>
      )}
    </div>
  );
}
