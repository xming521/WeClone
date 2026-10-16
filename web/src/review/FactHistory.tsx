import { useState } from 'react';
import { api } from '../api';
import { displayProfileText } from '../data';
import type { ProfileFact } from '../data';
import { STATUS } from './status';

interface HistoryEntry {
  at: string;
  action: string;
  before: ProfileFact | null;
  after: ProfileFact;
}
export function FactHistory({ fact }: { fact: ProfileFact }) {
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
        {open ? '收起修改历史' : '查看修改历史'}
      </button>
      {open && (
        <div>
          {error ||
            (rows?.length === 0
              ? '暂无修改记录'
              : rows?.map((row, i) => (
                  <div className="history-row" key={i}>
                    <time>{new Date(row.at).toLocaleString()}</time>
                    <p>
                      {row.action === 'review' ? '审核' : row.action === 'create' ? '新增' : '编辑'}：
                      {row.before ? STATUS[row.before.status] : '新记录'} → {STATUS[row.after.status]}
                    </p>
                    {row.action !== 'review' && (
                      <>
                        <p>{row.before && `修改前：${row.before.attr} · ${displayProfileText(row.before.value)}`}</p>
                        <p>
                          保存后：{row.after.attr} · {displayProfileText(row.after.value)}
                        </p>
                        {row.before?.group_id !== row.after.group_id && <p>归属位置已变更</p>}
                      </>
                    )}
                  </div>
                )))}
        </div>
      )}
    </div>
  );
}
