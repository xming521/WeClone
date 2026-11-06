// The files sent to weclone/server/chat_upload.py, which places the chats they hold into csv_folder.

export interface Picked {
  /** Relative to what was picked or dropped, the picked folder's own name first. */
  path: string;
  file: File;
}

export interface UploadResult {
  chats: number;
  replaced: number;
}

function hidden(name: string): boolean {
  return name.startsWith('.') || name === '__MACOSX';
}

/** Whether the server keeps the file at this path, by the rule the server applies again: the files a chat folder
 * is read from, and zips, which it opens. Files hidden by the system or an archiver are left out. */
export function uploadable(path: string): boolean {
  const parts = path.split(/[\\/]/).filter(Boolean);
  const name = parts.at(-1);
  if (!name || parts.some(hidden)) return false;
  return name === 'users.json' || name.endsWith('.csv') || name.toLowerCase().endsWith('.zip');
}

/** The files of an input, the ones of a picked folder with their path in it. */
export function pickedFiles(files: Iterable<File>): Picked[] {
  return Array.from(files, (file) => ({ path: file.webkitRelativePath || file.name, file })).filter((picked) =>
    uploadable(picked.path),
  );
}

function entries(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  // Each call gives the next batch, an empty one at the end.
  return new Promise((resolve, reject) => {
    const found: FileSystemEntry[] = [];
    const next = () =>
      reader.readEntries((batch) => {
        if (!batch.length) resolve(found);
        else {
          found.push(...batch);
          next();
        }
      }, reject);
    next();
  });
}

async function walk(entry: FileSystemEntry, found: Picked[]): Promise<void> {
  if (hidden(entry.name)) return;
  const path = entry.fullPath.replace(/^\/+/, '');
  if (entry.isDirectory) {
    for (const child of await entries((entry as FileSystemDirectoryEntry).createReader())) await walk(child, found);
  } else if (uploadable(path)) {
    const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
    found.push({ path, file });
  }
}

/** Files and folders dropped on the page. The entries are taken at once: the drop's items go stale after it. */
export function droppedFiles(transfer: DataTransfer): Promise<Picked[]> {
  const dropped = Array.from(transfer.items, (item) => item.webkitGetAsEntry()).filter(
    (entry): entry is FileSystemEntry => entry !== null,
  );
  return (async () => {
    const found: Picked[] = [];
    for (const entry of dropped) await walk(entry, found);
    return found;
  })();
}
