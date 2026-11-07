import assert from 'node:assert/strict';
import { test } from 'node:test';
import { pickedFiles, uploadable } from '../src/build/upload.ts';

test('uploads keep what a chat folder is read from, and zips', () => {
  for (const path of ['csv/alice/alice_0_730.csv', 'csv/alice/users.json', 'Chats.ZIP', 'csv\\alice\\alice_0_730.csv'])
    assert.equal(uploadable(path), true, path);
});

test('uploads leave out other files and what the system or an archiver hides', () => {
  for (const path of ['csv/alice/photo.jpg', 'csv/.DS_Store', 'csv/.git/x.csv', '__MACOSX/csv/alice/._a.csv', ''])
    assert.equal(uploadable(path), false, path);
});

test('a picked folder stays in the path', () => {
  const file = new File(['a'], 'alice_0_1.csv');
  Object.defineProperty(file, 'webkitRelativePath', { value: 'csv/alice/alice_0_1.csv' });
  const picked = pickedFiles([file, new File(['b'], 'qq.zip'), new File([''], 'x.png')]);
  assert.deepEqual(
    picked.map((item) => item.path),
    ['csv/alice/alice_0_1.csv', 'qq.zip'],
  );
});
