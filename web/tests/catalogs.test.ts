import assert from 'node:assert/strict';
import { test } from 'node:test';
import en from '../src/i18n/locales/en/translation.json' with { type: 'json' };
import zhCN from '../src/i18n/locales/zh-CN/translation.json' with { type: 'json' };

type Catalog = { [key: string]: string | Catalog };

function flatten(catalog: Catalog, prefix = ''): Map<string, string> {
  const entries = new Map<string, string>();
  for (const [key, value] of Object.entries(catalog)) {
    if (typeof value === 'string') entries.set(prefix + key, value);
    else for (const entry of flatten(value, `${prefix}${key}.`)) entries.set(...entry);
  }
  return entries;
}

// Chinese has one plural form, English two: a zh-CN `_other` key stands for both English ones.
function englishKeys(keys: Iterable<string>): string[] {
  return [...keys].flatMap((key) => (key.endsWith('_other') ? [key.replace(/_other$/, '_one'), key] : [key]));
}

const variables = (text: string) => [...text.matchAll(/{{\s*(\w+)/g)].map((match) => match[1]).sort();
const tags = (text: string) => [...text.matchAll(/<\/?(\w+)/g)].map((match) => match[0]).sort();

const zh = flatten(zhCN);
const english = flatten(en);

test('the English catalog has the keys of the zh-CN one', () => {
  assert.deepEqual([...english.keys()].sort(), englishKeys(zh.keys()).sort());
});

test('each English text keeps the variables and tags of its zh-CN text', () => {
  for (const [key, text] of english) {
    const source = zh.get(key.replace(/_one$/, '_other'))!;
    // A singular may name its one item instead of counting it.
    const expected = key.endsWith('_one')
      ? variables(source).filter((name) => variables(text).includes(name))
      : variables(source);
    assert.deepEqual(variables(text), expected, key);
    assert.deepEqual(tags(text), tags(source), key);
    assert.doesNotMatch(text, /zhNumeral/, key);
  }
});
