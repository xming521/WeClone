import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { DIMENSIONS, DOMAIN_COLORS, ROOT_COLOR, dimensionColor, sectionColor } from '../src/data.ts';

// The canvas map cannot read CSS in data.ts, which keeps plain values; these tests keep them equal
// to the design system's tokens (src/ds/tokens.css, light theme) and its section rule (components.css).
const tokensCss = readFileSync(new URL('../src/ds/tokens.css', import.meta.url), 'utf8');
const componentsCss = readFileSync(new URL('../src/ds/components.css', import.meta.url), 'utf8');
// The first declaration of a token is its light-theme one; the dark theme overrides it later.
const token = (name: string) => new RegExp(`${name}:\\s*([^;]+);`).exec(tokensCss)?.[1].trim();

test('dimension and centre colours are the design system tokens', () => {
  const names = ['identity', 'goal', 'condition', 'plan', 'preference'];
  assert.deepEqual(
    DIMENSIONS.map((entry) => entry.color),
    names.map((name) => token(`--dim-${name}`)),
  );
  assert.equal(ROOT_COLOR, token('--ink-3'));
});

test('the section rule matches the design system', () => {
  assert.equal(token('--sec-base'), 'oklch(0.585 0.11 182)');
  assert.match(componentsCss, /--wc-sec-span:268;--wc-sec-dl:\.09/);
  assert.match(componentsCss, /\.05 \* clamp\(0, \(var\(--wc-sec-t\) - \.8\) \/ \.2, 1\)/);
});

// The palette the rule was fitted to, as the view had it before the rule: the same to within 5/255 a channel.
const OLD_PALETTE = [
  '#058f80',
  '#1cabb1',
  '#058aa4',
  '#49a1d1',
  '#4a7ebb',
  '#7e92db',
  '#7a70b9',
  '#a984cc',
  '#9c64a0',
  '#c67ba9',
  '#b05e78',
  '#d27a7c',
  '#b3624c',
  '#cf8755',
  '#ae7a30',
  '#c2a44f',
];
const channels = (hex: string) => [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16));

test('section colours reproduce the 16-colour palette at n = 16', () => {
  OLD_PALETTE.forEach((old, index) => {
    const now = channels(DOMAIN_COLORS[index]);
    channels(old).forEach((value, channel) => assert.ok(Math.abs(value - now[channel]) <= 5, `${index}/${channel}`));
  });
});

test('section colours work for any count', () => {
  for (const count of [1, 2, 3, 8, 15, 16, 40]) {
    const colors = Array.from({ length: count }, (_, index) => sectionColor(index, count));
    assert.ok(
      colors.every((color) => /^#[0-9a-f]{6}$/.test(color)),
      `n = ${count}`,
    );
    assert.equal(new Set(colors).size, count, `distinct at n = ${count}`);
  }
  // An index past the count extends the walk instead of repeating.
  assert.notEqual(sectionColor(3, 3), sectionColor(2, 3));
  assert.equal(sectionColor(3, 3), sectionColor(3, 4));
  assert.equal(dimensionColor(100), sectionColor(0, 16));
});

test('the tokens the canvas reads exist', () => {
  const names = [
    '--paper',
    '--card',
    '--line-1',
    '--line-2',
    '--ink-1',
    '--ink-2',
    '--ink-3',
    '--ink-4',
    '--edge-1',
    '--edge-2',
    '--logo-you-1',
    '--logo-you-2',
    '--logo-you-eye-sm',
    '--logo-clone-1',
    '--logo-clone-2',
    '--logo-clone-eye-sm',
    '--text-node-root',
    '--text-node-dim',
    '--text-node-section',
    '--text-node-attr',
  ];
  for (const name of names) assert.ok(token(name), name);
});
