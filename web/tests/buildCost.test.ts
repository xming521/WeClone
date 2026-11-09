import assert from 'node:assert/strict';
import { test } from 'node:test';
import { money, span, tokens } from '../src/build/cost.ts';

test('money is in yuan on the Chinese page and dollars on the English one', () => {
  assert.equal(money(1, 1, 'zh-CN'), '¥7');
  assert.equal(money(1, 1, 'en'), '$1');
});

test('money keeps two significant digits below 10 and whole numbers from 10, and a single request always two digits', () => {
  assert.equal(money(0.123, 0.123, 'en'), '$0.12');
  assert.equal(money(9.87, 9.87, 'en'), '$9.9');
  assert.equal(money(1234.5, 1234.5, 'en'), '$1,235');
  assert.equal(money(0.0013 / 7, 0.0026 / 7, 'zh-CN', true), '¥0.0013–0.0026');
  assert.equal(money(30, 30, 'en', true), '$30');
});

test('money gives one figure when both ends read the same', () => {
  assert.equal(money(6 / 7, 23 / 7, 'zh-CN'), '¥6–23');
  assert.equal(money(100.1, 100.4, 'en'), '$100');
});

test('tokens are in 万 and 亿 on the Chinese page and compact on the English one', () => {
  assert.equal(tokens(10437000, 'zh-CN'), '1,040 万');
  assert.equal(tokens(1300000, 'zh-CN'), '130 万');
  assert.equal(tokens(250000000, 'zh-CN'), '2.5 亿');
  assert.equal(tokens(234, 'zh-CN'), '234');
  assert.equal(tokens(10437000, 'en'), '10.4M');
});

test('span is in minutes up to 90 minutes and in hours beyond', () => {
  assert.deepEqual(span(25 * 60, 45 * 60, 'en'), { unit: 'minutes', range: '25–45' });
  assert.deepEqual(span(10, 20, 'en'), { unit: 'minutes', range: '1' });
  assert.deepEqual(span(80 * 60, 100 * 60, 'en'), { unit: 'hours', range: '1.3–1.7' });
  assert.deepEqual(span(11 * 3600, 19 * 3600, 'en'), { unit: 'hours', range: '11–19' });
  assert.deepEqual(span(9 * 3600, 9 * 3600, 'zh-CN'), { unit: 'hours', range: '9' });
});
