// How the build page writes money, token counts and times. Plain functions of the page's language.

/** US dollars to yuan for the Chinese page: the rate LiteLLM converts prices listed in yuan at, so they show as listed. */
export const USD_TO_CNY = 7.0;

const isChinese = (language: string) => language.toLowerCase().startsWith('zh');

function number(value: number, language: string, options: Intl.NumberFormatOptions = {}): string {
  return new Intl.NumberFormat(language, options).format(value);
}

/** An amount in the page's currency: yuan on the Chinese page, dollars otherwise. A single request's price keeps two
 * significant digits whatever its size. A total of 10 or more is whole, a smaller one keeps two significant digits. */
function amount(usd: number, language: string, each: boolean): string {
  const value = isChinese(language) ? usd * USD_TO_CNY : usd;
  return each || value < 10
    ? number(value, language, { maximumSignificantDigits: 2 })
    : number(value, language, { maximumFractionDigits: 0 });
}

const symbol = (language: string) => (isChinese(language) ? '¥' : '$');

/** "¥6–23", or "¥212" when both ends read the same. `each` is for the price of one request. */
export function money(low: number, high: number, language: string, each = false): string {
  const [from, to] = [amount(low, language, each), amount(high, language, each)];
  return `${symbol(language)}${from === to ? from : `${from}–${to}`}`;
}

/** A token count in the page's compact form, with three significant digits: "1,040 万" in Chinese, "10.4M" in English.
 * The Chinese unit is set off from its number as the page's other counts are. */
export function tokens(count: number, language: string): string {
  // 'always' groups four digits too, as "1,040 万"; the TypeScript lib types the option as a boolean only.
  const options = { notation: 'compact', maximumSignificantDigits: 3, useGrouping: 'always' };
  const format = new Intl.NumberFormat(language, options as unknown as Intl.NumberFormatOptions);
  return format
    .formatToParts(count)
    .map((part) => (part.type === 'compact' && isChinese(language) ? ` ${part.value}` : part.value))
    .join('');
}

// Up to this long, a time is in minutes, beyond it in hours.
const MINUTES_UNTIL = 90 * 60;

/** A time or a range of times in minutes or, past 90 minutes, hours. `range` is the numbers, as "25–45". */
export function span(low: number, high: number, language: string): { unit: 'minutes' | 'hours'; range: string } {
  const minutes = high <= MINUTES_UNTIL;
  const write = (seconds: number) =>
    minutes
      ? number(Math.max(1, Math.round(seconds / 60)), language)
      : number(seconds / 3600, language, { maximumFractionDigits: seconds >= 36000 ? 0 : 1 });
  const [from, to] = [write(low), write(high)];
  return { unit: minutes ? 'minutes' : 'hours', range: from === to ? from : `${from}–${to}` };
}
