// Formatters the catalogs call by name. Import-free, so tests load it directly.

const CHINESE_NUMERALS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'];

/** Small counts in Chinese numerals, as in 五维总览; 100 and above stay digits. */
export function chineseNumeral(n: number): string {
  if (!Number.isInteger(n) || n < 0 || n >= 100) return String(n);
  if (n <= 10) return CHINESE_NUMERALS[n];
  const tens = Math.floor(n / 10);
  return `${tens > 1 ? CHINESE_NUMERALS[tens] : ''}十${n % 10 ? CHINESE_NUMERALS[n % 10] : ''}`;
}
