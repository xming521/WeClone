import i18n from '../i18n';
import { money, span, tokens } from './cost';
import type { Estimate, Usage } from './progress';

/** An amount of money in US dollars, as the page's currency writes it. */
export function amountText(usd: number, each = false): string {
  return money(usd, usd, i18n.language, each);
}

/** A time or a range of times, as "25–45 分钟" or "11–19 小时". */
export function timeText(lowSeconds: number, highSeconds: number): string {
  const { unit, range } = span(lowSeconds, highSeconds, i18n.language);
  return unit === 'minutes'
    ? i18n.t(($) => $.build.cost.minutes, { range })
    : i18n.t(($) => $.build.cost.hours, { range });
}

/** Tokens in and out, as "输入约 1,040 万、输出约 130 万 token". */
export function tokensText(input: number, output: number): string {
  return i18n.t(($) => $.build.cost.tokens, {
    input: tokens(input, i18n.language),
    output: tokens(output, i18n.language),
  });
}

/** What a build has used so far: its cost, or for a model with no price its tokens. `value` is the figure alone, `text`
 * with its "about". Null before any request, and when a service that returns no usage leaves nothing to show. */
export function spend(
  usage: Usage | null | undefined,
): { kind: 'cost' | 'tokens'; value: string; text: string } | null {
  if (!usage || usage.requests <= 0) return null;
  if (usage.cost !== null) {
    const value = amountText(usage.cost);
    return { kind: 'cost', value, text: i18n.t(($) => $.build.cost.about, { amount: value }) };
  }
  if (usage.input_tokens + usage.output_tokens <= 0) return null;
  const value = tokensText(usage.input_tokens, usage.output_tokens);
  return { kind: 'tokens', value, text: value };
}

/** What a build was expected to cost and take, as "¥6–23、25–45 分钟", for when it is too early to project from the
 * build itself. Null when the estimate has neither. */
export function estimatedTotal(estimate: Estimate | null | undefined): string | null {
  const parts = [
    estimate?.cost && money(estimate.cost.low, estimate.cost.high, i18n.language),
    estimate?.seconds && timeText(estimate.seconds.low, estimate.seconds.high),
  ].filter((part): part is string => Boolean(part));
  return parts.length
    ? i18n.t(($) => $.build.running.estimated, { parts: parts.join(i18n.t(($) => $.common.listSeparator)) })
    : null;
}
