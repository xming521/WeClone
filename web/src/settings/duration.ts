import i18n from '../i18n';

/** A length of time in the largest whole unit: minutes, hours or days. */
export function formatDuration(seconds: number): string {
  const t = i18n.t;
  if (seconds % 86400 === 0) return t(($) => $.common.duration.days, { count: seconds / 86400 });
  if (seconds % 3600 === 0) return t(($) => $.common.duration.hours, { count: seconds / 3600 });
  return t(($) => $.common.duration.minutes, { count: Math.round(seconds / 60) });
}

/** A server time (ISO text) as a local date and time. */
export function formatTime(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? null
    : date.toLocaleString(i18n.language, { dateStyle: 'medium', timeStyle: 'short' });
}
