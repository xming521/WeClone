import { LoaderCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import i18n from '../i18n';
import { ErrorDetails } from '../feedback/ErrorDetails';
import { money, tokens } from './cost';
import type { Provider } from '../settings/modelSettings';
import type { Estimate } from './progress';
import { timeText } from './spend';

/** The figures, the cost first: what the trial found the requests cost, as money or tokens, how long they take, and
 * the requests. A trial of the saved settings does not price changes still in the form. */
function estimateFigures(estimate: Estimate, unsaved: boolean): { value: string; label: string }[] {
  const { t, language } = i18n;
  const figures = [];
  const { cost, input_tokens: input, output_tokens: output, seconds } = estimate;
  if (estimate.probe && !unsaved) {
    if (cost)
      figures.push({ value: money(cost.low, cost.high, language), label: t(($) => $.build.estimate.figures.cost) });
    else if (input != null && output != null)
      figures.push(
        { value: tokens(input, language), label: t(($) => $.build.estimate.figures.input) },
        { value: tokens(output, language), label: t(($) => $.build.estimate.figures.output) },
      );
    if (seconds)
      figures.push({ value: timeText(seconds.low, seconds.high), label: t(($) => $.build.estimate.figures.time) });
  }
  figures.push({
    value: new Intl.NumberFormat(language).format(estimate.requests),
    label: t(($) => $.build.estimate.figures.requests),
  });
  return figures;
}

/** Whether a trial is still to run before the build starts: none has priced the settings, saved or still in the form,
 * and it has something to send. */
export function trialNeeded(estimate: Estimate | null, unsaved: boolean): boolean {
  return Boolean(estimate && (!estimate.probe || unsaved) && estimate.probe_requests > 0);
}

/** The sentences under the figures: what a trial sends, how the price was found, and what to know about it. */
function estimateHint(estimate: Estimate, unsaved: Provider | null): string[] {
  const { t } = i18n;
  const { probe, cost } = estimate;
  const codex = (unsaved ?? estimate.backend) === 'codex_exec';
  const sentences: string[] = [];
  if (!probe || unsaved) {
    const count = estimate.probe_requests;
    if (count > 0 && unsaved) sentences.push(t(($) => $.build.estimate.unsaved));
    if (count > 0)
      sentences.push(
        codex ? t(($) => $.build.estimate.inviteCodex, { count }) : t(($) => $.build.estimate.invite, { count }),
      );
    else sentences.push(t(($) => $.build.estimate.processed));
  } else {
    if (codex) sentences.push(t(($) => $.build.estimate.codex));
    else if (estimate.unmetered) sentences.push(t(($) => $.build.estimate.unmetered));
    else if (cost) {
      const vendor = cost.source === 'official' ? cost.vendor : null;
      const saving = Math.round((cost.off_peak_saving ?? 0) * 100);
      const share = cost.off_peak_share === null ? null : Math.round(cost.off_peak_share * 100);
      sentences.push(
        t(($) => $.build.estimate.priced, {
          source: vendor
            ? t(($) => $.build.estimate.sourceOfficial, { vendor: t(($) => $.settings.model.vendors[vendor]) })
            : t(($) => $.build.estimate.sourceMarket),
          each: money(cost.per_request.low, cost.per_request.high, i18n.language, true),
          offPeak:
            share === null || saving <= 0
              ? ''
              : share >= 100
                ? t(($) => $.build.estimate.offPeakAll)
                : share <= 0
                  ? t(($) => $.build.estimate.offPeakNone, { saving })
                  : t(($) => $.build.estimate.offPeakPart, { share, saving }),
          date: cost.prices_date,
        }),
      );
    } else sentences.push(t(($) => $.build.estimate.noPrice));
  }
  return sentences;
}

/** The estimate of what a build sends and costs, in the confirm step. It never blocks starting. */
export function EstimateRow({
  estimate,
  unsaved,
  estimating,
  unavailable,
  probing,
  probeError,
  onProbe,
  onRetry,
  onDismissProbeError,
}: {
  estimate: Estimate | null;
  /** The backend in the model form while it has changes not saved yet. */
  unsaved: Provider | null;
  estimating: boolean;
  /** Why there is no estimate, when the server could not make one. */
  unavailable: string;
  probing: boolean;
  probeError: { message: string; detail: string } | null;
  onProbe: () => void;
  onRetry: () => void;
  onDismissProbeError: () => void;
}) {
  const { t } = useTranslation();
  const sentences = estimate ? estimateHint(estimate, unsaved) : unavailable ? [unavailable] : [];
  const hint = probing
    ? t(($) => $.build.estimate.probingHint, { count: estimate?.probe_requests ?? 0 })
    : sentences.join(t(($) => $.common.sentenceSeparator));
  return (
    <>
      <div className="settings-row build-estimate">
        <div className="settings-row__text">
          <span className="settings-row__label">
            {t(($) => $.build.estimate.label)}
            {!estimate && (
              <span className="build-estimate__value" role="status">
                {estimating ? (
                  <>
                    <LoaderCircle className="build-spin" size={13} aria-hidden="true" />{' '}
                    {t(($) => $.build.estimate.estimating)}
                  </>
                ) : (
                  t(($) => $.build.estimate.unavailable)
                )}
              </span>
            )}
          </span>
          {estimate && (
            <div className="build-estimate__figures" role="status">
              {estimateFigures(estimate, Boolean(unsaved)).map((figure) => (
                <div key={figure.label}>
                  <strong>{figure.value}</strong>
                  <span>{figure.label}</span>
                </div>
              ))}
            </div>
          )}
          {hint && <span className="settings-row__hint">{hint}</span>}
        </div>
        {trialNeeded(estimate, Boolean(unsaved)) && (
          <div className="settings-row__control">
            <button className="wc-btn wc-btn--sm" type="button" disabled={probing} onClick={onProbe}>
              {probing && <LoaderCircle className="build-spin" size={13} aria-hidden="true" />}{' '}
              {probing
                ? t(($) => $.build.estimate.probing)
                : (unsaved ?? estimate?.backend) === 'codex_exec'
                  ? t(($) => $.build.estimate.probeUsage)
                  : t(($) => $.build.estimate.probe)}
            </button>
          </div>
        )}
        {!estimate && !estimating && (
          <div className="settings-row__control">
            <button className="wc-btn wc-btn--sm" type="button" onClick={onRetry}>
              {t(($) => $.common.retry)}
            </button>
          </div>
        )}
      </div>
      {probeError && (
        <div className="summary-error" role="alert">
          <span>
            {probeError.message}
            {probeError.detail && <ErrorDetails text={probeError.detail} />}
          </span>
          <button className="wc-btn wc-btn--ghost wc-btn--sm" type="button" onClick={onDismissProbeError}>
            {t(($) => $.common.dismiss)}
          </button>
        </div>
      )}
    </>
  );
}
