import { LoaderCircle, Sparkles } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useBuildStatus } from './buildState';
import { buildStep } from './progress';
import './build.css';

/** In the topbar of every page: a build that runs, or a finished one the server does not show yet. */
export function BuildIndicator() {
  const { t } = useTranslation();
  const status = useBuildStatus();
  if (status?.state === 'running') {
    const { step, steps } = buildStep(status);
    return (
      <a className="build-pill" href="#/build" title={t(($) => $.build.indicator.open)}>
        <LoaderCircle className="build-spin" size={14} strokeWidth={2} aria-hidden="true" />
        {t(($) => $.build.indicator.running, { step, steps })}
      </a>
    );
  }
  if (status?.state === 'succeeded' && !status.current)
    return (
      <a className="build-pill build-pill--done" href="#/build" title={t(($) => $.build.indicator.open)}>
        <Sparkles size={14} strokeWidth={2} aria-hidden="true" />
        {t(($) => $.build.indicator.done)}
      </a>
    );
  return null;
}
