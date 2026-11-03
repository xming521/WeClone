import { useEffect, useState } from 'react';
import { api } from '../api';
import type { RunInfo } from './SettingsApp';

/** GET /api/run, or null until it answers or when it fails: the run is extra information wherever it shows. */
export function useRun(): RunInfo | null {
  const [run, setRun] = useState<RunInfo | null>(null);
  useEffect(() => {
    let active = true;
    api.get<RunInfo>('run').then(
      (info) => active && setRun(info),
      () => undefined,
    );
    return () => {
      active = false;
    };
  }, []);
  return run;
}
