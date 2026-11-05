import { useEffect } from 'react';
import { create } from 'zustand';
import { api } from '../api';
import { useProfileStore } from '../store';
import type { BuildStatus } from './progress';

const POLL_MS = 3000;

interface BuildStore {
  /** The latest build, null when there was none, undefined until the server answers. */
  status: BuildStatus | null | undefined;
  refresh: () => Promise<void>;
}

let watchers = 0;
let timer: number | undefined;

/** While a build runs and the page is visible, the status is fetched again every few seconds. */
function schedule() {
  window.clearTimeout(timer);
  if (watchers && !document.hidden && useBuildStore.getState().status?.state === 'running')
    timer = window.setTimeout(() => void useBuildStore.getState().refresh(), POLL_MS);
}

export const useBuildStore = create<BuildStore>((set, get) => ({
  status: undefined,
  refresh: async () => {
    const before = get().status;
    try {
      const status = await api.get<BuildStatus | null>('build');
      set({ status });
      // A build the server opened on finishing replaces the profile on the map.
      if (before?.state === 'running' && status?.state === 'succeeded' && status.current) {
        const profile = useProfileStore.getState();
        if (profile.viewModel || profile.missing) void profile.reload();
      }
    } catch {
      // The status is extra information wherever it shows; the next poll tries again.
    }
    schedule();
  },
}));

function wake() {
  if (!document.hidden) void useBuildStore.getState().refresh();
}

/** The latest build, kept current while the calling component is mounted. */
export function useBuildStatus(): BuildStatus | null | undefined {
  useEffect(() => {
    watchers += 1;
    if (watchers === 1) {
      document.addEventListener('visibilitychange', wake);
      void useBuildStore.getState().refresh();
    }
    return () => {
      watchers -= 1;
      if (!watchers) {
        document.removeEventListener('visibilitychange', wake);
        window.clearTimeout(timer);
      }
    };
  }, []);
  return useBuildStore((state) => state.status);
}
