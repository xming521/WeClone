// Shapes of weclone/server/profile_build.py, with the status file of weclone/data/agent/profile_pipeline.py.
export type Purpose = 'assistant' | 'clone';
/** upload: the chats uploaded from the web. config: the platform and csv_folder of settings.jsonc. */
export type ChatSource = 'upload' | 'config';
export type StageKey =
  'snapshot' | 'distill_profile' | 'distill_event' | 'organize' | 'hierarchy' | 'summaries' | 'style';
export type StageState = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
/** `interrupted`: the build process ended without recording how, as when the machine went down. */
export type BuildState = 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

export interface BuildStage {
  /** The [n/N] of the command line. */
  step: number;
  key: StageKey;
  state: StageState;
  started_at: string | null;
  finished_at: string | null;
  /** Chat files done and in all, for the stages that count them. */
  done: number | null;
  total: number | null;
}

export interface BuildStatus {
  purpose: Purpose;
  state: BuildState;
  started_at: string;
  finished_at: string | null;
  /** Chat files the build read, known after its first stage. */
  chats: number | null;
  stages: BuildStage[];
  /** Where it stopped. `message` is empty in encrypted mode, which keeps only the exception class. */
  error: { step: number | null; key: StageKey | null; type: string; message: string } | null;
  /** Its result directory under the runs folder, and whether the server shows it now. */
  run: string;
  current: boolean;
  log: string;
}

export interface Problem {
  code: string;
  params: Record<string, unknown>;
  /** The server's Chinese text, for codes the page has no words for. */
  message: string;
  /** About the saved model service. */
  backend?: true;
}

export interface BuildCheck {
  backend: { provider: 'codex_exec' | 'api' | null; model: string | null; max_context_tokens: number | null };
  /** The chats the build would read: the uploaded ones when there are any. */
  chats: { source: ChatSource; platform: 'chat' | 'telegram' | null; folder: string | null; count: number };
  problems: Problem[];
}

export interface RunVersion {
  name: string;
  purpose: Purpose | null;
  state: BuildState;
  /** Whether it has a profile to open. */
  complete: boolean;
  current: boolean;
  /** When its build started, null for a run made before builds kept a status. */
  started_at: string | null;
  generated_at: string | null;
  reviewed_at: string | null;
  style: boolean;
}

/** The step a running build is on, and the last step. Steps count the stages in order, so a status file written
 * before every stage had its own number counts the same way. Between two stages it is on the next one, as the bar is. */
export function buildStep(status: BuildStatus): { step: number; steps: number } {
  const steps = status.stages.length;
  const running = status.stages.findIndex((stage) => stage.state === 'running');
  const done = status.stages.map((stage) => stage.state).lastIndexOf('succeeded');
  return { step: (running >= 0 ? running : Math.min(done + 1, steps - 1)) + 1, steps };
}

// A running stage that counts no chats closes in on CREEP_CAP of its share at this pace, and fills the rest when it
// finishes. The pace is about the length of such stages in the builds seen so far (half a minute to two minutes).
const CREEP_SECONDS = 60;
const CREEP_CAP = 0.9;

/** How far a build has come, from 0 to 1: each stage an equal share, the running one filled by the chats it has done,
 * or by its time when it counts none. */
export function buildProgress(status: BuildStatus, now: number): number {
  const shares = status.stages.map((stage) => {
    if (stage.state === 'succeeded') return 1;
    if (stage.state !== 'running') return 0;
    if (stage.total) return (stage.done ?? 0) / stage.total;
    const seconds = stage.started_at ? Math.max(0, (now - Date.parse(stage.started_at)) / 1000) : 0;
    return CREEP_CAP * (1 - Math.exp(-seconds / CREEP_SECONDS));
  });
  return shares.length ? shares.reduce((sum, share) => sum + share, 0) / shares.length : 0;
}

/** Seconds as m:ss, or h:mm:ss from an hour on. */
export function clock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  const [h, m, s] = [Math.floor(whole / 3600), Math.floor((whole % 3600) / 60), whole % 60];
  const pad = (value: number) => String(value).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}
