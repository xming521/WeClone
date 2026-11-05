// Shapes of weclone/server/profile_build.py, with the status file of weclone/data/agent/profile_pipeline.py.
export type Purpose = 'assistant' | 'clone';
export type StageKey =
  'snapshot' | 'distill_profile' | 'distill_event' | 'organize' | 'hierarchy' | 'summaries' | 'style';
export type StageState = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
/** `interrupted`: the build process ended without recording how, as when the machine went down. */
export type BuildState = 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

export interface BuildStage {
  /** The [n/N] of the command line: the two extraction stages share step 2. */
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
}

export interface BuildCheck {
  backend: { provider: 'codex_exec' | 'api' | null; model: string | null; max_context_tokens: number | null };
  chats: { platform: string | null; count: number };
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

/** The step a running build is on, and the last step. */
export function buildStep(status: BuildStatus): { step: number; steps: number } {
  const steps = status.stages.at(-1)?.step ?? 0;
  const running = status.stages.find((stage) => stage.state === 'running');
  const done = status.stages.filter((stage) => stage.state === 'succeeded').at(-1);
  return { step: running?.step ?? done?.step ?? 1, steps };
}

/** Seconds as m:ss, or h:mm:ss from an hour on. */
export function clock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  const [h, m, s] = [Math.floor(whole / 3600), Math.floor((whole % 3600) / 60), whole % 60];
  const pad = (value: number) => String(value).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}
