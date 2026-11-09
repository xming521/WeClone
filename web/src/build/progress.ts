// Shapes of weclone/server/profile_build.py, with the status file of weclone/data/agent/profile_pipeline.py.
import type { Vendor } from '../settings/modelSettings';

export type Purpose = 'assistant' | 'clone';
/** upload: the chats uploaded from the web. config: the platform and csv_folder of settings.jsonc. */
export type StageKey =
  'snapshot' | 'distill_profile' | 'distill_event' | 'organize' | 'hierarchy' | 'summaries' | 'style';
export type StageState = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
/** `interrupted`: the build process ended without recording how, as when the machine went down. */
export type BuildState = 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

/** What the model service has been asked and has answered so far. `cost` is in US dollars, null for a model the
 * price table does not have. `unmetered` counts the requests the service returned no usage for. */
export interface Usage {
  requests: number;
  unmetered: number;
  input_tokens: number;
  cached_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  cost: number | null;
}

/** What a build is expected to send and cost, from GET /api/build/estimate and POST /api/build/probe. Money is in
 * US dollars, time in seconds. `probe` is the average of the trial requests, null until they were sent with the
 * settings in force. */
export interface Estimate {
  backend: 'api' | 'codex_exec';
  /** Requests the build sends: the extraction ones with no answer yet, the later stages, the speaking style. */
  requests: number;
  /** The requests a trial would send now: none when every extraction request has its answer already. */
  probe_requests: number;
  /** `later` is organizing, hierarchy and summaries together. */
  plan: { distill_profile: number; distill_event: number; later: number; style: number };
  probe: {
    requests: number;
    at: string;
    input_tokens: number;
    cached_tokens: number;
    output_tokens: number;
    reasoning_tokens: number;
    seconds: number;
  } | null;
  /** The service returned no usage for the trial. */
  unmetered: boolean;
  input_tokens: number | null;
  output_tokens: number | null;
  cost: {
    low: number;
    high: number;
    per_request: { low: number; high: number };
    /** How much of the build's time, started now, falls in off-peak hours, for a price that has them. */
    off_peak_share: number | null;
    off_peak_saving: number | null;
    source: 'official' | 'market';
    vendor: Vendor | null;
    prices_date: string;
  } | null;
  seconds: { low: number; high: number } | null;
}

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
  /** What done and total count: `requests`. Null, or absent in a status file from before requests were counted, is
   * chat files. */
  unit?: 'requests' | null;
  /** Absent in a status file from before usage was kept, which reads as null. */
  usage?: Usage | null;
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
  /** The build so far, and what it was expected to be after its first step. Absent in older status files. */
  usage?: Usage | null;
  estimate?: Estimate | null;
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
  chats: { platform: 'chat' | 'telegram' | null; folder: string | null; count: number };
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

/** Fewer requests than this say too little about the pace and price of the rest. */
export const PROJECT_AFTER_REQUESTS = 20;
// A later stage (organizing, hierarchy, summaries) takes this many times as long per request as an extraction one,
// as LATER_SECONDS of weclone/data/agent/build_cost.py.
const LATER_SLOWDOWN = 2;

export interface Projection {
  /** Requests still to send, and the time and money they take, at the pace of the build so far. */
  requests: number;
  seconds: number;
  /** In US dollars, null when the model has no price. */
  cost: number | null;
  /** The spent and the rest together. */
  totalCost: number | null;
}

/** What is left of a running build, from its pace and price so far and the plan it started with. Null before it has
 * sent enough requests to tell, or when the status has no usage or estimate. */
export function projectBuild(status: BuildStatus, now: number): Projection | null {
  const { usage, estimate } = status;
  if (!usage || !estimate || usage.requests < PROJECT_AFTER_REQUESTS) return null;
  const find = (key: BuildStage['key']) => status.stages.find((stage) => stage.key === key);
  const started = find('distill_profile')?.started_at;
  if (!started) return null;
  const perRequest = Math.max(0, now - Date.parse(started)) / 1000 / usage.requests;

  const extraction = (['distill_profile', 'distill_event'] as const).reduce((sum, key) => {
    const stage = find(key);
    if (stage?.state === 'succeeded') return sum;
    if (stage?.state === 'running') return sum + Math.max(0, (stage.total ?? estimate.plan[key]) - (stage.done ?? 0));
    return sum + estimate.plan[key];
  }, 0);
  const sent = (...keys: BuildStage['key'][]) => keys.reduce((sum, key) => sum + (find(key)?.usage?.requests ?? 0), 0);
  const later =
    find('summaries')?.state === 'succeeded'
      ? 0
      : Math.max(0, estimate.plan.later - sent('organize', 'hierarchy', 'summaries'));
  const style =
    !find('style') || find('style')?.state === 'succeeded' ? 0 : Math.max(0, estimate.plan.style - sent('style'));

  const requests = extraction + later + style;
  const cost = usage.cost === null ? null : (usage.cost / usage.requests) * requests;
  return {
    requests,
    seconds: (extraction + style + later * LATER_SLOWDOWN) * perRequest,
    cost,
    totalCost: usage.cost === null || cost === null ? null : usage.cost + cost,
  };
}

/** Whether a build costs far more than it was expected to: the spent and the rest together past this many times the
 * top of the estimate. */
export const OVERSPEND_FACTOR = 2;

export function overspent(status: BuildStatus, projection: Projection | null): boolean {
  const high = status.estimate?.cost?.high;
  return projection?.totalCost != null && high != null && projection.totalCost > OVERSPEND_FACTOR * high;
}
