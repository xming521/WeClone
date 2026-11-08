// Shapes of weclone/server/model_settings.py, shared by the settings page and the build page.
export type Provider = 'codex_exec' | 'api';
export type StructuredOutput = 'json_schema' | 'json_object';

export interface Saved {
  version: string;
  provider: Provider;
  codex: { model: string; effort: string; batch_size: number; timeout: number };
  api: {
    base_url: string;
    key: string;
    model: string;
    /** Null for the default window. */
    context_window: number | null;
    reasoning_effort: string | null;
    extra_body: Record<string, unknown> | null;
    /** Null asks for json_schema. */
    structured_output: StructuredOutput | null;
    batch_size: number;
    timeout: number;
  };
}

export const PROVIDERS: Provider[] = ['codex_exec', 'api'];
// Suggestions only, any value is sent as it is: Codex's model_reasoning_effort values, and for the API the union of
// the reasoning_effort values providers documented as of 2026-11, each accepting its own subset.
export const CODEX_EFFORTS = ['minimal', 'low', 'medium', 'high', 'xhigh'];
export const API_EFFORTS = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];

export type Vendor = 'deepseek' | 'qwen' | 'kimi' | 'glm' | 'siliconflow' | 'openrouter' | 'openai';
/** Services the API form offers by name, so a first setup needs only the key. Each is the OpenAI-compatible base
 * URL its provider documented as of 2026-11; where a provider runs a mainland-China and an international service,
 * the China one is first, picked for the Chinese page, the other for the English one. Any other address is typed. */
export const VENDORS: Record<Vendor, string[]> = {
  deepseek: ['https://api.deepseek.com'],
  qwen: ['https://dashscope.aliyuncs.com/compatible-mode/v1', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'],
  kimi: ['https://api.moonshot.cn/v1', 'https://api.moonshot.ai/v1'],
  glm: ['https://open.bigmodel.cn/api/paas/v4', 'https://api.z.ai/api/paas/v4'],
  siliconflow: ['https://api.siliconflow.cn/v1', 'https://api.siliconflow.com/v1'],
  openrouter: ['https://openrouter.ai/api/v1'],
  openai: ['https://api.openai.com/v1'],
};
/** The format a service is set to when picked, for the services known not to take json_schema: DeepSeek answers it
 * with 400 "This response_format type is unavailable now" (as of 2026-11). The others are left to the connection
 * test. */
export const VENDOR_OUTPUTS: Partial<Record<Vendor, StructuredOutput>> = { deepseek: 'json_object' };

/** The named service an address belongs to, either of its regions, trailing slashes aside. */
export function vendorOf(baseUrl: string): Vendor | null {
  const url = baseUrl.trim().replace(/\/+$/, '');
  return (Object.keys(VENDORS) as Vendor[]).find((vendor) => VENDORS[vendor].includes(url)) ?? null;
}

/** The request body that saves the settings as they were read. */
export function savedBody(saved: Saved) {
  return {
    version: saved.version,
    provider: saved.provider,
    codex: { ...saved.codex },
    api: { ...saved.api },
  };
}
