import { useCallback, useEffect, useId, useImperativeHandle, useRef, useState, type ReactNode, type Ref } from 'react';
import { CircleAlert, CircleCheck, Eye, EyeOff } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { api, ApiError, errorCode } from '../api';
import { ComboInput, SelectInput } from '../Dropdown';
import { ProblemText } from '../build/problems';
import type { Problem } from '../build/progress';
import { toast } from '../feedback/toast';
import {
  API_EFFORTS,
  CODEX_EFFORTS,
  PROVIDERS,
  savedBody,
  vendorOf,
  VENDOR_OUTPUTS,
  VENDORS,
  type Provider,
  type Saved,
  type StructuredOutput,
  type Vendor,
} from './modelSettings';

type TestResult =
  | { ok: boolean; problem: Problem | null }
  | {
      ok: boolean;
      seconds: number;
      models: string[] | null;
      /** The format the endpoint answered in. */
      structured_output: StructuredOutput | null;
      /** The model's window, when the endpoint's list states it. */
      context_window: number | null;
      error: string | null;
    };

// The empty choice saves no format.
const OUTPUTS = ['', 'json_schema', 'json_object'] as const;
// The request fields extra_body cannot set, as RESERVED_BODY on the server.
const RESERVED_BODY = ['model', 'messages', 'stream'];

/** The form as typed: every field a string. */
interface Draft {
  provider: Provider;
  codex: { model: string; effort: string; batch_size: string; timeout: string };
  api: {
    base_url: string;
    model: string;
    key: string;
    context_window: string;
    reasoning_effort: string;
    extra_body: string;
    structured_output: string;
    batch_size: string;
    timeout: string;
  };
}

function toDraft(saved: Saved): Draft {
  const { api, codex } = saved;
  return {
    provider: saved.provider,
    codex: { ...codex, batch_size: String(codex.batch_size), timeout: String(codex.timeout) },
    api: {
      base_url: api.base_url,
      model: api.model,
      key: api.key,
      context_window: api.context_window == null ? '' : String(api.context_window),
      reasoning_effort: api.reasoning_effort ?? '',
      extra_body: api.extra_body ? JSON.stringify(api.extra_body, null, 2) : '',
      structured_output: api.structured_output ?? '',
      batch_size: String(api.batch_size),
      timeout: String(api.timeout),
    },
  };
}

type FieldKey = `codex.${keyof Draft['codex']}` | `api.${keyof Draft['api']}`;
type FieldError = 'required' | 'number' | 'url' | 'json' | 'reserved';
// The fields only the settings page shows.
const TUNING: FieldKey[] = [
  'codex.batch_size',
  'codex.timeout',
  'api.context_window',
  'api.extra_body',
  'api.structured_output',
  'api.batch_size',
  'api.timeout',
];

function positive(text: string): number | null {
  return /^\d+$/.test(text.trim()) && Number(text) > 0 ? Number(text) : null;
}

/** The request body for the draft, and what is wrong with it field by field. Only the backend in use has to be
 * complete; the other one is sent as typed when that is valid, else as saved. */
function parse(draft: Draft, saved: Saved) {
  const errors: Partial<Record<FieldKey, FieldError>> = {};
  const codexActive = draft.provider === 'codex_exec';
  const check = (key: FieldKey, problem: FieldError | null) => {
    if (problem) errors[key] = problem;
  };
  const required = (value: string, active: boolean) => (active && !value.trim() ? 'required' : null);
  const number = (key: FieldKey, value: string, optional = false) => {
    if (optional && !value.trim()) return null;
    const parsed = positive(value);
    check(key, parsed === null ? 'number' : null);
    return parsed;
  };

  const { codex, api } = draft;
  check('codex.model', required(codex.model, codexActive));
  check('codex.effort', required(codex.effort, codexActive));
  const codexBody = {
    model: codex.model.trim(),
    effort: codex.effort.trim(),
    batch_size: number('codex.batch_size', codex.batch_size),
    timeout: number('codex.timeout', codex.timeout),
  };

  const baseUrl = api.base_url.trim();
  check('api.base_url', required(baseUrl, !codexActive) ?? (baseUrl && !/^https?:\/\//.test(baseUrl) ? 'url' : null));
  check('api.model', required(api.model, !codexActive));
  check('api.key', required(api.key, !codexActive));
  let extraBody: Record<string, unknown> | null = null;
  if (api.extra_body.trim()) {
    try {
      const value: unknown = JSON.parse(api.extra_body);
      if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error();
      extraBody = value as Record<string, unknown>;
      check('api.extra_body', RESERVED_BODY.some((key) => key in value) ? 'reserved' : null);
    } catch {
      check('api.extra_body', 'json');
    }
  }
  const apiBody = {
    base_url: baseUrl,
    model: api.model.trim(),
    key: api.key.trim(),
    context_window: number('api.context_window', api.context_window, true),
    reasoning_effort: api.reasoning_effort.trim() || null,
    extra_body: extraBody,
    structured_output: (api.structured_output || null) as StructuredOutput | null,
    batch_size: number('api.batch_size', api.batch_size),
    timeout: number('api.timeout', api.timeout),
  };

  const section = codexActive ? 'codex.' : 'api.';
  const blocking = (Object.keys(errors) as FieldKey[]).some((key) => key.startsWith(section));
  // A test can go without a model: it then lists the endpoint's models to choose from.
  const blockingTest = (Object.keys(errors) as FieldKey[]).some(
    (key) => key.startsWith(section) && key !== 'api.model',
  );
  const otherInvalid = (Object.keys(errors) as FieldKey[]).some((key) => !key.startsWith(section));
  const body = {
    provider: draft.provider,
    codex: codexActive || !otherInvalid ? codexBody : saved.codex,
    api: !codexActive || !otherInvalid ? apiBody : savedBody(saved).api,
  };
  return { body, errors, blocking, blockingTest };
}

/** The form on the build page: the fields a build cannot go without, the rest left to the settings page. */
interface BuildUse {
  /** The saved backend cannot run here, so the form starts on the API. */
  preferApi: boolean;
  /** What the check found, shown where the settings cannot be read, as in the demo. */
  facts: ReactNode;
  /** The backend the form would save when it has changes not saved yet, else null. A build saves them first. */
  onPending: (provider: Provider | null) => void;
  /** Set to the form's save: true once saved, false when a field is wrong or saving failed. */
  saver: Ref<() => Promise<boolean>>;
  onSaved: () => void;
}

/** The backend profile building uses and its request settings, kept in settings.jsonc. */
export function ModelSettings({ demo = false, build }: { demo?: boolean; build?: BuildUse }) {
  const { t, i18n } = useTranslation();
  const [saved, setSaved] = useState<Saved | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState<'save' | 'test' | null>(null);
  const [result, setResult] = useState<TestResult | null>(null);
  // The models the endpoint last listed, offered until the address changes.
  const [models, setModels] = useState<string[] | null>(null);
  // Another service picked while its address is still empty.
  const [custom, setCustom] = useState(false);
  const [shown, setShown] = useState(false);
  const pending = Boolean(saved && draft && JSON.stringify(draft) !== JSON.stringify(toDraft(saved)));
  const pendingProvider = pending && draft ? draft.provider : null;
  const onPending = build?.onPending;
  useEffect(() => {
    onPending?.(pendingProvider);
  }, [onPending, pendingProvider]);

  const load = useCallback(() => {
    api.get<Saved>('settings/model').then(
      (value) => {
        setSaved(value);
        setDraft(toDraft(value));
      },
      (reason: Error) => setError(reason.message),
    );
  }, []);

  useEffect(() => {
    if (!demo) load();
  }, [demo, load]);

  async function save(): Promise<boolean> {
    if (!saved || !draft) return false;
    const { body, blocking } = parse(draft, saved);
    setShown(true);
    if (blocking) return false;
    setBusy('save');
    setError('');
    try {
      const value = await api.put<Saved>('settings/model', { ...body, version: saved.version });
      setSaved(value);
      setDraft(toDraft(value));
      setShown(false);
      toast(t(($) => $.settings.model.saved));
      build?.onSaved();
      return true;
    } catch (reason) {
      setError((reason as Error).message);
      // Changed elsewhere: show the file as it is now.
      if (errorCode(reason) === 'settings_changed') load();
      return false;
    } finally {
      setBusy(null);
    }
  }
  useImperativeHandle(build?.saver, () => save);

  // Done once: later the owner's own choice stands, Codex included.
  const preferApi = build?.preferApi ?? false;
  const preferred = useRef(false);
  useEffect(() => {
    if (!preferApi || !saved || preferred.current) return;
    preferred.current = true;
    if (saved.provider === 'codex_exec') setDraft((current) => current && { ...current, provider: 'api' });
  }, [preferApi, saved]);

  if (demo) return <p className="summary-placeholder">{t(($) => $.settings.model.demo)}</p>;
  if (!saved || !draft) {
    if (build) return error ? build.facts : null;
    return error ? <ErrorLine text={error} /> : null;
  }

  const { body, errors, blockingTest } = parse(draft, saved);
  const changed = pending;
  // Only the settings page shows the request tuning.
  const full = !build;
  const codex = draft.provider === 'codex_exec';
  // Field errors show once the owner has tried to save or test.
  const fieldError = (key: FieldKey) => {
    const kind = shown ? errors[key] : undefined;
    return kind && t(($) => $.settings.model.invalid[kind]);
  };
  // The build page has no place for these errors, so it names them in a line of its own.
  const tuningWrong = !full && shown && TUNING.some((key) => key.startsWith(codex ? 'codex.' : 'api.') && errors[key]);

  function edit(update: (draft: Draft) => Draft) {
    setDraft((current) => current && update(current));
    setResult(null);
  }
  const codexField = (key: keyof Draft['codex']) => ({
    value: draft.codex[key],
    error: fieldError(`codex.${key}`),
    onChange: (value: string) => edit((current) => ({ ...current, codex: { ...current.codex, [key]: value } })),
  });
  const apiField = (key: keyof Draft['api']) => ({
    value: draft.api[key],
    error: fieldError(`api.${key}`),
    onChange: (value: string) => edit((current) => ({ ...current, api: { ...current.api, [key]: value } })),
  });
  // Another endpoint lists other models.
  const urlField = {
    ...apiField('base_url'),
    onChange: (value: string) => {
      apiField('base_url').onChange(value);
      setModels(null);
    },
  };
  const vendor = vendorOf(draft.api.base_url);
  const choice = vendor ?? (custom || draft.api.base_url.trim() ? 'custom' : '');
  function pickVendor(value: string) {
    setCustom(value === 'custom');
    if (value === 'custom') return urlField.onChange('');
    const urls = VENDORS[value as Vendor];
    const url = i18n.language.startsWith('zh') ? urls[0] : urls[urls.length - 1];
    // The format kept for the previous service does not carry over.
    const output = VENDOR_OUTPUTS[value as Vendor] ?? '';
    edit((current) => ({ ...current, api: { ...current.api, base_url: url, structured_output: output } }));
    setModels(null);
  }

  async function test() {
    if (blockingTest) {
      setShown(true);
      return;
    }
    setBusy('test');
    setError('');
    setResult(null);
    try {
      const value = await api.post<TestResult>('settings/model/test', body);
      setResult(value);
      if ('models' in value) {
        if (value.models) setModels(value.models);
        // What the endpoint told about the model, filled in to keep once saved.
        const { structured_output: output, context_window: contextWindow } = value;
        setDraft(
          (current) =>
            current && {
              ...current,
              api: {
                ...current.api,
                ...(output && { structured_output: output }),
                ...(contextWindow && { context_window: String(contextWindow) }),
              },
            },
        );
      }
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : String(reason));
    } finally {
      setBusy(null);
    }
  }

  const base = draft.api.base_url.trim().replace(/\/+$/, '');
  const requests = base && t(($) => $.settings.model.baseUrlHint, { url: `${base}/chat/completions` });
  // The settings page always shows the address; the build page only for a service not offered by name.
  const urlShown = full || choice === 'custom';
  return (
    <div className={full ? 'settings-model' : 'settings-model settings-model--build'}>
      {full && <p className="settings-section__hint">{t(($) => $.settings.model.hint)}</p>}
      <div className="settings-row">
        <div className="settings-row__text">
          <span className="settings-row__label">
            {full ? t(($) => $.settings.model.backend) : t(($) => $.build.prepare.backend)}
          </span>
          <span className="settings-row__hint">{t(($) => $.settings.model.backendHints[draft.provider])}</span>
          {!full && (
            <span className="settings-row__hint">
              <Trans
                i18nKey={($) => $.build.prepare.backendChoiceHint}
                components={{ a: <a className="wc-link" href="#/settings" /> }}
              />
            </span>
          )}
        </div>
        <div className="settings-row__control">
          <div className="wc-seg" role="radiogroup" aria-label={t(($) => $.settings.model.backend)}>
            {PROVIDERS.map((provider) => (
              <button
                key={provider}
                type="button"
                role="radio"
                aria-checked={draft.provider === provider}
                disabled={busy !== null}
                onClick={() => edit((current) => ({ ...current, provider }))}
              >
                {t(($) => $.settings.model.backends[provider])}
              </button>
            ))}
          </div>
        </div>
      </div>

      {codex ? (
        <div className="settings-model__fields">
          <Field label={t(($) => $.settings.model.model)} {...codexField('model')} />
          <Field
            label={t(($) => $.settings.model.effort)}
            hint={t(($) => $.settings.model.codexEffortHint)}
            options={CODEX_EFFORTS}
            {...codexField('effort')}
          />
          {full && (
            <>
              <Field label={t(($) => $.settings.model.batchSize)} numeric {...codexField('batch_size')} />
              <Field label={t(($) => $.settings.model.timeout)} numeric {...codexField('timeout')} />
            </>
          )}
        </div>
      ) : (
        <div className="settings-model__fields">
          <Pick
            label={t(($) => $.settings.model.vendor)}
            placeholder={t(($) => $.settings.model.vendorPick)}
            options={[...Object.keys(VENDORS), 'custom'].map((value) => ({
              value,
              label: t(($) => $.settings.model.vendors[value as Vendor | 'custom']),
            }))}
            hint={urlShown ? undefined : requests || undefined}
            error={urlShown ? undefined : fieldError('api.base_url')}
            disabled={busy !== null}
            value={choice}
            onChange={pickVendor}
          />
          {urlShown && (
            <Field
              label={t(($) => $.settings.model.baseUrl)}
              hint={requests || t(($) => $.settings.model.baseUrlEmpty)}
              placeholder="https://api.example.com/v1"
              {...urlField}
            />
          )}
          <Field label={t(($) => $.settings.model.key)} secret {...apiField('key')} />
          <Field
            label={t(($) => $.settings.model.model)}
            hint={models?.length ? t(($) => $.settings.model.models, { count: models.length }) : undefined}
            options={models ?? undefined}
            {...apiField('model')}
          />
          {full && (
            <Field
              label={t(($) => $.settings.model.context)}
              hint={t(($) => $.settings.model.contextHint)}
              numeric
              placeholder="128000"
              {...apiField('context_window')}
            />
          )}
          <Field
            label={t(($) => $.settings.model.effort)}
            hint={t(($) => $.settings.model.effortHint)}
            options={API_EFFORTS}
            {...apiField('reasoning_effort')}
          />
          {full && (
            <>
              <Field label={t(($) => $.settings.model.batchSize)} numeric {...apiField('batch_size')} />
              <Field label={t(($) => $.settings.model.timeout)} numeric {...apiField('timeout')} />
              <Choice
                label={t(($) => $.settings.model.output)}
                hint={t(($) => $.settings.model.outputHint)}
                options={OUTPUTS.map((value) => ({
                  value,
                  label: t(($) => $.settings.model.outputs[value || 'auto']),
                }))}
                disabled={busy !== null}
                {...apiField('structured_output')}
              />
              <Field
                wide
                area
                label={t(($) => $.settings.model.extraBody)}
                hint={t(($) => $.settings.model.extraBodyHint)}
                placeholder={'{"thinking": {"type": "disabled"}}'}
                {...apiField('extra_body')}
              />
            </>
          )}
        </div>
      )}

      <div className="settings-model__actions">
        <button
          className="wc-btn wc-btn--primary"
          type="button"
          disabled={!changed || busy !== null}
          onClick={() => void save()}
        >
          {t(($) => $.settings.model.save)}
        </button>
        <button className="wc-btn" type="button" disabled={busy !== null} onClick={() => void test()}>
          {busy === 'test' ? t(($) => $.settings.model.testing) : t(($) => $.settings.model.test)}
        </button>
        {result && <TestLine result={result} picking={!draft.api.model.trim()} />}
      </div>
      {tuningWrong && (
        <ErrorLine
          text={
            <Trans
              i18nKey={($) => $.build.prepare.tuningInvalid}
              components={{ a: <a className="wc-link" href="#/settings" /> }}
            />
          }
        />
      )}
      {error && <ErrorLine text={error} />}
    </div>
  );
}

function ErrorLine({ text }: { text: ReactNode }) {
  return (
    <p className="wc-field__hint" data-tone="danger" role="alert">
      {text}
    </p>
  );
}

/** The test's outcome. `picking`: the test ran without a model, so it only listed them. */
function TestLine({ result, picking }: { result: TestResult; picking: boolean }) {
  const { t } = useTranslation();
  let text: ReactNode;
  if ('problem' in result) {
    text = result.problem ? <ProblemText problem={result.problem} /> : t(($) => $.settings.model.testCodex);
  } else if (result.ok && picking) {
    text = result.models?.length
      ? t(($) => $.settings.model.testListed, { count: result.models.length })
      : t(($) => $.settings.model.testUnlisted);
  } else {
    text = result.ok
      ? t(($) => $.settings.model.testOk, {
          seconds: result.seconds,
          output: t(($) => $.settings.model.outputs[result.structured_output ?? 'auto']),
        })
      : t(($) => $.settings.model.testFailed, { error: result.error ?? '' });
  }
  const Icon = result.ok ? CircleCheck : CircleAlert;
  return (
    <p className="settings-model__result" data-tone={result.ok ? undefined : 'danger'} role="status">
      <Icon size={15} strokeWidth={1.8} aria-hidden="true" />
      <span>{text}</span>
    </p>
  );
}

/** A field whose control is a list, showing a placeholder until something is picked. */
function Pick({
  label,
  hint,
  error,
  value,
  onChange,
  placeholder,
  options,
  disabled,
}: {
  label: string;
  hint?: string;
  error?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  options: { value: string; label: string }[];
  disabled: boolean;
}) {
  const id = useId();
  return (
    <div className="wc-field">
      <label className="wc-field__label" htmlFor={id}>
        {label}
      </label>
      <SelectInput
        id={id}
        value={value}
        onChange={onChange}
        options={options}
        placeholder={placeholder}
        disabled={disabled}
        invalid={Boolean(error)}
      />
      {(error || hint) && (
        <span className="wc-field__hint" data-tone={error ? 'danger' : undefined}>
          {error || hint}
        </span>
      )}
    </div>
  );
}

/** A field across the form whose control is a segmented choice. */
function Choice({
  label,
  hint,
  value,
  onChange,
  options,
  disabled,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  disabled: boolean;
}) {
  const id = useId();
  return (
    <div className="wc-field settings-model__wide">
      <span className="wc-field__label" id={id}>
        {label}
      </span>
      <div className="wc-seg" role="radiogroup" aria-labelledby={id}>
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={value === option.value}
            disabled={disabled}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
      <span className="wc-field__hint">{hint}</span>
    </div>
  );
}

function Field({
  label,
  hint,
  error,
  value,
  onChange,
  placeholder,
  options,
  numeric = false,
  secret = false,
  area = false,
  wide = false,
}: {
  label: string;
  hint?: ReactNode;
  error?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  /** Offered while typing; any other value can still be typed. */
  options?: string[];
  numeric?: boolean;
  secret?: boolean;
  area?: boolean;
  wide?: boolean;
}) {
  const { t } = useTranslation();
  const id = useId();
  const [shown, setShown] = useState(false);
  const shared = {
    id,
    value,
    placeholder,
    spellCheck: false,
    'aria-invalid': error ? true : undefined,
  };
  return (
    <div className={wide ? 'wc-field settings-model__wide' : 'wc-field'}>
      <label className="wc-field__label" htmlFor={id}>
        {label}
      </label>
      {options?.length ? (
        <ComboInput {...shared} invalid={Boolean(error)} autoComplete="off" options={options} onChange={onChange} />
      ) : (
        <div className={area ? 'wc-input wc-input--area' : 'wc-input'} aria-invalid={error ? true : undefined}>
          {area ? (
            <textarea {...shared} rows={4} onChange={(event) => onChange(event.target.value)} />
          ) : (
            <input
              {...shared}
              // A key is text masked by CSS, not a password field, which browsers fill with saved passwords.
              type="text"
              data-masked={secret && !shown ? true : undefined}
              inputMode={numeric ? 'numeric' : undefined}
              autoComplete="off"
              onChange={(event) => onChange(event.target.value)}
            />
          )}
          {secret && (
            <button
              type="button"
              className="wc-iconbtn wc-iconbtn--sm"
              // Keeps the caret in the field, so typing goes on after a look.
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => setShown(!shown)}
              title={shown ? t(($) => $.settings.model.hideKey) : t(($) => $.settings.model.showKey)}
              aria-label={shown ? t(($) => $.settings.model.hideKey) : t(($) => $.settings.model.showKey)}
            >
              {shown ? <EyeOff size={16} strokeWidth={1.8} /> : <Eye size={16} strokeWidth={1.8} />}
            </button>
          )}
        </div>
      )}
      {(error || hint) && (
        <span className="wc-field__hint" data-tone={error ? 'danger' : undefined}>
          {error || hint}
        </span>
      )}
    </div>
  );
}
