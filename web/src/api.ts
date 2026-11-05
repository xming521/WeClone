import i18n from './i18n';

async function request<T>(
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  path: string,
  body?: unknown,
): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    cache: 'no-store',
    ...(method === 'GET'
      ? {}
      : {
          method,
          headers: { 'Content-Type': 'application/json', 'X-WeClone-Request': '1' },
          body: JSON.stringify(body),
        }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    // A wrong password on sign-in or password change is no expired session.
    if (response.status === 401 && data?.code !== 'password_wrong')
      window.dispatchEvent(new Event('weclone-unauthorized'));
    throw new ApiError(errorMessage(data, response.status), data);
  }
  return response.json() as Promise<T>;
}

/** A failed request, with the server's code and any fields it sent beside it. */
export class ApiError extends Error {
  code?: string;
  data: Record<string, unknown>;

  constructor(message: string, data: Record<string, unknown> | null) {
    super(message);
    this.data = data ?? {};
    this.code = typeof data?.code === 'string' ? data.code : undefined;
  }
}

export function errorCode(reason: unknown): string | undefined {
  return reason instanceof ApiError ? reason.code : undefined;
}

/** The server's error in the page's language: known errors by `code`, others by the server's own `detail`. */
function errorMessage(data: { code?: unknown; detail?: unknown } | null, status: number) {
  const key = `errors.${String(data?.code)}`;
  if (typeof data?.code === 'string' && i18n.exists(key)) return i18n.t(key);
  return typeof data?.detail === 'string' ? data.detail : i18n.t(($) => $.common.httpError, { status });
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body: unknown = {}) => request<T>('POST', path, body),
  put: <T>(path: string, body: unknown) => request<T>('PUT', path, body),
  patch: <T>(path: string, body: unknown) => request<T>('PATCH', path, body),
  delete: <T>(path: string) => request<T>('DELETE', path),
};
