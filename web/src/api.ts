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
    if (response.status === 401 && path !== 'auth/login') window.dispatchEvent(new Event('weclone-unauthorized'));
    const data = await response.json().catch(() => null);
    throw new Error(typeof data?.detail === 'string' ? data.detail : `请求失败：HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body: unknown = {}) => request<T>('POST', path, body),
  put: <T>(path: string, body: unknown) => request<T>('PUT', path, body),
  patch: <T>(path: string, body: unknown) => request<T>('PATCH', path, body),
  delete: <T>(path: string) => request<T>('DELETE', path),
};
