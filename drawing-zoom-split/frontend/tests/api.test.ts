/** Tests for the typed API client (axios methods mocked, no network). */

import { AxiosError, type AxiosResponse } from 'axios';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  api,
  errorMessage,
  updateDeviceLine,
  uploadPdf,
} from '../src/api/client';

function response<T>(data: T): AxiosResponse<T> {
  return { data } as AxiosResponse<T>;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('uploadPdf', () => {
  it('posts the file as multipart form data', async () => {
    const post = vi
      .spyOn(api, 'post')
      .mockResolvedValue(response({ project_id: 'abc', status: 'pending' }));

    const file = new File(['%PDF-1.7'], 'layout.pdf', { type: 'application/pdf' });
    const result = await uploadPdf(file);

    expect(result.project_id).toBe('abc');
    expect(post).toHaveBeenCalledTimes(1);
    const [url, body] = post.mock.calls[0]!;
    expect(url).toBe('/upload');
    expect(body).toBeInstanceOf(FormData);
    expect((body as FormData).get('file')).toBe(file);
  });
});

describe('updateDeviceLine', () => {
  it('patches the line and returns the refreshed project', async () => {
    const detail = { id: 'p1', grand_total: 42 };
    const patch = vi.spyOn(api, 'patch').mockResolvedValue(response(detail));

    const result = await updateDeviceLine('p1', 'l1', { count: 4 });

    expect(patch).toHaveBeenCalledWith('/projects/p1/lines/l1', { count: 4 });
    expect(result).toEqual(detail);
  });
});

describe('errorMessage', () => {
  it('prefers the API detail message', () => {
    const error = new AxiosError('Request failed');
    error.response = { data: { detail: 'Only PDF files are accepted.' } } as never;

    expect(errorMessage(error)).toBe('Only PDF files are accepted.');
  });

  it('falls back to the generic message', () => {
    expect(errorMessage(new Error('boom'))).toBe('boom');
    expect(errorMessage('weird')).toBe('Unexpected error');
  });
});
