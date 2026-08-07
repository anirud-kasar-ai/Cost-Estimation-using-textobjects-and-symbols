/** Tests for the React Query hooks (API layer mocked). */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { useProject, useUpdateDeviceLine } from '../src/hooks/useProjects';
import type { ProjectDetail } from '../src/types';

vi.mock('../src/api/client', () => ({
  getProject: vi.fn(),
  listProjects: vi.fn(),
  deleteProject: vi.fn(),
  updateDeviceLine: vi.fn(),
  uploadPdf: vi.fn(),
}));

import { getProject, updateDeviceLine } from '../src/api/client';

beforeEach(() => {
  vi.clearAllMocks();
});

const DETAIL = {
  id: 'p1',
  filename: 'layout.pdf',
  status: 'done',
  error_message: null,
  created_at: '2026-07-28T00:00:00Z',
  metadata: {
    title: null,
    client: 'ACME',
    architect: null,
    engineer: null,
    project_address: null,
    due_date: null,
  },
  device_lines: [],
  grand_total: 100,
  currency: 'USD',
  page_count: 1,
  has_requirement_pdf: false,
  requirement_provider: null,
  pages_truncated: false,
} satisfies ProjectDetail;

function createWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

describe('useProject', () => {
  it('fetches project detail when an id is provided', async () => {
    vi.mocked(getProject).mockResolvedValue(DETAIL);
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    const { result } = renderHook(() => useProject('p1'), {
      wrapper: createWrapper(queryClient),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.metadata.client).toBe('ACME');
  });

  it('stays idle without an id', () => {
    const queryClient = new QueryClient();
    const { result } = renderHook(() => useProject(null), {
      wrapper: createWrapper(queryClient),
    });

    expect(result.current.fetchStatus).toBe('idle');
    expect(getProject).not.toHaveBeenCalled();
  });
});

describe('useUpdateDeviceLine', () => {
  it('writes the refreshed project into the query cache', async () => {
    const updated = { ...DETAIL, grand_total: 999 };
    vi.mocked(updateDeviceLine).mockResolvedValue(updated);
    const queryClient = new QueryClient();
    queryClient.setQueryData(['project', 'p1'], DETAIL);

    const { result } = renderHook(() => useUpdateDeviceLine('p1'), {
      wrapper: createWrapper(queryClient),
    });
    await result.current.mutateAsync({ lineId: 'l1', payload: { count: 4 } });

    expect(updateDeviceLine).toHaveBeenCalledWith('p1', 'l1', { count: 4 });
    expect(
      (queryClient.getQueryData(['project', 'p1']) as ProjectDetail).grand_total,
    ).toBe(999);
  });
});
