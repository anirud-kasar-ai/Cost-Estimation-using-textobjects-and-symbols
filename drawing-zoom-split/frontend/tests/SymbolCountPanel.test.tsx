/** SymbolCountPanel renders pivot rows from report data. */

import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, expect, it, vi } from 'vitest';

import { SymbolCountPanel } from '../src/components/SymbolCountPanel';
import type { JobDetail } from '../src/types';

vi.mock('../src/api/client', () => ({
  downloadSymbolCountCsv: vi.fn(),
  downloadSymbolCountPdf: vi.fn(),
  fetchSymbolCountReport: vi.fn(() =>
    Promise.resolve({
      status: 'done',
      columns: ['B-WING-EAST (page_020)'],
      rows: [
        {
          symbol: 'R',
          description: 'DATA RACK',
          mfg_model: 'ACME',
          part_number: 'WM5500',
          counts: { 'B-WING-EAST (page_020)': 3 },
          total: 3,
        },
      ],
    }),
  ),
  errorMessage: (error: unknown) => String(error),
}));

const baseJob: JobDetail = {
  id: 'demo-job',
  filename: 'demo.pdf',
  status: 'done',
  created_at: new Date().toISOString(),
  pages: [],
  symbol_count_status: 'done',
  symbol_count_instances: 1,
};

function renderPanel(job: JobDetail) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <SymbolCountPanel job={job} />
    </QueryClientProvider>,
  );
}

describe('SymbolCountPanel', () => {
  it('shows pivot table headers and counts', async () => {
    renderPanel(baseJob);
    expect(await screen.findByText('Symbol Count by Wing')).toBeInTheDocument();
    expect(await screen.findByText('DATA RACK')).toBeInTheDocument();
    expect(await screen.findByText('B-WING-EAST (page_020)')).toBeInTheDocument();
    expect(screen.getAllByText('3')).toHaveLength(2);
  });

  it('shows counting state while processing', () => {
    renderPanel({
      ...baseJob,
      status: 'processing',
      stage: 'counting_symbols',
      symbol_count_status: undefined,
    });
    expect(screen.getByText(/Counting symbols on wing zoom tiles/i)).toBeInTheDocument();
  });
});
