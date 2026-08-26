/** Tests for the editable costing report grid. */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { CostingReportTable, formatMoney } from '../src/components/CostingReportTable';
import type { DeviceLine } from '../src/types';

vi.mock('../src/api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/api/client')>();
  return {
    ...actual,
    getLineReview: vi.fn(),
    verifyInstance: vi.fn(),
  };
});

import { getLineReview } from '../src/api/client';

const LINES: DeviceLine[] = [
  {
    id: 'line-1',
    device_type: 'supply_air_diffuser',
    display_name: 'Supply Air Diffuser',
    count: 6,
    unit_cost: 185,
    detected_count: 6,
    default_unit_cost: 185,
    needs_review: false,
    line_total: 1110,
    category: 'Diffusers & Grilles',
    unit: 'EA',
    mfg: 'Titus',
    part_number: 'TMS-AA',
    locations: 'ROOM 101',
    sample_detection_id: 'det-1',
    no_price_set: false,
    verified_count: 6,
    instance_count: 6,
    all_verified: true,
  },
  {
    id: 'line-2',
    device_type: 'smoke_damper',
    display_name: 'Smoke Damper',
    count: 3,
    unit_cost: 100,
    detected_count: 2,
    default_unit_cost: 100,
    needs_review: true,
    line_total: 300,
    category: 'Uncategorized',
    unit: 'EA',
    mfg: null,
    part_number: null,
    locations: null,
    sample_detection_id: null,
    no_price_set: false,
    verified_count: 0,
    instance_count: 2,
    all_verified: false,
  },
];

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function renderTable(onUpdateLine = vi.fn(), lines: DeviceLine[] = LINES) {
  render(
    <Wrapper>
      <CostingReportTable
        projectId="proj-1"
        lines={lines}
        currency="USD"
        grandTotal={1410}
        onUpdateLine={onUpdateLine}
      />
    </Wrapper>,
  );
  return onUpdateLine;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('CostingReportTable', () => {
  it('renders device rows and the grand total', () => {
    renderTable();

    expect(screen.getByText('Supply Air Diffuser')).toBeInTheDocument();
    expect(screen.getByText('Smoke Damper')).toBeInTheDocument();
    expect(screen.getByText(formatMoney(1410, 'USD'))).toBeInTheDocument();
  });

  it('shows verified and unverified badges', () => {
    renderTable();
    expect(screen.getByText('verified')).toBeInTheDocument();
    expect(screen.getByText('unverified')).toBeInTheDocument();
  });

  it('shows no price set badge and blanks unit cost', () => {
    const noPrice: DeviceLine = {
      ...LINES[0]!,
      id: 'line-np',
      no_price_set: true,
      unit_cost: null,
      default_unit_cost: null,
      line_total: 0,
      all_verified: false,
      verified_count: 0,
      instance_count: 1,
    };
    renderTable(vi.fn(), [noPrice]);
    expect(screen.getByText('no price set')).toBeInTheDocument();
    expect(screen.getByLabelText('Unit cost for Supply Air Diffuser')).toHaveTextContent('—');
  });

  it('commits an edited count on blur', async () => {
    const user = userEvent.setup();
    const onUpdateLine = renderTable();

    const input = screen.getByLabelText('Count for Supply Air Diffuser');
    await user.clear(input);
    await user.type(input, '9');
    await user.tab();

    expect(onUpdateLine).toHaveBeenCalledWith('line-1', { count: 9 });
  });

  it('commits an edited unit cost on Enter', async () => {
    const user = userEvent.setup();
    const onUpdateLine = renderTable();

    const input = screen.getByLabelText('Unit cost for Smoke Damper');
    await user.clear(input);
    await user.type(input, '250.5{Enter}');

    expect(onUpdateLine).toHaveBeenCalledWith('line-2', { unit_cost: 250.5 });
  });

  it('does not commit unchanged or invalid values', async () => {
    const user = userEvent.setup();
    const onUpdateLine = renderTable();

    const input = screen.getByLabelText('Count for Supply Air Diffuser');
    // Unchanged value
    await user.click(input);
    await user.tab();
    // Cleared (NaN) value
    await user.clear(input);
    await user.tab();

    expect(onUpdateLine).not.toHaveBeenCalled();
  });

  it('does not commit values above the sanity caps', async () => {
    const user = userEvent.setup();
    const onUpdateLine = renderTable();

    const count = screen.getByLabelText('Count for Supply Air Diffuser');
    await user.clear(count);
    await user.type(count, '999999999');
    await user.tab();

    const unitCost = screen.getByLabelText('Unit cost for Smoke Damper');
    await user.clear(unitCost);
    await user.type(unitCost, '99999999999{Enter}');

    expect(onUpdateLine).not.toHaveBeenCalled();
  });

  it('shows a reset link when a count is overridden and resets to detected', async () => {
    const user = userEvent.setup();
    const onUpdateLine = renderTable();

    const reset = screen.getByRole('button', { name: /detected 2 — reset/ });
    await user.click(reset);

    expect(onUpdateLine).toHaveBeenCalledWith('line-2', { count: 2 });
  });

  it('opens line review and loads crop via getLineReview', async () => {
    const user = userEvent.setup();
    vi.mocked(getLineReview).mockResolvedValue({
      detection_id: 'det-1',
      device_line_id: 'line-1',
      page_number: 1,
      device_type: 'supply_air_diffuser',
      confidence: 0.91,
      room_label: 'ROOM 101',
      quantity: 1,
      unit: 'each',
      bbox: [10, 20, 30, 40],
      crop_url: '/api/projects/proj-1/detections/det-1/crop',
      page_image_url: null,
      needs_review: false,
      verified: false,
    });

    renderTable();
    await user.click(screen.getByText('Supply Air Diffuser'));

    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    expect(getLineReview).toHaveBeenCalledWith('proj-1', 'line-1');
    expect(await screen.findByText('91%')).toBeInTheDocument();
  });

  it('renders an empty state when there are no lines', () => {
    render(
      <Wrapper>
        <CostingReportTable
          projectId="proj-1"
          lines={[]}
          currency="USD"
          grandTotal={0}
          onUpdateLine={vi.fn()}
        />
      </Wrapper>,
    );
    expect(screen.getByText('No devices detected.')).toBeInTheDocument();
  });

  it('formatMoney renders em dash for null', () => {
    expect(formatMoney(null, 'USD')).toBe('—');
  });
});
