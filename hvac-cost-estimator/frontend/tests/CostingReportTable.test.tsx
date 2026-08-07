/** Tests for the editable costing report grid. */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { CostingReportTable, formatMoney } from '../src/components/CostingReportTable';
import type { DeviceLine } from '../src/types';

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
  },
];

function renderTable(onUpdateLine = vi.fn()) {
  render(
    <CostingReportTable
      lines={LINES}
      currency="USD"
      grandTotal={1410}
      onUpdateLine={onUpdateLine}
    />,
  );
  return onUpdateLine;
}

describe('CostingReportTable', () => {
  it('renders device rows and the grand total', () => {
    renderTable();

    expect(screen.getByText('Supply Air Diffuser')).toBeInTheDocument();
    expect(screen.getByText('Smoke Damper')).toBeInTheDocument();
    expect(screen.getByText(formatMoney(1410, 'USD'))).toBeInTheDocument();
  });

  it('flags lines that need review', () => {
    renderTable();
    expect(screen.getByText('needs review')).toBeInTheDocument();
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

  it('renders an empty state when there are no lines', () => {
    render(
      <CostingReportTable lines={[]} currency="USD" grandTotal={0} onUpdateLine={vi.fn()} />,
    );
    expect(screen.getByText('No devices detected.')).toBeInTheDocument();
  });
});
