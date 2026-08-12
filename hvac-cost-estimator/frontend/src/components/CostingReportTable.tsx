/** Editable costing report grid (TanStack Table).
 *
 * Count and unit cost are editable; edits are committed on blur/Enter and
 * PATCHed to the backend, which recalculates totals server-side.
 * Click a device name to open the sheet-crop review view.
 */

import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { useMemo, useState } from 'react';

import type { DeviceLine, DeviceLineUpdate } from '../types';
import { LineReviewModal } from './LineReviewModal';

// Sanity caps mirroring the backend validation (backend/schemas/project.py).
export const MAX_LINE_COUNT = 100_000;
export const MAX_UNIT_COST = 10_000_000;

export function formatMoney(value: number, currency: string): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(value);
}

interface EditableNumberCellProps {
  value: number;
  max: number;
  integer?: boolean;
  label: string;
  onCommit: (value: number) => void;
}

function EditableNumberCell({
  value,
  max,
  integer = false,
  label,
  onCommit,
}: EditableNumberCellProps) {
  const [draft, setDraft] = useState<string | null>(null);

  const commit = () => {
    if (draft === null) return;
    const parsed = integer ? parseInt(draft, 10) : parseFloat(draft);
    setDraft(null);
    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= max && parsed !== value) {
      onCommit(parsed);
    }
  };

  return (
    <input
      type="number"
      aria-label={label}
      min={0}
      max={max}
      step={integer ? 1 : 0.01}
      value={draft ?? String(value)}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') (event.target as HTMLInputElement).blur();
        if (event.key === 'Escape') setDraft(null);
      }}
      className="w-24 rounded-md border border-slate-200 px-2 py-1 text-right text-sm focus:border-sky-500 focus:outline-none"
    />
  );
}

interface CostingReportTableProps {
  projectId: string;
  lines: DeviceLine[];
  currency: string;
  grandTotal: number;
  onUpdateLine: (lineId: string, payload: DeviceLineUpdate) => void;
}

export function CostingReportTable({
  projectId,
  lines,
  currency,
  grandTotal,
  onUpdateLine,
}: CostingReportTableProps) {
  const [reviewLine, setReviewLine] = useState<DeviceLine | null>(null);
  const columnHelper = createColumnHelper<DeviceLine>();

  const columns = useMemo(
    () => [
      columnHelper.accessor('category', {
        header: 'Category',
        cell: (info) => (
          <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
            {info.getValue() || 'Uncategorized'}
          </span>
        ),
      }),
      columnHelper.accessor('display_name', {
        header: 'Device',
        cell: (info) => {
          const line = info.row.original;
          return (
            <div>
              <button
                type="button"
                onClick={() => setReviewLine(line)}
                className="text-left font-medium text-sky-700 hover:underline"
              >
                {info.getValue()}
              </button>
              <div className="mt-0.5 text-xs text-slate-400">
                {[line.mfg, line.part_number].filter(Boolean).join(' / ') || line.device_type}
                {line.locations ? ` · ${line.locations}` : ''}
              </div>
              {line.needs_review && (
                <span className="mt-1 inline-block rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
                  needs review
                </span>
              )}
            </div>
          );
        },
      }),
      columnHelper.accessor('unit', {
        header: 'Unit',
        cell: (info) => <span className="text-slate-500">{info.getValue() || 'EA'}</span>,
      }),
      columnHelper.accessor('count', {
        header: () => <span className="block text-right">Qty</span>,
        cell: (info) => {
          const line = info.row.original;
          return (
            <div className="flex flex-col items-end">
              <EditableNumberCell
                value={line.count}
                max={MAX_LINE_COUNT}
                integer
                label={`Count for ${line.display_name}`}
                onCommit={(count) => onUpdateLine(line.id, { count })}
              />
              {line.count !== line.detected_count && (
                <button
                  type="button"
                  onClick={() => onUpdateLine(line.id, { count: line.detected_count })}
                  className="mt-1 text-xs text-sky-600 hover:underline"
                >
                  detected {line.detected_count} — reset
                </button>
              )}
            </div>
          );
        },
      }),
      columnHelper.accessor('unit_cost', {
        header: () => <span className="block text-right">Unit Cost</span>,
        cell: (info) => {
          const line = info.row.original;
          return (
            <div className="flex flex-col items-end">
              <EditableNumberCell
                value={line.unit_cost}
                max={MAX_UNIT_COST}
                label={`Unit cost for ${line.display_name}`}
                onCommit={(unit_cost) => onUpdateLine(line.id, { unit_cost })}
              />
              {line.unit_cost !== line.default_unit_cost && (
                <button
                  type="button"
                  onClick={() => onUpdateLine(line.id, { unit_cost: line.default_unit_cost })}
                  className="mt-1 text-xs text-sky-600 hover:underline"
                >
                  rate {formatMoney(line.default_unit_cost, currency)} — reset
                </button>
              )}
            </div>
          );
        },
      }),
      columnHelper.accessor('line_total', {
        header: () => <span className="block text-right">Total Cost</span>,
        cell: (info) => (
          <span className="block text-right font-semibold text-slate-800">
            {formatMoney(info.getValue(), currency)}
          </span>
        ),
      }),
    ],
    [columnHelper, currency, onUpdateLine],
  );

  const table = useReactTable({
    data: lines,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.id,
  });

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <h2 className="border-b border-slate-100 px-5 py-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Detailed Costing Report
      </h2>
      <p className="border-b border-slate-50 px-5 py-2 text-xs text-slate-400">
        Click a device name to open the sheet crop for human verification.
      </p>
      <table className="w-full text-sm">
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id} className="border-b border-slate-100 text-left">
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  className="px-5 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400"
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.length === 0 ? (
            <tr>
              <td colSpan={6} className="px-5 py-8 text-center italic text-slate-400">
                No devices detected.
              </td>
            </tr>
          ) : (
            table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="border-b border-slate-50 hover:bg-slate-50/60">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-5 py-3 align-top">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
        <tfoot>
          <tr className="bg-amber-50">
            <td className="px-5 py-4 font-semibold text-slate-700" colSpan={5}>
              Grand Total
            </td>
            <td className="px-5 py-4 text-right text-base font-bold text-slate-900">
              {formatMoney(grandTotal, currency)}
            </td>
          </tr>
        </tfoot>
      </table>
      {reviewLine && (
        <LineReviewModal
          projectId={projectId}
          line={reviewLine}
          onClose={() => setReviewLine(null)}
        />
      )}
    </section>
  );
}
