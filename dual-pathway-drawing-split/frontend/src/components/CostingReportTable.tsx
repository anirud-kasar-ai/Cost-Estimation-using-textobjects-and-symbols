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

export function formatMoney(value: number | null | undefined, currency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(value);
}

interface EditableNumberCellProps {
  value: number;
  max: number;
  integer?: boolean;
  label: string;
  disabled?: boolean;
  onCommit: (value: number) => void;
}

function EditableNumberCell({
  value,
  max,
  integer = false,
  label,
  disabled = false,
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
      disabled={disabled}
      value={draft ?? String(value)}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') (event.target as HTMLInputElement).blur();
        if (event.key === 'Escape') setDraft(null);
      }}
      className="w-24 rounded-md border border-slate-200 px-2 py-1 text-right text-sm focus:border-sky-500 focus:outline-none disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400"
    />
  );
}

interface CostingReportTableProps {
  projectId: string;
  lines: DeviceLine[];
  currency: string;
  grandTotal: number;
  onUpdateLine: (lineId: string, payload: DeviceLineUpdate) => void;
  onReviewDone?: () => void;
  /** When true, only rows with all_verified (or legend rows with no instances) */
  verifiedOnly?: boolean;
  onVerifiedOnlyChange?: (value: boolean) => void;
}

export function CostingReportTable({
  projectId,
  lines,
  currency,
  grandTotal,
  onUpdateLine,
  onReviewDone,
  verifiedOnly = false,
  onVerifiedOnlyChange,
}: CostingReportTableProps) {
  const [reviewLine, setReviewLine] = useState<DeviceLine | null>(null);
  const columnHelper = createColumnHelper<DeviceLine>();

  const visibleLines = useMemo(() => {
    if (!verifiedOnly) return lines;
    return lines.filter((line) => {
      // Legend SKUs (no detections yet) stay visible so estimators can edit qty
      if (line.instance_count === 0) return true;
      return line.all_verified;
    });
  }, [lines, verifiedOnly]);

  const filteredTotal = useMemo(
    () => visibleLines.reduce((sum, line) => sum + (line.line_total || 0), 0),
    [visibleLines],
  );

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
          const canReview = Boolean(line.sample_detection_id) || line.instance_count > 0;
          return (
            <div>
              {canReview ? (
                <button
                  type="button"
                  onClick={() => setReviewLine(line)}
                  className="text-left font-medium text-sky-700 hover:underline"
                >
                  {info.getValue()}
                </button>
              ) : (
                <span className="font-medium text-slate-800">{info.getValue()}</span>
              )}
              <div className="mt-0.5 text-xs text-slate-400">
                {[line.mfg, line.part_number].filter(Boolean).join(' / ') || line.device_type}
                {line.locations ? ` · ${line.locations}` : ''}
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {line.no_price_set && (
                  <span className="inline-block rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
                    no price set
                  </span>
                )}
                {line.instance_count === 0 ? (
                  <span className="inline-block rounded-full bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-700">
                    legend SKU
                  </span>
                ) : line.all_verified ? (
                  <span className="inline-block rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
                    verified
                  </span>
                ) : (
                  <span className="inline-block rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
                    unverified
                  </span>
                )}
              </div>
            </div>
          );
        },
      }),
      columnHelper.accessor('unit', {
        header: 'Unit',
        cell: (info) => <span className="text-slate-500">{info.getValue() || 'each'}</span>,
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
          if (line.no_price_set || line.unit_cost === null) {
            return (
              <div className="flex flex-col items-end">
                <span className="block w-24 text-right text-sm text-slate-400" aria-label={`Unit cost for ${line.display_name}`}>
                  —
                </span>
              </div>
            );
          }
          return (
            <div className="flex flex-col items-end">
              <EditableNumberCell
                value={line.unit_cost}
                max={MAX_UNIT_COST}
                label={`Unit cost for ${line.display_name}`}
                disabled={line.no_price_set}
                onCommit={(unit_cost) => onUpdateLine(line.id, { unit_cost })}
              />
              {line.default_unit_cost !== null &&
                line.unit_cost !== line.default_unit_cost && (
                  <button
                    type="button"
                    onClick={() =>
                      onUpdateLine(line.id, { unit_cost: line.default_unit_cost })
                    }
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
        header: () => <span className="block text-right">Total</span>,
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
    data: visibleLines,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.id,
  });

  const displayTotal = verifiedOnly ? filteredTotal : grandTotal;

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-5 py-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Detailed Costing Report
        </h2>
        {onVerifiedOnlyChange && (
          <label className="flex items-center gap-2 text-xs text-slate-600">
            <input
              type="checkbox"
              checked={verifiedOnly}
              onChange={(e) => onVerifiedOnlyChange(e.target.checked)}
              className="rounded border-slate-300"
            />
            Verified only
          </label>
        )}
      </div>
      <p className="border-b border-slate-50 px-5 py-2 text-xs text-slate-400">
        {verifiedOnly
          ? 'Showing verified detector lines (legend SKUs without instances remain).'
          : 'Click a device with a sample crop to open sheet verification. Legend SKUs are edited by qty / pricing.'}
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
                {verifiedOnly ? 'No verified lines yet.' : 'No devices detected.'}
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
              {verifiedOnly ? 'Verified Subtotal' : 'Grand Total'}
            </td>
            <td className="px-5 py-4 text-right text-base font-bold text-slate-900">
              {formatMoney(displayTotal, currency)}
            </td>
          </tr>
        </tfoot>
      </table>
      {reviewLine && (
        <LineReviewModal
          projectId={projectId}
          line={reviewLine}
          onClose={() => setReviewLine(null)}
          onReviewDone={() => {
            setReviewLine(null);
            onReviewDone?.();
          }}
        />
      )}
    </section>
  );
}
