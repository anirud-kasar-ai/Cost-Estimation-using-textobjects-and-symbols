/** Pivot table of symbol counts per wing instance + CSV/PDF downloads. */

import { useMemo, useState } from 'react';

import {
  downloadSymbolCountCsv,
  downloadSymbolCountPdf,
  errorMessage,
} from '../api/client';
import { useSymbolCountReport } from '../hooks/useJobs';
import type { JobDetail } from '../types';

interface SymbolCountPanelProps {
  job: JobDetail;
}

const STAGE_LABELS: Record<string, string> = {
  counting_symbols: 'Counting symbols on wing images…',
};

export function SymbolCountPanel({ job }: SymbolCountPanelProps) {
  const [showZeroCounts, setShowZeroCounts] = useState(false);
  const [busy, setBusy] = useState<'csv' | 'pdf' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isCounting = job.status === 'processing' && job.stage === 'counting_symbols';
  const reportQuery = useSymbolCountReport(job.id, job.symbol_count_status);

  const visibleRows = useMemo(() => {
    const rows = reportQuery.data?.rows ?? [];
    if (showZeroCounts) return rows;
    return rows.filter((row) => row.total > 0);
  }, [reportQuery.data?.rows, showZeroCounts]);

  if (!isCounting && !job.symbol_count_status) {
    return null;
  }

  const handleDownload = async (kind: 'csv' | 'pdf') => {
    setBusy(kind);
    setError(null);
    try {
      if (kind === 'csv') {
        await downloadSymbolCountCsv(job.id, job.filename);
      } else {
        await downloadSymbolCountPdf(job.id, job.filename);
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Symbol Count by Wing
          </h2>
          {isCounting && (
            <p className="flex items-center gap-2 text-sm text-sky-700">
              <span className="h-2 w-2 animate-pulse rounded-full bg-sky-500" />
              {STAGE_LABELS.counting_symbols}
            </p>
          )}
          {job.symbol_count_status === 'done' && (
            <p className="text-sm text-slate-700">
              Detected symbols across {job.symbol_count_instances ?? 0} wing instance(s).
            </p>
          )}
          {job.symbol_count_status === 'skipped' && (
            <p className="text-sm text-amber-700">
              Symbol counting skipped
              {job.symbol_count_reason ? `: ${job.symbol_count_reason}` : '.'}
            </p>
          )}
          {job.symbol_count_status === 'failed' && (
            <p className="text-sm text-red-700">
              Symbol counting failed
              {job.symbol_count_error ? `: ${job.symbol_count_error}` : '.'}
            </p>
          )}
        </div>
        {job.symbol_count_status === 'done' && (
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => void handleDownload('csv')}
              disabled={busy !== null}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {busy === 'csv' ? 'Downloading…' : 'Download CSV'}
            </button>
            <button
              type="button"
              onClick={() => void handleDownload('pdf')}
              disabled={busy !== null}
              className="rounded-lg bg-slate-800 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-900 disabled:opacity-50"
            >
              {busy === 'pdf' ? 'Downloading…' : 'Download PDF'}
            </button>
          </div>
        )}
      </div>

      {error && (
        <p role="alert" className="mb-3 text-sm text-red-600">
          {error}
        </p>
      )}

      {job.symbol_count_status === 'done' && reportQuery.isLoading && (
        <p className="text-sm text-slate-400">Loading symbol counts…</p>
      )}

      {job.symbol_count_status === 'done' && reportQuery.isError && (
        <p role="alert" className="text-sm text-red-600">
          {errorMessage(reportQuery.error)}
        </p>
      )}

      {job.symbol_count_status === 'done' && reportQuery.data && (
        <>
          <div className="mb-3 flex items-center gap-2">
            <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-600">
              <input
                type="checkbox"
                checked={showZeroCounts}
                onChange={(event) => setShowZeroCounts(event.target.checked)}
                className="rounded border-slate-300"
              />
              Show zero counts
            </label>
            <span className="text-xs text-slate-400">
              {visibleRows.length} of {reportQuery.data.rows.length} symbols shown
            </span>
          </div>

          <div className="overflow-x-auto rounded-lg border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-emerald-50">
                <tr>
                  <th className="whitespace-nowrap px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-emerald-900">
                    Symbol
                  </th>
                  <th className="min-w-[160px] px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-emerald-900">
                    Description
                  </th>
                  <th className="min-w-[120px] px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-emerald-900">
                    MFG/Model
                  </th>
                  <th className="min-w-[90px] px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-emerald-900">
                    Part Number
                  </th>
                  {reportQuery.data.columns.map((column) => (
                    <th
                      key={column}
                      className="whitespace-nowrap px-3 py-2 text-center text-xs font-semibold uppercase tracking-wide text-emerald-900"
                    >
                      {column}
                    </th>
                  ))}
                  <th className="whitespace-nowrap px-3 py-2 text-center text-xs font-semibold uppercase tracking-wide text-emerald-900">
                    Total
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {visibleRows.length === 0 ? (
                  <tr>
                    <td
                      colSpan={5 + reportQuery.data.columns.length}
                      className="px-3 py-6 text-center text-slate-400"
                    >
                      No symbols detected on plan drawings.
                    </td>
                  </tr>
                ) : (
                  visibleRows.map((row) => (
                    <tr key={`${row.symbol}-${row.description}`} className="hover:bg-slate-50">
                      <td className="whitespace-nowrap px-3 py-2 font-medium text-slate-800">
                        {row.symbol || '—'}
                      </td>
                      <td className="px-3 py-2 text-slate-700">{row.description}</td>
                      <td className="px-3 py-2 text-slate-600">{row.mfg_model || '—'}</td>
                      <td className="px-3 py-2 text-slate-600">{row.part_number || '—'}</td>
                      {reportQuery.data.columns.map((column) => (
                        <td
                          key={column}
                          className="px-3 py-2 text-center tabular-nums text-slate-800"
                        >
                          {row.counts[column] ?? 0}
                        </td>
                      ))}
                      <td className="px-3 py-2 text-center font-semibold tabular-nums text-slate-900">
                        {row.total}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
