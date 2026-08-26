/** General / sheet / demo-tag notes extract + download of the generated PDF. */

import { useState } from 'react';

import { downloadSheetNotesPdf, errorMessage } from '../api/client';

interface SheetNotesPanelProps {
  jobId: string;
  filename: string;
  hasSheetNotesPdf: boolean;
  itemCount?: number;
}

export function SheetNotesPanel({
  jobId,
  filename,
  hasSheetNotesPdf,
  itemCount,
}: SheetNotesPanelProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!hasSheetNotesPdf) {
    return null;
  }

  const handleDownload = async () => {
    setBusy(true);
    setError(null);
    try {
      await downloadSheetNotesPdf(jobId, filename);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Sheet Notes Extract
          </h2>
          <p className="text-sm text-slate-700">
            General notes, sheet notes (boxed callout numbers), and demo tag notes
            scanned from every page.
            {itemCount != null && itemCount > 0 ? ` ${itemCount} items found.` : ''}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Saved in this job folder as “{filename.replace(/\.pdf$/i, '')} sheet notes.pdf”,
            with per-diagram notes under 05_metadata/
          </p>
        </div>
        <button
          type="button"
          onClick={() => void handleDownload()}
          disabled={busy}
          className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? 'Downloading…' : 'Download Sheet Notes PDF'}
        </button>
      </div>
      {error && (
        <p role="alert" className="mt-2 text-sm text-red-600">
          {error}
        </p>
      )}
    </section>
  );
}
