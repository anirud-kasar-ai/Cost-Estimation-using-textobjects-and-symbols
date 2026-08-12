/** Technical symbol legend extract + download of the generated PDF. */

import { useState } from 'react';

import { downloadTechnicalSymbolPdf, errorMessage } from '../api/client';

interface TechnicalSymbolPanelProps {
  projectId: string;
  filename: string;
  hasTechnicalSymbolPdf: boolean;
}

export function TechnicalSymbolPanel({
  projectId,
  filename,
  hasTechnicalSymbolPdf,
}: TechnicalSymbolPanelProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!hasTechnicalSymbolPdf) {
    return null;
  }

  const handleDownload = async () => {
    setBusy(true);
    setError(null);
    try {
      await downloadTechnicalSymbolPdf(projectId, filename);
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
            Technical Symbol Extract
          </h2>
          <p className="text-sm text-slate-700">
            TECHNOLOGY SYMBOL LEGEND / SYMBOL LEGEND tables scanned from every page.
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Saved as “{filename.replace(/\.pdf$/i, '')} technical symbol.pdf”
          </p>
        </div>
        <button
          type="button"
          onClick={() => void handleDownload()}
          disabled={busy}
          className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? 'Downloading…' : 'Download Technical Symbol PDF'}
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
