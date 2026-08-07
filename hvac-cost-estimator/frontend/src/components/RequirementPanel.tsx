/** Requirement provider summary + download of the extracted requirement PDF. */

import { useState } from 'react';

import { downloadRequirementPdf, errorMessage } from '../api/client';

interface RequirementPanelProps {
  projectId: string;
  filename: string;
  provider: string | null;
  hasRequirementPdf: boolean;
  pagesTruncated?: boolean;
}

export function RequirementPanel({
  projectId,
  filename,
  provider,
  hasRequirementPdf,
  pagesTruncated = false,
}: RequirementPanelProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!hasRequirementPdf) {
    return null;
  }

  const handleDownload = async () => {
    setBusy(true);
    setError(null);
    try {
      await downloadRequirementPdf(projectId, filename);
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
            Requirement Extract
          </h2>
          <p className="text-sm text-slate-700">
            <span className="font-medium text-slate-500">Provider: </span>
            {provider ?? <span className="italic text-slate-400">Not identified</span>}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Saved as “{filename.replace(/\.pdf$/i, '')} requirement.pdf”
          </p>
          {pagesTruncated && (
            <p className="mt-2 text-xs text-amber-700">
              Drawing has more pages than the CV render limit — requirement text was still
              extracted from the full PDF.
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={() => void handleDownload()}
          disabled={busy}
          className="rounded-lg bg-teal-700 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? 'Downloading…' : 'Download Requirement PDF'}
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
