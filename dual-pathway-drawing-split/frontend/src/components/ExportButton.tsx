/** Downloads the costing report as CSV. */

import { useState } from 'react';

import { downloadReportCsv, errorMessage } from '../api/client';

interface ExportButtonProps {
  projectId: string;
  filename: string;
  disabled?: boolean;
}

export function ExportButton({ projectId, filename, disabled = false }: ExportButtonProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    setBusy(true);
    setError(null);
    try {
      await downloadReportCsv(projectId, filename);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={() => void handleExport()}
        disabled={disabled || busy}
        className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {busy ? 'Exporting…' : 'Export CSV'}
      </button>
      {error && (
        <span role="alert" className="text-sm text-red-600">
          {error}
        </span>
      )}
    </div>
  );
}
