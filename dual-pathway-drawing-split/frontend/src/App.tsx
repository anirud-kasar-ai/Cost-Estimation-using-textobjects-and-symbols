import { useEffect, useState } from 'react';

import { errorMessage, getHealth } from './api/client';
import { JobList } from './components/JobList';
import { MetadataPanel } from './components/MetadataPanel';
import { RequirementPanel } from './components/RequirementPanel';
import { SheetNotesPanel } from './components/SheetNotesPanel';
import { SymbolCountPanel } from './components/SymbolCountPanel';
import { TechnicalSymbolPanel } from './components/TechnicalSymbolPanel';
import { UploadZone } from './components/UploadZone';
import { useJob, useJobs, useUploadPdf } from './hooks/useJobs';
import type { JobDetail } from './types';

const STAGE_LABELS: Record<string, string> = {
  extracting: 'Extracting requirements, symbols, and sheet notes…',
  rendering: 'Rendering PDF pages…',
  loading_model: 'Connecting to vision model…',
  classifying: 'Classifying plan pages…',
  drawing_crop: 'Cropping diagrams…',
  wing_map: 'Mapping wing regions…',
  wing_crop: 'Cutting wing images and zoom tiles…',
  counting_symbols: 'Counting symbols on wing images with vision model…',
};

function stageLabel(job: JobDetail): string {
  if (job.message?.trim()) {
    return job.message;
  }
  if (job.stage && STAGE_LABELS[job.stage]) {
    return STAGE_LABELS[job.stage];
  }
  return job.stage || job.status;
}

function JobProgressBanner({ job }: { job: JobDetail }) {
  const active = job.status === 'queued' || job.status === 'processing';
  if (!active) return null;

  return (
    <div
      role="status"
      className="rounded-xl border border-sky-200 bg-sky-50 p-6 text-sky-800"
    >
      <div className="flex items-center gap-3">
        <span className="h-3 w-3 animate-pulse rounded-full bg-sky-500" />
        <div>
          <p className="font-medium">Processing “{job.filename}”</p>
          <p className="mt-0.5 text-sm">{stageLabel(job)}</p>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [healthLine, setHealthLine] = useState('Checking model…');

  const jobs = useJobs();
  const job = useJob(selectedId);
  const upload = useUploadPdf(setSelectedId);

  useEffect(() => {
    void getHealth()
      .then((health) => {
        const keyOk = health.runtime === 'groq' ? health.groq_key_set : health.token_set;
        setHealthLine(`${health.runtime} · ${keyOk ? 'key set' : 'set API key in .env'} · ${health.model}`);
      })
      .catch(() => setHealthLine('API not reachable'));
  }, []);

  const detail = job.data;
  const isProcessing = detail?.status === 'queued' || detail?.status === 'processing';

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="border-b border-slate-200 bg-white px-8 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <h1 className="text-lg font-bold text-slate-800">
              <span className="text-sky-600">Dual-Pathway</span> · Drawing Split
            </h1>
            <p className="text-sm text-slate-400">
              Upload a construction PDF to extract requirements, split wings, and count symbols per wing.
            </p>
          </div>
          <p className="text-xs text-slate-400">{healthLine}</p>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-6 p-6 lg:grid-cols-[340px_1fr]">
        <aside className="space-y-6">
          <UploadZone
            onUpload={(file) => upload.mutate(file)}
            uploading={upload.isPending}
            error={upload.isError ? errorMessage(upload.error) : null}
          />
          <section>
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Recent Jobs
            </h2>
            <JobList
              jobs={jobs.data ?? []}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          </section>
        </aside>

        <section className="space-y-6">
          {!selectedId && (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white p-16 text-center text-slate-400">
              Upload a PDF or select a recent job to view extraction results and symbol counts.
            </div>
          )}

          {selectedId && job.isLoading && (
            <div className="rounded-xl border border-slate-200 bg-white p-16 text-center text-slate-400">
              Loading job…
            </div>
          )}

          {detail && isProcessing && <JobProgressBanner job={detail} />}

          {detail?.status === 'failed' && (
            <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-6 text-red-700">
              <p className="font-semibold">Processing failed</p>
              <p className="mt-1 text-sm">{detail.error ?? detail.message ?? 'Unknown error.'}</p>
            </div>
          )}

          {detail && (detail.status === 'done' || detail.status === 'processing') && (
            <>
              <RequirementPanel
                jobId={detail.id}
                filename={detail.filename}
                provider={detail.requirement_provider ?? null}
                hasRequirementPdf={Boolean(detail.has_requirement_pdf)}
              />
              <TechnicalSymbolPanel
                jobId={detail.id}
                filename={detail.filename}
                hasTechnicalSymbolPdf={Boolean(detail.has_technical_symbol_pdf)}
                entryCount={detail.symbol_entry_count}
              />
              <SymbolCountPanel job={detail} />
              <SheetNotesPanel
                jobId={detail.id}
                filename={detail.filename}
                hasSheetNotesPdf={Boolean(detail.has_sheet_notes_pdf)}
                itemCount={detail.sheet_notes_item_count}
              />
              {detail.metadata && <MetadataPanel metadata={detail.metadata} />}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
