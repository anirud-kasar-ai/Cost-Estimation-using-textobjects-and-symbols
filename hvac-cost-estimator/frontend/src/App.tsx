/** i-TAB dashboard: upload a layout PDF, review extracted metadata and the
 * editable costing report, export CSV. */

import { useState } from 'react';

import { errorMessage } from './api/client';
import { CostingReportTable } from './components/CostingReportTable';
import { ExportButton } from './components/ExportButton';
import { MetadataPanel } from './components/MetadataPanel';
import { ProjectList } from './components/ProjectList';
import { RequirementPanel } from './components/RequirementPanel';
import { UploadZone } from './components/UploadZone';
import {
  useDeleteProject,
  useProject,
  useProjects,
  useUpdateDeviceLine,
  useUploadPdf,
} from './hooks/useProjects';

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const projects = useProjects();
  const project = useProject(selectedId);
  const upload = useUploadPdf(setSelectedId);
  const deleteProject = useDeleteProject((deletedId) => {
    setSelectedId((current) => (current === deletedId ? null : current));
  });

  const detail = project.data;
  const isProcessing = detail?.status === 'pending' || detail?.status === 'processing';

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="border-b border-slate-200 bg-white px-8 py-4">
        <h1 className="text-lg font-bold text-slate-800">
          <span className="text-sky-600">i-TAB</span> · Mechanical Device Cost Estimator
        </h1>
        <p className="text-sm text-slate-400">
          Upload an HVAC layout PDF to extract metadata, count devices, and cost the project.
        </p>
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
              Recent Uploads
            </h2>
            <ProjectList
              projects={projects.data ?? []}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onDelete={(projectId) => deleteProject.mutate(projectId)}
            />
          </section>
        </aside>

        <section className="space-y-6">
          {!selectedId && (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white p-16 text-center text-slate-400">
              Upload a layout PDF or select a recent upload to view its costing report.
            </div>
          )}

          {selectedId && project.isLoading && (
            <div className="rounded-xl border border-slate-200 bg-white p-16 text-center text-slate-400">
              Loading project…
            </div>
          )}

          {detail && isProcessing && (
            <div
              role="status"
              className="flex items-center gap-3 rounded-xl border border-sky-200 bg-sky-50 p-6 text-sky-800"
            >
              <span className="h-3 w-3 animate-pulse rounded-full bg-sky-500" />
              Processing “{detail.filename}” — extracting metadata and detecting devices…
            </div>
          )}

          {detail?.status === 'failed' && (
            <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-6 text-red-700">
              <p className="font-semibold">Processing failed</p>
              <p className="mt-1 text-sm">{detail.error_message ?? 'Unknown error.'}</p>
            </div>
          )}

          {detail?.status === 'done' && (
            <ProjectReport detailId={detail.id} />
          )}
        </section>
      </main>
    </div>
  );
}

/** Rendered once a project reaches "done"; owns the line-override mutation. */
function ProjectReport({ detailId }: { detailId: string }) {
  const { data: detail } = useProject(detailId);
  const updateLine = useUpdateDeviceLine(detailId);

  if (!detail) return null;

  return (
    <>
      <RequirementPanel
        projectId={detail.id}
        filename={detail.filename}
        provider={detail.requirement_provider}
        hasRequirementPdf={detail.has_requirement_pdf}
        pagesTruncated={detail.pages_truncated}
      />
      <MetadataPanel metadata={detail.metadata} />
      <CostingReportTable
        lines={detail.device_lines}
        currency={detail.currency}
        grandTotal={detail.grand_total}
        onUpdateLine={(lineId, payload) => updateLine.mutate({ lineId, payload })}
      />
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          {detail.page_count} page{detail.page_count === 1 ? '' : 's'} processed · counts and
          unit costs are editable
        </p>
        <ExportButton projectId={detail.id} filename={detail.filename} />
      </div>
      {updateLine.isError && (
        <p role="alert" className="text-sm text-red-600">
          {errorMessage(updateLine.error)}
        </p>
      )}
    </>
  );
}
