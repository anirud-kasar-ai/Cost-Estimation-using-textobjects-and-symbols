import { createFileRoute, Link } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { Check, Download, FileUp, Loader2, UploadCloud, FileText } from "lucide-react";
import { AppShell, APP_NAME, PageHeader } from "@/components/app-shell";
import { SortableTh, TablePagination, useTable } from "@/components/table-utils";
import { useEstimator } from "@/lib/estimator-data";
import { formatBytes, formatDateTime, formatDuration, invoicePdfUrl, type Job } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: `Upload Drawing — ${APP_NAME}` },
      {
        name: "description",
        content:
          "Upload a construction drawing PDF to extract the symbol legend, detect symbols and generate a cost estimate.",
      },
      {
        property: "og:title",
        content: `Upload Drawing — ${APP_NAME}`,
      },
      {
        property: "og:description",
        content: "Extract symbol counts from drawing PDFs and generate costed reports.",
      },
    ],
  }),
  component: UploadPage,
});

/** Ordered pipeline steps; each backend stage maps to one step. */
const STEPS: { label: string; stages: string[] }[] = [
  {
    label: "Extracting Legend & Requirements",
    stages: ["queued", "extracting", "rendering", "loading_model"],
  },
  { label: "Classifying Pages", stages: ["classifying"] },
  { label: "Cropping Diagrams & Wings", stages: ["drawing_crop", "wing_map", "wing_crop"] },
  { label: "Detecting Symbols (YOLO)", stages: ["counting"] },
  { label: "Pricing & Invoice", stages: ["pricing"] },
];

function stepIndex(stage: string): number {
  const i = STEPS.findIndex((s) => s.stages.includes(stage));
  if (i >= 0) return i;
  if (stage === "done") return STEPS.length;
  return 0;
}

const statusStyles: Record<string, string> = {
  processing: "bg-warning/15 text-warning border-warning/30",
  queued: "bg-warning/15 text-warning border-warning/30",
  done: "bg-brand/10 text-brand border-brand/30 dark:text-foreground",
  failed: "bg-danger/15 text-danger border-danger/30",
};

const statusLabels: Record<string, string> = {
  processing: "Processing",
  queued: "Queued",
  done: "Complete",
  failed: "Failed",
};

/** Time a finished job took, e.g. "12m 30s" — "—" while still running. */
function jobDurationLabel(job: Job): string {
  if (job.actual_seconds != null && job.actual_seconds > 0) {
    return formatDuration(job.actual_seconds);
  }
  if (job.status === "done" || job.status === "failed") {
    const start = Date.parse(job.created_at);
    const end = Date.parse(job.updated_at);
    if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
      return formatDuration((end - start) / 1000);
    }
  }
  return "—";
}

/** Sortable value per Recent Uploads column. */
function jobSortValue(job: Job, key: string): unknown {
  switch (key) {
    case "filename":
      return job.filename || job.id;
    case "created_at":
      return job.created_at;
    case "size_bytes":
      return job.size_bytes ?? null;
    case "duration": {
      if (job.actual_seconds != null) return job.actual_seconds;
      const start = Date.parse(job.created_at);
      const end = Date.parse(job.updated_at);
      return Number.isFinite(start) && Number.isFinite(end) && end > start
        ? (end - start) / 1000
        : null;
    }
    case "status":
      return job.status;
    default:
      return null;
  }
}

function UploadPage() {
  const { jobs, jobsLoading, uploadFile, uploading, selectJob, selectedJob } = useEstimator();
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [trackedId, setTrackedId] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  const tracked: Job | null = trackedId ? (jobs.find((j) => j.id === trackedId) ?? null) : null;
  const done = tracked?.status === "done";
  const failed = tracked?.status === "failed";
  const running = tracked && !done && !failed;
  const activeStep = tracked ? stepIndex(tracked.stage) : -1;
  const progressPct = Math.round(((tracked?.progress ?? 0) as number) * 100);

  const table = useTable(jobs, jobSortValue, {
    pageSize: 10,
    initialSort: { key: "created_at", asc: false },
  });

  const pickFile = (list: FileList | File[] | null | undefined) => {
    if (!list) return;
    const files = [...list];
    const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length !== files.length) toast.error("Only .pdf files are supported");
    if (pdfs.length === 0) return;
    if (pdfs.length > 1) toast.info("One PDF at a time — using the first file");
    setFile(pdfs[0] ?? null);
    setTrackedId(null);
  };

  const processDrawing = async () => {
    if (!file) return;
    try {
      const job = await uploadFile(file);
      setTrackedId(job.id);
      selectJob(job.id);
      toast.success(`Upload accepted — ${job.message}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    }
  };

  return (
    <AppShell>
      <PageHeader
        title="Process Construction Drawings"
        subtitle="Upload one construction drawing PDF — it becomes a project with its own report and invoice"
      />

      <div className="mx-auto max-w-3xl">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            pickFile(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "card-surface flex cursor-pointer flex-col items-center justify-center gap-3 border-2 border-dashed px-6 py-12 text-center transition-colors",
            drag ? "border-orange bg-orange/5" : "border-border hover:border-brand/50",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={(e) => {
              pickFile(e.target.files);
              e.target.value = "";
            }}
          />
          <UploadCloud className="h-8 w-8 text-brand" />
          <p className="text-sm font-medium">
            Drop a construction drawing PDF here or click to browse
          </p>
          <p className="text-xs text-muted-foreground">
            One drawing set · full report &amp; invoice
          </p>
        </div>

        {file && (
          <div className="card-surface mt-3 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4 p-4">
            <div className="flex min-w-0 items-center gap-3">
              <div className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-brand/10 text-brand">
                <FileText className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">{file.name}</p>
                <p className="text-xs text-muted-foreground">File size: {formatBytes(file.size)}</p>
              </div>
            </div>
            <button
              onClick={processDrawing}
              disabled={uploading || !!running}
              className="shrink-0 rounded-md bg-orange px-4 py-2 text-sm font-medium text-orange-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {uploading ? "Uploading…" : done || failed ? "Re-process" : "Process Drawing"}
            </button>
          </div>
        )}

        {tracked && (
          <div className="card-surface mt-3 p-5">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm font-medium">Processing — {tracked.id}</p>
              {running && tracked.eta_seconds != null && (
                <p className="text-xs text-muted-foreground tabular-nums">
                  ETA ~{formatDuration(tracked.eta_seconds)}
                </p>
              )}
            </div>
            <p className="mb-4 text-xs text-muted-foreground">{tracked.message}</p>
            {running && (
              <div className="mb-4 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-orange transition-all"
                  style={{ width: `${Math.max(2, progressPct)}%` }}
                />
              </div>
            )}
            <ol className="space-y-2.5">
              {STEPS.map((s, i) => (
                <li key={s.label} className="flex items-center gap-3 text-sm">
                  {done || i < activeStep ? (
                    <Check className="h-4 w-4 shrink-0 text-success" />
                  ) : i === activeStep && running ? (
                    <Loader2 className="h-4 w-4 shrink-0 animate-spin text-orange" />
                  ) : (
                    <span className="h-4 w-4 shrink-0 rounded-full border border-border" />
                  )}
                  <span
                    className={
                      done || i <= activeStep ? "text-foreground" : "text-muted-foreground"
                    }
                  >
                    {s.label}
                  </span>
                </li>
              ))}
            </ol>
            {failed && (
              <p className="mt-4 text-sm text-danger">
                Processing failed: {tracked.error || tracked.message}
              </p>
            )}
            {done && (
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <Link
                  to="/report"
                  onClick={() => selectJob(tracked.id)}
                  className="inline-flex rounded-md bg-brand px-4 py-2 text-sm font-medium text-brand-foreground"
                >
                  View Report
                </Link>
                <span className="text-xs text-muted-foreground">
                  Finished in {formatDuration(tracked.actual_seconds)}
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      <section className="mt-10">
        <h2 className="mb-3 text-sm font-semibold tracking-tight">Recent Uploads</h2>
        <div className="card-surface overflow-x-auto">
          {jobsLoading ? (
            <div className="flex items-center justify-center gap-2 px-6 py-12 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading jobs…
            </div>
          ) : jobs.length === 0 ? (
            <div className="flex flex-col items-center gap-2 px-6 py-12 text-center">
              <FileUp className="h-7 w-7 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                No uploads yet — process a drawing to see it here
              </p>
            </div>
          ) : (
            <>
              <table className="w-full min-w-[760px] text-sm">
                <thead className="border-b border-border text-left text-xs text-muted-foreground uppercase">
                  <tr>
                    <SortableTh
                      label="Drawing"
                      sortKey="filename"
                      sort={table.sort}
                      onSort={table.toggleSort}
                    />
                    <SortableTh
                      label="Uploaded"
                      sortKey="created_at"
                      sort={table.sort}
                      onSort={table.toggleSort}
                    />
                    <SortableTh
                      label="Size"
                      sortKey="size_bytes"
                      sort={table.sort}
                      onSort={table.toggleSort}
                    />
                    <SortableTh
                      label="Time Taken"
                      sortKey="duration"
                      sort={table.sort}
                      onSort={table.toggleSort}
                    />
                    <SortableTh
                      label="Status"
                      sortKey="status"
                      sort={table.sort}
                      onSort={table.toggleSort}
                    />
                    <th className="px-4 py-3 font-medium">Invoice</th>
                    <th className="px-4 py-3 font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {table.rows.map((u) => (
                    <tr
                      key={u.id}
                      className={cn(
                        "border-b border-border/60 last:border-0 hover:bg-accent/50",
                        selectedJob?.id === u.id && "bg-accent/40",
                      )}
                    >
                      <td className="px-4 py-3">
                        <div className="flex min-w-0 items-center gap-3">
                          <div className="grid h-9 w-9 shrink-0 place-items-center rounded bg-muted text-muted-foreground">
                            <FileText className="h-4 w-4" />
                          </div>
                          <div className="min-w-0">
                            <span className="block truncate font-medium">{u.filename || u.id}</span>
                            <span className="block truncate text-xs text-muted-foreground">
                              {u.message}
                            </span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-muted-foreground whitespace-nowrap">
                        {formatDateTime(u.created_at)}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground tabular-nums">
                        {formatBytes(u.size_bytes)}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground tabular-nums">
                        {jobDurationLabel(u)}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={cn(
                            "rounded-full border px-2.5 py-0.5 text-xs font-medium",
                            statusStyles[u.status] ?? statusStyles["processing"],
                          )}
                        >
                          {statusLabels[u.status] ?? u.status}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        {u.status === "done" ? (
                          <a
                            href={invoicePdfUrl(u.id)}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1.5 text-sm font-medium text-orange hover:underline"
                            title="Download invoice PDF"
                          >
                            <Download className="h-3.5 w-3.5" /> PDF
                          </a>
                        ) : (
                          <span className="text-sm text-muted-foreground">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        {u.status === "done" ? (
                          <Link
                            to="/report"
                            onClick={() => selectJob(u.id)}
                            className="text-sm font-medium text-orange hover:underline"
                          >
                            View Report
                          </Link>
                        ) : (
                          <span className="text-sm text-muted-foreground">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <TablePagination
                page={table.page}
                pageCount={table.pageCount}
                total={table.total}
                pageSize={table.pageSize}
                onPage={table.setPage}
              />
            </>
          )}
        </div>
      </section>
    </AppShell>
  );
}
