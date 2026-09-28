import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  AlertTriangle,
  BookOpen,
  Check,
  Download,
  FileSpreadsheet,
  FolderOpen,
  Loader2,
  Pencil,
  Receipt,
  Save,
} from "lucide-react";
import { AppShell, APP_NAME, PageHeader } from "@/components/app-shell";
import { SortableTh, TablePagination, useTable } from "@/components/table-utils";
import { currency, useEstimator, type LineItem } from "@/lib/estimator-data";
import {
  formatDateTime,
  getOverlay,
  getSymbolTable,
  invoiceCsvUrl,
  invoicePdfUrl,
  jobFileUrl,
  legendPdfUrl,
  type LegendEntry,
  type OverlayWing,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { colorForClass } from "@/lib/class-colors";

export const Route = createFileRoute("/report")({
  head: () => ({
    meta: [
      { title: `Project Report — ${APP_NAME}` },
      {
        name: "description",
        content: "Detected drawing symbols, project metadata and a live device-costed estimate.",
      },
      { property: "og:title", content: `Project Report — ${APP_NAME}` },
      {
        property: "og:description",
        content: "Detection confidence and detailed costing report for a construction drawing.",
      },
    ],
  }),
  component: ReportPage,
});

/** Wing image with detection boxes drawn on top — visual verification of counts.
 *  Review mode turns it into click-to-correct: click a box to remove a false
 *  positive (−1 count), click the drawing to add a missed symbol (+1 count). */
function WingOverlay({ jobId, wing }: { jobId: string; wing: OverlayWing }) {
  const { lineItems, updateLineItem } = useEstimator();
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [showUnmatched, setShowUnmatched] = useState(false);
  const [reviewMode, setReviewMode] = useState(false);
  const [addTargetId, setAddTargetId] = useState("");
  /** Detection indexes removed during this review session (visual only). */
  const [removedIdx, setRemovedIdx] = useState<ReadonlySet<number>>(new Set());
  /** Markers for symbols added by clicking the drawing (visual only). */
  const [added, setAdded] = useState<{ x: number; y: number; color: string; label: string }[]>([]);

  const wingLabel = `${wing.wing ?? "wing"} page ${wing.page ?? "?"}`;

  const detections = useMemo(
    () =>
      wing.detections
        .map((d, idx) => ({ d, idx }))
        .filter(({ d, idx }) => (showUnmatched || d.legend_number != null) && !removedIdx.has(idx)),
    [wing, showUnmatched, removedIdx],
  );
  const unmatchedCount =
    wing.detections.length - wing.detections.filter((d) => d.legend_number != null).length;

  const findLineItem = (className: string) =>
    lineItems.find((li) => li.yoloClasses.includes(className));

  const removeDetection = (idx: number, className: string) => {
    const li = findLineItem(className);
    if (!li) {
      toast.error(`No priced line matches "${className}" — nothing to decrement`);
      return;
    }
    const next = Math.max(0, li.count - 1);
    updateLineItem(li.id, {
      count: next,
      note: li.note || `Overlay review: removed false positive on ${wingLabel}`,
    });
    setRemovedIdx((prev) => new Set(prev).add(idx));
    toast.success(`${li.deviceClass}: count ${li.count} → ${next} (auto-saved)`);
  };

  const addMissedSymbol = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!reviewMode || !addTargetId) return;
    const li = lineItems.find((l) => l.id === addTargetId);
    if (!li) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    const color = colorForClass(li.yoloClasses[0] ?? li.deviceClass);
    setAdded((prev) => [...prev, { x, y, color, label: li.symbol }]);
    updateLineItem(li.id, {
      count: li.count + 1,
      note: li.note || `Overlay review: added missed symbol on ${wingLabel}`,
    });
    toast.success(`${li.deviceClass}: count ${li.count} → ${li.count + 1} (auto-saved)`);
  };

  /** Per-class counts for the legend chips (colors come from colorForClass,
   * so every class keeps one stable, distinct color across all wings). */
  const classColor = useMemo(() => {
    const tally = new Map<string, number>();
    for (const d of wing.detections) tally.set(d.class_name, (tally.get(d.class_name) ?? 0) + 1);
    const ordered = [...tally.entries()].sort((a, b) => b[1] - a[1]);
    return { ordered };
  }, [wing]);

  if (!wing.image) {
    return (
      <p className="px-6 py-10 text-center text-sm text-muted-foreground">
        Wing image not found on disk for this job.
      </p>
    );
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <label className="flex cursor-pointer items-center gap-1.5 text-xs font-medium">
          <input
            type="checkbox"
            checked={reviewMode}
            onChange={(e) => setReviewMode(e.target.checked)}
            className="accent-orange"
          />
          Review mode
        </label>
        {reviewMode && (
          <>
            <select
              value={addTargetId}
              onChange={(e) => setAddTargetId(e.target.value)}
              className="max-w-[240px] truncate rounded-md border border-border bg-card px-2 py-1 text-xs outline-none focus:border-orange"
            >
              <option value="">— symbol to add on click —</option>
              {lineItems.map((li) => (
                <option key={li.id} value={li.id}>
                  {li.deviceClass}
                </option>
              ))}
            </select>
            <span className="text-xs text-muted-foreground">
              Click a box to remove a false positive (−1) · pick a symbol, then click the drawing to
              add a missed one (+1). Counts auto-save.
            </span>
          </>
        )}
      </div>
      <div className="relative overflow-auto rounded-md border border-border bg-white">
        <div
          className={cn("relative w-full", reviewMode && addTargetId && "cursor-crosshair")}
          onClick={addMissedSymbol}
        >
          <img
            src={jobFileUrl(jobId, wing.image)}
            alt={`${wing.wing ?? "wing"} page ${wing.page ?? ""}`}
            className="block w-full select-none"
            onLoad={(e) =>
              setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
            }
            draggable={false}
          />
          {size &&
            detections.map(({ d, idx }) => {
              const color = colorForClass(d.class_name);
              return (
                <div
                  key={idx}
                  title={
                    reviewMode
                      ? `Remove ${d.class_name} (−1 count)`
                      : `${d.class_name} — ${(d.confidence * 100).toFixed(0)}%${d.legend_number == null ? " (not matched to legend)" : ""}`
                  }
                  onClick={
                    reviewMode
                      ? (e) => {
                          e.stopPropagation();
                          removeDetection(idx, d.class_name);
                        }
                      : undefined
                  }
                  className={cn("absolute", reviewMode && "cursor-pointer hover:bg-danger/20")}
                  style={{
                    left: `${(d.x1 / size.w) * 100}%`,
                    top: `${(d.y1 / size.h) * 100}%`,
                    width: `${((d.x2 - d.x1) / size.w) * 100}%`,
                    height: `${((d.y2 - d.y1) / size.h) * 100}%`,
                    border: `2px ${d.legend_number == null ? "dashed" : "solid"} ${color}`,
                    background: "transparent",
                    boxShadow: "0 0 0 1px rgba(255,255,255,0.6)",
                  }}
                />
              );
            })}
          {added.map((m, i) => (
            <span
              key={i}
              title="Added during overlay review (+1 count)"
              className="absolute grid h-5 w-5 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full text-[9px] font-bold text-white shadow"
              style={{ left: `${m.x}%`, top: `${m.y}%`, background: m.color }}
            >
              {m.label.slice(0, 2)}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        {classColor.ordered.map(([name, count]) => (
          <span key={name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: colorForClass(name) }} />
            {name} × {count}
          </span>
        ))}
        {unmatchedCount > 0 && (
          <label className="ml-auto flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={showUnmatched}
              onChange={(e) => setShowUnmatched(e.target.checked)}
              className="accent-orange"
            />
            Show {unmatchedCount} unmatched (dashed, not priced)
          </label>
        )}
      </div>
    </div>
  );
}

function DetectionOverlaySection({ jobId }: { jobId: string }) {
  const overlayQuery = useQuery({
    queryKey: ["overlay", jobId],
    queryFn: () => getOverlay(jobId),
    staleTime: 5 * 60_000,
  });
  const wings = overlayQuery.data?.wings ?? [];
  const [active, setActive] = useState(0);
  const wing = wings[Math.min(active, Math.max(0, wings.length - 1))];

  return (
    <section className="mt-8">
      <h2 className="mb-3 text-sm font-semibold">
        Detection Overlay{" "}
        <span className="font-normal text-muted-foreground">
          — verify counts visually on each wing
        </span>
      </h2>
      <div className="card-surface p-5">
        {overlayQuery.isLoading ? (
          <div className="flex items-center justify-center gap-2 px-6 py-10 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading wing images…
          </div>
        ) : wings.length === 0 ? (
          <p className="px-6 py-10 text-center text-sm text-muted-foreground">
            No wing detections available for this drawing.
          </p>
        ) : (
          <>
            {wings.length > 1 && (
              <div className="mb-4 flex flex-wrap gap-2">
                {wings.map((w, i) => (
                  <button
                    key={`${w.folder}:${i}`}
                    onClick={() => setActive(i)}
                    className={cn(
                      "rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                      i === active
                        ? "border-orange bg-orange/10 text-orange"
                        : "border-border hover:bg-muted",
                    )}
                  >
                    {w.wing ?? "Wing"} · page {w.page ?? "?"} ({w.matched}/{w.yolo_count})
                  </button>
                ))}
              </div>
            )}
            {wing && <WingOverlay jobId={jobId} wing={wing} />}
          </>
        )}
      </div>
    </section>
  );
}

/** Sortable value per Symbol Legend column. */
function legendSortValue(entry: LegendEntry & { index: number }, key: string): unknown {
  switch (key) {
    case "index":
      return entry.index;
    case "symbol":
      return entry.symbol ?? "";
    case "description":
      return entry.description;
    case "mfg_model":
      return entry.mfg_model ?? "";
    case "part_number":
      return entry.part_number ?? "";
    default:
      return null;
  }
}

/** Sortable value per Detailed Costing Report column. */
function lineSortValue(li: LineItem, key: string): unknown {
  switch (key) {
    case "symbol":
      return li.symbol;
    case "deviceClass":
      return li.deviceClass;
    case "count":
      return li.count;
    case "unitCost":
      return li.unitCost;
    case "total":
      return li.count * li.unitCost;
    default:
      return null;
  }
}

function ReportPage() {
  const {
    jobs,
    selectedJob,
    selectJob,
    counts,
    countsLoading,
    metadata,
    setMetadata,
    lineItems,
    updateLineItem,
    correctionsSavedAt,
    saveCorrections,
    savingCorrections,
    grandTotal,
    projectTitle,
  } = useEstimator();
  const navigate = useNavigate();
  const [editCell, setEditCell] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const startEdit = (key: string, value: string | number) => {
    setEditCell(key);
    setDraft(String(value));
  };

  /** Only finished drawings can show a report — used by the project dropdown. */
  const doneJobs = useMemo(() => jobs.filter((j) => j.status === "done"), [jobs]);

  /** Only detections that matched a legend row — unmatched symbols are hidden everywhere. */
  const matchedDetections = useMemo(
    () => (counts?.yolo_detections ?? []).filter((d) => d.legend_number != null),
    [counts],
  );

  /** Per-class detection distribution with EXACT pipeline class names. */
  const classDistribution = useMemo(() => {
    const tally = new Map<string, number>();
    for (const det of matchedDetections) {
      tally.set(det.class_name, (tally.get(det.class_name) ?? 0) + 1);
    }
    return [...tally.entries()].sort((a, b) => b[1] - a[1]);
  }, [matchedDetections]);

  const confidenceStats = useMemo(() => {
    const dets = matchedDetections;
    if (dets.length === 0) return null;
    const avg = dets.reduce((s, d) => s + d.confidence, 0) / dets.length;
    const byClass = new Map<string, { count: number; sum: number; min: number }>();
    for (const d of dets) {
      const cur = byClass.get(d.class_name) ?? { count: 0, sum: 0, min: 1 };
      byClass.set(d.class_name, {
        count: cur.count + 1,
        sum: cur.sum + d.confidence,
        min: Math.min(cur.min, d.confidence),
      });
    }
    const perClass = [...byClass.entries()]
      .map(([name, s]) => ({
        name,
        count: s.count,
        avg: s.sum / s.count,
        min: s.min,
      }))
      .sort((a, b) => b.count - a.count);
    return { avg, perClass };
  }, [matchedDetections]);

  const skipped = counts?.method?.startsWith("skipped");

  /** Symbols that ended up unpriced: generic-fallback lines + legend-less detections. */
  const unpricedInfo = useMemo(() => {
    const fallbackLines = (counts?.invoice?.lines ?? []).filter((l) => l.device_key === "DEFAULT");
    const fallbackQty = fallbackLines.reduce((s, l) => s + l.quantity, 0);
    const unmatched = counts?.yolo_unmatched ?? [];
    const unmatchedQty = unmatched.reduce((s, u) => s + u.count, 0);
    if (fallbackQty === 0 && unmatchedQty === 0) return null;
    return {
      fallbackLines,
      fallbackQty,
      unmatchedQty,
      unmatchedNames: unmatched.map((u) => `${u.class_name} × ${u.count}`),
    };
  }, [counts]);

  const legendQuery = useQuery({
    queryKey: ["legend", selectedJob?.id],
    queryFn: () => getSymbolTable(selectedJob!.id),
    enabled: !!selectedJob && selectedJob.status === "done",
    staleTime: 5 * 60_000,
  });
  const legendTable = legendQuery.data ?? null;

  const legendRows = useMemo(
    () => (legendTable?.entries ?? []).map((e, index) => ({ ...e, index: index + 1 })),
    [legendTable],
  );
  const legendGrid = useTable(legendRows, legendSortValue, { pageSize: 8 });
  const costGrid = useTable(lineItems, lineSortValue, { pageSize: 10 });

  /** Project selector — lets the user open any processed PDF's report. */
  const projectSelect = doneJobs.length > 0 && (
    <label className="flex items-center gap-2 text-xs text-muted-foreground">
      <FolderOpen className="h-4 w-4 shrink-0" />
      <span className="hidden sm:inline">Project</span>
      <select
        value={selectedJob?.id ?? ""}
        onChange={(e) => selectJob(e.target.value)}
        className="max-w-[260px] truncate rounded-md border border-border bg-card px-2 py-1.5 text-xs font-medium text-foreground outline-none focus:border-orange focus:ring-1 focus:ring-orange"
      >
        {selectedJob && selectedJob.status !== "done" && (
          <option value={selectedJob.id}>{selectedJob.filename || selectedJob.id}</option>
        )}
        {doneJobs.map((j) => (
          <option key={j.id} value={j.id}>
            {j.filename || j.id}
          </option>
        ))}
      </select>
    </label>
  );

  if (!selectedJob) {
    return (
      <AppShell>
        <PageHeader
          title="Project Report"
          subtitle="No processed drawing selected yet"
          actions={projectSelect}
        />
        <div className="card-surface flex flex-col items-center gap-3 px-6 py-14 text-center">
          <p className="text-sm text-muted-foreground">
            Upload and process a drawing first, then open its report here.
          </p>
          <Link
            to="/"
            className="rounded-md bg-orange px-4 py-2 text-sm font-medium text-orange-foreground"
          >
            Go to Upload
          </Link>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <PageHeader
        title={projectTitle}
        subtitle={`${selectedJob.message}${counts?.yolo_model ? ` · model ${counts.yolo_model}` : ""}`}
        actions={
          <>
            {projectSelect}
            <span
              className={cn(
                "rounded-full border px-2.5 py-1 text-xs font-medium",
                selectedJob.status === "done"
                  ? "border-success/30 bg-success/15 text-success"
                  : "border-warning/30 bg-warning/15 text-warning",
              )}
            >
              {selectedJob.status === "done" ? "Complete" : selectedJob.status}
            </span>
          </>
        }
      />

      {countsLoading && (
        <div className="card-surface mb-6 flex items-center gap-2 p-4 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading detection results…
        </div>
      )}

      {skipped && (
        <div className="card-surface mb-6 flex items-start gap-3 border-warning/40 p-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
          <p className="text-sm text-warning">{counts?.note}</p>
        </div>
      )}

      <div className="card-surface p-5">
        <h2 className="mb-4 text-sm font-semibold">Project Metadata</h2>
        <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(metadata).map(([k, v]) => (
            <div key={k} className="min-w-0">
              <dt className="text-xs tracking-wide text-muted-foreground uppercase">{k}</dt>
              <dd className="group mt-1">
                {editCell === `m:${k}` ? (
                  <input
                    autoFocus
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={() => {
                      setMetadata(k, draft);
                      setEditCell(null);
                      toast.success("Values updated");
                    }}
                    onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                    className="w-full rounded border border-orange bg-card px-2 py-1 text-sm outline-none ring-1 ring-orange"
                  />
                ) : (
                  <button
                    onClick={() => startEdit(`m:${k}`, v)}
                    className="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left text-sm hover:bg-muted"
                  >
                    <span className="truncate">{v}</span>
                    <Pencil className="h-3 w-3 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100" />
                  </button>
                )}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <div className="card-surface p-5">
          <h2 className="mb-4 text-sm font-semibold">Detected Symbols (by model class)</h2>
          {classDistribution.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {skipped ? "Detection was skipped for this drawing set." : "No symbols detected."}
            </p>
          ) : (
            <ul className="space-y-2.5">
              {classDistribution.map(([name, count]) => (
                <li
                  key={name}
                  className="flex items-center gap-3 border-b border-border/60 pb-2 text-sm last:border-0 last:pb-0"
                >
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-sm"
                    style={{ background: colorForClass(name) }}
                  />
                  <span className="min-w-0 flex-1 truncate font-medium" title={name}>
                    {name}
                  </span>
                  <span className="shrink-0 text-right text-base font-semibold tabular-nums">
                    {count}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card-surface p-5">
          <h2 className="mb-4 text-sm font-semibold">Detection Confidence</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-md border border-border p-4">
              <p className="text-xs text-muted-foreground uppercase">Total symbols counted</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">{counts?.total_count ?? 0}</p>
            </div>
            <div className="rounded-md border border-border p-4">
              <p className="text-xs text-muted-foreground uppercase">Avg. confidence</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">
                {confidenceStats ? `${(confidenceStats.avg * 100).toFixed(1)}%` : "—"}
              </p>
            </div>
          </div>

          {confidenceStats && confidenceStats.perClass.length > 0 && (
            <div className="mt-4 max-h-[300px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-card text-left text-xs text-muted-foreground uppercase">
                  <tr className="border-b border-border">
                    <th className="py-2 pr-2 font-medium">Symbol</th>
                    <th className="py-2 pr-2 text-right font-medium">Count</th>
                    <th className="py-2 pr-2 text-right font-medium">Avg</th>
                    <th className="py-2 text-right font-medium">Min</th>
                  </tr>
                </thead>
                <tbody>
                  {confidenceStats.perClass.map((c) => (
                    <tr key={c.name} className="border-b border-border/60 last:border-0">
                      <td className="max-w-[200px] truncate py-2 pr-2" title={c.name}>
                        {c.name}
                      </td>
                      <td className="py-2 pr-2 text-right tabular-nums">{c.count}</td>
                      <td className="py-2 pr-2 text-right tabular-nums">
                        {(c.avg * 100).toFixed(0)}%
                      </td>
                      <td className="py-2 text-right tabular-nums">{(c.min * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {selectedJob.status === "done" && !skipped && (
        <DetectionOverlaySection jobId={selectedJob.id} />
      )}

      <section className="mt-8">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold">
            Symbol Legend (extracted from the PDF)
            {legendTable && (
              <span className="ml-2 font-normal text-muted-foreground">
                {legendTable.entry_count} row(s)
                {legendTable.entries[0]?.legend_title
                  ? ` · ${legendTable.entries[0].legend_title}`
                  : ""}
                {legendTable.legend_pages?.length
                  ? ` · page ${legendTable.legend_pages.join(", ")}`
                  : ""}
              </span>
            )}
          </h2>
          {selectedJob && (
            <a
              href={legendPdfUrl(selectedJob.id)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
            >
              <BookOpen className="h-3.5 w-3.5" /> Open legend PDF
            </a>
          )}
        </div>
        <div className="card-surface overflow-x-auto">
          {legendQuery.isLoading ? (
            <div className="flex items-center justify-center gap-2 px-6 py-10 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading symbol legend…
            </div>
          ) : legendRows.length === 0 ? (
            <p className="px-6 py-10 text-center text-sm text-muted-foreground">
              No symbol legend was extracted for this drawing.
            </p>
          ) : (
            <>
              <table className="w-full min-w-[760px] text-sm">
                <thead className="border-b border-border text-left text-xs text-muted-foreground uppercase">
                  <tr>
                    <SortableTh
                      label="#"
                      sortKey="index"
                      sort={legendGrid.sort}
                      onSort={legendGrid.toggleSort}
                    />
                    <SortableTh
                      label="Symbol"
                      sortKey="symbol"
                      sort={legendGrid.sort}
                      onSort={legendGrid.toggleSort}
                    />
                    <SortableTh
                      label="Description"
                      sortKey="description"
                      sort={legendGrid.sort}
                      onSort={legendGrid.toggleSort}
                    />
                    <SortableTh
                      label="Mfg / Model"
                      sortKey="mfg_model"
                      sort={legendGrid.sort}
                      onSort={legendGrid.toggleSort}
                    />
                    <SortableTh
                      label="Part Number"
                      sortKey="part_number"
                      sort={legendGrid.sort}
                      onSort={legendGrid.toggleSort}
                    />
                    <th className="px-4 py-3 font-medium">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {legendGrid.rows.map((entry) => (
                    <tr
                      key={`${entry.description}:${entry.index}`}
                      className="border-b border-border/60 last:border-0 hover:bg-accent/50"
                    >
                      <td className="px-4 py-3 text-muted-foreground tabular-nums">
                        {entry.index}
                      </td>
                      <td className="px-4 py-3">
                        <span className="grid h-8 w-8 place-items-center rounded bg-brand/10 text-xs font-bold text-brand">
                          {entry.symbol || "—"}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-medium">{entry.description}</td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {entry.mfg_model || <span className="text-xs italic">Not available</span>}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {entry.part_number || <span className="text-xs italic">Not available</span>}
                      </td>
                      <td className="max-w-[260px] px-4 py-3 text-xs text-muted-foreground">
                        <span className="line-clamp-2">
                          {(entry.notes || []).join(", ") || (
                            <span className="italic">Not available</span>
                          )}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <TablePagination
                page={legendGrid.page}
                pageCount={legendGrid.pageCount}
                total={legendGrid.total}
                pageSize={legendGrid.pageSize}
                onPage={legendGrid.setPage}
              />
            </>
          )}
        </div>
      </section>

      <section className="mt-8">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold">
            Detailed Costing Report{" "}
            <span className="font-normal text-muted-foreground">— edits auto-save</span>
          </h2>
          <div className="flex items-center gap-2">
            {correctionsSavedAt && (
              <span className="flex items-center gap-1 rounded-full border border-success/30 bg-success/10 px-2.5 py-1 text-xs font-medium text-success">
                <Check className="h-3 w-3" /> Verified {formatDateTime(correctionsSavedAt)}
              </span>
            )}
            {lineItems.length > 0 && (
              <button
                onClick={() => {
                  void saveCorrections().then(
                    () => toast.success("Corrections saved — model accuracy updated"),
                    (e: unknown) => toast.error(e instanceof Error ? e.message : "Save failed"),
                  );
                }}
                disabled={savingCorrections}
                className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted disabled:opacity-50"
              >
                {savingCorrections ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Save className="h-3.5 w-3.5" />
                )}
                Save corrections
              </button>
            )}
          </div>
        </div>

        {unpricedInfo && (
          <div className="mb-3 flex items-start gap-3 rounded-md border border-warning/30 bg-warning/10 p-4">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
            <div className="text-sm text-warning">
              {unpricedInfo.fallbackQty > 0 && (
                <p>
                  {unpricedInfo.fallbackQty} symbol(s) have no price in the catalog and were billed
                  at the generic default rate:{" "}
                  {unpricedInfo.fallbackLines.map((l) => l.description).join(", ")}. Add real prices
                  on the Pricing Catalog page.
                </p>
              )}
              {unpricedInfo.unmatchedQty > 0 && (
                <p className={cn(unpricedInfo.fallbackQty > 0 && "mt-1")}>
                  {unpricedInfo.unmatchedQty} detection(s) matched no legend row and are NOT priced:{" "}
                  {unpricedInfo.unmatchedNames.join(", ")}.
                </p>
              )}
            </div>
          </div>
        )}

        <div className="card-surface overflow-x-auto">
          {lineItems.length === 0 ? (
            <p className="px-6 py-10 text-center text-sm text-muted-foreground">
              No priced symbols for this drawing.
            </p>
          ) : (
            <>
              <table className="w-full min-w-[760px] text-sm">
                <thead className="border-b border-border text-left text-xs text-muted-foreground uppercase">
                  <tr>
                    <SortableTh
                      label="Symbol"
                      sortKey="symbol"
                      sort={costGrid.sort}
                      onSort={costGrid.toggleSort}
                    />
                    <SortableTh
                      label="Device Type (legend)"
                      sortKey="deviceClass"
                      sort={costGrid.sort}
                      onSort={costGrid.toggleSort}
                    />
                    <SortableTh
                      label="Count"
                      sortKey="count"
                      sort={costGrid.sort}
                      onSort={costGrid.toggleSort}
                    />
                    <SortableTh
                      label="Unit Cost"
                      sortKey="unitCost"
                      sort={costGrid.sort}
                      onSort={costGrid.toggleSort}
                    />
                    <SortableTh
                      label="Total Cost"
                      sortKey="total"
                      sort={costGrid.sort}
                      onSort={costGrid.toggleSort}
                      align="right"
                    />
                  </tr>
                </thead>
                <tbody>
                  {costGrid.rows.map((li) => (
                    <tr key={li.id} className="border-b border-border/60 hover:bg-accent/50">
                      <td className="px-4 py-3">
                        <span className="grid h-8 w-8 place-items-center rounded bg-brand/10 text-xs font-bold text-brand">
                          {li.symbol}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="font-medium">{li.deviceClass}</span>
                        {li.yoloClasses.length > 0 && (
                          <span className="mt-0.5 block text-xs text-muted-foreground">
                            model class: {li.yoloClasses.join(", ")}
                          </span>
                        )}
                        {editCell === `n:${li.id}` ? (
                          <input
                            autoFocus
                            value={draft}
                            onChange={(e) => setDraft(e.target.value)}
                            onBlur={() => {
                              updateLineItem(li.id, { note: draft.trim() });
                              setEditCell(null);
                            }}
                            onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                            placeholder="Reason — e.g. page 9 has a missed symbol (optional)"
                            className="mt-1 w-full max-w-[340px] rounded border border-orange bg-card px-2 py-1 text-xs outline-none ring-1 ring-orange"
                          />
                        ) : li.note ? (
                          <button
                            onClick={() => startEdit(`n:${li.id}`, li.note)}
                            className="mt-1 block max-w-[340px] truncate rounded px-1 py-0.5 text-left text-xs text-orange italic hover:bg-muted"
                            title={`Correction note: ${li.note} (click to edit)`}
                          >
                            note: {li.note}
                          </button>
                        ) : li.overridden ? (
                          <button
                            onClick={() => startEdit(`n:${li.id}`, "")}
                            className="mt-1 rounded px-1 py-0.5 text-xs text-muted-foreground italic hover:bg-muted hover:text-foreground"
                          >
                            + add note (why corrected — optional)
                          </button>
                        ) : null}
                      </td>
                      <td className="px-4 py-3">
                        {editCell === `c:${li.id}` ? (
                          <input
                            autoFocus
                            value={draft}
                            onChange={(e) => setDraft(e.target.value)}
                            onBlur={() => {
                              const n = parseInt(draft, 10);
                              if (!isNaN(n) && n >= 0) {
                                updateLineItem(li.id, { count: n });
                                // Changed vs the model's count → ask for a reason (optional).
                                if (n !== li.modelCount) {
                                  toast.info(
                                    `Count changed ${li.modelCount} → ${n} — add a reason (optional)`,
                                  );
                                  setEditCell(`n:${li.id}`);
                                  setDraft(li.note);
                                  return;
                                }
                                toast.success("Values updated");
                              }
                              setEditCell(null);
                            }}
                            onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                            className="w-20 rounded border border-orange bg-card px-2 py-1 text-sm outline-none ring-1 ring-orange"
                          />
                        ) : (
                          <span>
                            <button
                              onClick={() => startEdit(`c:${li.id}`, li.count)}
                              className="rounded px-1 py-0.5 tabular-nums hover:bg-muted"
                            >
                              {li.count}
                            </button>
                            {li.count !== li.modelCount && (
                              <span
                                className="block text-xs text-muted-foreground"
                                title="Count originally detected by the model"
                              >
                                model: {li.modelCount}
                              </span>
                            )}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {editCell === `u:${li.id}` ? (
                            <input
                              autoFocus
                              value={draft}
                              onChange={(e) => setDraft(e.target.value)}
                              onBlur={() => {
                                const n = parseFloat(draft);
                                if (!isNaN(n) && n >= 0) {
                                  updateLineItem(li.id, { unitCost: n });
                                  toast.success("Values updated");
                                }
                                setEditCell(null);
                              }}
                              onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                              className="w-28 rounded border border-orange bg-card px-2 py-1 text-sm outline-none ring-1 ring-orange"
                            />
                          ) : (
                            <button
                              onClick={() => startEdit(`u:${li.id}`, li.unitCost)}
                              className="rounded px-1 py-0.5 tabular-nums hover:bg-muted"
                            >
                              {currency(li.unitCost)}
                            </button>
                          )}
                          <span
                            className={cn(
                              "rounded-full border px-2 py-0.5 text-[11px]",
                              li.overridden
                                ? "border-orange/40 bg-orange/10 text-orange"
                                : "border-border bg-muted text-muted-foreground",
                            )}
                          >
                            {li.overridden ? "project override" : "from pricing table"}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right font-medium tabular-nums">
                        {currency(li.count * li.unitCost)}
                      </td>
                    </tr>
                  ))}
                  <tr className="border-t-2 border-foreground/40">
                    <td colSpan={4} className="px-4 py-4 text-base font-semibold">
                      Grand Total
                    </td>
                    <td className="px-4 py-4 text-right text-lg font-bold tabular-nums">
                      {currency(grandTotal)}
                    </td>
                  </tr>
                </tbody>
              </table>
              <TablePagination
                page={costGrid.page}
                pageCount={costGrid.pageCount}
                total={costGrid.total}
                pageSize={costGrid.pageSize}
                onPage={costGrid.setPage}
              />
            </>
          )}
        </div>
      </section>

      <div className="mt-6 flex flex-wrap items-center justify-end gap-3">
        <a
          href={invoiceCsvUrl(selectedJob.id)}
          className="inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted"
        >
          <FileSpreadsheet className="h-4 w-4" /> Export as CSV
        </a>
        <button
          onClick={() => navigate({ to: "/invoice" })}
          className="inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted"
        >
          <Receipt className="h-4 w-4" /> View Invoice
        </button>
        <a
          href={invoicePdfUrl(selectedJob.id)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-lg bg-orange px-4 py-2 text-sm font-semibold text-orange-foreground shadow-sm transition-all duration-150 hover:opacity-90"
        >
          <Download className="h-4 w-4" /> Download Invoice (PDF)
        </a>
      </div>
    </AppShell>
  );
}
