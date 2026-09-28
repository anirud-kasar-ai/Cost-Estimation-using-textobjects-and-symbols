import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Download, GitCompareArrows, Loader2, Minus } from "lucide-react";
import { AppShell, APP_NAME, PageHeader } from "@/components/app-shell";
import { currency, useEstimator } from "@/lib/estimator-data";
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  getCounts,
  type InvoiceLine,
  type Job,
} from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/compare")({
  head: () => ({
    meta: [
      { title: `Project Comparison — ${APP_NAME}` },
      {
        name: "description",
        content: "Side-by-side comparison of symbol counts and costs between two projects.",
      },
      { property: "og:title", content: `Project Comparison — ${APP_NAME}` },
      {
        property: "og:description",
        content: "Compare detected symbols, quantities and estimated costs of two drawings.",
      },
    ],
  }),
  component: ComparePage,
});

function ProjectSelect({
  label,
  jobs,
  value,
  exclude,
  onChange,
}: {
  label: string;
  jobs: Job[];
  value: string;
  exclude?: string;
  onChange: (id: string) => void;
}) {
  return (
    <label className="flex min-w-0 flex-1 flex-col gap-1.5">
      <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full truncate rounded-md border border-border bg-card px-3 py-2 text-sm font-medium outline-none focus:border-orange focus:ring-1 focus:ring-orange"
      >
        <option value="">— Select a project —</option>
        {jobs
          .filter((j) => j.id !== exclude)
          .map((j) => (
            <option key={j.id} value={j.id}>
              {j.filename || j.id}
            </option>
          ))}
      </select>
    </label>
  );
}

/** Signed difference chip with percentage: green when B is lower, red when higher. */
function DeltaChip({
  a,
  b,
  money,
  qtyDecimals,
}: {
  a: number;
  b: number;
  money?: boolean;
  qtyDecimals?: boolean;
}) {
  const diff = b - a;
  if (Math.abs(diff) < 1e-9) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        <Minus className="h-3 w-3" /> same
      </span>
    );
  }
  const up = diff > 0;
  const pct = a !== 0 ? (diff / a) * 100 : null;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 text-xs font-medium tabular-nums",
        up ? "text-danger" : "text-success",
      )}
    >
      {up ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
      {money
        ? currency(Math.abs(diff))
        : qtyDecimals
          ? Math.abs(diff).toFixed(2)
          : Math.abs(diff).toLocaleString()}
      {pct != null && (
        <span className="text-[10px] opacity-80">
          ({pct > 0 ? "+" : ""}
          {pct.toFixed(1)}%)
        </span>
      )}
    </span>
  );
}

/** RFC-4180-ish CSV cell escaping. */
function csvCell(value: string | number): string {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function ComparePage() {
  const { jobs, selectedJob } = useEstimator();
  const doneJobs = useMemo(() => jobs.filter((j) => j.status === "done"), [jobs]);

  // Project A defaults to the currently selected project (e.g. from the upload page).
  const [idA, setIdA] = useState<string>(() =>
    selectedJob?.status === "done" ? selectedJob.id : "",
  );
  const [idB, setIdB] = useState<string>("");

  const jobA = doneJobs.find((j) => j.id === idA) ?? null;
  const jobB = doneJobs.find((j) => j.id === idB) ?? null;

  const countsA = useQuery({
    queryKey: ["counts", idA, "done"],
    queryFn: () => getCounts(idA),
    enabled: !!jobA,
    staleTime: 60_000,
  });
  const countsB = useQuery({
    queryKey: ["counts", idB, "done"],
    queryFn: () => getCounts(idB),
    enabled: !!jobB,
    staleTime: 60_000,
  });

  const ready = jobA && jobB && countsA.data && countsB.data;
  const loading = (!!jobA && countsA.isLoading) || (!!jobB && countsB.isLoading);

  /** Per-page normalization: makes different-sized drawing sets comparable. */
  const [perPage, setPerPage] = useState(false);
  const pagesA = jobA?.page_count ?? 0;
  const pagesB = jobB?.page_count ?? 0;
  const canNormalize = pagesA > 0 && pagesB > 0;
  const fA = perPage && canNormalize ? 1 / pagesA : 1;
  const fB = perPage && canNormalize ? 1 / pagesB : 1;
  const fmtQty = (n: number) => (perPage ? n.toFixed(2) : n.toLocaleString());

  /** Union of device classes across both invoices with counts + costs per side.
   *  Rows merge on the canonical pricing device_key (not legend wording), so
   *  "RD — Roof Drain" and "ROOF DRAIN" land on one row. Generic-rate lines
   *  (device_key DEFAULT) fall back to a normalized description key. */
  const comparisonRows = useMemo(() => {
    if (!countsA.data || !countsB.data) return [];
    const normKey = (s: string) =>
      s
        .toUpperCase()
        .replace(/[^A-Z0-9]+/g, " ")
        .trim();
    const keyFor = (ln: InvoiceLine) =>
      ln.device_key && ln.device_key !== "DEFAULT" ? ln.device_key : normKey(ln.description);
    type Side = { qty: number; total: number };
    const map = new Map<string, { name: string; a: Side; b: Side }>();
    const add = (side: "a" | "b", lines: InvoiceLine[]) => {
      for (const ln of lines ?? []) {
        const key = keyFor(ln);
        const cur = map.get(key) ?? {
          name: ln.description,
          a: { qty: 0, total: 0 },
          b: { qty: 0, total: 0 },
        };
        cur[side] = {
          qty: cur[side].qty + ln.quantity,
          total: cur[side].total + ln.line_total_usd,
        };
        map.set(key, cur);
      }
    };
    // A first so its legend wording names the row; B fills gaps.
    add("a", countsA.data.invoice?.lines ?? []);
    add("b", countsB.data.invoice?.lines ?? []);
    return [...map.values()].sort((x, y) => y.a.total + y.b.total - (x.a.total + x.b.total));
  }, [countsA.data, countsB.data]);

  const totals = useMemo(() => {
    const a = countsA.data;
    const b = countsB.data;
    return {
      symbolsA: a?.total_count ?? 0,
      symbolsB: b?.total_count ?? 0,
      subtotalA: a?.invoice?.subtotal_usd ?? 0,
      subtotalB: b?.invoice?.subtotal_usd ?? 0,
      grandA: a?.invoice?.grand_total_usd ?? 0,
      grandB: b?.invoice?.grand_total_usd ?? 0,
    };
  }, [countsA.data, countsB.data]);

  /** Side-by-side facts about each drawing set. */
  const factRows: { label: string; a: string; b: string }[] =
    jobA && jobB
      ? [
          { label: "Source file", a: jobA.filename || jobA.id, b: jobB.filename || jobB.id },
          {
            label: "Uploaded",
            a: formatDateTime(jobA.created_at),
            b: formatDateTime(jobB.created_at),
          },
          {
            label: "File size",
            a: formatBytes(jobA.size_bytes),
            b: formatBytes(jobB.size_bytes),
          },
          {
            label: "Pages",
            a: String(jobA.page_count ?? "—"),
            b: String(jobB.page_count ?? "—"),
          },
          {
            label: "Processing time",
            a: formatDuration(jobA.actual_seconds),
            b: formatDuration(jobB.actual_seconds),
          },
          {
            label: "Detection model",
            a: countsA.data?.yolo_model ?? "—",
            b: countsB.data?.yolo_model ?? "—",
          },
        ]
      : [];

  /** Download the device comparison as a CSV (amounts in USD). */
  const exportCsv = () => {
    if (!jobA || !jobB) return;
    const pct = (a: number, b: number) => (a !== 0 ? (((b - a) / a) * 100).toFixed(1) : "");
    const lines: string[] = [
      `Project A,${csvCell(jobA.filename || jobA.id)}`,
      `Project B,${csvCell(jobB.filename || jobB.id)}`,
      "",
      "device_type,qty_a,qty_b,qty_delta,qty_delta_pct,cost_a_usd,cost_b_usd,cost_delta_usd,cost_delta_pct",
      ...comparisonRows.map((r) =>
        [
          csvCell(r.name),
          r.a.qty,
          r.b.qty,
          r.b.qty - r.a.qty,
          pct(r.a.qty, r.b.qty),
          r.a.total.toFixed(2),
          r.b.total.toFixed(2),
          (r.b.total - r.a.total).toFixed(2),
          pct(r.a.total, r.b.total),
        ].join(","),
      ),
      [
        "GRAND TOTAL",
        totals.symbolsA,
        totals.symbolsB,
        totals.symbolsB - totals.symbolsA,
        pct(totals.symbolsA, totals.symbolsB),
        totals.grandA.toFixed(2),
        totals.grandB.toFixed(2),
        (totals.grandB - totals.grandA).toFixed(2),
        pct(totals.grandA, totals.grandB),
      ].join(","),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `comparison_${jobA.id}_vs_${jobB.id}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <AppShell>
      <PageHeader
        title="Project Comparison"
        subtitle="Compare two drawing revisions to see the cost impact of design changes, or benchmark similar projects against each other"
        actions={
          ready ? (
            <button
              onClick={exportCsv}
              className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
            >
              <Download className="h-3.5 w-3.5" /> Export CSV
            </button>
          ) : undefined
        }
      />

      {doneJobs.length < 2 ? (
        <div className="card-surface flex flex-col items-center gap-3 px-6 py-14 text-center">
          <GitCompareArrows className="h-8 w-8 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            You need at least two processed drawings to run a comparison.
          </p>
          <Link
            to="/"
            className="rounded-md bg-orange px-4 py-2 text-sm font-medium text-orange-foreground"
          >
            Go to Upload
          </Link>
        </div>
      ) : (
        <>
          <div className="card-surface flex flex-col gap-4 p-5 sm:flex-row sm:items-end">
            <ProjectSelect
              label="Project A"
              jobs={doneJobs}
              value={idA}
              exclude={idB}
              onChange={setIdA}
            />
            <GitCompareArrows className="hidden h-5 w-5 shrink-0 text-muted-foreground sm:mb-2.5 sm:block" />
            <ProjectSelect
              label="Project B"
              jobs={doneJobs}
              value={idB}
              exclude={idA}
              onChange={setIdB}
            />
          </div>

          {loading && (
            <div className="card-surface mt-4 flex items-center justify-center gap-2 px-6 py-10 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading project results…
            </div>
          )}

          {!loading && (!jobA || !jobB) && (
            <p className="mt-6 text-center text-sm text-muted-foreground">
              Select two projects above to see the comparison.
            </p>
          )}

          {ready && (
            <>
              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                <div className="card-surface p-5">
                  <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                    Total Symbols
                  </p>
                  <div className="mt-2 flex items-baseline justify-between gap-2">
                    <p className="text-2xl font-semibold tabular-nums">{totals.symbolsA}</p>
                    <p className="text-2xl font-semibold tabular-nums">{totals.symbolsB}</p>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>Project A</span>
                    <DeltaChip a={totals.symbolsA} b={totals.symbolsB} />
                    <span>Project B</span>
                  </div>
                </div>
                <div className="card-surface p-5">
                  <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                    Subtotal
                  </p>
                  <div className="mt-2 flex items-baseline justify-between gap-2">
                    <p className="text-xl font-semibold tabular-nums">
                      {currency(totals.subtotalA)}
                    </p>
                    <p className="text-xl font-semibold tabular-nums">
                      {currency(totals.subtotalB)}
                    </p>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>Project A</span>
                    <DeltaChip a={totals.subtotalA} b={totals.subtotalB} money />
                    <span>Project B</span>
                  </div>
                </div>
                <div className="card-surface p-5">
                  <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                    Grand Total (incl. tax)
                  </p>
                  <div className="mt-2 flex items-baseline justify-between gap-2">
                    <p className="text-xl font-semibold tabular-nums">{currency(totals.grandA)}</p>
                    <p className="text-xl font-semibold tabular-nums">{currency(totals.grandB)}</p>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>Project A</span>
                    <DeltaChip a={totals.grandA} b={totals.grandB} money />
                    <span>Project B</span>
                  </div>
                </div>
              </div>

              <section className="mt-8">
                <h2 className="mb-3 text-sm font-semibold">Project Details</h2>
                <div className="card-surface overflow-x-auto">
                  <table className="w-full min-w-[640px] text-sm">
                    <thead className="border-b border-border text-left text-xs text-muted-foreground uppercase">
                      <tr>
                        <th className="px-4 py-3 font-medium"></th>
                        <th className="px-4 py-3 font-medium">Project A</th>
                        <th className="px-4 py-3 font-medium">Project B</th>
                      </tr>
                    </thead>
                    <tbody>
                      {factRows.map((row) => (
                        <tr key={row.label} className="border-b border-border/60 last:border-0">
                          <td className="px-4 py-3 text-xs tracking-wide text-muted-foreground uppercase">
                            {row.label}
                          </td>
                          <td className="px-4 py-3">{row.a}</td>
                          <td className="px-4 py-3">{row.b}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="mt-8">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <h2 className="text-sm font-semibold">
                    Device Cost Comparison{" "}
                    <span className="font-normal text-muted-foreground">
                      — matched on canonical device keys, so differing legend wording still lands on
                      one row
                    </span>
                  </h2>
                  <label
                    className={cn(
                      "flex items-center gap-1.5 text-xs font-medium",
                      canNormalize ? "cursor-pointer" : "cursor-not-allowed opacity-50",
                    )}
                    title={
                      canNormalize
                        ? `Divide by page count (A: ${pagesA} pages, B: ${pagesB} pages) to compare different-sized drawing sets`
                        : "Page counts unavailable for one of the projects"
                    }
                  >
                    <input
                      type="checkbox"
                      checked={perPage}
                      onChange={(e) => setPerPage(e.target.checked)}
                      disabled={!canNormalize}
                      className="accent-orange"
                    />
                    Normalize per page
                  </label>
                </div>
                <div className="card-surface overflow-x-auto">
                  {comparisonRows.length === 0 ? (
                    <p className="px-6 py-10 text-center text-sm text-muted-foreground">
                      Neither project has priced invoice lines.
                    </p>
                  ) : (
                    <table className="w-full min-w-[820px] text-sm">
                      <thead className="border-b border-border text-left text-xs text-muted-foreground uppercase">
                        <tr>
                          <th className="px-4 py-3 font-medium">Device Type</th>
                          <th className="px-4 py-3 text-right font-medium">
                            Qty A{perPage ? " /pg" : ""}
                          </th>
                          <th className="px-4 py-3 text-right font-medium">
                            Qty B{perPage ? " /pg" : ""}
                          </th>
                          <th className="px-4 py-3 text-right font-medium">Qty Δ</th>
                          <th className="px-4 py-3 text-right font-medium">
                            Cost A{perPage ? " /pg" : ""}
                          </th>
                          <th className="px-4 py-3 text-right font-medium">
                            Cost B{perPage ? " /pg" : ""}
                          </th>
                          <th className="px-4 py-3 text-right font-medium">Cost Δ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonRows.map((row) => (
                          <tr
                            key={row.name}
                            className="border-b border-border/60 last:border-0 hover:bg-accent/50"
                          >
                            <td className="px-4 py-3 font-medium">{row.name}</td>
                            <td className="px-4 py-3 text-right tabular-nums">
                              {fmtQty(row.a.qty * fA)}
                            </td>
                            <td className="px-4 py-3 text-right tabular-nums">
                              {fmtQty(row.b.qty * fB)}
                            </td>
                            <td className="px-4 py-3 text-right">
                              <DeltaChip
                                a={row.a.qty * fA}
                                b={row.b.qty * fB}
                                qtyDecimals={perPage}
                              />
                            </td>
                            <td className="px-4 py-3 text-right tabular-nums">
                              {currency(row.a.total * fA)}
                            </td>
                            <td className="px-4 py-3 text-right tabular-nums">
                              {currency(row.b.total * fB)}
                            </td>
                            <td className="px-4 py-3 text-right">
                              <DeltaChip a={row.a.total * fA} b={row.b.total * fB} money />
                            </td>
                          </tr>
                        ))}
                        <tr className="border-t-2 border-foreground/40 font-semibold">
                          <td className="px-4 py-4">Grand Total{perPage ? " (per page)" : ""}</td>
                          <td className="px-4 py-4 text-right tabular-nums">
                            {fmtQty(totals.symbolsA * fA)}
                          </td>
                          <td className="px-4 py-4 text-right tabular-nums">
                            {fmtQty(totals.symbolsB * fB)}
                          </td>
                          <td className="px-4 py-4 text-right">
                            <DeltaChip
                              a={totals.symbolsA * fA}
                              b={totals.symbolsB * fB}
                              qtyDecimals={perPage}
                            />
                          </td>
                          <td className="px-4 py-4 text-right tabular-nums">
                            {currency(totals.grandA * fA)}
                          </td>
                          <td className="px-4 py-4 text-right tabular-nums">
                            {currency(totals.grandB * fB)}
                          </td>
                          <td className="px-4 py-4 text-right">
                            <DeltaChip a={totals.grandA * fA} b={totals.grandB * fB} money />
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  )}
                </div>
              </section>
            </>
          )}
        </>
      )}
    </AppShell>
  );
}
