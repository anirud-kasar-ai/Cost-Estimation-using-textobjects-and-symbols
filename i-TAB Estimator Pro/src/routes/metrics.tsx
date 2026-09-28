import { createFileRoute } from "@tanstack/react-router";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Loader2 } from "lucide-react";
import { AppShell, APP_NAME, PageHeader } from "@/components/app-shell";
import { SortableTh, TablePagination, useTable } from "@/components/table-utils";
import { formatDuration, getStats, type StatsClass } from "@/lib/api";
import { colorForClass } from "@/lib/class-colors";

export const Route = createFileRoute("/metrics")({
  head: () => ({
    meta: [
      { title: `Model Performance Dashboard — ${APP_NAME}` },
      {
        name: "description",
        content: `Detection totals, per-class distribution and available model weights for ${APP_NAME}.`,
      },
      {
        property: "og:title",
        content: `Model Performance Dashboard — ${APP_NAME}`,
      },
      {
        property: "og:description",
        content: "Per-class detections, confidence and model inventory for the detection pipeline.",
      },
    ],
  }),
  component: MetricsPage,
});

function Kpi({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="card-surface p-5">
      <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{label}</p>
      <p className="mt-2 text-2xl font-semibold tabular-nums">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

function Meter({ value }: { value: number }) {
  return (
    <div className="flex min-w-[110px] items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-brand" style={{ width: `${value * 100}%` }} />
      </div>
      <span className="w-11 shrink-0 text-right text-xs tabular-nums">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

/** Standard section heading: clear title + plain-language description of the analysis. */
function SectionHeading({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-4">
      <h2 className="text-sm font-semibold">{title}</h2>
      <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
    </div>
  );
}

/** Sortable value per class-performance column. */
function classSortValue(row: StatsClass, key: string): unknown {
  return row[key as keyof StatsClass] as unknown;
}

function MetricsPage() {
  const statsQuery = useQuery({ queryKey: ["stats"], queryFn: getStats, staleTime: 30_000 });
  const stats = statsQuery.data;

  // Only classes the CURRENT model files are trained on — classes left over
  // from older/replaced weights are hidden.
  const classes = useMemo(() => (stats?.classes ?? []).filter((c) => c.models.length > 0), [stats]);

  const classGrid = useTable(classes, classSortValue, {
    pageSize: 10,
    initialSort: { key: "detections", asc: false },
  });

  const topClasses = [...classes]
    .sort((a, b) => b.detections - a.detections)
    .slice(0, 12)
    .map((c) => ({ name: c.class_name, detections: c.detections }));

  const donutData = classes.slice(0, 10).map((c) => ({ name: c.class_name, value: c.detections }));

  const activeModel = stats?.models.find((m) => m.active) ?? stats?.models[0];

  return (
    <AppShell>
      <PageHeader
        title="Model Performance Dashboard"
        subtitle="Detection totals, class-wise accuracy and model inventory across all processed drawings"
      />

      {statsQuery.isLoading ? (
        <div className="card-surface flex items-center justify-center gap-2 px-6 py-14 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading statistics…
        </div>
      ) : !stats ? (
        <div className="card-surface px-6 py-14 text-center text-sm text-muted-foreground">
          Could not load stats — is the backend running?
        </div>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Kpi
              label="Total Symbols Detected"
              value={stats.total_symbols.toLocaleString()}
              hint={`All-time, across ${stats.jobs_done} processed drawing(s)`}
            />
            <Kpi
              label="Drawings Processed"
              value={String(stats.jobs_done)}
              hint="Drawing sets completed by the pipeline"
            />
            <Kpi
              label="Average Processing Time"
              value={formatDuration(stats.avg_processing_seconds)}
              hint="Per drawing, upload to finished invoice"
            />
            <Kpi
              label="Active Model"
              value={activeModel?.name?.replace(/\.pt$/, "") ?? "—"}
              hint={
                activeModel
                  ? `${activeModel.classes.length} classes · ${stats.models.length} model(s) available`
                  : "No model loaded"
              }
            />
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <div className="card-surface p-5">
              <SectionHeading
                title="Symbol Detections by Class"
                description="Detection count for the 12 most frequently found symbol classes across all processed drawings"
              />
              <div className="h-[340px] overflow-y-auto pr-1">
                <ol className="divide-y divide-border/60">
                  {topClasses.map((c, i) => (
                    <li
                      key={c.name}
                      className="flex items-center justify-between gap-3 py-2 text-sm"
                    >
                      <span className="flex min-w-0 items-center gap-2.5">
                        <span className="w-6 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
                          {i + 1}.
                        </span>
                        <span className="truncate">{c.name}</span>
                      </span>
                      <span className="shrink-0 font-semibold tabular-nums">
                        {c.detections.toLocaleString()}
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>

            <div className="card-surface p-5">
              <SectionHeading
                title="Detection Share by Symbol Class"
                description="How the total detections are split between the top 10 symbol classes"
              />
              <div className="h-[340px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={donutData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={70}
                      outerRadius={110}
                      paddingAngle={2}
                      stroke="var(--card)"
                    >
                      {donutData.map((d, i) => (
                        <Cell key={i} fill={colorForClass(d.name)} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: "var(--popover)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        color: "var(--popover-foreground)",
                        fontSize: 12,
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
                {donutData.map((d) => (
                  <span
                    key={d.name}
                    className="flex items-center gap-1.5 text-xs text-muted-foreground"
                  >
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ background: colorForClass(d.name) }}
                    />
                    {d.name}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <section className="mt-8">
            <SectionHeading
              title="Class-wise Model Performance"
              description={`Training accuracy (mAP50, Precision, Recall) for each of the ${classes.length} symbol classes the detection models are trained on`}
            />
            <div className="card-surface overflow-x-auto">
              {classes.length === 0 ? (
                <p className="px-6 py-10 text-center text-sm text-muted-foreground">
                  No models loaded — is the backend running?
                </p>
              ) : (
                <>
                  <table className="w-full min-w-[880px] text-sm">
                    <thead className="border-b border-border text-left text-xs text-muted-foreground uppercase">
                      <tr>
                        <SortableTh
                          label="Symbol Class"
                          sortKey="class_name"
                          sort={classGrid.sort}
                          onSort={classGrid.toggleSort}
                        />
                        <th className="px-4 py-3 font-medium">Trained In</th>
                        <SortableTh
                          label="Training mAP50"
                          sortKey="train_map50"
                          sort={classGrid.sort}
                          onSort={classGrid.toggleSort}
                        />
                        <SortableTh
                          label="Precision"
                          sortKey="train_precision"
                          sort={classGrid.sort}
                          onSort={classGrid.toggleSort}
                        />
                        <SortableTh
                          label="Recall"
                          sortKey="train_recall"
                          sort={classGrid.sort}
                          onSort={classGrid.toggleSort}
                        />
                      </tr>
                    </thead>
                    <tbody>
                      {classGrid.rows.map((c) => {
                        return (
                          <tr
                            key={c.class_name}
                            className="border-b border-border/60 last:border-0 hover:bg-accent/50"
                          >
                            <td className="px-4 py-3 font-medium">{c.class_name}</td>
                            <td className="px-4 py-3">
                              <div className="flex flex-wrap gap-1">
                                {c.models.map((m) => (
                                  <span
                                    key={m}
                                    className="rounded-full border border-border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground"
                                  >
                                    {m.replace(/\.pt$/, "")}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-4 py-3">
                              {c.train_map50 != null ? (
                                <Meter value={c.train_map50} />
                              ) : (
                                <span className="text-xs text-muted-foreground">—</span>
                              )}
                            </td>
                            <td className="px-4 py-3">
                              {c.train_precision != null ? (
                                <Meter value={c.train_precision} />
                              ) : (
                                <span className="text-xs text-muted-foreground">—</span>
                              )}
                            </td>
                            <td className="px-4 py-3">
                              {c.train_recall != null ? (
                                <Meter value={c.train_recall} />
                              ) : (
                                <span className="text-xs text-muted-foreground">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <TablePagination
                    page={classGrid.page}
                    pageCount={classGrid.pageCount}
                    total={classGrid.total}
                    pageSize={classGrid.pageSize}
                    onPage={classGrid.setPage}
                  />
                </>
              )}
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              What the metrics mean — <strong>mAP50</strong>: overall detection accuracy (higher is
              better) · <strong>Precision</strong>: of everything the model detected, how much was
              correct · <strong>Recall</strong>: of all symbols actually present, how many the model
              found. Values come from each model&apos;s training validation set and apply to all
              classes of that model.
            </p>
          </section>

          <section className="mt-8">
            <SectionHeading
              title="Overall Model Accuracy"
              description="Average training accuracy across all available model weight files"
            />
            {(() => {
              const withMetrics = stats.models.filter((m) => m.train_metrics);
              const avg = (pick: (m: (typeof withMetrics)[number]) => number) =>
                withMetrics.length
                  ? withMetrics.reduce((s, m) => s + pick(m), 0) / withMetrics.length
                  : null;
              const avgMap50 = avg((m) => m.train_metrics!.map50);
              const avgPrecision = avg((m) => m.train_metrics!.precision);
              const avgRecall = avg((m) => m.train_metrics!.recall);
              const totalClasses = stats.models.reduce((s, m) => s + m.classes.length, 0);
              const cells: { label: string; value: number | null }[] = [
                { label: "mAP50", value: avgMap50 },
                { label: "Precision", value: avgPrecision },
                { label: "Recall", value: avgRecall },
              ];
              return (
                <div className="card-surface p-5">
                  <div className="grid gap-6 sm:grid-cols-3">
                    {cells.map((c) => (
                      <div key={c.label}>
                        <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                          {c.label}
                        </p>
                        <p className="mt-1 text-2xl font-semibold tabular-nums">
                          {c.value != null ? `${(c.value * 100).toFixed(1)}%` : "—"}
                        </p>
                        {c.value != null && (
                          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full bg-brand"
                              style={{ width: `${c.value * 100}%` }}
                            />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  <p className="mt-4 text-xs text-muted-foreground">
                    Average across {withMetrics.length} model weight file(s) · {totalClasses}{" "}
                    trained classes · metrics from each training run&apos;s validation set. The
                    pipeline picks the model whose classes best match each PDF&apos;s symbol legend
                    automatically.
                  </p>
                </div>
              );
            })()}
          </section>
        </>
      )}
    </AppShell>
  );
}
