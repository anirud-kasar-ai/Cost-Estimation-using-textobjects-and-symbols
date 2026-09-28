import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Loader2, Lock, Search } from "lucide-react";
import { AppShell, APP_NAME, PageHeader } from "@/components/app-shell";
import { SortableTh, TablePagination, useTable } from "@/components/table-utils";
import { currency, useEstimator } from "@/lib/estimator-data";
import { classAbbrev, formatDateTime, updatePrice, type PriceItem } from "@/lib/api";

export const Route = createFileRoute("/pricing")({
  head: () => ({
    meta: [
      { title: `Pricing Catalog — ${APP_NAME}` },
      {
        name: "description",
        content: "Unit costs used for cost estimation across all detected symbol classes.",
      },
      { property: "og:title", content: `Pricing Catalog — ${APP_NAME}` },
      {
        property: "og:description",
        content: "Searchable unit-cost reference table for detected symbol classes.",
      },
    ],
  }),
  component: PricingPage,
});

/** Sortable value per Pricing Catalog column. */
function priceSortValue(row: PriceItem, key: string): unknown {
  switch (key) {
    case "device_key":
      return row.device_key;
    case "description":
      return row.description;
    case "unit_price_usd":
      return row.unit_price_usd;
    case "updated_at":
      return row.updated_at;
    default:
      return null;
  }
}

function PricingPage() {
  const { pricing, pricingLoading, pricingCsv } = useEstimator();
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [editKey, setEditKey] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [reason, setReason] = useState("");
  const [saving, setSaving] = useState(false);

  const commitPrice = async (deviceKey: string) => {
    const n = parseFloat(draft);
    if (isNaN(n) || n < 0) {
      toast.error("Enter a valid price");
      return;
    }
    setEditKey(null);
    setSaving(true);
    try {
      await updatePrice(deviceKey, n, reason.trim());
      await queryClient.invalidateQueries({ queryKey: ["pricing"] });
      toast.success(`${deviceKey} price set to ${currency(n)} (locked)`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Price update failed");
    } finally {
      setSaving(false);
    }
  };

  const filtered = useMemo(
    () =>
      pricing.filter(
        (r) =>
          r.device_key.toLowerCase().includes(query.toLowerCase()) ||
          r.description.toLowerCase().includes(query.toLowerCase()),
      ),
    [pricing, query],
  );

  const grid = useTable(filtered, priceSortValue, {
    pageSize: 10,
    initialSort: { key: "device_key", asc: true },
  });

  const lastUpdated = formatDateTime(pricing[0]?.updated_at);

  return (
    <AppShell>
      <PageHeader
        title="Pricing Catalog"
        subtitle="Unit costs from the pipeline pricing catalog (pricing/device_prices.csv) — refreshed on every invoice build"
        actions={<span className="text-xs text-muted-foreground">Last updated: {lastUpdated}</span>}
      />

      <div className="relative mb-4 max-w-sm">
        <Search className="absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter by symbol / device class…"
          className="w-full rounded-md border border-border bg-card py-2 pr-3 pl-9 text-sm outline-none focus:border-orange focus:ring-1 focus:ring-orange"
        />
      </div>

      <div className="card-surface overflow-x-auto">
        {pricingLoading ? (
          <div className="flex items-center justify-center gap-2 px-6 py-14 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading pricing catalog…
          </div>
        ) : grid.total === 0 ? (
          <div className="flex flex-col items-center gap-3 px-6 py-14 text-center">
            <p className="text-sm text-muted-foreground">
              {pricing.length === 0
                ? "Pricing catalog is empty — is the backend running?"
                : "No rows match your filter."}
            </p>
          </div>
        ) : (
          <>
            <table className="w-full min-w-[820px] text-sm">
              <thead className="border-b border-border text-left text-xs text-muted-foreground uppercase">
                <tr>
                  <th className="px-4 py-3 font-medium">Symbol</th>
                  <SortableTh
                    label="Device Class (exact key)"
                    sortKey="device_key"
                    sort={grid.sort}
                    onSort={grid.toggleSort}
                  />
                  <SortableTh
                    label="Description"
                    sortKey="description"
                    sort={grid.sort}
                    onSort={grid.toggleSort}
                  />
                  <th className="px-4 py-3 font-medium">Unit</th>
                  <SortableTh
                    label="Unit Cost"
                    sortKey="unit_price_usd"
                    sort={grid.sort}
                    onSort={grid.toggleSort}
                  />
                  <SortableTh
                    label="Last Updated"
                    sortKey="updated_at"
                    sort={grid.sort}
                    onSort={grid.toggleSort}
                  />
                </tr>
              </thead>
              <tbody>
                {grid.rows.map((r) => (
                  <tr
                    key={r.device_key}
                    className="border-b border-border/60 last:border-0 hover:bg-accent/50"
                  >
                    <td className="px-4 py-3">
                      <span className="grid h-8 w-8 place-items-center rounded bg-brand/10 text-xs font-bold text-brand">
                        {classAbbrev(r.device_key)}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-medium">{r.device_key}</td>
                    <td className="px-4 py-3 text-muted-foreground">{r.description}</td>
                    <td className="px-4 py-3 text-muted-foreground">{r.unit}</td>
                    <td className="px-4 py-3">
                      {editKey === r.device_key ? (
                        <div className="flex flex-col gap-1.5">
                          <input
                            autoFocus
                            value={draft}
                            onChange={(e) => setDraft(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && void commitPrice(r.device_key)}
                            placeholder="New price"
                            className="w-32 rounded border border-orange bg-card px-2 py-1 text-sm outline-none ring-1 ring-orange"
                          />
                          <input
                            value={reason}
                            onChange={(e) => setReason(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && void commitPrice(r.device_key)}
                            placeholder="Reason for update (optional)"
                            className="w-56 rounded border border-border bg-card px-2 py-1 text-xs outline-none focus:border-orange focus:ring-1 focus:ring-orange"
                          />
                          <div className="flex gap-2">
                            <button
                              onClick={() => void commitPrice(r.device_key)}
                              className="rounded bg-orange px-2.5 py-1 text-xs font-medium text-orange-foreground hover:opacity-90"
                            >
                              Save
                            </button>
                            <button
                              onClick={() => setEditKey(null)}
                              className="rounded border border-border px-2.5 py-1 text-xs font-medium hover:bg-muted"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <span className="flex flex-col gap-0.5">
                          <span className="flex items-center gap-2">
                            <button
                              disabled={saving}
                              onClick={() => {
                                setEditKey(r.device_key);
                                setDraft(String(r.unit_price_usd));
                                setReason(r.override_reason ?? "");
                              }}
                              className="rounded px-1 py-0.5 tabular-nums hover:bg-muted disabled:opacity-50"
                              title="Click to edit — edited prices are locked and never drift"
                            >
                              {currency(r.unit_price_usd)}
                            </button>
                            {r.overridden && (
                              <span
                                className="flex items-center gap-1 rounded-full border border-orange/40 bg-orange/10 px-2 py-0.5 text-[11px] font-medium text-orange"
                                title={
                                  r.override_reason
                                    ? `Manually set — reason: ${r.override_reason}`
                                    : "Manually set — excluded from synthetic price drift"
                                }
                              >
                                <Lock className="h-2.5 w-2.5" /> manual
                              </span>
                            )}
                          </span>
                          {r.override_reason && (
                            <span
                              className="max-w-[240px] truncate text-xs text-muted-foreground italic"
                              title={r.override_reason}
                            >
                              {r.override_reason}
                            </span>
                          )}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground whitespace-nowrap">
                      {formatDateTime(r.updated_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <TablePagination
              page={grid.page}
              pageCount={grid.pageCount}
              total={grid.total}
              pageSize={grid.pageSize}
              onPage={grid.setPage}
            />
          </>
        )}
      </div>

      <p className="mt-4 text-xs text-muted-foreground">
        Source: <code>{pricingCsv || "pricing/device_prices.csv"}</code> — click a unit cost to edit
        it. Prices are stable between runs; edited prices are saved to the CSV and marked
        &quot;manual&quot;.
      </p>
    </AppShell>
  );
}
