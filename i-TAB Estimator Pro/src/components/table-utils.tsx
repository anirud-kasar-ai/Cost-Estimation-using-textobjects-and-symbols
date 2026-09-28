import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";

export type SortState = { key: string; asc: boolean } | null;

/**
 * Client-side sorting + pagination for data tables.
 * `getValue` extracts the sortable value for a column key from a row.
 */
export function useTable<T>(
  rows: T[],
  getValue: (row: T, key: string) => unknown,
  opts?: { pageSize?: number; initialSort?: { key: string; asc: boolean } },
) {
  const [sort, setSort] = useState<SortState>(opts?.initialSort ?? null);
  const [page, setPage] = useState(1);
  const pageSize = opts?.pageSize ?? 10;

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const dir = sort.asc ? 1 : -1;
    return [...rows].sort((a, b) => {
      const x = getValue(a, sort.key);
      const y = getValue(b, sort.key);
      if (x == null && y == null) return 0;
      if (x == null) return dir;
      if (y == null) return -dir;
      if (typeof x === "number" && typeof y === "number") return (x - y) * dir;
      return String(x).localeCompare(String(y)) * dir;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, sort]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize));
  const safePage = Math.min(page, pageCount);
  const pageRows = useMemo(
    () => sorted.slice((safePage - 1) * pageSize, safePage * pageSize),
    [sorted, safePage, pageSize],
  );

  const toggleSort = (key: string) => {
    setSort((s) => (s?.key === key ? { key, asc: !s.asc } : { key, asc: true }));
    setPage(1);
  };

  return {
    rows: pageRows,
    allRows: sorted,
    sort,
    toggleSort,
    page: safePage,
    pageCount,
    setPage,
    pageSize,
    total: sorted.length,
  };
}

/** Sortable table header cell with a direction indicator. */
export function SortableTh({
  label,
  sortKey,
  sort,
  onSort,
  className,
  align = "left",
}: {
  label: string;
  sortKey: string;
  sort: SortState;
  onSort: (key: string) => void;
  className?: string;
  align?: "left" | "right";
}) {
  const active = sort?.key === sortKey;
  return (
    <th
      className={cn(
        "cursor-pointer px-4 py-3 font-medium select-none",
        align === "right" && "text-right",
        className,
      )}
      onClick={() => onSort(sortKey)}
      title={`Sort by ${label}`}
    >
      <span
        className={cn(
          "inline-flex items-center gap-1",
          align === "right" && "flex-row-reverse",
          active && "text-foreground",
        )}
      >
        {label}
        {active ? (
          sort!.asc ? (
            <ArrowUp className="h-3 w-3 shrink-0" />
          ) : (
            <ArrowDown className="h-3 w-3 shrink-0" />
          )
        ) : (
          <ChevronsUpDown className="h-3 w-3 shrink-0 opacity-40" />
        )}
      </span>
    </th>
  );
}

/** Footer pagination bar — hidden automatically when everything fits one page. */
export function TablePagination({
  page,
  pageCount,
  total,
  pageSize,
  onPage,
}: {
  page: number;
  pageCount: number;
  total: number;
  pageSize: number;
  onPage: (page: number) => void;
}) {
  if (total <= pageSize) return null;
  const from = (page - 1) * pageSize + 1;
  const to = Math.min(page * pageSize, total);
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-4 py-3">
      <p className="text-xs text-muted-foreground tabular-nums">
        Showing {from}–{to} of {total}
      </p>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onPage(page - 1)}
          disabled={page <= 1}
          aria-label="Previous page"
          className="rounded-md border border-border p-1.5 transition-colors hover:bg-muted disabled:opacity-40"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        {Array.from({ length: pageCount }, (_, i) => i + 1)
          .filter((p) => p === 1 || p === pageCount || Math.abs(p - page) <= 1)
          .map((p, idx, arr) => (
            <span key={p} className="flex items-center">
              {idx > 0 && arr[idx - 1] !== p - 1 && (
                <span className="px-1 text-xs text-muted-foreground">…</span>
              )}
              <button
                onClick={() => onPage(p)}
                className={cn(
                  "min-w-8 rounded-md border px-2 py-1 text-xs font-medium tabular-nums transition-colors",
                  p === page
                    ? "border-orange bg-orange/10 text-orange"
                    : "border-border hover:bg-muted",
                )}
              >
                {p}
              </button>
            </span>
          ))}
        <button
          onClick={() => onPage(page + 1)}
          disabled={page >= pageCount}
          aria-label="Next page"
          className="rounded-md border border-border p-1.5 transition-colors hover:bg-muted disabled:opacity-40"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
