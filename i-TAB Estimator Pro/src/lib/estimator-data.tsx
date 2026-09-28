import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { toast } from "sonner";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  classAbbrev,
  formatBytes,
  formatDateTime,
  formatDuration,
  getCorrections,
  getCounts,
  getPricing,
  listJobs,
  saveCorrections as apiSaveCorrections,
  uploadPdf,
  type CorrectionLine,
  type Counts,
  type Job,
  type PriceItem,
} from "./api";

/** One editable costing line derived from the backend invoice. */
export type LineItem = {
  id: string;
  deviceClass: string;
  symbol: string;
  count: number;
  /** Count originally reported by the model (before any human correction). */
  modelCount: number;
  unitCost: number;
  unit: string;
  overridden: boolean;
  yoloClasses: string[];
  /** Optional reason for a manual correction. */
  note: string;
};

export type Metadata = Record<string, string>;

export type InvoiceRecord = {
  number: string;
  generatedAt: string;
};

/* ---------- Currency formatting ---------- */

/** Currencies the user can display estimates in (base prices are USD). */
export const CURRENCIES = [
  { code: "USD", symbol: "$", label: "US Dollar", locale: "en-US", rate: 1 },
  { code: "EUR", symbol: "€", label: "Euro", locale: "de-DE", rate: 0.92 },
  { code: "GBP", symbol: "£", label: "British Pound", locale: "en-GB", rate: 0.79 },
  { code: "INR", symbol: "₹", label: "Indian Rupee", locale: "en-IN", rate: 88.0 },
  { code: "AED", symbol: "AED", label: "UAE Dirham", locale: "en-AE", rate: 3.67 },
] as const;

export type CurrencyCode = (typeof CURRENCIES)[number]["code"];

const CURRENCY_STORAGE_KEY = "estimator.currency";

let activeCurrency: (typeof CURRENCIES)[number] = CURRENCIES[0];

/** Format a USD amount in the currently selected display currency. */
export const currency = (usd: number) =>
  (usd * activeCurrency.rate).toLocaleString(activeCurrency.locale, {
    style: "currency",
    currency: activeCurrency.code,
    maximumFractionDigits: 2,
  });

type Ctx = {
  /* jobs */
  jobs: Job[];
  jobsLoading: boolean;
  selectedJob: Job | null;
  selectJob: (id: string) => void;
  /* live upload */
  uploadFile: (file: File, opts?: { select?: boolean }) => Promise<Job>;
  uploading: boolean;
  /* results for selected job */
  counts: Counts | null;
  countsLoading: boolean;
  lineItems: LineItem[];
  updateLineItem: (
    id: string,
    patch: Partial<Pick<LineItem, "count" | "unitCost" | "note">>,
  ) => void;
  /* human-verified corrections */
  correctionsSavedAt: string | null;
  saveCorrections: () => Promise<void>;
  savingCorrections: boolean;
  grandTotal: number;
  taxRate: number;
  taxAmount: number;
  invoiceTotal: number;
  metadata: Metadata;
  setMetadata: (key: string, value: string) => void;
  projectTitle: string;
  invoice: InvoiceRecord | null;
  /* pricing table */
  pricing: PriceItem[];
  pricingLoading: boolean;
  pricingCsv: string;
  /* display currency */
  currencyCode: CurrencyCode;
  setCurrencyCode: (code: CurrencyCode) => void;
};

const DataContext = createContext<Ctx | null>(null);

export function EstimatorProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [currencyCode, setCurrencyCodeState] = useState<CurrencyCode>("USD");

  // Restore the persisted currency choice on the client.
  useEffect(() => {
    const saved =
      typeof window !== "undefined" ? window.localStorage.getItem(CURRENCY_STORAGE_KEY) : null;
    if (saved && CURRENCIES.some((c) => c.code === saved)) {
      setCurrencyCodeState(saved as CurrencyCode);
    }
  }, []);

  // Keep the module-level formatter in sync so `currency()` works everywhere.
  activeCurrency = CURRENCIES.find((c) => c.code === currencyCode) ?? CURRENCIES[0];

  const setCurrencyCode = useCallback((code: CurrencyCode) => {
    setCurrencyCodeState(code);
    try {
      window.localStorage.setItem(CURRENCY_STORAGE_KEY, code);
    } catch {
      /* storage unavailable — selection lasts for the session only */
    }
  }, []);
  const [uploading, setUploading] = useState(false);
  /** Client-side edits, keyed per job so switching projects keeps them separate. */
  const [lineOverrides, setLineOverrides] = useState<
    Record<string, Record<string, { count?: number; unitCost?: number; note?: string }>>
  >({});
  const [metaOverrides, setMetaOverrides] = useState<Record<string, Metadata>>({});

  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: listJobs,
    refetchInterval: (query) => {
      const jobs = query.state.data ?? [];
      const busy = jobs.some((j) => j.status === "processing" || j.status === "queued");
      return busy ? 2500 : 15000;
    },
  });
  const jobs = useMemo(() => jobsQuery.data ?? [], [jobsQuery.data]);

  // Default selection: newest finished job, else newest job.
  const selectedJob = useMemo(() => {
    if (selectedJobId) return jobs.find((j) => j.id === selectedJobId) ?? null;
    return jobs.find((j) => j.status === "done") ?? jobs[0] ?? null;
  }, [jobs, selectedJobId]);

  const countsQuery = useQuery({
    queryKey: ["counts", selectedJob?.id, selectedJob?.status],
    queryFn: () => getCounts(selectedJob!.id),
    enabled: !!selectedJob && selectedJob.status === "done",
    staleTime: 60_000,
  });
  const counts = countsQuery.data ?? null;

  const pricingQuery = useQuery({
    queryKey: ["pricing"],
    queryFn: getPricing,
    staleTime: 5 * 60_000,
  });

  const correctionsQuery = useQuery({
    queryKey: ["corrections", selectedJob?.id],
    queryFn: () => getCorrections(selectedJob!.id),
    enabled: !!selectedJob && selectedJob.status === "done",
    staleTime: 60_000,
  });
  const savedCorrections = useMemo(() => {
    const map = new Map<string, CorrectionLine>();
    for (const ln of correctionsQuery.data?.lines ?? []) map.set(ln.key, ln);
    return map;
  }, [correctionsQuery.data]);

  const uploadFile = useCallback(
    async (file: File, opts?: { select?: boolean }) => {
      setUploading(true);
      try {
        const job = await uploadPdf(file);
        if (opts?.select !== false) setSelectedJobId(job.id);
        await queryClient.invalidateQueries({ queryKey: ["jobs"] });
        return job;
      } finally {
        setUploading(false);
      }
    },
    [queryClient],
  );

  // Invalidate cached counts when the selected job finishes a fresh run.
  useEffect(() => {
    if (selectedJob?.status === "done") {
      void queryClient.invalidateQueries({ queryKey: ["counts", selectedJob.id] });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedJob?.status, selectedJob?.id]);

  const jobKey = selectedJob?.id ?? "";
  const jobLineOverrides = lineOverrides[jobKey] ?? {};

  const lineItems: LineItem[] = useMemo(() => {
    const lines = counts?.invoice?.lines ?? [];
    return lines.map((ln, i) => {
      const id = `${ln.device_key}:${i}`;
      const o = jobLineOverrides[id] ?? {};
      // Priority: this-session edit > saved human correction > model value.
      const saved = savedCorrections.get(id);
      const count = o.count ?? saved?.corrected_count ?? ln.quantity;
      const unitCost =
        o.unitCost ??
        (saved && saved.corrected_unit_price > 0 ? saved.corrected_unit_price : undefined) ??
        ln.unit_price_usd;
      return {
        id,
        deviceClass: ln.description,
        symbol: classAbbrev(ln.description),
        count,
        modelCount: ln.quantity,
        unitCost,
        unit: ln.unit || "EA",
        overridden: count !== ln.quantity || unitCost !== ln.unit_price_usd,
        yoloClasses: ln.yolo_classes ?? [],
        note: o.note ?? saved?.note ?? "",
      };
    });
  }, [counts, jobLineOverrides, savedCorrections]);

  const [savingCorrections, setSavingCorrections] = useState(false);
  const saveCorrections = useCallback(async () => {
    if (!selectedJob || !counts?.invoice) return;
    const lines = (counts.invoice.lines ?? []).map((ln, i) => {
      const id = `${ln.device_key}:${i}`;
      const item = lineItems.find((li) => li.id === id);
      return {
        key: id,
        description: ln.description,
        model_count: ln.quantity,
        corrected_count: item?.count ?? ln.quantity,
        model_unit_price: ln.unit_price_usd,
        corrected_unit_price: item?.unitCost ?? ln.unit_price_usd,
        note: item?.note ?? "",
      };
    });
    setSavingCorrections(true);
    try {
      await apiSaveCorrections(selectedJob.id, lines);
      await queryClient.invalidateQueries({ queryKey: ["corrections", selectedJob.id] });
      await queryClient.invalidateQueries({ queryKey: ["stats"] });
    } finally {
      setSavingCorrections(false);
    }
  }, [selectedJob, counts, lineItems, queryClient]);

  const updateLineItem = useCallback(
    (id: string, patch: Partial<Pick<LineItem, "count" | "unitCost" | "note">>) => {
      if (!jobKey) return;
      setLineOverrides((prev) => ({
        ...prev,
        [jobKey]: {
          ...(prev[jobKey] ?? {}),
          [id]: { ...(prev[jobKey]?.[id] ?? {}), ...patch },
        },
      }));
    },
    [jobKey],
  );

  // Auto-save corrections: edits used to live only in memory until the user
  // pressed "Save corrections" — a refresh silently lost them. Debounce 2.5s
  // after the last edit, then persist to the backend.
  const saveRef = useRef(saveCorrections);
  saveRef.current = saveCorrections;
  const jobEditCount = Object.keys(jobLineOverrides).length;
  useEffect(() => {
    if (jobEditCount === 0) return;
    const timer = setTimeout(() => {
      void saveRef.current().catch((e: unknown) => {
        toast.error(
          e instanceof Error
            ? `Auto-save failed: ${e.message}`
            : "Auto-save failed — use the Save corrections button",
        );
      });
    }, 2500);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobLineOverrides]);

  const grandTotal = lineItems.reduce((s, i) => s + i.count * i.unitCost, 0);
  const taxRate = counts?.invoice?.tax_rate ?? 0;
  const taxAmount = grandTotal * taxRate;
  const invoiceTotal = grandTotal + taxAmount;

  const baseMetadata: Metadata = useMemo(() => {
    if (!selectedJob) return {};
    const meta: Metadata = {
      "Source File": selectedJob.filename,
      "File Size": formatBytes(selectedJob.size_bytes),
      Pages: String(selectedJob.page_count ?? "—"),
      Status: selectedJob.status,
      Uploaded: formatDateTime(selectedJob.created_at),
    };
    if (counts) {
      meta["Detection Model"] = counts.yolo_model ?? "—";
      meta["Legend Rows"] = String(counts.legend_entries ?? "—");
      meta["Wings Scanned"] = String(counts.per_wing?.length ?? "—");
    }
    if (selectedJob.actual_seconds) {
      meta["Processing Time"] = formatDuration(selectedJob.actual_seconds);
    }
    return meta;
  }, [selectedJob, counts]);

  const metadata: Metadata = useMemo(
    () => ({ ...baseMetadata, ...(metaOverrides[jobKey] ?? {}) }),
    [baseMetadata, metaOverrides, jobKey],
  );

  const setMetadata = useCallback(
    (key: string, value: string) => {
      if (!jobKey) return;
      setMetaOverrides((prev) => ({
        ...prev,
        [jobKey]: { ...(prev[jobKey] ?? {}), [key]: value },
      }));
    },
    [jobKey],
  );

  const projectTitle = selectedJob
    ? (selectedJob.filename || selectedJob.id).replace(/\.(pdf|png|jpe?g|webp)$/i, "")
    : "No project selected";

  const invoice: InvoiceRecord | null = useMemo(() => {
    if (!counts?.invoice) return null;
    return {
      number: `EST-${(selectedJob?.id ?? "JOB").toUpperCase().slice(0, 24)}`,
      generatedAt: counts.invoice.generated_at,
    };
  }, [counts, selectedJob]);

  const value: Ctx = {
    jobs,
    jobsLoading: jobsQuery.isLoading,
    selectedJob,
    selectJob: setSelectedJobId,
    uploadFile,
    uploading,
    counts,
    countsLoading: countsQuery.isLoading,
    lineItems,
    updateLineItem,
    correctionsSavedAt: correctionsQuery.data?.saved_at ?? null,
    saveCorrections,
    savingCorrections,
    grandTotal,
    taxRate,
    taxAmount,
    invoiceTotal,
    metadata,
    setMetadata,
    projectTitle,
    invoice,
    pricing: pricingQuery.data?.items ?? [],
    pricingLoading: pricingQuery.isLoading,
    pricingCsv: pricingQuery.data?.csv ?? "",
    currencyCode,
    setCurrencyCode,
  };

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}

export function useEstimator() {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error("useEstimator must be used within EstimatorProvider");
  return ctx;
}
