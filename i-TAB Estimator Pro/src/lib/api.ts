/** Typed client for the drawing-zoom-split FastAPI backend (proxied at /api). */

export const API_BASE = "/api";

/* ---------- Backend response shapes ---------- */

export type Job = {
  id: string;
  status: "queued" | "processing" | "done" | "failed" | string;
  stage: string;
  message: string;
  filename: string;
  /** Size of the uploaded source file in bytes (new jobs only). */
  size_bytes?: number;
  /** Bill-To client block shown on the estimate (set from the Invoice page). */
  client?: { name?: string; address?: string; contact?: string };
  created_at: string;
  updated_at: string;
  error: string | null;
  page_count?: number;
  progress?: number;
  eta_seconds?: number;
  estimated_total_seconds?: number;
  actual_seconds?: number;
  diagram_pages?: number;
  wings_total?: number;
  pages?: unknown[];
};

export type InvoiceLine = {
  description: string;
  device_key: string;
  quantity: number;
  unit: string;
  unit_price_usd: number;
  line_total_usd: number;
  yolo_classes: string[];
};

export type Invoice = {
  job_id: string;
  currency: string;
  generated_at: string;
  tax_rate: number;
  subtotal_usd: number;
  tax_usd: number;
  grand_total_usd: number;
  line_count: number;
  lines: InvoiceLine[];
};

export type CountRow = {
  number: number;
  description: string;
  detail_ref: string | null;
  count: number;
  yolo_count: number;
  yolo_classes: string[];
  method: string;
};

export type UnmatchedRow = {
  class_name: string;
  count: number;
  reason: string;
};

export type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  image_name: string;
  /** Legend row this detection was matched to; null = not in the legend (not priced). */
  legend_number: number | null;
};

export type Counts = {
  method: string;
  yolo_model: string | null;
  yolo_conf?: number;
  legend_entries?: number;
  zoom_folders_processed?: number;
  total_count: number;
  total_yolo?: number;
  rows: CountRow[];
  yolo_unmatched: UnmatchedRow[];
  per_wing: { wing: string; page: number; yolo_count: number; matched: number }[];
  yolo_detections?: Detection[];
  invoice: Invoice | null;
  note?: string;
};

export type PriceItem = {
  device_key: string;
  description: string;
  unit: string;
  unit_price_usd: number;
  currency: string;
  updated_at: string;
  /** True when the price was manually edited (locked — never drifts). */
  overridden?: boolean;
  /** Optional reason entered when the price was manually updated. */
  override_reason?: string | null;
};

export type OverlayDetection = {
  class_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  legend_number: number | null;
};

export type OverlayWing = {
  wing: string | null;
  page: number | null;
  folder: string;
  /** Relative path of the wing image, served via jobFileUrl(). Null if missing. */
  image: string | null;
  yolo_count: number;
  matched: number;
  detections: OverlayDetection[];
};

export type Overlay = {
  job_id: string;
  yolo_model: string | null;
  wings: OverlayWing[];
};

export type CorrectionLine = {
  key: string;
  description: string;
  model_count: number;
  corrected_count: number;
  model_unit_price: number;
  corrected_unit_price: number;
  /** Optional reason for the manual correction (e.g. which page was miscounted). */
  note?: string;
};

export type Corrections = {
  saved_at: string | null;
  lines: CorrectionLine[];
};

export type StatsVerification = {
  jobs_verified: number;
  lines_verified: number;
  model_count_total: number;
  human_count_total: number;
  count_accuracy: number | null;
};

export type StatsClass = {
  class_name: string;
  /** Which model weight file(s) this class is trained in. */
  models: string[];
  detections: number;
  matched: number;
  match_rate: number | null;
  avg_confidence: number | null;
  /** Training metrics of the model this class belongs to (model-level). */
  train_map50: number | null;
  train_precision: number | null;
  train_recall: number | null;
};

export type ModelTrainMetrics = {
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
  epochs: number | null;
};

export type StatsModel = {
  name: string;
  classes: string[];
  size_bytes: number;
  modified_at: string;
  active: boolean;
  train_metrics: ModelTrainMetrics | null;
};

export type Stats = {
  jobs_done: number;
  total_symbols: number;
  avg_processing_seconds: number | null;
  classes: StatsClass[];
  models: StatsModel[];
  verification: StatsVerification | null;
};

export type LegendEntry = {
  symbol: string | null;
  description: string;
  notes: string[];
  source_page: number;
  legend_title: string | null;
  mfg_model: string | null;
  part_number: string | null;
  has_glyph: boolean;
};

export type SymbolTable = {
  entry_count: number;
  legend_pages: number[];
  total_pages: number;
  notes: string[];
  entries: LegendEntry[];
};

export type Health = {
  ok: boolean;
  model: string;
  yolo: boolean;
  yolo_model: string | null;
  yolo_models: string[];
  yolo_classes: string[];
};

/* ---------- Fetch helpers ---------- */

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`GET ${path} failed (${res.status}): ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`PUT ${path} failed (${res.status}): ${text.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const listJobs = () => getJson<Job[]>("/jobs");
export const getJob = (id: string) => getJson<Job>(`/jobs/${encodeURIComponent(id)}`);
export const getCounts = (id: string) => getJson<Counts>(`/jobs/${encodeURIComponent(id)}/counts`);
export const getPricing = () =>
  getJson<{ csv: string; count: number; items: PriceItem[] }>("/pricing");
export const getSymbolTable = (id: string) =>
  getJson<SymbolTable>(`/jobs/${encodeURIComponent(id)}/files/symbol_table.json`);
export const getStats = () => getJson<Stats>("/stats");
export const getHealth = () => getJson<Health>("/health");
export const getOverlay = (id: string) =>
  getJson<Overlay>(`/jobs/${encodeURIComponent(id)}/overlay`);
export const getCorrections = (id: string) =>
  getJson<Corrections>(`/jobs/${encodeURIComponent(id)}/corrections`);
export const saveCorrections = (id: string, lines: CorrectionLine[]) =>
  putJson<Corrections>(`/jobs/${encodeURIComponent(id)}/corrections`, { lines });
export const updatePrice = (deviceKey: string, unitPriceUsd: number, reason?: string) =>
  putJson<PriceItem>(`/pricing/${encodeURIComponent(deviceKey)}`, {
    unit_price_usd: unitPriceUsd,
    reason: reason ?? "",
  });
export const jobFileUrl = (id: string, rel: string) =>
  `${API_BASE}/jobs/${encodeURIComponent(id)}/files/${rel}`;
/** Set a custom tax rate (fraction, e.g. 0.045 for 4.5%) — rebuilds invoice JSON/PDF/CSV. */
export const updateTaxRate = (id: string, taxRate: number) =>
  putJson<Invoice>(`/jobs/${encodeURIComponent(id)}/tax`, { tax_rate: taxRate });
/** Save the Bill-To client block for a job — regenerates the estimate PDF. */
export const updateClient = (
  id: string,
  client: { name: string; address: string; contact: string },
) => putJson<Job>(`/jobs/${encodeURIComponent(id)}/client`, client);

export async function uploadPdf(file: File): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/jobs`, { method: "POST", body: form });
  if (!res.ok) {
    let detail = `Upload failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<Job>;
}

export const invoicePdfUrl = (id: string) =>
  `${API_BASE}/jobs/${encodeURIComponent(id)}/invoice.pdf`;
export const invoiceCsvUrl = (id: string) =>
  `${API_BASE}/jobs/${encodeURIComponent(id)}/invoice.csv`;
export const jobZipUrl = (id: string) => `${API_BASE}/jobs/${encodeURIComponent(id)}/zip`;
export const legendPdfUrl = (id: string) =>
  `${API_BASE}/jobs/${encodeURIComponent(id)}/technical-symbol.pdf`;

/* ---------- Small shared utils ---------- */

/** Short letter code shown as a "symbol" badge for a device class name. */
export function classAbbrev(name: string): string {
  const words = (name || "?")
    .replace(/[^A-Za-z0-9 ]+/g, " ")
    .trim()
    .split(/\s+/);
  const first = words[0] ?? "?";
  const second = words[1];
  if (!second) return first.slice(0, 2).toUpperCase() || "?";
  return `${first.charAt(0)}${second.charAt(0)}`.toUpperCase();
}

/** Human-readable file size, e.g. "2.41 MB". */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !isFinite(bytes) || bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

/** Standard app-wide date format, e.g. "24 Sep 2026". */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

/** Standard app-wide date-time format, e.g. "24 Sep 2026, 02:40 PM".
 *  Date-only strings (no time part) fall back to the date format. */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  if (!iso.includes("T") && !iso.includes(":")) return formatDate(iso);
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const date = d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  const time = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: true });
  return `${date}, ${time}`;
}

export function formatDuration(totalSeconds: number | null | undefined): string {
  if (totalSeconds == null || !isFinite(totalSeconds)) return "—";
  const s = Math.max(0, Math.round(totalSeconds));
  if (s < 60) return `${s} sec`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${s % 60} sec`;
  return `${Math.floor(m / 60)} hr ${m % 60} min`;
}
