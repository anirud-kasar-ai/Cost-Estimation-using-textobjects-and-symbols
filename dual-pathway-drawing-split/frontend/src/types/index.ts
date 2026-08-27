export type JobStatus = 'queued' | 'processing' | 'done' | 'failed';

export type SymbolCountStatus = 'done' | 'skipped' | 'failed';

export interface SymbolCountRow {
  symbol: string;
  description: string;
  mfg_model: string;
  part_number: string;
  counts: Record<string, number>;
  total: number;
}

export interface SymbolCountReport {
  status: string;
  columns: string[];
  rows: SymbolCountRow[];
}

export interface ProjectMetadata {
  title: string | null;
  client: string | null;
  architect: string | null;
  engineer: string | null;
  project_address: string | null;
  due_date: string | null;
}

export interface WingOutput {
  index: number;
  name: string;
  image: string;
  roi_overlay_image?: string | null;
}

export interface JobPage {
  page: number;
  page_key: string;
  kept: boolean;
  reason?: string | null;
  page_image?: string | null;
  diagram?: string | null;
  wings?: WingOutput[];
}

export interface JobSummary {
  id: string;
  filename: string;
  status: JobStatus;
  stage?: string | null;
  message?: string | null;
  created_at: string;
  updated_at?: string | null;
}

export interface JobDetail extends JobSummary {
  pages: JobPage[];
  error?: string | null;
  page_count?: number;
  drawing_count?: number;
  wing_count?: number;
  has_requirement_pdf?: boolean;
  has_technical_symbol_pdf?: boolean;
  has_sheet_notes_pdf?: boolean;
  requirement_provider?: string | null;
  symbol_entry_count?: number;
  sheet_notes_item_count?: number;
  symbol_count_status?: SymbolCountStatus | null;
  symbol_count_instances?: number;
  symbol_count_report_csv?: string;
  symbol_count_report_pdf?: string;
  symbol_count_report_json?: string;
  symbol_count_reason?: string;
  symbol_count_error?: string;
  metadata?: ProjectMetadata;
}
