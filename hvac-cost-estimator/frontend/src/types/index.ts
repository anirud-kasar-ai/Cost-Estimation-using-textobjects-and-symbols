/** API types mirroring the backend Pydantic schemas (backend/schemas/). */

export type ProjectStatus = 'pending' | 'processing' | 'done' | 'failed';

export interface ProjectMetadata {
  title: string | null;
  client: string | null;
  architect: string | null;
  engineer: string | null;
  project_address: string | null;
  due_date: string | null;
}

export interface DeviceLine {
  id: string;
  device_type: string;
  display_name: string;
  count: number;
  unit_cost: number;
  detected_count: number;
  default_unit_cost: number;
  needs_review: boolean;
  line_total: number;
}

export interface ProjectSummary {
  id: string;
  filename: string;
  status: ProjectStatus;
  error_message: string | null;
  created_at: string;
}

export interface ProjectDetail extends ProjectSummary {
  metadata: ProjectMetadata;
  device_lines: DeviceLine[];
  grand_total: number;
  currency: string;
  page_count: number;
  has_requirement_pdf: boolean;
  requirement_provider: string | null;
  pages_truncated: boolean;
}

export interface UploadResponse {
  project_id: string;
  status: ProjectStatus;
}

export interface DeviceLineUpdate {
  count?: number;
  unit_cost?: number;
}
