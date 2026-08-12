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
  category: string;
  unit: string;
  mfg: string | null;
  part_number: string | null;
  locations: string | null;
  sample_detection_id: string | null;
}

export interface DetectionReview {
  detection_id: string;
  device_line_id: string | null;
  page_number: number;
  device_type: string;
  confidence: number;
  room_label: string | null;
  quantity: number;
  unit: string;
  bbox: [number, number, number, number];
  crop_url: string;
  page_image_url: string | null;
  needs_review: boolean;
}

export interface PricingItem {
  id: string;
  mfg: string;
  part_number: string;
  display_name: string;
  category: string;
  unit: string;
  unit_cost: number;
  device_type: string | null;
  updated_at: string | null;
}

export interface PricingItemUpdate {
  mfg?: string;
  part_number?: string;
  display_name?: string;
  category?: string;
  unit?: string;
  unit_cost?: number;
  device_type?: string | null;
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
  has_technical_symbol_pdf: boolean;
}

export interface UploadResponse {
  project_id: string;
  status: ProjectStatus;
}

export interface DeviceLineUpdate {
  count?: number;
  unit_cost?: number;
}
