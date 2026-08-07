/** Typed API client. All calls go through the Vite dev proxy (`/api`). */

import axios from 'axios';

import type {
  DeviceLineUpdate,
  ProjectDetail,
  ProjectSummary,
  UploadResponse,
} from '../types';

export const api = axios.create({ baseURL: '/api' });

/** Extract a human-readable message from an API error. */
export function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail: unknown = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    return error.message;
  }
  return error instanceof Error ? error.message : 'Unexpected error';
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<UploadResponse>('/upload', form);
  return data;
}

export async function listProjects(): Promise<ProjectSummary[]> {
  const { data } = await api.get<ProjectSummary[]>('/projects');
  return data;
}

export async function getProject(projectId: string): Promise<ProjectDetail> {
  const { data } = await api.get<ProjectDetail>(`/projects/${projectId}`);
  return data;
}

export async function deleteProject(projectId: string): Promise<void> {
  await api.delete(`/projects/${projectId}`);
}

export async function updateDeviceLine(
  projectId: string,
  lineId: string,
  payload: DeviceLineUpdate,
): Promise<ProjectDetail> {
  const { data } = await api.patch<ProjectDetail>(
    `/projects/${projectId}/lines/${lineId}`,
    payload,
  );
  return data;
}

/** Download the CSV report and trigger a browser "save file" action. */
export async function downloadReportCsv(projectId: string, filename: string): Promise<void> {
  const { data } = await api.get<Blob>(`/projects/${projectId}/report/csv`, {
    responseType: 'blob',
  });
  const url = URL.createObjectURL(data);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${filename.replace(/\.pdf$/i, '')}_costing_report.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/** Download the extracted ``<name> requirement.pdf``. */
export async function downloadRequirementPdf(
  projectId: string,
  filename: string,
): Promise<void> {
  const { data } = await api.get<Blob>(`/projects/${projectId}/requirement.pdf`, {
    responseType: 'blob',
  });
  const url = URL.createObjectURL(data);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${filename.replace(/\.pdf$/i, '')} requirement.pdf`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
