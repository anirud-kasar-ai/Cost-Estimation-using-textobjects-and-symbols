/** Typed API client for dual-pathway-drawing-split backend. */

import axios from 'axios';

import type { JobDetail, JobSummary, SymbolCountReport } from '../types';

export const api = axios.create({ baseURL: '/api' });

export function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail: unknown = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    return error.message;
  }
  return error instanceof Error ? error.message : 'Unexpected error';
}

export async function uploadPdf(file: File): Promise<JobDetail> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<JobDetail>('/jobs', form);
  return data;
}

export async function listJobs(): Promise<JobSummary[]> {
  const { data } = await api.get<JobSummary[]>('/jobs');
  return data;
}

export async function getJob(jobId: string): Promise<JobDetail> {
  const { data } = await api.get<JobDetail>(`/jobs/${jobId}`);
  return data;
}

export function jobFileUrl(jobId: string, relPath: string): string {
  return `/api/jobs/${jobId}/files/${relPath}`;
}

export function jobZipUrl(jobId: string): string {
  return `/api/jobs/${jobId}/zip`;
}

async function downloadPdf(url: string, downloadName: string): Promise<void> {
  const { data } = await api.get<Blob>(url, { responseType: 'blob' });
  const blobUrl = URL.createObjectURL(data);
  const anchor = document.createElement('a');
  anchor.href = blobUrl;
  anchor.download = downloadName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(blobUrl);
}

export async function downloadRequirementPdf(jobId: string, filename: string): Promise<void> {
  await downloadPdf(
    `/jobs/${jobId}/requirement.pdf`,
    `${filename.replace(/\.pdf$/i, '')} requirement.pdf`,
  );
}

export async function downloadTechnicalSymbolPdf(jobId: string, filename: string): Promise<void> {
  await downloadPdf(
    `/jobs/${jobId}/technical-symbol.pdf`,
    `${filename.replace(/\.pdf$/i, '')} technical symbol.pdf`,
  );
}

export async function downloadSheetNotesPdf(jobId: string, filename: string): Promise<void> {
  await downloadPdf(
    `/jobs/${jobId}/sheet-notes.pdf`,
    `${filename.replace(/\.pdf$/i, '')} sheet notes.pdf`,
  );
}

export async function fetchSymbolCountReport(jobId: string): Promise<SymbolCountReport> {
  const { data } = await api.get<SymbolCountReport>(`/jobs/${jobId}/symbol-count.json`);
  return data;
}

export async function downloadSymbolCountCsv(jobId: string, filename: string): Promise<void> {
  await downloadPdf(
    `/jobs/${jobId}/symbol-count.csv`,
    `${filename.replace(/\.pdf$/i, '')} symbol count.csv`,
  );
}

export async function downloadSymbolCountPdf(jobId: string, filename: string): Promise<void> {
  await downloadPdf(
    `/jobs/${jobId}/symbol-count.pdf`,
    `${filename.replace(/\.pdf$/i, '')} symbol count.pdf`,
  );
}

export interface HealthResponse {
  ok: boolean;
  model: string;
  runtime: string;
  groq_key_set: boolean;
  token_set: boolean;
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}
