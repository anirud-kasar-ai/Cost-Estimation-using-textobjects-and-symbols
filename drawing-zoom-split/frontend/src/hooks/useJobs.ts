import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { getJob, listJobs, uploadPdf, fetchSymbolCountReport } from '../api/client';
import type { JobDetail, SymbolCountStatus } from '../types';

export function useJobs() {
  return useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
    refetchInterval: (query) => {
      const jobs = query.state.data;
      if (!jobs?.some((job) => job.status === 'queued' || job.status === 'processing')) {
        return false;
      }
      return 2000;
    },
  });
}

export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const job = query.state.data as JobDetail | undefined;
      if (!job || job.status === 'done' || job.status === 'failed') return false;
      return 1500;
    },
  });
}

export function useUploadPdf(onCreated?: (jobId: string) => void) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: uploadPdf,
    onSuccess: (job) => {
      void queryClient.invalidateQueries({ queryKey: ['jobs'] });
      void queryClient.setQueryData(['job', job.id], job);
      onCreated?.(job.id);
    },
  });
}

export function useSymbolCountReport(
  jobId: string | null,
  symbolCountStatus?: SymbolCountStatus | null,
) {
  return useQuery({
    queryKey: ['symbolCountReport', jobId],
    queryFn: () => fetchSymbolCountReport(jobId!),
    enabled: Boolean(jobId) && symbolCountStatus === 'done',
  });
}
