/** React Query hooks wrapping the API client. */

import {
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';

import {
  calculateCosting,
  deleteProject,
  getProject,
  getProjectJobs,
  listProjects,
  setCostingMode,
  updateDeviceLine,
  uploadPdf,
  verifyInstance,
} from '../api/client';
import type { CostingMode, DeviceLineUpdate, ProjectDetail } from '../types';

export function useProjects() {
  return useQuery({ queryKey: ['projects'], queryFn: listProjects });
}

/** Project detail, polling while the pipeline or latest job is still running. */
export function useProject(projectId: string | null) {
  return useQuery({
    queryKey: ['project', projectId],
    queryFn: () => getProject(projectId!),
    enabled: projectId !== null,
    refetchInterval: (query) => {
      const data = query.state.data;
      const status = data?.status;
      const jobStatus = data?.latest_job?.status;
      if (status === 'pending' || status === 'processing') return 1500;
      if (jobStatus === 'queued' || jobStatus === 'running') return 1500;
      return false;
    },
  });
}

export function useJobs(projectId: string | null) {
  return useQuery({
    queryKey: ['jobs', projectId],
    queryFn: () => getProjectJobs(projectId!),
    enabled: projectId !== null,
  });
}

export function useUploadPdf(onUploaded: (projectId: string) => void) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: uploadPdf,
    onSuccess: (response) => {
      void queryClient.invalidateQueries({ queryKey: ['projects'] });
      onUploaded(response.project_id);
    },
  });
}

export function useUpdateDeviceLine(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ lineId, payload }: { lineId: string; payload: DeviceLineUpdate }) =>
      updateDeviceLine(projectId, lineId, payload),
    onSuccess: (detail: ProjectDetail) => {
      queryClient.setQueryData(['project', projectId], detail);
    },
  });
}

export function useVerifyInstance(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ instanceId, verified }: { instanceId: string; verified: boolean }) =>
      verifyInstance(projectId, instanceId, verified),
    onSuccess: (detail: ProjectDetail) => {
      queryClient.setQueryData(['project', projectId], detail);
    },
  });
}

export function useCalculateCosting(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (mode?: CostingMode) => calculateCosting(projectId, mode),
    onSuccess: (detail: ProjectDetail) => {
      queryClient.setQueryData(['project', projectId], detail);
      void queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });
}

export function useSetCostingMode(projectId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (mode: CostingMode) => setCostingMode(projectId, mode),
    onSuccess: (detail: ProjectDetail) => {
      queryClient.setQueryData(['project', projectId], detail);
      void queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });
}

export function useDeleteProject(onDeleted: (projectId: string) => void) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteProject,
    onSuccess: (_data, projectId) => {
      void queryClient.invalidateQueries({ queryKey: ['projects'] });
      queryClient.removeQueries({ queryKey: ['project', projectId] });
      onDeleted(projectId);
    },
  });
}
