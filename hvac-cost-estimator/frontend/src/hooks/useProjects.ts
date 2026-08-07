/** React Query hooks wrapping the API client. */

import {
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';

import {
  deleteProject,
  getProject,
  listProjects,
  updateDeviceLine,
  uploadPdf,
} from '../api/client';
import type { DeviceLineUpdate, ProjectDetail } from '../types';

export function useProjects() {
  return useQuery({ queryKey: ['projects'], queryFn: listProjects });
}

/** Project detail, polling while the pipeline is still running. */
export function useProject(projectId: string | null) {
  return useQuery({
    queryKey: ['project', projectId],
    queryFn: () => getProject(projectId!),
    enabled: projectId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'pending' || status === 'processing' ? 1500 : false;
    },
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
