/** Recent uploads list with status pills and delete action. */

import type { ProjectStatus, ProjectSummary } from '../types';

const STATUS_STYLES: Record<ProjectStatus, string> = {
  pending: 'bg-slate-100 text-slate-600',
  processing: 'bg-sky-100 text-sky-700',
  done: 'bg-emerald-100 text-emerald-700',
  failed: 'bg-red-100 text-red-700',
};

interface ProjectListProps {
  projects: ProjectSummary[];
  selectedId: string | null;
  onSelect: (projectId: string) => void;
  onDelete: (projectId: string) => void;
}

export function ProjectList({ projects, selectedId, onSelect, onDelete }: ProjectListProps) {
  if (projects.length === 0) {
    return <p className="text-sm italic text-slate-400">No uploads yet.</p>;
  }

  return (
    <ul className="space-y-2">
      {projects.map((project) => (
        <li
          key={project.id}
          className={`flex items-center gap-2 rounded-lg border px-3 py-2 transition-colors ${
            project.id === selectedId
              ? 'border-sky-300 bg-sky-50'
              : 'border-slate-200 bg-white hover:border-sky-200'
          }`}
        >
          <button
            type="button"
            onClick={() => onSelect(project.id)}
            className="min-w-0 flex-1 text-left"
          >
            <span className="block truncate text-sm font-medium text-slate-700">
              {project.filename}
            </span>
            <span className="text-xs text-slate-400">
              {new Date(project.created_at).toLocaleString()}
            </span>
          </button>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[project.status]}`}
          >
            {project.status}
          </span>
          <button
            type="button"
            aria-label={`Delete ${project.filename}`}
            onClick={() => onDelete(project.id)}
            className="text-slate-300 transition-colors hover:text-red-500"
          >
            ✕
          </button>
        </li>
      ))}
    </ul>
  );
}
