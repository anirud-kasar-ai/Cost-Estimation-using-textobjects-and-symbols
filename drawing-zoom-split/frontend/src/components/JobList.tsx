import type { JobStatus, JobSummary } from '../types';

const STATUS_STYLES: Record<JobStatus, string> = {
  queued: 'bg-slate-100 text-slate-600',
  processing: 'bg-sky-100 text-sky-700',
  done: 'bg-emerald-100 text-emerald-700',
  failed: 'bg-red-100 text-red-700',
};

interface JobListProps {
  jobs: JobSummary[];
  selectedId: string | null;
  onSelect: (jobId: string) => void;
}

export function JobList({ jobs, selectedId, onSelect }: JobListProps) {
  if (jobs.length === 0) {
    return <p className="text-sm italic text-slate-400">No uploads yet.</p>;
  }

  return (
    <ul className="space-y-2">
      {jobs.map((job) => (
        <li key={job.id}>
          <button
            type="button"
            onClick={() => onSelect(job.id)}
            className={`flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left transition-colors ${
              job.id === selectedId
                ? 'border-sky-300 bg-sky-50'
                : 'border-slate-200 bg-white hover:border-sky-200'
            }`}
          >
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium text-slate-700">
                {job.filename}
              </span>
              <span className="text-xs text-slate-400">
                {new Date(job.created_at).toLocaleString()}
              </span>
            </span>
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[job.status]}`}
            >
              {job.status}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
