/** Title-block metadata extracted by the text branch of the pipeline. */

import type { ProjectMetadata } from '../types';

const FIELDS: { key: keyof ProjectMetadata; label: string }[] = [
  { key: 'title', label: 'Project Title' },
  { key: 'client', label: 'Client' },
  { key: 'architect', label: 'Architect' },
  { key: 'engineer', label: 'Engineer' },
  { key: 'project_address', label: 'Project Address' },
  { key: 'due_date', label: 'Due Date' },
];

export function MetadataPanel({ metadata }: { metadata: ProjectMetadata }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Project Metadata
      </h2>
      <dl className="grid grid-cols-1 gap-x-8 gap-y-3 sm:grid-cols-2">
        {FIELDS.map(({ key, label }) => (
          <div key={key}>
            <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
              {label}
            </dt>
            <dd className="mt-0.5 text-sm text-slate-800">
              {metadata[key] ?? <span className="italic text-slate-400">Not found</span>}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
