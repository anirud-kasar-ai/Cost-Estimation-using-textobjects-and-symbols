import { jobFileUrl, jobZipUrl } from '../api/client';
import type { JobDetail } from '../types';

function Figure({ src, caption }: { src: string; caption: string }) {
  return (
    <figure className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      <img src={src} alt={caption} className="block h-40 w-full bg-slate-50 object-contain" />
      <figcaption className="border-t border-slate-200 px-3 py-2 text-xs text-slate-500">
        {caption}
      </figcaption>
    </figure>
  );
}

export function DrawingResultsPanel({ job }: { job: JobDetail }) {
  const keptPages = job.pages?.filter((page) => page.kept) ?? [];
  if (!keptPages.length) {
    return null;
  }

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            Drawing Split Results
          </h2>
          <p className="text-sm text-slate-600">
            {job.drawing_count ?? 0} diagram(s), {job.wing_count ?? 0} wing image(s)
          </p>
        </div>
        {job.status === 'done' && (
          <a
            href={jobZipUrl(job.id)}
            className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-700"
          >
            Download zip
          </a>
        )}
      </div>

      {keptPages.map((page) => (
        <article
          key={page.page_key}
          className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
        >
          <div className="mb-4 flex items-center gap-2">
            <h3 className="text-base font-semibold text-slate-800">Page {page.page}</h3>
            <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
              DRAWING
            </span>
          </div>

          {page.page_image && (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-3">
              <Figure src={jobFileUrl(job.id, page.page_image)} caption="Full sheet" />
            </div>
          )}

          {page.diagram && (
            <>
              <h4 className="mb-2 mt-5 text-sm font-semibold text-slate-700">Diagram crop</h4>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-3">
                <Figure src={jobFileUrl(job.id, page.diagram)} caption="Full diagram" />
              </div>
            </>
          )}

          {page.wings && page.wings.length > 0 && (
            <>
              <h4 className="mb-2 mt-5 text-sm font-semibold text-slate-700">
                Wings ({page.wings.length})
              </h4>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-3">
                {page.wings.flatMap((wing) => {
                  const items = [
                    <Figure
                      key={`${wing.name}-wing`}
                      src={jobFileUrl(job.id, wing.image)}
                      caption={`${wing.name} wing`}
                    />,
                  ];
                  if (wing.roi_overlay_image) {
                    items.unshift(
                      <Figure
                        key={`${wing.name}-roi`}
                        src={jobFileUrl(job.id, wing.roi_overlay_image)}
                        caption={`${wing.name} ROI overlay`}
                      />,
                    );
                  }
                  return items;
                })}
              </div>
            </>
          )}
        </article>
      ))}
    </section>
  );
}
