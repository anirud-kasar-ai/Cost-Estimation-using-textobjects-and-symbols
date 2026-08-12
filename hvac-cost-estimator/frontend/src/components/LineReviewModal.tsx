/** Human review: sheet crop for a costing line item. */

import { useEffect, useState } from 'react';

import { errorMessage, getLineReview } from '../api/client';
import type { DetectionReview, DeviceLine } from '../types';

interface LineReviewModalProps {
  projectId: string;
  line: DeviceLine;
  onClose: () => void;
}

export function LineReviewModal({ projectId, line, onClose }: LineReviewModalProps) {
  const [review, setReview] = useState<DetectionReview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void getLineReview(projectId, line.id)
      .then((data) => {
        if (!cancelled) setReview(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, line.id]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Review ${line.display_name}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[90vh] w-full max-w-2xl overflow-auto rounded-xl bg-white shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between border-b border-slate-100 px-5 py-4">
          <div>
            <h3 className="text-base font-semibold text-slate-800">{line.display_name}</h3>
            <p className="mt-1 text-sm text-slate-500">
              {line.category}
              {line.mfg ? ` · ${line.mfg}` : ''}
              {line.part_number ? ` / ${line.part_number}` : ''}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-sm text-slate-500 hover:bg-slate-100"
          >
            Close
          </button>
        </div>

        <div className="space-y-4 px-5 py-4">
          {loading && <p className="text-sm text-slate-400">Loading sheet crop…</p>}
          {error && (
            <p role="alert" className="text-sm text-red-600">
              {error}
            </p>
          )}
          {review && (
            <>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <div>
                  <dt className="text-slate-400">Page</dt>
                  <dd className="font-medium text-slate-700">{review.page_number}</dd>
                </div>
                <div>
                  <dt className="text-slate-400">Confidence</dt>
                  <dd className="font-medium text-slate-700">
                    {(review.confidence * 100).toFixed(0)}%
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-400">Location</dt>
                  <dd className="font-medium text-slate-700">
                    {review.room_label ?? line.locations ?? '—'}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-400">Quantity</dt>
                  <dd className="font-medium text-slate-700">
                    {review.quantity} {review.unit}
                  </dd>
                </div>
                <div className="col-span-2">
                  <dt className="text-slate-400">Bounding box</dt>
                  <dd className="font-mono text-xs text-slate-600">
                    [{review.bbox.map((v) => Math.round(v)).join(', ')}]
                  </dd>
                </div>
              </dl>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Sheet crop
                </p>
                <img
                  src={review.crop_url}
                  alt={`Detection crop for ${line.display_name}`}
                  className="mx-auto max-h-72 object-contain"
                />
              </div>
              <p className="text-xs text-slate-400">
                Verify this detection before trusting the quantity in the costing report.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
