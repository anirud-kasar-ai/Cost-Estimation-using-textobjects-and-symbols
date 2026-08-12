/** Editable SQLite pricing catalog (mfg + part → unit cost). */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

import { errorMessage, listPricing, updatePricingItem } from '../api/client';
import { formatMoney } from './CostingReportTable';

export function PricingPanel() {
  const queryClient = useQueryClient();
  const pricing = useQuery({ queryKey: ['pricing'], queryFn: listPricing });
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftCost, setDraftCost] = useState('');

  const save = useMutation({
    mutationFn: ({ id, unit_cost }: { id: string; unit_cost: number }) =>
      updatePricingItem(id, { unit_cost }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['pricing'] });
      setEditingId(null);
    },
  });

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-5 py-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Pricing Catalog
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          SQLite table keyed by manufacturer + part number. Seeded manually — edit unit costs
          here; new uploads join detections against these rows.
        </p>
      </div>
      {pricing.isError && (
        <p role="alert" className="px-5 py-3 text-sm text-red-600">
          {errorMessage(pricing.error)}
        </p>
      )}
      <div className="max-h-72 overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 text-left text-xs uppercase tracking-wide text-slate-400">
              <th className="px-5 py-2">Category</th>
              <th className="px-3 py-2">Mfg / Part</th>
              <th className="px-3 py-2">Device</th>
              <th className="px-3 py-2 text-right">Unit</th>
              <th className="px-5 py-2 text-right">Cost</th>
            </tr>
          </thead>
          <tbody>
            {(pricing.data ?? []).map((item) => (
              <tr key={item.id} className="border-b border-slate-50">
                <td className="px-5 py-2 text-slate-500">{item.category}</td>
                <td className="px-3 py-2">
                  <div className="font-medium text-slate-700">{item.mfg}</div>
                  <div className="text-xs text-slate-400">{item.part_number}</div>
                </td>
                <td className="px-3 py-2 text-slate-700">{item.display_name}</td>
                <td className="px-3 py-2 text-right text-slate-500">{item.unit}</td>
                <td className="px-5 py-2 text-right">
                  {editingId === item.id ? (
                    <input
                      type="number"
                      min={0}
                      step={0.01}
                      aria-label={`Unit cost for ${item.display_name}`}
                      value={draftCost}
                      onChange={(e) => setDraftCost(e.target.value)}
                      onBlur={() => {
                        const parsed = parseFloat(draftCost);
                        if (Number.isFinite(parsed) && parsed >= 0) {
                          save.mutate({ id: item.id, unit_cost: parsed });
                        } else {
                          setEditingId(null);
                        }
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                      className="w-24 rounded-md border border-slate-200 px-2 py-1 text-right"
                      autoFocus
                    />
                  ) : (
                    <button
                      type="button"
                      className="text-sky-700 hover:underline"
                      onClick={() => {
                        setEditingId(item.id);
                        setDraftCost(String(item.unit_cost));
                      }}
                    >
                      {formatMoney(item.unit_cost, 'USD')}
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {(pricing.data ?? []).length === 0 && !pricing.isLoading && (
              <tr>
                <td colSpan={5} className="px-5 py-6 text-center italic text-slate-400">
                  No pricing rows yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
