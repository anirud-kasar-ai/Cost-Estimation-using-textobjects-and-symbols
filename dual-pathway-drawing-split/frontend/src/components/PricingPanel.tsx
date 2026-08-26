/** Editable SQLite pricing catalog (mfg + part → unit cost). */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, type FormEvent } from 'react';

import {
  addPricingFromLegend,
  createPricingItem,
  errorMessage,
  listPricing,
  seedSuggestedPricing,
  updatePricingItem,
} from '../api/client';
import type { PricingUnit } from '../types';
import { formatMoney } from './CostingReportTable';

const EMPTY_FORM = {
  mfg: '',
  part_number: '',
  description: '',
  unit: 'each' as PricingUnit,
  unit_cost: '',
};

interface PricingPanelProps {
  /** When set, shows “Add missing legend SKUs” for that project. */
  projectId?: string | null;
}

export function PricingPanel({ projectId = null }: PricingPanelProps) {
  const queryClient = useQueryClient();
  const pricing = useQuery({ queryKey: ['pricing'], queryFn: listPricing });
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftCost, setDraftCost] = useState('');
  const [form, setForm] = useState(EMPTY_FORM);

  const save = useMutation({
    mutationFn: ({ id, unit_cost }: { id: string; unit_cost: number }) =>
      updatePricingItem(id, { unit_cost }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['pricing'] });
      setEditingId(null);
    },
  });

  const create = useMutation({
    mutationFn: createPricingItem,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['pricing'] });
      setForm(EMPTY_FORM);
    },
  });

  const seedSuggested = useMutation({
    mutationFn: () => seedSuggestedPricing(),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['pricing'] });
    },
  });

  const addFromLegend = useMutation({
    mutationFn: () => addPricingFromLegend(projectId!),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['pricing'] });
      if (projectId) {
        void queryClient.invalidateQueries({ queryKey: ['project', projectId] });
      }
    },
  });

  const submitCreate = (event: FormEvent) => {
    event.preventDefault();
    const unit_cost = parseFloat(form.unit_cost);
    if (
      !form.mfg.trim() ||
      !form.part_number.trim() ||
      !Number.isFinite(unit_cost) ||
      unit_cost < 0
    ) {
      return;
    }
    create.mutate({
      mfg: form.mfg.trim(),
      part_number: form.part_number.trim(),
      description: form.description.trim(),
      unit: form.unit,
      unit_cost,
    });
  };

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-5 py-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Pricing Catalog
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          Keyed by manufacturer + part. For tech legends (1948-class), add SKUs from the
          symbol legend, then enter unit costs. Suggested HVAC rates are for detector mode
          only.
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          {projectId && (
            <button
              type="button"
              onClick={() => addFromLegend.mutate()}
              className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-50"
            >
              {addFromLegend.isPending ? 'Adding…' : 'Add missing legend SKUs'}
            </button>
          )}
          <button
            type="button"
            onClick={() => seedSuggested.mutate()}
            className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-sky-700 hover:bg-sky-50"
          >
            {seedSuggested.isPending ? 'Seeding…' : 'Seed suggested HVAC rates'}
          </button>
        </div>
        {addFromLegend.isSuccess && (
          <p className="mt-1 text-xs text-slate-500">
            Inserted {addFromLegend.data.inserted} legend SKU stub(s). Edit costs, then
            recalculate costing.
          </p>
        )}
        {seedSuggested.isSuccess && (
          <p className="mt-1 text-xs text-slate-500">
            Inserted {seedSuggested.data.inserted} suggested HVAC row(s).
          </p>
        )}
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
              <th className="px-5 py-2">Mfg / Part</th>
              <th className="px-3 py-2">Description</th>
              <th className="px-3 py-2 text-right">Unit</th>
              <th className="px-5 py-2 text-right">Cost</th>
            </tr>
          </thead>
          <tbody>
            {(pricing.data ?? []).map((item) => (
              <tr key={item.id} className="border-b border-slate-50">
                <td className="px-5 py-2">
                  <div className="font-medium text-slate-700">{item.mfg}</div>
                  <div className="text-xs text-slate-400">{item.part_number}</div>
                  {item.source === 'legend_needs_price' && (
                    <span className="mt-0.5 inline-block text-[10px] uppercase tracking-wide text-amber-600">
                      needs price
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-slate-700">{item.description || '—'}</td>
                <td className="px-3 py-2 text-right text-slate-500">{item.unit}</td>
                <td className="px-5 py-2 text-right">
                  {editingId === item.id ? (
                    <input
                      type="number"
                      min={0}
                      step={0.01}
                      aria-label={`Unit cost for ${item.description || item.part_number}`}
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
                      {item.source === 'legend_needs_price' && item.unit_cost === 0
                        ? '—'
                        : formatMoney(item.unit_cost, 'USD')}
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {(pricing.data ?? []).length === 0 && !pricing.isLoading && (
              <tr>
                <td colSpan={4} className="px-5 py-6 text-center italic text-slate-400">
                  No pricing rows — add mfg/part costs or legend SKUs.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <form
        onSubmit={submitCreate}
        className="space-y-2 border-t border-slate-100 px-5 py-4"
      >
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Add pricing row
        </p>
        <div className="grid grid-cols-2 gap-2">
          <input
            required
            placeholder="Mfg"
            value={form.mfg}
            onChange={(e) => setForm((f) => ({ ...f, mfg: e.target.value }))}
            className="rounded-md border border-slate-200 px-2 py-1.5 text-sm"
          />
          <input
            required
            placeholder="Part number"
            value={form.part_number}
            onChange={(e) => setForm((f) => ({ ...f, part_number: e.target.value }))}
            className="rounded-md border border-slate-200 px-2 py-1.5 text-sm"
          />
        </div>
        <input
          placeholder="Description"
          value={form.description}
          onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
          className="w-full rounded-md border border-slate-200 px-2 py-1.5 text-sm"
        />
        <div className="flex gap-2">
          <select
            value={form.unit}
            onChange={(e) =>
              setForm((f) => ({ ...f, unit: e.target.value as PricingUnit }))
            }
            className="rounded-md border border-slate-200 px-2 py-1.5 text-sm"
            aria-label="Unit"
          >
            <option value="each">each</option>
            <option value="ft">ft</option>
          </select>
          <input
            required
            type="number"
            min={0}
            step={0.01}
            placeholder="Unit cost"
            value={form.unit_cost}
            onChange={(e) => setForm((f) => ({ ...f, unit_cost: e.target.value }))}
            className="min-w-0 flex-1 rounded-md border border-slate-200 px-2 py-1.5 text-sm"
          />
          <button
            type="submit"
            disabled={create.isPending}
            className="rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-700 disabled:opacity-50"
          >
            Add
          </button>
        </div>
        {create.isError && (
          <p role="alert" className="text-sm text-red-600">
            {errorMessage(create.error)}
          </p>
        )}
      </form>
    </section>
  );
}
