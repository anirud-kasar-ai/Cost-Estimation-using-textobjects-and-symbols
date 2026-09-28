import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ArrowLeft, Download, Loader2, Pencil, Printer } from "lucide-react";
import { AppShell, APP_NAME } from "@/components/app-shell";
import { currency, useEstimator } from "@/lib/estimator-data";
import { formatDate, formatDateTime, invoicePdfUrl, updateClient, updateTaxRate } from "@/lib/api";

/** Days an estimate stays valid — keep in sync with ESTIMATE_VALID_DAYS on the backend. */
const VALID_DAYS = 30;

export const Route = createFileRoute("/invoice")({
  head: () => ({
    meta: [
      { title: `Invoice — ${APP_NAME}` },
      {
        name: "description",
        content: "Printable device-costed invoice generated from the drawing estimate.",
      },
      { property: "og:title", content: `Invoice — ${APP_NAME}` },
      {
        property: "og:description",
        content: "Line-item invoice with subtotal, tax and total for the detected symbols.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: InvoicePage,
});

function InvoicePage() {
  const {
    selectedJob,
    counts,
    lineItems,
    grandTotal,
    taxRate,
    taxAmount,
    invoiceTotal,
    metadata,
    projectTitle,
    invoice,
    currencyCode,
  } = useEstimator();
  const queryClient = useQueryClient();
  const [editingTax, setEditingTax] = useState(false);
  const [taxDraft, setTaxDraft] = useState("");
  const [savingTax, setSavingTax] = useState(false);
  const [editingClient, setEditingClient] = useState(false);
  const [clientDraft, setClientDraft] = useState({ name: "", address: "", contact: "" });
  const [savingClient, setSavingClient] = useState(false);

  const commitClient = async () => {
    if (!selectedJob) return;
    setSavingClient(true);
    try {
      await updateClient(selectedJob.id, clientDraft);
      await queryClient.invalidateQueries({ queryKey: ["jobs"] });
      setEditingClient(false);
      toast.success("Client details saved — estimate PDF regenerated");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Saving client details failed");
    } finally {
      setSavingClient(false);
    }
  };

  const commitTax = async () => {
    if (!selectedJob) return;
    const pct = parseFloat(taxDraft);
    setEditingTax(false);
    if (isNaN(pct) || pct < 0 || pct > 100) {
      toast.error("Enter a tax percentage between 0 and 100");
      return;
    }
    setSavingTax(true);
    try {
      await updateTaxRate(selectedJob.id, pct / 100);
      await queryClient.invalidateQueries({ queryKey: ["counts", selectedJob.id] });
      toast.success(`Tax updated to ${pct}% — invoice PDF regenerated`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Tax update failed");
    } finally {
      setSavingTax(false);
    }
  };

  if (!selectedJob) {
    return (
      <AppShell>
        <div className="card-surface flex flex-col items-center gap-3 px-6 py-14 text-center">
          <p className="text-sm text-muted-foreground">
            No processed drawing selected — upload and analyze a PDF first.
          </p>
          <Link
            to="/"
            className="rounded-md bg-orange px-4 py-2 text-sm font-medium text-orange-foreground"
          >
            Go to Upload
          </Link>
        </div>
      </AppShell>
    );
  }

  const record = invoice ?? { number: `EST-${selectedJob.id.toUpperCase()}`, generatedAt: "" };
  const validUntil = record.generatedAt
    ? new Date(new Date(record.generatedAt).getTime() + VALID_DAYS * 86_400_000).toISOString()
    : null;
  const client = selectedJob.client ?? {};

  return (
    <AppShell>
      <div className="no-print sticky top-[65px] z-20 -mx-4 mb-6 flex flex-wrap items-center justify-between gap-3 border-b border-border bg-background/90 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6">
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold tracking-tight">Cost Estimate</h1>
          <p className="truncate text-sm text-muted-foreground">
            {record.number} · {projectTitle}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Link
            to="/report"
            className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-colors duration-150 hover:bg-accent/10 hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" /> Back to Report
          </Link>
          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted"
          >
            <Printer className="h-4 w-4" /> Print
          </button>
          <a
            href={invoicePdfUrl(selectedJob.id)}
            target="_blank"
            rel="noreferrer"
            onClick={() => toast.success("Opening estimate PDF")}
            className="inline-flex items-center gap-2 rounded-lg bg-orange px-4 py-2 text-sm font-semibold text-orange-foreground shadow-sm transition-all duration-150 hover:opacity-90"
          >
            <Download className="h-4 w-4" /> Download Estimate (PDF)
          </a>
        </div>
      </div>

      <div className="mx-auto w-full max-w-[800px] rounded-[10px] bg-white text-[#1A1F2B] shadow-[0_10px_40px_rgba(15,20,32,0.18)] ring-1 ring-black/5">
        <div className="p-8 sm:p-10">
          <header className="flex flex-wrap items-start justify-between gap-6 border-b border-[#E5E7EB] pb-6">
            <div className="flex items-center gap-3">
              <img
                src="/aziro-logo.png"
                alt="Aziro"
                className="h-14 w-14 rounded-[10px] border border-[#E5E7EB] bg-white object-contain"
              />
              <div>
                <p className="font-display text-base font-bold">{APP_NAME}</p>
                <p className="text-xs text-[#6B7280]">Construction Symbol Cost Estimator</p>
              </div>
            </div>
            <div className="text-right">
              <p className="font-display text-lg font-bold tracking-tight">COST ESTIMATE</p>
              <p className="mt-1 text-sm font-semibold tabular-nums">{record.number}</p>
              <p className="text-xs text-[#6B7280] tabular-nums">
                Issued {formatDateTime(record.generatedAt)}
              </p>
              <p className="text-xs text-[#6B7280] tabular-nums">
                Valid until {validUntil ? formatDate(validUntil) : "—"}
              </p>
            </div>
          </header>

          {/* ----- Parties: From / Bill To ----- */}
          <section className="grid gap-6 border-b border-[#E5E7EB] py-6 sm:grid-cols-2">
            <div>
              <p className="text-[11px] font-semibold tracking-wide text-[#6B7280] uppercase">
                From
              </p>
              <p className="mt-2 text-sm font-semibold">Aziro</p>
              <p className="mt-0.5 text-sm text-[#4B5563]">{APP_NAME}</p>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <p className="text-[11px] font-semibold tracking-wide text-[#6B7280] uppercase">
                  Bill To
                </p>
                {!editingClient && (
                  <button
                    onClick={() => {
                      setClientDraft({
                        name: client.name ?? "",
                        address: client.address ?? "",
                        contact: client.contact ?? "",
                      });
                      setEditingClient(true);
                    }}
                    disabled={savingClient}
                    aria-label="Edit client details"
                    className="no-print rounded p-0.5 text-[#6B7280] hover:bg-[#F3F4F6] disabled:opacity-50"
                  >
                    {savingClient ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Pencil className="h-3 w-3" />
                    )}
                  </button>
                )}
              </div>
              {editingClient ? (
                <div className="no-print mt-2 flex max-w-[300px] flex-col gap-1.5">
                  {(
                    [
                      ["name", "Client / company name"],
                      ["address", "Address"],
                      ["contact", "Email / phone"],
                    ] as const
                  ).map(([key, placeholder]) => (
                    <input
                      key={key}
                      value={clientDraft[key]}
                      onChange={(e) => setClientDraft((d) => ({ ...d, [key]: e.target.value }))}
                      placeholder={placeholder}
                      className="rounded border border-[#1F4E8C] px-2 py-1 text-sm outline-none"
                    />
                  ))}
                  <div className="flex gap-2">
                    <button
                      onClick={() => void commitClient()}
                      disabled={savingClient}
                      className="rounded bg-[#1F4E8C] px-2.5 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditingClient(false)}
                      className="rounded border border-[#D1D5DB] px-2.5 py-1 text-xs font-medium hover:bg-[#F3F4F6]"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : client.name || client.address || client.contact ? (
                <div className="mt-2 text-sm">
                  {client.name && <p className="font-semibold">{client.name}</p>}
                  {client.address && <p className="mt-0.5 text-[#4B5563]">{client.address}</p>}
                  {client.contact && <p className="mt-0.5 text-[#4B5563]">{client.contact}</p>}
                </div>
              ) : (
                <p className="mt-2 text-sm text-[#9CA3AF] italic">
                  Not specified — click the pencil to add client details
                </p>
              )}
            </div>
          </section>

          <section className="grid gap-6 border-b border-[#E5E7EB] py-6 sm:grid-cols-2">
            <div>
              <p className="text-[11px] font-semibold tracking-wide text-[#6B7280] uppercase">
                Source Drawing
              </p>
              <p className="mt-2 text-sm font-semibold">{metadata["Source File"]}</p>
              <p className="mt-1 text-sm text-[#4B5563]">{metadata["Pages"]} page(s)</p>
            </div>
            <div className="sm:text-right">
              <p className="text-[11px] font-semibold tracking-wide text-[#6B7280] uppercase">
                Project
              </p>
              <p className="mt-2 text-sm font-semibold">{projectTitle}</p>
              <p className="mt-1 text-sm text-[#4B5563]">Date {metadata["Uploaded"]}</p>
            </div>
          </section>

          <div className="overflow-x-auto py-6">
            {lineItems.length === 0 ? (
              <p className="py-6 text-center text-sm text-[#6B7280]">
                No priced symbol lines for this drawing.
                {counts?.note ? ` ${counts.note}` : ""}
              </p>
            ) : (
              <table className="w-full min-w-[620px] text-sm">
                <thead>
                  <tr className="border-b border-[#D1D5DB] text-[11px] tracking-wide text-[#6B7280] uppercase">
                    <th className="py-2 pr-4 text-left font-semibold">Device Type (legend)</th>
                    <th className="py-2 pr-4 text-left font-semibold">Model Class</th>
                    <th className="py-2 pr-4 text-right font-semibold">Qty</th>
                    <th className="py-2 pr-4 text-right font-semibold">Unit Price</th>
                    <th className="py-2 text-right font-semibold">Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {lineItems.map((item, i) => (
                    <tr key={item.id} className={i % 2 === 1 ? "bg-[#F7F8FA]" : undefined}>
                      <td className="py-2.5 pr-4 font-medium">
                        <span className="mr-2 inline-grid h-6 w-6 place-items-center rounded bg-[#1F4E8C]/10 text-[10px] font-bold text-[#1F4E8C] align-middle">
                          {item.symbol}
                        </span>
                        {item.deviceClass}
                      </td>
                      <td className="py-2.5 pr-4 text-[#4B5563]">
                        {item.yoloClasses.join(", ") || "—"}
                      </td>
                      <td className="py-2.5 pr-4 text-right tabular-nums">{item.count}</td>
                      <td className="py-2.5 pr-4 text-right tabular-nums">
                        {currency(item.unitCost)}
                      </td>
                      <td className="py-2.5 text-right font-medium tabular-nums">
                        {currency(item.count * item.unitCost)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="flex justify-end">
            <dl className="w-full max-w-xs space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-[#6B7280]">Subtotal</dt>
                <dd className="tabular-nums">{currency(grandTotal)}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-[#6B7280]">
                  {editingTax ? (
                    <span className="no-print inline-flex items-center gap-1">
                      Tax
                      <input
                        autoFocus
                        value={taxDraft}
                        onChange={(e) => setTaxDraft(e.target.value)}
                        onBlur={() => void commitTax()}
                        onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                        className="w-16 rounded border border-[#1F4E8C] px-1.5 py-0.5 text-sm tabular-nums outline-none"
                      />
                      %
                    </span>
                  ) : (
                    <button
                      onClick={() => {
                        setTaxDraft((taxRate * 100).toFixed(2).replace(/\.?0+$/, ""));
                        setEditingTax(true);
                      }}
                      disabled={savingTax}
                      className="group inline-flex items-center gap-1.5 rounded px-1 py-0.5 hover:bg-[#F3F4F6] disabled:opacity-60"
                      title="Click to change the tax rate"
                    >
                      Tax ({(taxRate * 100).toFixed(2)}%)
                      {savingTax ? (
                        <Loader2 className="no-print h-3 w-3 animate-spin" />
                      ) : (
                        <Pencil className="no-print h-3 w-3 opacity-0 group-hover:opacity-100" />
                      )}
                    </button>
                  )}
                </dt>
                <dd className="tabular-nums">{currency(taxAmount)}</dd>
              </div>
              <div className="flex justify-between border-t-2 border-[#1A1F2B] pt-2 text-base font-bold">
                <dt>Grand Total</dt>
                <dd className="tabular-nums">{currency(invoiceTotal)}</dd>
              </div>
            </dl>
          </div>

          <footer className="mt-10 border-t border-[#E5E7EB] pt-4 text-xs text-[#9CA3AF]">
            <p>
              All amounts shown in {currencyCode} · Estimate valid for {VALID_DAYS} days from issue
              date · This is a computer-generated cost estimate, not a demand for payment.
            </p>
            <p className="mt-1">
              Aziro · {APP_NAME} · {record.number} · Issued {formatDateTime(record.generatedAt)}
            </p>
          </footer>
        </div>
      </div>
    </AppShell>
  );
}
