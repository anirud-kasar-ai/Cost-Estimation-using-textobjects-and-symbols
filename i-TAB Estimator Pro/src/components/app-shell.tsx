import { Link, useRouterState } from "@tanstack/react-router";
import { useState, type ReactNode } from "react";
import {
  BarChart3,
  ChevronRight,
  FileText,
  GitCompareArrows,
  Home,
  Menu,
  Moon,
  Receipt,
  Sun,
  Tags,
  UploadCloud,
  X,
} from "lucide-react";
import { useTheme } from "@/lib/theme";
import { CURRENCIES, useEstimator, type CurrencyCode } from "@/lib/estimator-data";
import { cn } from "@/lib/utils";

export const APP_NAME = "Cost Estimate Using Site Construction Drawing";

const nav = [
  {
    to: "/",
    label: "Upload Drawings",
    icon: UploadCloud,
    description: "Upload a construction drawing PDF and track its processing",
  },
  {
    to: "/metrics",
    label: "Model Performance",
    icon: BarChart3,
    description: "Detection accuracy, class distribution and model inventory",
  },
  {
    to: "/pricing",
    label: "Pricing Catalog",
    icon: Tags,
    description: "Unit costs for every device class used in estimates",
  },
  {
    to: "/report",
    label: "Project Report",
    icon: FileText,
    description: "Detected symbols, legend and costed estimate per project",
  },
  {
    to: "/compare",
    label: "Project Comparison",
    icon: GitCompareArrows,
    description: "Side-by-side symbol counts and costs of two projects",
  },
  {
    to: "/invoice",
    label: "Invoice",
    icon: Receipt,
    description: "Printable invoice with subtotal, tax and grand total",
  },
] as const;

/** Breadcrumb label for each route. */
const pageLabels: Record<string, string> = {
  "/": "Upload Drawings",
  "/pricing": "Pricing Catalog",
  "/metrics": "Model Performance",
  "/report": "Project Report",
  "/compare": "Project Comparison",
  "/invoice": "Invoice",
};

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <div className="flex h-full flex-col bg-sidebar text-sidebar-foreground">
      <nav className="flex-1 space-y-1 p-3">
        {nav.map((item) => {
          const active = pathname === item.to;
          return (
            <Link
              key={item.to}
              to={item.to}
              onClick={onNavigate}
              title={item.description}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-brand text-brand-foreground"
                  : "text-sidebar-muted hover:bg-sidebar-accent hover:text-sidebar-foreground",
              )}
            >
              <item.icon className="h-4 w-4 shrink-0" />
              <span className="truncate">{item.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-sidebar-border p-4 text-[11px] text-sidebar-muted">
        YOLO symbol detection · auto model selection
      </div>
    </div>
  );
}

function Breadcrumbs() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const label = pageLabels[pathname] ?? pathname.replace("/", "");
  return (
    <nav aria-label="Breadcrumb" className="no-print flex min-w-0 items-center gap-1.5 text-sm">
      <Link
        to="/"
        className="flex shrink-0 items-center gap-1 text-muted-foreground transition-colors hover:text-foreground"
      >
        <Home className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">Home</span>
      </Link>
      {pathname !== "/" && (
        <>
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/60" />
          <span className="truncate font-medium text-foreground">{label}</span>
        </>
      )}
    </nav>
  );
}

function CurrencySelect() {
  const { currencyCode, setCurrencyCode } = useEstimator();
  return (
    <label className="no-print flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
      <span className="hidden sm:inline">Currency</span>
      <select
        value={currencyCode}
        onChange={(e) => setCurrencyCode(e.target.value as CurrencyCode)}
        aria-label="Display currency"
        className="rounded-md border border-border bg-card px-2 py-1.5 text-xs font-medium text-foreground outline-none focus:border-orange focus:ring-1 focus:ring-orange"
      >
        {CURRENCIES.map((c) => (
          <option key={c.code} value={c.code}>
            {c.code} ({c.symbol})
          </option>
        ))}
      </select>
    </label>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme();
  const [open, setOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background">
      {/* Single common header — full width, fixed on top, identical on every page */}
      <header className="no-print fixed top-0 right-0 left-0 z-40 border-b border-border bg-background/95 backdrop-blur">
        <div className="grid h-16 grid-cols-[minmax(0,1fr)_auto] items-center gap-4 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <button
              className="shrink-0 rounded-md border border-border p-2 lg:hidden"
              onClick={() => setOpen(true)}
              aria-label="Open menu"
            >
              <Menu className="h-4 w-4" />
            </button>
            <img
              src="/aziro-logo.png"
              alt="Aziro"
              className="h-12 w-12 shrink-0 rounded-md bg-white object-contain"
            />
            <h1 className="truncate text-lg font-bold tracking-tight sm:text-xl">{APP_NAME}</h1>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <CurrencySelect />
            <button
              onClick={toggle}
              aria-label="Toggle theme"
              className="shrink-0 rounded-md border border-border p-2 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </header>

      {/* Desktop sidebar — starts right below the shared header */}
      <aside className="hidden w-60 shrink-0 lg:block">
        <div className="fixed top-[65px] bottom-0 left-0 w-60">
          <SidebarContent />
        </div>
      </aside>

      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <div className="absolute inset-y-0 left-0 w-64">
            <SidebarContent onNavigate={() => setOpen(false)} />
          </div>
          <button
            aria-label="Close menu"
            onClick={() => setOpen(false)}
            className="absolute top-4 right-4 rounded-md bg-card p-2 text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Spacer matching the fixed header height (64px bar + border) */}
        <div className="h-[65px] shrink-0" aria-hidden />

        <div className="no-print mx-auto w-full max-w-7xl px-4 pt-4 sm:px-6">
          <Breadcrumbs />
        </div>

        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">{children}</main>

        {/* Common footer */}
        <footer className="no-print mt-auto border-t border-border bg-card/60">
          <div className="mx-auto flex w-full max-w-7xl items-center justify-center gap-3 px-4 py-5 sm:px-6">
            <img
              src="/aziro-logo.png"
              alt="Aziro"
              className="h-10 w-10 shrink-0 rounded-lg bg-white object-contain"
            />
            <span className="text-sm font-bold text-brand dark:text-primary">AI</span>
            <p className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} Aziro. All rights reserved.
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-8 grid grid-cols-[minmax(0,1fr)_auto] items-start gap-4 sm:flex sm:flex-wrap sm:items-center sm:justify-between">
      <div className="min-w-0">
        <h1 className="truncate text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
