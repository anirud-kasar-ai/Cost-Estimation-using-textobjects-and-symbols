import { useState } from "react";
import { classAbbrev, pricingSymbolUrl } from "@/lib/api";

/** Real sample of a device symbol (legend glyph / detection crop served by the
 *  backend). Hover to enlarge; falls back to a letter badge when no drawing or
 *  dataset contains the symbol yet. */
export function SymbolThumb({ deviceKey }: { deviceKey: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <span
        className="grid h-10 w-10 place-items-center rounded bg-brand/10 text-xs font-bold text-brand"
        title="No sample of this symbol yet — process a drawing containing it"
      >
        {classAbbrev(deviceKey)}
      </span>
    );
  }
  return (
    <span className="group relative inline-block">
      <img
        src={pricingSymbolUrl(deviceKey)}
        alt={`${deviceKey} symbol`}
        loading="lazy"
        onError={() => setFailed(true)}
        className="h-10 w-10 rounded border border-border bg-white object-contain p-0.5"
      />
      {/* Enlarged preview on hover */}
      <span className="pointer-events-none absolute top-1/2 left-12 z-30 hidden -translate-y-1/2 group-hover:block">
        <img
          src={pricingSymbolUrl(deviceKey)}
          alt=""
          className="h-36 w-36 rounded-md border border-border bg-white object-contain p-2 shadow-lg"
        />
      </span>
    </span>
  );
}
