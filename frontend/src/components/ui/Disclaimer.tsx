import { AlertTriangle } from "lucide-react";

export function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300 ${compact ? "text-xs" : "text-sm"}`}>
      <AlertTriangle size={compact ? 14 : 18} className="mt-0.5 shrink-0" />
      <p>
        This system is an <strong>AI-assisted research and educational tool</strong>, not a
        medical diagnosis. Predictions may be inaccurate and should never replace evaluation
        by a qualified eye-care professional.
      </p>
    </div>
  );
}
