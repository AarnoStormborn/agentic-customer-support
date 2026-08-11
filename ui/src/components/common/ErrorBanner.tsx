/** common/ErrorBanner.tsx — inline error with optional retry (ui.md §2.6). */
import { AlertTriangle } from "lucide-react";

export function ErrorBanner({
  title = "Something went wrong",
  message,
  retryLabel = "Retry",
  onRetry,
  className = "",
}: {
  title?: string;
  message?: string;
  retryLabel?: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={`flex items-start gap-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2.5 ${className}`}
    >
      <AlertTriangle size={16} className="mt-0.5 shrink-0 text-danger" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-text">{title}</p>
        {message && <p className="mt-0.5 break-words text-xs text-text-dim">{message}</p>}
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="shrink-0 rounded px-2 py-1 text-xs font-medium text-danger hover:bg-danger/15"
        >
          {retryLabel}
        </button>
      )}
    </div>
  );
}
