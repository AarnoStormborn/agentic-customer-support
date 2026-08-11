/** common/EmptyState.tsx — welcome hero + empty views (ui.md §2.6). */
import type { ReactNode } from "react";

export function EmptyState({
  icon,
  title,
  description,
  actions,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
      {icon && <div className="text-accent/70">{icon}</div>}
      <h2 className="text-xl font-semibold text-text">{title}</h2>
      {description && <p className="max-w-md text-sm text-text-dim">{description}</p>}
      {actions && <div className="mt-2 flex flex-wrap items-center justify-center gap-2">{actions}</div>}
    </div>
  );
}
