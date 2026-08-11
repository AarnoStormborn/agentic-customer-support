/**
 * Model selection helpers.
 *
 * The SDK's Model type lives in @earendil-works/pi-ai (a transitive dep, not
 * directly importable), so we derive it from ModelRuntime.getAvailable()'s
 * return type instead of importing the type package.
 */

import type { ModelRuntime } from "@earendil-works/pi-coding-agent";

export type AvailableModel = Awaited<ReturnType<ModelRuntime["getAvailable"]>>[number];
export type ModelLike = AvailableModel;

/** Supervisor preference order (prefix match on "provider/id"). */
const PREFERRED_SUPERVISOR = [
  "anthropic/claude-sonnet-4-5",
  "anthropic/claude-haiku-4-5",
  "openai/gpt-5",
  "google/gemini-3-pro",
];

/** Specialist (child session) preference: cheap + fast. */
const PREFERRED_SPECIALIST = [
  "anthropic/claude-haiku-4-5",
  "anthropic/claude-3-5-haiku",
  "openai/gpt-5-mini",
  "openai/gpt-4.1-mini",
  "google/gemini-2.5-flash",
  "google/gemini-3-flash",
];

export function modelId(m: ModelLike): string {
  return `${m.provider}/${m.id}`;
}

function pick(
  available: readonly ModelLike[],
  explicit: string | undefined,
  envVar: string | undefined,
  preferred: string[],
  fallback: ModelLike,
): ModelLike {
  const requested = explicit ?? (envVar && envVar.trim() !== "" ? envVar.trim() : undefined);
  if (requested) {
    // Exact "provider/id" match first, then bare-id match (e.g. "deepseek-v4-flash"
    // → "opencode-go/deepseek-v4-flash").
    const exact = available.find((m) => modelId(m) === requested);
    if (exact) return exact;
    const byId = available.filter((m) => m.id === requested);
    if (byId.length === 1) return byId[0]!;
    if (byId.length > 1) {
      throw new Error(
        `Model '${requested}' is ambiguous (matches ${byId.map(modelId).join(", ")}) — use provider/model form.`,
      );
    }
    throw new Error(
      `Model '${requested}' not found among available models: ${available.map(modelId).join(", ")}`,
    );
  }
  return (
    available.find((m) => preferred.some((p) => modelId(m).startsWith(p))) ??
    fallback
  );
}

/**
 * Resolve the supervisor model: explicit opts.model → PI_MODEL env → preferred
 * list → first available. Throws when nothing is available.
 */
export async function resolveSupervisorModel(
  modelRuntime: ModelRuntime,
  explicit?: string,
): Promise<ModelLike> {
  const available = await modelRuntime.getAvailable();
  if (available.length === 0) {
    throw new Error(
      "No authenticated models found. Configure ~/.pi/agent/auth.json or a provider API key in .env.",
    );
  }
  return pick(available, explicit, process.env.PI_MODEL, PREFERRED_SUPERVISOR, available[0]!);
}

/**
 * Resolve the specialist (child) model: PI_SPECIALIST_MODEL env → cheap-model
 * preference → the supervisor model.
 */
export async function resolveSpecialistModel(
  modelRuntime: ModelRuntime,
  supervisor: ModelLike,
): Promise<ModelLike> {
  const available = await modelRuntime.getAvailable();
  if (available.length === 0) return supervisor;
  return pick(available, undefined, process.env.PI_SPECIALIST_MODEL, PREFERRED_SPECIALIST, supervisor);
}
