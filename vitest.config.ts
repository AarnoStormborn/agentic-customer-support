import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    testTimeout: 30_000,
    hookTimeout: 30_000,
    // DB-dependent integration tests guard themselves via env flags (see tests/retrieval.test.ts)
    env: {
      // Deterministic small rate limit so the 429 test is fast
      RATE_CHAT_MAX: "10",
    },
  },
});
