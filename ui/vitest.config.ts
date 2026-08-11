import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// UI unit tests: fast, no network, no database. happy-dom gives us a DOM for
// component tests; fetch / EventSource are mocked in lib tests.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "happy-dom",
    globals: true, // @testing-library/react auto-cleanup relies on global afterEach
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
