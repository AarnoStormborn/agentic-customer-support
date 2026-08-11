/** Debug: inspect tool_execution_end result shape for sources collection. */
import "dotenv/config";
import { createSupportRuntime } from "../runtime/index.js";

const rt = await createSupportRuntime({});
rt.subscribe((e: any) => {
  if (e.type === "tool_execution_end") {
    const r = e.result;
    console.log("tool_execution_end:", e.toolName);
    console.log("  result keys:", r ? Object.keys(r) : "NO RESULT");
    if (r?.details) console.log("  details keys:", Object.keys(r.details), "| sources:", JSON.stringify(r.details.sources)?.slice(0, 200));
  }
  if (e.type === "agent_settled") {
    console.log("agent_settled sources:", JSON.stringify((e as any).sources)?.slice(0, 300));
  }
});
await rt.prompt("Use kb_search with query 'lg tv wifi reset' and report the sources");
rt.dispose();
console.log("DEBUG DONE");
