/**
 * src/server/routes/data.ts — read endpoints the UI needs (Phase 5b.1).
 *
 *   GET /api/models            available models ("provider/id") for the model picker
 *   GET /api/tickets           searchable/paginated tickets (server-side search)
 *   GET /api/tickets/:id       single ticket (TicketDrawer)
 *   GET /api/manuals           manual list (+ chunk counts)
 *   GET /api/manuals/:id/chunks  ordered chunks of one manual (manual browser)
 *
 * All queries are parameterized and read-only (shared pool from src/db/pool.ts).
 */
import type { FastifyPluginAsync } from "fastify";
import { env } from "../../config/env.js";
import { getPool } from "../../db/pool.js";
import { listAvailableModels } from "../../runtime/model.js";

const DEFAULT_PAGE_SIZE = 25;
const MAX_PAGE_SIZE = 100;

interface TicketQuery {
  q?: string;
  status?: string;
  priority?: string;
  source?: string;
  page?: string;
  pageSize?: string;
}

export const dataRoutes: FastifyPluginAsync = async (app) => {
  // --- models -----------------------------------------------------------
  app.get("/api/models", async () => {
    try {
      const models = await listAvailableModels();
      return { models, default: env.PI_MODEL || models[0] || null };
    } catch (err) {
      app.log.warn({ err }, "listAvailableModels failed");
      return { models: [], default: env.PI_MODEL || null, error: "model_runtime_unavailable" };
    }
  });

  // --- tickets ----------------------------------------------------------
  app.get<{ Querystring: TicketQuery }>("/api/tickets", async (request, reply) => {
    const { q, status, priority, source } = request.query;
    const page = Math.max(1, Number(request.query.page) || 1);
    const pageSize = Math.min(MAX_PAGE_SIZE, Math.max(1, Number(request.query.pageSize) || DEFAULT_PAGE_SIZE));
    const offset = (page - 1) * pageSize;

    const where: string[] = [];
    const params: unknown[] = [];
    let n = 1;

    if (q && q.trim()) {
      const like = `%${q.trim()}%`;
      where.push(`(product_purchased ILIKE $${n} OR ticket_subject ILIKE $${n} OR complaint_narrative ILIKE $${n} OR ticket_type ILIKE $${n})`);
      params.push(like);
      n += 1;
    }
    if (status) {
      where.push(`status ILIKE $${n}`);
      params.push(status);
      n += 1;
    }
    if (priority) {
      where.push(`ticket_priority ILIKE $${n}`);
      params.push(priority);
      n += 1;
    }
    if (source) {
      where.push(`source = $${n}`);
      params.push(source);
      n += 1;
    }

    const whereSql = where.length > 0 ? `WHERE ${where.join(" AND ")}` : "";

    const pool = getPool();
    const [rowsRes, countRes] = await Promise.all([
      pool.query(
        `SELECT ticket_id, source, customer_name, product_purchased, date_of_purchase,
                ticket_type, ticket_priority, ticket_channel, ticket_subject,
                complaint_narrative, company, status, is_synthetic, created_at
         FROM tickets ${whereSql}
         ORDER BY ticket_id DESC
         LIMIT $${n} OFFSET $${n + 1}`,
        [...params, pageSize, offset],
      ),
      pool.query(`SELECT COUNT(*)::int AS total FROM tickets ${whereSql}`, params),
    ]);

    return {
      rows: rowsRes.rows,
      total: countRes.rows[0]?.total ?? 0,
      page,
      pageSize,
    };
  });

  app.get<{ Params: { id: string } }>("/api/tickets/:id", async (request, reply) => {
    const id = Number(request.params.id);
    if (!Number.isInteger(id) || id <= 0) {
      return reply.code(400).send({ error: "invalid_id", message: "ticket id must be a positive integer" });
    }
    const pool = getPool();
    const res = await pool.query("SELECT * FROM tickets WHERE ticket_id = $1", [id]);
    if (res.rows.length === 0) {
      return reply.code(404).send({ error: "ticket_not_found", message: `No ticket #${id}` });
    }
    return res.rows[0];
  });

  // --- manuals ----------------------------------------------------------
  app.get("/api/manuals", async () => {
    const pool = getPool();
    const res = await pool.query(
      `SELECT d.doc_id, d.doc_name, d.file_path, d.doc_type, d.created_at,
              COUNT(c.chunk_id)::int AS chunk_count
       FROM documents d
       LEFT JOIN document_chunks c ON c.doc_id = d.doc_id
       GROUP BY d.doc_id
       ORDER BY d.created_at DESC`,
    );
    return { manuals: res.rows };
  });

  app.get<{ Params: { id: string }; Querystring: { limit?: string } }>(
    "/api/manuals/:id/chunks",
    async (request, reply) => {
      const id = Number(request.params.id);
      if (!Number.isInteger(id) || id <= 0) {
        return reply.code(400).send({ error: "invalid_id", message: "manual id must be a positive integer" });
      }
      const limit = Math.min(500, Math.max(1, Number(request.query.limit) || 200));
      const pool = getPool();
      const res = await pool.query(
        `SELECT chunk_id, doc_id, chunk_index, chunk_text, section, heading_path,
                page_start, page_end
         FROM document_chunks
         WHERE doc_id = $1
         ORDER BY chunk_index
         LIMIT $2`,
        [id, limit],
      );
      if (res.rows.length === 0) {
        const doc = await pool.query("SELECT doc_id FROM documents WHERE doc_id = $1", [id]);
        if (doc.rows.length === 0) {
          return reply.code(404).send({ error: "manual_not_found", message: `No manual #${id}` });
        }
      }
      return { manualId: id, chunks: res.rows };
    },
  );
};
