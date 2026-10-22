import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import pg from 'pg';
import { DomainError, initialState } from '@restoradar/domain';
import type { RestaurantState } from '@restoradar/domain';
export const { Pool } = pg;
export type Client = pg.PoolClient;
export type DatabasePool = pg.Pool;

export function createPool(connectionString = process.env.DATABASE_URL) {
  if (!connectionString) throw new Error('DATABASE_URL is required. See .env.example.');
  return new Pool({ connectionString, max: 10 });
}
export async function migrate(pool: DatabasePool) {
  const sql = await readFile(new URL('../migrations/001_foundation.sql', import.meta.url), 'utf8');
  const checksum = createHash('sha256').update(sql).digest('hex');
  const c = await pool.connect();
  try {
    await c.query('BEGIN');
    await c.query('SELECT pg_advisory_xact_lock(108294601)');
    await c.query(
      'CREATE TABLE IF NOT EXISTS schema_migrations (version integer PRIMARY KEY, checksum text NOT NULL)',
    );
    const existing = await c.query('SELECT checksum FROM schema_migrations WHERE version=1');
    if (existing.rowCount) {
      if (existing.rows[0].checksum !== checksum)
        throw new Error('Applied migration checksum mismatch. Add a new migration instead.');
    } else {
      await c.query(sql);
      await c.query('INSERT INTO schema_migrations VALUES (1,$1)', [checksum]);
    }
    await c.query('COMMIT');
  } catch (e) {
    await c.query('ROLLBACK');
    throw e;
  } finally {
    c.release();
  }
}

const collections = {
  tables: 'dining_tables',
  sessions: 'occupancy_sessions',
  assignments: 'session_assignments',
  cleaningCycles: 'cleaning_cycles',
  tasks: 'waiter_tasks',
  attempts: 'assignment_attempts',
  alerts: 'manager_alerts',
  adjustments: 'session_adjustments',
  monitoringGaps: 'monitoring_gaps',
  events: 'domain_events',
  audit: 'audit_log',
} as const;
const appendOnly = new Set<string>(['adjustments', 'monitoringGaps', 'events', 'audit']);

export async function loadState(c: Client, restaurantId: string): Promise<RestaurantState> {
  const r = await c.query('SELECT id,name,version,policy FROM restaurants WHERE id=$1', [
    restaurantId,
  ]);
  if (!r.rowCount) throw new DomainError('NOT_FOUND', 'المطعم غير موجود', 404);
  const row = r.rows[0];
  const state = initialState(row.id, row.name);
  state.version = row.version;
  state.policy = row.policy;
  for (const [key, table] of Object.entries(collections) as [keyof typeof collections, string][]) {
    const result = await c.query(
      `SELECT data FROM ${table} WHERE restaurant_id=$1 ${key === 'events' ? 'ORDER BY sequence' : ''}`,
      [restaurantId],
    );
    Object.assign(state, { [key]: result.rows.map((r) => r.data) });
  }
  return state;
}

export async function persistState(c: Client, previous: RestaurantState, next: RestaurantState) {
  if (previous === next) return;
  await c.query('UPDATE restaurants SET version=$2,policy=$3 WHERE id=$1', [
    next.id,
    next.version,
    next.policy,
  ]);
  for (const [key, table] of Object.entries(collections) as [keyof typeof collections, string][]) {
    const old = new Map(previous[key].map((item) => [item.id, JSON.stringify(item)]));
    for (const item of next[key]) {
      const encoded = JSON.stringify(item);
      if (old.get(item.id) === encoded) continue;
      if (appendOnly.has(key) && old.has(item.id))
        throw new Error(`Attempt to mutate append-only ${key}`);
      if (key === 'assignments' && 'sessionId' in item && 'startedAt' in item) {
        await c.query(
          `INSERT INTO session_assignments(id,restaurant_id,session_id,table_id,started_at,ended_at,data)
          VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(id) DO UPDATE SET started_at=EXCLUDED.started_at,ended_at=EXCLUDED.ended_at,data=EXCLUDED.data`,
          [item.id, next.id, item.sessionId, item.tableId, item.startedAt, item.endedAt, encoded],
        );
      } else if (key === 'events' && 'sequence' in item) {
        await c.query(
          'INSERT INTO domain_events(id,restaurant_id,sequence,data) VALUES($1,$2,$3,$4)',
          [item.id, next.id, item.sequence, encoded],
        );
        await c.query('INSERT INTO synchronization_outbox(event_id,restaurant_id) VALUES($1,$2)', [
          item.id,
          next.id,
        ]);
      } else if (appendOnly.has(key)) {
        await c.query(`INSERT INTO ${table}(id,restaurant_id,data) VALUES($1,$2,$3)`, [
          item.id,
          next.id,
          encoded,
        ]);
      } else {
        await c.query(
          `INSERT INTO ${table}(id,restaurant_id,data) VALUES($1,$2,$3) ON CONFLICT(id) DO UPDATE SET data=EXCLUDED.data`,
          [item.id, next.id, encoded],
        );
      }
    }
  }
}

export class Repository {
  constructor(public pool: DatabasePool) {}
  async read(restaurantId: string) {
    const c = await this.pool.connect();
    try {
      await c.query('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY');
      const state = await loadState(c, restaurantId);
      await c.query('COMMIT');
      return state;
    } catch (e) {
      await c.query('ROLLBACK');
      throw e;
    } finally {
      c.release();
    }
  }
  async update(
    restaurantId: string,
    fn: (state: RestaurantState, c: Client) => RestaurantState | Promise<RestaurantState>,
    receipt?: { commandId: string; fingerprint: string },
  ) {
    const c = await this.pool.connect();
    try {
      await c.query('BEGIN');
      const locked = await c.query('SELECT id FROM restaurants WHERE id=$1 FOR UPDATE', [
        restaurantId,
      ]);
      if (!locked.rowCount) throw new DomainError('NOT_FOUND', 'المطعم غير موجود', 404);
      if (receipt) {
        const seen = await c.query(
          'SELECT fingerprint FROM command_receipts WHERE restaurant_id=$1 AND command_id=$2',
          [restaurantId, receipt.commandId],
        );
        if (seen.rowCount) {
          if (seen.rows[0].fingerprint !== receipt.fingerprint)
            throw new DomainError('IDEMPOTENCY_CONFLICT', 'معرّف العملية مستخدم لطلب مختلف');
          await c.query('COMMIT');
          return { replayed: true };
        }
      }
      const previous = await loadState(c, restaurantId);
      const next = await fn(previous, c);
      await persistState(c, previous, next);
      if (receipt)
        await c.query(
          'INSERT INTO command_receipts(restaurant_id,command_id,fingerprint,applied_version) VALUES($1,$2,$3,$4)',
          [restaurantId, receipt.commandId, receipt.fingerprint, next.version],
        );
      await c.query('COMMIT');
      return { replayed: false };
    } catch (e) {
      await c.query('ROLLBACK');
      throw e;
    } finally {
      c.release();
    }
  }
}
