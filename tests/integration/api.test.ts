import { randomUUID, randomBytes } from 'node:crypto';
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { Pool, Repository, migrate, type DatabasePool } from '@restoradar/database';
import type { Command, Dashboard } from '@restoradar/contracts';
import { processDeadlines } from '@restoradar/domain';
import { buildApp } from '../../apps/edge-api/src/app.ts';
import { hashPassword } from '../../apps/edge-api/src/auth.ts';

if (!process.env.TEST_DATABASE_URL)
  throw new Error(
    'TEST_DATABASE_URL is required. Integration tests create and remove only their own isolated schema.',
  );
const schema = `test_${randomUUID().replaceAll('-', '')}`;
const admin = new Pool({ connectionString: process.env.TEST_DATABASE_URL });
let pool: DatabasePool, app: Awaited<ReturnType<typeof buildApp>>, repository: Repository;
const password = randomBytes(24).toString('hex');
const identities = Object.fromEntries(
  ['owner', 'manager', 'manager2', 'waiter', 'waiter2'].map((k) => [
    k,
    { id: randomUUID(), cookie: '' },
  ]),
);
let restaurantId: string, tableId: string;
const policy = {
  CUSTOMER_RECEPTION: { acceptanceSeconds: 10, completionSeconds: 20 },
  TABLE_CLEANING: { acceptanceSeconds: 15, completionSeconds: 30 },
};
function command(cmd: Command, who = 'owner', commandId = randomUUID()) {
  return app.inject({
    method: 'POST',
    url: `/api/restaurants/${restaurantId}/commands`,
    headers: { cookie: identities[who].cookie },
    payload: { commandId, command: cmd },
  });
}
async function getDashboard(who = 'owner', id = restaurantId) {
  return app.inject({
    method: 'GET',
    url: `/api/restaurants/${id}/dashboard`,
    headers: { cookie: identities[who].cookie },
  });
}
async function snapshot(who = 'owner'): Promise<Dashboard> {
  return (await getDashboard(who)).json();
}
async function seat() {
  const now = Date.now() - 100;
  const response = await command({
    type: 'simulation.transition',
    tableId,
    occupancy: 'OCCUPIED',
    observedAt: new Date(now - 5000).toISOString(),
    confirmedAt: new Date(now).toISOString(),
  });
  expect(response.statusCode).toBe(200);
  return snapshot();
}
beforeAll(async () => {
  await admin.query(`CREATE SCHEMA ${schema}`);
  pool = new Pool({
    connectionString: process.env.TEST_DATABASE_URL,
    options: `-c search_path=${schema},public`,
    max: 10,
  });
  await migrate(pool);
  await migrate(pool); // Installation must be repeatable.
  repository = new Repository(pool);
  const hashed = await hashPassword(password);
  for (const [name, user] of Object.entries(identities))
    await pool.query('INSERT INTO users(id,email,name,password_hash) VALUES($1,$2,$3,$4)', [
      user.id,
      `${name}@test.invalid`,
      name,
      hashed,
    ]);
  app = await buildApp(pool, { simulatorEnabled: true });
  for (const [name, user] of Object.entries(identities)) {
    const response = await app.inject({
      method: 'POST',
      url: '/api/auth/login',
      payload: { email: `${name}@test.invalid`, password },
    });
    expect(response.statusCode).toBe(200);
    user.cookie = String(response.headers['set-cookie']).split(';')[0];
    expect(response.headers['set-cookie']).toContain('HttpOnly');
    expect(response.headers['set-cookie']).toContain('SameSite=Strict');
  }
});
beforeEach(async () => {
  restaurantId = randomUUID();
  await pool.query('INSERT INTO restaurants(id,name) VALUES($1,$2)', [
    restaurantId,
    'Integration restaurant',
  ]);
  for (const [name, user] of Object.entries(identities))
    await pool.query('INSERT INTO memberships(restaurant_id,user_id,role) VALUES($1,$2,$3)', [
      restaurantId,
      user.id,
      name === 'owner' ? 'OWNER' : name.startsWith('manager') ? 'MANAGER' : 'WAITER',
    ]);
  expect((await command({ type: 'table.create', name: 'Table 1' })).statusCode).toBe(200);
  tableId = (await snapshot()).tables[0].id;
});
afterAll(async () => {
  if (app) await app.close();
  if (pool) await pool.end();
  await admin.query(`DROP SCHEMA IF EXISTS ${schema} CASCADE`);
  await admin.end();
});

describe('real PostgreSQL and authenticated API', () => {
  it('shuts down the real server gracefully, releasing its worker before closing the database pool', async () => {
    const url = new URL(process.env.TEST_DATABASE_URL!);
    url.searchParams.set('options', `-c search_path=${schema},public`);
    const child = spawn(process.execPath, ['--import', 'tsx', 'apps/edge-api/src/main.ts'], {
      cwd: process.cwd(),
      env: { ...process.env, DATABASE_URL: url.toString(), HOST: '127.0.0.1', PORT: '0' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const watchdog = setTimeout(() => child.kill('SIGKILL'), 10000);
    const exited = new Promise<{ code: number | null; signal: string | null }>((resolve) =>
      child.once('exit', (code, signal) => resolve({ code, signal })),
    );
    try {
      await new Promise<void>((resolve, reject) => {
        let output = '';
        child.stdout.on('data', (data) => {
          output += data.toString();
          if (output.includes('Server listening')) resolve();
        });
        child.once('error', reject);
        child.once('exit', () => reject(new Error('Server exited before becoming ready')));
      });
      const forcedStop = setTimeout(() => child.kill('SIGKILL'), 5000);
      try {
        child.kill('SIGTERM');
        const result = await exited;
        expect(result).toEqual({ code: 0, signal: null });
      } finally {
        clearTimeout(forcedStop);
      }
    } finally {
      clearTimeout(watchdog);
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
    }
  });
  it('requires authentication and enforces tenant isolation', async () => {
    expect(
      (await app.inject({ url: `/api/restaurants/${restaurantId}/dashboard` })).statusCode,
    ).toBe(401);
    const other = randomUUID();
    await pool.query('INSERT INTO restaurants(id,name) VALUES($1,$2)', [other, 'Other tenant']);
    expect((await getDashboard('owner', other)).statusCode).toBe(403);
    expect(
      (
        await app.inject({
          method: 'POST',
          url: `/api/restaurants/${other}/commands`,
          headers: { cookie: identities.owner.cookie },
          payload: {
            commandId: randomUUID(),
            command: { type: 'table.create', name: 'Cross-tenant' },
          },
        })
      ).statusCode,
    ).toBe(403);
  });
  it('does not leak password hashes, tokens, staff history or other waiter tasks', async () => {
    await command({ type: 'policy.configure', policy });
    const data = await seat();
    await command({
      type: 'task.assign',
      kind: 'CUSTOMER_RECEPTION',
      sourceId: data.sessions[0].id,
      waiterId: identities.waiter.id,
      expectedVersion: 0,
    });
    const other = await snapshot('waiter2');
    expect(other.tasks).toHaveLength(0);
    expect(other.attempts).toHaveLength(0);
    expect(other.alerts).toHaveLength(0);
    expect(other.audit).toHaveLength(0);
    expect(other.sessions).toHaveLength(0);
    expect(other.members.map((m) => m.id)).toEqual([identities.waiter2.id]);
    expect(JSON.stringify(await snapshot())).not.toMatch(
      /password_hash|token_hash|rr_session|scrypt:/,
    );
  });
  it('rejects non-owner settings/user creation and forged waiter management commands', async () => {
    expect((await command({ type: 'policy.configure', policy }, 'manager')).statusCode).toBe(403);
    expect(
      (
        await command(
          {
            type: 'table.correct',
            tableId,
            expectedVersion: 0,
            occupancy: 'OCCUPIED',
            reason: 'Forged request',
          },
          'waiter',
        )
      ).statusCode,
    ).toBe(403);
    expect(
      (
        await app.inject({
          method: 'POST',
          url: `/api/restaurants/${restaurantId}/users`,
          headers: { cookie: identities.manager.cookie },
          payload: {},
        })
      ).statusCode,
    ).toBe(403);
  });
  it('first concurrent manager acknowledgement wins atomically and preserves identity', async () => {
    const data = await seat(),
      id = data.alerts[0].id;
    const responses = await Promise.all([
      command({ type: 'alert.acknowledge', alertId: id }, 'manager'),
      command({ type: 'alert.acknowledge', alertId: id }, 'manager2'),
    ]);
    expect(responses.map((r) => r.statusCode)).toEqual([200, 200]);
    const a = (await snapshot()).alerts[0];
    expect([identities.manager.id, identities.manager2.id]).toContain(a.acknowledgedBy);
    const recorded = a.acknowledgedBy;
    await command({ type: 'alert.acknowledge', alertId: id });
    expect((await snapshot()).alerts[0].acknowledgedBy).toBe(recorded);
    expect((await snapshot()).audit.filter((a) => a.action === 'alert.acknowledge')).toHaveLength(
      1,
    );
  });
  it('replays simultaneous commands exactly once and rejects key reuse with a different payload', async () => {
    const id = randomUUID(),
      cmd: Command = { type: 'table.create', name: 'Idempotent table' };
    const results = await Promise.all([command(cmd, 'owner', id), command(cmd, 'owner', id)]);
    expect(results.map((r) => r.statusCode)).toEqual([200, 200]);
    expect(results.map((r) => r.json().replayed).sort()).toEqual([false, true]);
    expect((await snapshot()).tables.filter((t) => t.name === 'Idempotent table')).toHaveLength(1);
    expect(
      (await command({ type: 'table.create', name: 'Different payload' }, 'owner', id)).statusCode,
    ).toBe(409);
  });
  it('rejects one of two stale concurrent manual corrections', async () => {
    const cmd: Command = {
      type: 'table.correct',
      tableId,
      expectedVersion: 0,
      occupancy: 'OCCUPIED',
      reason: 'Manager confirmed guests',
    };
    const results = await Promise.all([command(cmd, 'manager'), command(cmd, 'manager2')]);
    expect(results.map((r) => r.statusCode).sort()).toEqual([200, 409]);
    expect((await snapshot()).sessions).toHaveLength(1);
  });
  it('requires owner-entered policy and allows only one initial assignment during a race', async () => {
    const data = await seat();
    const cmd: Command = {
      type: 'task.assign',
      kind: 'CUSTOMER_RECEPTION',
      sourceId: data.sessions[0].id,
      waiterId: identities.waiter.id,
      expectedVersion: 0,
    };
    expect((await command(cmd, 'manager')).json().code).toBe('POLICY_REQUIRED');
    await command({ type: 'policy.configure', policy });
    const results = await Promise.all([
      command(cmd, 'manager'),
      command({ ...cmd, waiterId: identities.waiter2.id }, 'manager2'),
    ]);
    expect(results.map((r) => r.statusCode).sort()).toEqual([200, 409]);
    expect((await snapshot()).tasks).toHaveLength(1);
  });
  it('revokes former assignee and makes repeated Accept/Complete commands idempotent', async () => {
    await command({ type: 'policy.configure', policy });
    const data = await seat();
    const base = {
      type: 'task.assign' as const,
      kind: 'CUSTOMER_RECEPTION' as const,
      sourceId: data.sessions[0].id,
    };
    await command({ ...base, waiterId: identities.waiter.id, expectedVersion: 0 });
    await command({ ...base, waiterId: identities.waiter2.id, expectedVersion: 1 });
    const t = (await snapshot()).tasks[0];
    expect(
      (await command({ type: 'task.accept', taskId: t.id, expectedVersion: t.version }, 'waiter'))
        .statusCode,
    ).toBe(403);
    const id = randomUUID(),
      accept: Command = { type: 'task.accept', taskId: t.id, expectedVersion: t.version };
    expect((await command(accept, 'waiter2', id)).statusCode).toBe(200);
    expect((await command(accept, 'waiter2', id)).json().replayed).toBe(true);
    const current = (await snapshot()).tasks[0],
      completeId = randomUUID();
    const complete: Command = {
      type: 'task.complete',
      taskId: current.id,
      expectedVersion: current.version,
    };
    expect((await command(complete, 'waiter2', completeId)).statusCode).toBe(200);
    expect((await command(complete, 'waiter2', completeId)).json().replayed).toBe(true);
    expect((await snapshot()).sessions[0].endedAt).toBeNull();
  });
  it('rejects a waiter from another restaurant even with a valid account', async () => {
    await command({ type: 'policy.configure', policy });
    const data = await seat();
    await pool.query('DELETE FROM memberships WHERE restaurant_id=$1 AND user_id=$2', [
      restaurantId,
      identities.waiter2.id,
    ]);
    expect(
      (
        await command({
          type: 'task.assign',
          kind: 'CUSTOMER_RECEPTION',
          sourceId: data.sessions[0].id,
          waiterId: identities.waiter2.id,
          expectedVersion: 0,
        })
      ).statusCode,
    ).toBe(400);
  });
  it('stores state, immutable audit and queued outbox events atomically', async () => {
    await seat();
    const events = await pool.query('SELECT id FROM domain_events WHERE restaurant_id=$1', [
      restaurantId,
    ]);
    const outbox = await pool.query(
      'SELECT event_id FROM synchronization_outbox WHERE restaurant_id=$1',
      [restaurantId],
    );
    expect(outbox.rows.map((r) => r.event_id).sort()).toEqual(events.rows.map((r) => r.id).sort());
    await expect(
      pool.query('DELETE FROM audit_log WHERE restaurant_id=$1', [restaurantId]),
    ).rejects.toThrow('append-only');
    await expect(
      pool.query('UPDATE domain_events SET sequence=100 WHERE restaurant_id=$1', [restaurantId]),
    ).rejects.toThrow('append-only');
    expect((await snapshot()).sessions).toHaveLength(1);
  });
  it('preserves durable timestamps across a new app instance and rejects unsafe simulation by default', async () => {
    const initial = await seat();
    const second = await buildApp(pool);
    try {
      const response = await second.inject({
        url: `/api/restaurants/${restaurantId}/dashboard`,
        headers: { cookie: identities.owner.cookie },
      });
      expect(response.json().sessions[0].startedAt).toBe(initial.sessions[0].startedAt);
      expect(response.json().simulatorEnabled).toBe(false);
      const rejected = await second.inject({
        method: 'POST',
        url: `/api/restaurants/${restaurantId}/commands`,
        headers: { cookie: identities.owner.cookie },
        payload: {
          commandId: randomUUID(),
          command: { type: 'simulation.vision', tableId, state: 'UNKNOWN' },
        },
      });
      expect(rejected.statusCode).toBe(403);
    } finally {
      await second.close();
    }
  });
  it('deduplicates persisted deadline alerts across repeated workers', async () => {
    await command({ type: 'policy.configure', policy });
    const data = await seat();
    await command({
      type: 'task.assign',
      kind: 'CUSTOMER_RECEPTION',
      sourceId: data.sessions[0].id,
      waiterId: identities.waiter.id,
      expectedVersion: 0,
    });
    const future = new Date(Date.now() + 60000).toISOString();
    await repository.update(restaurantId, (s) => processDeadlines(s, future));
    await repository.update(restaurantId, (s) => processDeadlines(s, future));
    expect((await snapshot()).alerts.filter((a) => a.kind === 'ACCEPTANCE_OVERDUE')).toHaveLength(
      1,
    );
  });

  it('recovers monitoring gaps and manual overrides, and permits only one deadline worker', async () => {
    const initial = await seat();
    await command(
      {
        type: 'table.correct',
        tableId,
        expectedVersion: initial.tables[0].version,
        occupancy: 'OCCUPIED',
        reason: 'Confirmed locally',
      },
      'manager',
    );
    await pool.query('UPDATE restaurants SET last_heartbeat=$2 WHERE id=$1', [
      restaurantId,
      new Date(Date.now() - 120000),
    ]);
    const worker = await buildApp(pool, { workers: true });
    try {
      const recovered = await repository.read(restaurantId);
      expect(recovered.monitoringGaps).toHaveLength(1);
      expect(recovered.tables[0].aiObservedState).toBe('UNKNOWN');
      expect(recovered.tables[0].override?.reason).toBe('Confirmed locally');
      expect(recovered.sessions[0].startedAt).toBe(initial.sessions[0].startedAt);
      expect(recovered.sessions[0].endedAt).toBeNull();
      await expect(buildApp(pool, { workers: true })).rejects.toThrow('already running');
    } finally {
      await worker.close();
    }
    // Legacy/just-created restaurants can lack a periodic heartbeat.
    await pool.query('UPDATE restaurants SET last_heartbeat=NULL WHERE id=$1', [restaurantId]);
    const restarted = await buildApp(pool, { workers: true });
    await restarted.close();
    expect((await repository.read(restaurantId)).monitoringGaps).toHaveLength(2);
  });
  it('enforces overlap constraints in PostgreSQL independently of the domain', async () => {
    const data = await seat(),
      a = data.assignments[0];
    await expect(
      pool.query(
        'INSERT INTO session_assignments(id,restaurant_id,session_id,table_id,started_at,data) VALUES($1,$2,$3,$4,$5,$6)',
        [randomUUID(), restaurantId, a.sessionId, tableId, a.startedAt, {}],
      ),
    ).rejects.toThrow();
  });
  it('blocks cross-origin and malformed requests without applying commands', async () => {
    const before = (await snapshot()).version;
    const response = await app.inject({
      method: 'POST',
      url: `/api/restaurants/${restaurantId}/commands`,
      headers: {
        cookie: identities.owner.cookie,
        origin: 'https://untrusted.invalid',
        'sec-fetch-site': 'cross-site',
      },
      payload: { commandId: randomUUID(), command: { type: 'table.create', name: 'CSRF' } },
    });
    expect(response.statusCode).toBe(403);
    expect(
      (
        await command({
          type: 'table.correct',
          tableId,
          expectedVersion: 0,
          occupancy: 'OCCUPIED',
          reason: '',
        })
      ).statusCode,
    ).toBe(400);
    expect((await snapshot()).version).toBe(before);
  });
  it('creates staff without persisting their password in audit or responses', async () => {
    const response = await app.inject({
      method: 'POST',
      url: `/api/restaurants/${restaurantId}/users`,
      headers: { cookie: identities.owner.cookie },
      payload: {
        name: 'New waiter',
        email: `${randomUUID()}@test.invalid`,
        password,
        role: 'WAITER',
      },
    });
    expect(response.statusCode).toBe(200);
    expect(response.body).not.toContain(password);
    const state = await snapshot();
    expect(JSON.stringify(state.audit)).not.toContain(password);
    expect(state.members.some((m) => m.id === response.json().id && m.role === 'WAITER')).toBe(
      true,
    );
  });
  it('invalid credentials are rejected and logout invalidates the database session', async () => {
    const invalid = await app.inject({
      method: 'POST',
      url: '/api/auth/login',
      payload: { email: 'missing@test.invalid', password: 'not-the-password' },
    });
    expect(invalid.statusCode).toBe(401);
    const login = await app.inject({
      method: 'POST',
      url: '/api/auth/login',
      payload: { email: 'owner@test.invalid', password },
    });
    const cookie = String(login.headers['set-cookie']).split(';')[0];
    expect((await app.inject({ url: '/api/me', headers: { cookie } })).statusCode).toBe(200);
    expect(
      (
        await app.inject({
          method: 'POST',
          url: '/api/auth/logout',
          headers: { cookie },
          payload: {},
        })
      ).statusCode,
    ).toBe(200);
    expect((await app.inject({ url: '/api/me', headers: { cookie } })).statusCode).toBe(401);
  });

  it('bootstraps an owner through the documented CLI without defaults or duplicate initialization', async () => {
    const bootstrapSchema = `bootstrap_${randomUUID().replaceAll('-', '')}`;
    await admin.query(`CREATE SCHEMA ${bootstrapSchema}`);
    const bootstrapUrl = new URL(process.env.TEST_DATABASE_URL!);
    bootstrapUrl.searchParams.set('options', `-c search_path=${bootstrapSchema},public`);
    const bootstrapPool = new Pool({ connectionString: bootstrapUrl.toString() });
    try {
      await migrate(bootstrapPool);
      const env = {
        ...process.env,
        DATABASE_URL: bootstrapUrl.toString(),
        RESTAURANT_NAME: 'CLI restaurant',
        OWNER_NAME: 'CLI owner',
        OWNER_EMAIL: 'bootstrap@test.invalid',
        OWNER_PASSWORD: password,
      };
      const run = () =>
        promisify(execFile)(
          process.execPath,
          ['--import', 'tsx', 'apps/edge-api/src/setup-owner.ts'],
          { env, cwd: process.cwd() },
        );
      const result = await run();
      expect(result.stdout).toContain('Owner created');
      expect(result.stdout + result.stderr).not.toContain(password);
      await expect(run()).rejects.toThrow('Already initialized');
      const users = await bootstrapPool.query('SELECT password_hash FROM users');
      expect(users.rows[0].password_hash).toMatch(/^scrypt:/);
      expect(users.rows[0].password_hash).not.toContain(password);
      expect(
        (await bootstrapPool.query('SELECT policy FROM restaurants')).rows[0].policy,
      ).toBeNull();
      expect(
        (await bootstrapPool.query('SELECT last_heartbeat FROM restaurants')).rows[0]
          .last_heartbeat,
      ).not.toBeNull();
      const audit = (await bootstrapPool.query('SELECT id,data FROM audit_log')).rows[0];
      expect(audit.id).toBe(audit.data.id);
    } finally {
      await bootstrapPool.end();
      await admin.query(`DROP SCHEMA IF EXISTS ${bootstrapSchema} CASCADE`);
    }
  });
});
