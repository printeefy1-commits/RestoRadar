import { randomBytes, randomUUID } from 'node:crypto';
import Fastify from 'fastify';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Pool, migrate, type DatabasePool } from '@restoradar/database';
import type { TrialJob, TrialReport } from '@restoradar/contracts';
import { buildApp } from '../../apps/edge-api/src/app.ts';
import { hashPassword } from '../../apps/edge-api/src/auth.ts';

if (!process.env.TEST_DATABASE_URL) throw new Error('TEST_DATABASE_URL is required');
const schema = `video_${randomUUID().replaceAll('-', '')}`;
const admin = new Pool({ connectionString: process.env.TEST_DATABASE_URL });
const tenant = randomUUID(),
  otherTenant = randomUUID(),
  tableId = randomUUID();
const identities = Object.fromEntries(
  ['OWNER', 'MANAGER', 'WAITER'].map((role) => [role, { id: randomUUID(), cookie: '' }]),
);
const backend = Fastify();
let pool: DatabasePool,
  app: Awaited<ReturnType<typeof buildApp>>,
  calls = 0;
let storedJob: TrialJob | null = null;
const sourceId = 'a'.repeat(64);
const trial = {
  sourceId,
  startSeconds: 0,
  durationSeconds: 10,
  regions: [{ id: randomUUID(), tableId, x: 0.1, y: 0.2, width: 0.2, height: 0.2 }],
};
beforeAll(async () => {
  backend.addHook('onRequest', async () => {
    calls++;
  });
  backend.get('/status', () => ({
    enabled: true,
    models: 'READY',
    source: { id: sourceId, duration: 60, width: 640, height: 480, bytes: 100 },
  }));
  backend.get('/video', (request, reply) => {
    expect(request.headers.range).toBe('bytes=0-3');
    return reply
      .code(206)
      .headers({
        'Content-Type': 'video/mp4',
        'Content-Range': 'bytes 0-3/100',
        'Accept-Ranges': 'bytes',
      })
      .send(Buffer.from('test'));
  });
  backend.post('/restaurants/:restaurantId/jobs', (request, reply) => {
    expect(request.params).toEqual({ restaurantId: tenant });
    const mode = (request.body as { detectionMode?: string }).detectionMode;
    expect(['FULL_FRAME', 'SEAT_FOCUS', undefined]).toContain(mode);
    expect(request.body).toEqual(mode ? { ...trial, detectionMode: mode } : trial);
    return reply.code(202).send({ id: randomUUID(), status: 'RUNNING', progress: 0 });
  });
  backend.get('/restaurants/:restaurantId/jobs/latest', () => storedJob);
  backend.get('/restaurants/:restaurantId/jobs/:jobId', (request) => {
    expect(request.params).toEqual({ restaurantId: tenant, jobId: storedJob?.id });
    return storedJob;
  });
  const url = await backend.listen({ host: '127.0.0.1', port: 0 });
  await admin.query(`CREATE SCHEMA ${schema}`);
  pool = new Pool({
    connectionString: process.env.TEST_DATABASE_URL,
    options: `-c search_path=${schema},public`,
  });
  await migrate(pool);
  await pool.query('INSERT INTO restaurants(id,name) VALUES($1,$2),($3,$4)', [
    tenant,
    'Video restaurant',
    otherTenant,
    'Other tenant',
  ]);
  const password = randomBytes(24).toString('hex');
  const hash = await hashPassword(password);
  for (const [role, user] of Object.entries(identities)) {
    await pool.query('INSERT INTO users(id,email,name,password_hash) VALUES($1,$2,$3,$4)', [
      user.id,
      `${role.toLowerCase()}@video.invalid`,
      role,
      hash,
    ]);
    await pool.query('INSERT INTO memberships(restaurant_id,user_id,role) VALUES($1,$2,$3)', [
      tenant,
      user.id,
      role,
    ]);
  }
  await pool.query('INSERT INTO memberships(restaurant_id,user_id,role) VALUES($1,$2,$3)', [
    otherTenant,
    identities.OWNER.id,
    'OWNER',
  ]);
  app = await buildApp(pool, { videoLabUrl: url, videoLabRestaurantId: tenant });
  for (const [role, user] of Object.entries(identities)) {
    const response = await app.inject({
      method: 'POST',
      url: '/api/auth/login',
      payload: { email: `${role}@video.invalid`, password },
    });
    user.cookie = String(response.headers['set-cookie']).split(';')[0];
  }
  await pool.query('INSERT INTO dining_tables(id,restaurant_id,data) VALUES($1,$2,$3)', [
    tableId,
    tenant,
    JSON.stringify({
      id: tableId,
      name: 'Mapped table',
      version: 0,
      occupancy: 'EMPTY',
      aiObservedState: 'UNKNOWN',
      lastVerifiedAiState: null,
      override: null,
    }),
  ]);
});
afterAll(async () => {
  await app?.close();
  await backend.close();
  await pool?.end();
  await admin.query(`DROP SCHEMA IF EXISTS ${schema} CASCADE`);
  await admin.end();
});
function get(path = '', role = 'OWNER', id = tenant) {
  return app.inject({
    url: `/api/restaurants/${id}/video-lab${path}`,
    headers: { cookie: identities[role].cookie },
  });
}
describe('owner-only recorded-video gateway', () => {
  it('blocks unauthenticated users, other roles and other restaurants before contacting the worker', async () => {
    const initial = calls;
    expect((await app.inject({ url: `/api/restaurants/${tenant}/video-lab` })).statusCode).toBe(
      401,
    );
    for (const role of ['MANAGER', 'WAITER']) {
      for (const path of ['', '/video', '/jobs/latest'])
        expect((await get(path, role)).statusCode).toBe(403);
      expect(
        (
          await app.inject({
            method: 'POST',
            url: `/api/restaurants/${tenant}/video-lab/jobs`,
            headers: { cookie: identities[role].cookie },
            payload: trial,
          })
        ).statusCode,
      ).toBe(403);
    }
    expect((await get('', 'OWNER', otherTenant)).statusCode).toBe(403);
    expect(calls).toBe(initial);
  });
  it('streams authenticated ranges and rejects multipart/multiple ranges', async () => {
    const response = await app.inject({
      url: `/api/restaurants/${tenant}/video-lab/video`,
      headers: { cookie: identities.OWNER.cookie, range: 'bytes=0-3' },
    });
    expect(response.statusCode).toBe(206);
    expect(response.body).toBe('test');
    expect(response.headers['content-range']).toBe('bytes 0-3/100');
    expect(response.headers['cache-control']).toBe('no-store');
    expect(response.headers['content-security-policy']).toContain("media-src 'self'");
    expect(
      (
        await app.inject({
          url: `/api/restaurants/${tenant}/video-lab/video`,
          headers: { cookie: identities.OWNER.cookie, range: 'bytes=0-2,4-5' },
        })
      ).statusCode,
    ).toBe(416);
  });
  it('forwards an owner’s explicit focused mode and rejects unsupported modes', async () => {
    const send = (detectionMode: string) =>
      app.inject({
        method: 'POST',
        url: `/api/restaurants/${tenant}/video-lab/jobs`,
        headers: { cookie: identities.OWNER.cookie },
        payload: { ...trial, detectionMode },
      });
    const initial = calls;
    expect((await send('unexpected')).statusCode).toBe(400);
    expect(calls).toBe(initial);
    expect((await send('SEAT_FOCUS')).statusCode).toBe(202);
  });
  it('preserves focused merge diagnostics and historical reports through both job routes', async () => {
    const report: TrialReport = {
      mode: 'RECORDED_VIDEO_DRY_RUN',
      detectionMode: 'SEAT_FOCUS',
      sourceId,
      models: {
        detector: 'test detector',
        pose: 'test pose',
        tracker: 'test tracker',
        artifacts: {},
      },
      startSeconds: 0,
      endSeconds: 10,
      analyzedSamples: 21,
      processingSeconds: 5,
      samplesPerProcessingSecond: 4.2,
      departureEvaluated: false,
      regions: trial.regions,
      events: [],
      samples: [{ seconds: 10, focusCrops: 1, people: [], tables: { [tableId]: 'UNKNOWN' } }],
    };
    try {
      for (const currentReport of [
        report,
        { ...report, focusMergePolicy: 'ALIGNED_CONTAINMENT_V1' as const },
      ]) {
        storedJob = { id: randomUUID(), status: 'COMPLETED', progress: 1, report: currentReport };
        for (const path of ['/jobs/latest', `/jobs/${storedJob.id}`]) {
          const response = await get(path);
          expect(response.statusCode).toBe(200);
          expect(response.json()).toEqual(storedJob);
        }
      }
    } finally {
      storedJob = null;
    }
  });
  it('validates table ownership, bounds and same-origin without mutating live state', async () => {
    const send = (payload: unknown, headers = {}) =>
      app.inject({
        method: 'POST',
        url: `/api/restaurants/${tenant}/video-lab/jobs`,
        headers: {
          cookie: identities.OWNER.cookie,
          'Content-Type': 'application/json',
          ...headers,
        },
        payload: JSON.stringify(payload),
      });
    const initial = calls;
    expect(
      (await send({ ...trial, regions: [{ ...trial.regions[0], tableId: randomUUID() }] }))
        .statusCode,
    ).toBe(400);
    expect(
      (await send({ ...trial, regions: [{ ...trial.regions[0], width: 1 }] })).statusCode,
    ).toBe(400);
    expect((await send(trial, { origin: 'https://attacker.invalid' })).statusCode).toBe(403);
    expect(calls).toBe(initial);
    const before = (
      await app.inject({
        url: `/api/restaurants/${tenant}/dashboard`,
        headers: { cookie: identities.OWNER.cookie },
      })
    ).json();
    expect((await send(trial)).statusCode).toBe(202);
    const after = (
      await app.inject({
        url: `/api/restaurants/${tenant}/dashboard`,
        headers: { cookie: identities.OWNER.cookie },
      })
    ).json();
    expect(after.sessions).toEqual(before.sessions);
    expect(after.alerts).toEqual(before.alerts);
    expect(after.version).toBe(before.version);
  });
});
