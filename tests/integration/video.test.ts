import { randomBytes, randomUUID } from 'node:crypto';
import { request as httpRequest } from 'node:http';
import { Readable } from 'node:stream';
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
let lastStarted: unknown;
let uploadedBytes = 0;
let firstUploadChunk: (() => void) | undefined;
let uploadFailure = false;
let uploadedName = '';
const sourceId = 'a'.repeat(64);
const trial = {
  sourceId,
  startSeconds: 0,
  durationSeconds: 10,
  regions: [{ id: randomUUID(), tableId, x: 0.1, y: 0.2, width: 0.2, height: 0.2 }],
};
const wristOptions = {
  wristTrial: true,
  tabletopRegions: [{ ...trial.regions[0], id: randomUUID() }],
};
beforeAll(async () => {
  backend.addHook('onRequest', async () => {
    calls++;
  });
  backend.get('/restaurants/:restaurantId/status', () => ({
    enabled: true,
    models: 'READY',
    source: { id: sourceId, duration: 60, width: 640, height: 480, bytes: 100 },
  }));
  backend.get('/restaurants/:restaurantId/video', (request, reply) => {
    expect((request.params as { restaurantId: string }).restaurantId).toBe(tenant);
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
  backend.addContentTypeParser('video/mp4', (_request, payload, done) => done(null, payload));
  backend.post('/restaurants/:restaurantId/video', async (request, reply) => {
    expect(request.params).toEqual({ restaurantId: tenant });
    uploadedBytes = 0;
    uploadedName = String(request.headers['x-video-name']);
    try {
      for await (const chunk of request.body as Readable) {
        uploadedBytes += chunk.length;
        firstUploadChunk?.();
      }
    } catch {
      return reply.code(400).send({ detail: 'incomplete' });
    }
    if (uploadFailure) return reply.code(422).send({ detail: 'فيديو غير صالح' });
    return reply.code(201).send({ id: sourceId, uploaded: true, name: 'camera.mp4' });
  });
  backend.post('/restaurants/:restaurantId/video/delete', (request) => {
    expect(request.params).toEqual({ restaurantId: tenant });
    expect(request.body).toEqual({ sourceId });
    return { ok: true };
  });
  backend.post('/restaurants/:restaurantId/jobs', (request, reply) => {
    lastStarted = request.body;
    expect(request.params).toEqual({ restaurantId: tenant });
    const { detectionMode, poseModel, wristTrial, tabletopRegions, seatTrial } = request.body as {
      detectionMode?: string;
      poseModel?: string;
      wristTrial?: boolean;
      seatTrial?: boolean;
      tabletopRegions?: unknown[];
    };
    expect(['FULL_FRAME', 'SEAT_FOCUS', undefined]).toContain(detectionMode);
    expect([
      'RTMPOSE_TINY',
      'RTMPOSE_MEDIUM',
      'YOLO26_NANO_POSE',
      'YOLO26_SMALL_POSE',
      undefined,
    ]).toContain(poseModel);
    expect(request.body).toEqual({
      ...trial,
      ...(detectionMode === undefined ? {} : { detectionMode }),
      ...(poseModel === undefined ? {} : { poseModel }),
      ...(wristTrial === undefined ? {} : { wristTrial }),
      ...(seatTrial === undefined ? {} : { seatTrial }),
      ...(tabletopRegions === undefined ? {} : { tabletopRegions }),
    });
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
  it('blocks unauthorized upload/deletion and cross-origin binary requests before the worker', async () => {
    const initial = calls;
    for (const role of ['', 'MANAGER', 'WAITER']) {
      const headers = role ? { cookie: identities[role].cookie } : {};
      for (const action of ['upload', 'delete']) {
        const response = await app.inject({
          method: 'POST',
          url: `/api/restaurants/${tenant}/video-lab/video${action === 'delete' ? '/delete' : ''}`,
          headers: {
            ...headers,
            'Content-Type': action === 'upload' ? 'video/mp4' : 'application/json',
          },
          payload: action === 'upload' ? Buffer.from('video') : JSON.stringify({ sourceId }),
        });
        expect(response.statusCode).toBe(role ? 403 : 401);
      }
    }
    for (const extra of [
      { origin: 'https://attacker.invalid' },
      { 'sec-fetch-site': 'cross-site' },
    ]) {
      expect(
        (
          await app.inject({
            method: 'POST',
            url: `/api/restaurants/${tenant}/video-lab/video`,
            headers: { cookie: identities.OWNER.cookie, 'Content-Type': 'video/mp4', ...extra },
            payload: Buffer.from('video'),
          })
        ).statusCode,
      ).toBe(403);
    }
    expect(
      (
        await app.inject({
          method: 'POST',
          url: `/api/restaurants/${otherTenant}/video-lab/video`,
          headers: { cookie: identities.OWNER.cookie, 'Content-Type': 'video/mp4' },
          payload: Buffer.from('video'),
        })
      ).statusCode,
    ).toBe(403);
    expect(calls).toBe(initial);
  });
  it('streams a chunked upload to the worker before the browser finishes sending', async () => {
    // Use a real HTTP socket so this verifies streaming, not only inject buffering.
    const address = await app.listen({ host: '127.0.0.1', port: 0 });
    let resolveChunk!: () => void;
    const arrived = new Promise<void>((resolve) => {
      resolveChunk = resolve;
    });
    firstUploadChunk = resolveChunk;
    const req = httpRequest(`${address}/api/restaurants/${tenant}/video-lab/video`, {
      method: 'POST',
      headers: {
        cookie: identities.OWNER.cookie,
        'Content-Type': 'video/mp4',
        'X-Video-Name': 'camera.mp4',
      },
    });
    const completed = new Promise<{ status: number; body: string }>((resolve, reject) => {
      req.on('error', reject);
      req.on('response', (response) => {
        let body = '';
        response.on('data', (chunk) => {
          body += chunk;
        });
        response.on('end', () => resolve({ status: response.statusCode!, body }));
      });
    });
    try {
      req.write(Buffer.alloc(32 * 1024, 1));
      await arrived;
      expect(uploadedBytes).toBe(32 * 1024);
      req.end(Buffer.alloc(32 * 1024, 2));
      const response = await completed;
      expect(response.status).toBe(201);
      expect(JSON.parse(response.body).uploaded).toBe(true);
      expect(uploadedBytes).toBe(64 * 1024);
      expect(uploadedName).toBe('camera.mp4');
    } finally {
      firstUploadChunk = undefined;
      req.destroy();
    }
  });
  it('enforces upload bounds, preserves JSON protections and forwards validation errors', async () => {
    const send = (headers = {}) =>
      app.inject({
        method: 'POST',
        url: `/api/restaurants/${tenant}/video-lab/video`,
        headers: { cookie: identities.OWNER.cookie, 'Content-Type': 'video/mp4', ...headers },
        payload: Buffer.from('invalid'),
      });
    const initial = calls;
    expect((await send({ 'Content-Length': String(100 * 1024 * 1024 + 1) })).statusCode).toBe(413);
    expect((await send({ 'Content-Type': 'application/json' })).statusCode).toBe(415);
    expect(
      (
        await app.inject({
          method: 'POST',
          url: '/api/auth/login',
          headers: { 'Content-Type': 'video/mp4' },
          payload: Buffer.from('invalid'),
        })
      ).statusCode,
    ).toBe(415);
    expect(calls).toBe(initial);
    uploadFailure = true;
    try {
      const failed = await send();
      expect(failed.statusCode).toBe(422);
      expect(failed.json().message).toBe('فيديو غير صالح');
    } finally {
      uploadFailure = false;
    }
  });
  it('rejects an oversized chunked stream even without a declared length', async () => {
    const address = `http://127.0.0.1:${(app.server.address() as { port: number }).port}`;
    const req = httpRequest(`${address}/api/restaurants/${tenant}/video-lab/video`, {
      method: 'POST',
      headers: { cookie: identities.OWNER.cookie, 'Content-Type': 'video/mp4' },
    });
    const stream = Readable.from(
      (function* () {
        const chunk = Buffer.alloc(64 * 1024, 1);
        for (let i = 0; i <= 1600; i++) yield chunk;
      })(),
    );
    const response = new Promise<number>((resolve, reject) => {
      req.on('error', reject);
      req.on('response', (res) => {
        res.resume();
        res.on('end', () => resolve(res.statusCode!));
      });
    });
    try {
      stream.pipe(req);
      expect(await response).toBe(413);
      expect(uploadedBytes).toBeLessThanOrEqual(100 * 1024 * 1024);
    } finally {
      stream.destroy();
      req.destroy();
    }
  });
  it('checks the current source on deletion and rejects malformed source selectors', async () => {
    const initial = calls;
    expect((await get('/video?source=../../etc/passwd')).statusCode).toBe(400);
    const send = (body: unknown) =>
      app.inject({
        method: 'POST',
        url: `/api/restaurants/${tenant}/video-lab/video/delete`,
        headers: { cookie: identities.OWNER.cookie },
        payload: body as object,
      });
    expect((await send({ sourceId: 'bad' })).statusCode).toBe(400);
    expect((await send({ sourceId, path: '/recordings/restaurant-demo.mp4' })).statusCode).toBe(
      400,
    );
    expect(calls).toBe(initial);
    expect((await send({ sourceId })).json()).toEqual({ ok: true });
  });
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
  it('forwards explicit detector and pose choices while rejecting unsupported models and URLs', async () => {
    const send = (options: Record<string, unknown>) =>
      app.inject({
        method: 'POST',
        url: `/api/restaurants/${tenant}/video-lab/jobs`,
        headers: { cookie: identities.OWNER.cookie },
        payload: { ...trial, ...options },
      });
    const initial = calls;
    for (const invalid of [
      { seatTrial: 'true' },
      { seatTrial: 1 },
      { seatTrial: null },
      { detectionMode: 'unexpected' },
      { poseModel: 'unexpected' },
      { poseModel: 'https://example.invalid/model.onnx' },
      { poseModel: 'RTMPOSE_MEDIUM', poseUrl: 'https://example.invalid/model.onnx' },
    ])
      expect((await send(invalid)).statusCode).toBe(400);
    expect(calls).toBe(initial);
    for (const valid of [
      {},
      { seatTrial: true },
      { seatTrial: false },
      { detectionMode: 'SEAT_FOCUS' },
      { poseModel: 'RTMPOSE_TINY' },
      { poseModel: 'RTMPOSE_MEDIUM' },
      { poseModel: 'YOLO26_NANO_POSE' },
      { poseModel: 'YOLO26_SMALL_POSE' },
      { detectionMode: 'SEAT_FOCUS', poseModel: 'YOLO26_SMALL_POSE' },
      { detectionMode: 'SEAT_FOCUS', poseModel: 'YOLO26_NANO_POSE' },
      { detectionMode: 'SEAT_FOCUS', poseModel: 'RTMPOSE_MEDIUM' },
    ])
      expect((await send(valid)).statusCode).toBe(202);
  });
  it('protects seat trials from non-owner access before contacting the worker', async () => {
    const initial = calls;
    for (const role of ['MANAGER', 'WAITER']) {
      const response = await app.inject({
        method: 'POST',
        url: `/api/restaurants/${tenant}/video-lab/jobs`,
        headers: { cookie: identities[role].cookie },
        payload: { ...trial, seatTrial: true },
      });
      expect(response.statusCode).toBe(403);
    }
    expect(calls).toBe(initial);
  });
  it('validates tabletop geometry, activation and table scope before starting a wrist trial', async () => {
    const send = (body: Record<string, unknown>) =>
      app.inject({
        method: 'POST',
        url: `/api/restaurants/${tenant}/video-lab/jobs`,
        headers: { cookie: identities.OWNER.cookie },
        payload: body,
      });
    const top = wristOptions.tabletopRegions[0];
    const initial = calls;
    for (const extra of [
      { wristTrial: true },
      { ...wristOptions, wristTrial: 'true' },
      { tabletopRegions: [top] },
      { wristTrial: true, tabletopRegions: [trial.regions[0]] },
      { wristTrial: true, tabletopRegions: [{ ...top, tableId: randomUUID() }] },
      { wristTrial: true, tabletopRegions: [{ ...top, x: 0.95 }] },
      {
        wristTrial: true,
        tabletopRegions: Array.from({ length: 41 }, () => ({ ...top, id: randomUUID() })),
      },
    ])
      expect((await send({ ...trial, ...extra })).statusCode).toBe(400);
    const foreign = randomUUID();
    expect(
      (
        await send({
          ...trial,
          wristTrial: true,
          regions: [{ ...trial.regions[0], tableId: foreign }],
          tabletopRegions: [{ ...top, tableId: foreign }],
        })
      ).statusCode,
    ).toBe(400);
    expect(calls).toBe(initial);
    expect((await send({ ...trial, ...wristOptions })).statusCode).toBe(202);
    expect(lastStarted).toEqual({ ...trial, ...wristOptions });
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
        {
          ...report,
          poseModel: 'RTMPOSE_MEDIUM' as const,
          ...wristOptions,
          focusMergePolicy: 'ALIGNED_CONTAINMENT_V1' as const,
        },
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
