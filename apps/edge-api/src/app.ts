import Fastify, { type FastifyRequest } from 'fastify';
import cookie from '@fastify/cookie';
import rateLimit from '@fastify/rate-limit';
import staticFiles from '@fastify/static';
import { randomUUID, createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { z, ZodError } from 'zod';
import { commandEnvelopeSchema, roleSchema } from '@restoradar/contracts';
import type { Dashboard, Member, Role, RestaurantState } from '@restoradar/contracts';
import {
  DomainError,
  executeCommand,
  markMonitoringGap,
  processDeadlines,
} from '@restoradar/domain';
import { Repository, type Client, type DatabasePool } from '@restoradar/database';
import { hashPassword, hashToken, newToken, verifyPassword } from './auth.ts';

interface Identity {
  id: string;
  name: string;
}
const uuid = z.uuid();
const credentials = z
  .object({
    email: z
      .email()
      .max(254)
      .transform((v) => v.toLowerCase()),
    password: z.string().min(1).max(256),
  })
  .strict();
const newUser = z
  .object({
    email: z
      .email()
      .max(254)
      .transform((v) => v.toLowerCase()),
    name: z.string().trim().min(1).max(80),
    role: roleSchema,
    password: z.string().min(12).max(256),
  })
  .strict();
export interface AppOptions {
  simulatorEnabled?: boolean;
  secureCookies?: boolean;
  workers?: boolean;
  logger?: boolean;
  publicOrigin?: string;
}
const alertOrder = {
  NEW_CUSTOMERS: 0,
  ACCEPTANCE_OVERDUE: 1,
  COMPLETION_OVERDUE: 2,
  CLEANING_REQUIRED: 3,
};

export function dashboard(
  state: RestaurantState,
  members: Member[],
  userId: string,
  role: Role,
  simulatorEnabled: boolean,
  now = new Date().toISOString(),
): Dashboard {
  const { events: _, ...s } = structuredClone(state);
  if (role === 'WAITER') {
    s.policy = null;
    s.audit = [];
    s.alerts = [];
    s.adjustments = [];
    s.sessions = [];
    s.assignments = [];
    s.monitoringGaps = [];
    s.tasks = s.tasks.filter(
      (t) => s.attempts.find((a) => a.id === t.currentAttemptId)?.waiterId === userId,
    );
    const attemptIds = new Set(s.tasks.map((t) => t.currentAttemptId));
    s.attempts = s.attempts.filter((a) => attemptIds.has(a.id));
    // Manual reasons and actor IDs belong to management history, not waiter status cards.
    s.tables = s.tables.map((t) => ({ ...t, override: null }));
    s.cleaningCycles = s.cleaningCycles.map((c) => ({ ...c, readyBy: null }));
    members = members.filter((m) => m.id === userId);
  }
  s.alerts.sort(
    (a, b) => alertOrder[a.kind] - alertOrder[b.kind] || a.createdAt.localeCompare(b.createdAt),
  );
  s.audit.sort((a, b) => b.at.localeCompare(a.at));
  return {
    ...s,
    members,
    serverTime: now,
    simulatorEnabled: role === 'OWNER' && simulatorEnabled,
    sync: 'LOCAL_ONLY',
  };
}

export async function buildApp(pool: DatabasePool, options: AppOptions = {}) {
  const app = Fastify({
    bodyLimit: 16384,
    logger: options.logger
      ? { redact: ['req.headers.cookie', 'req.headers.authorization'] }
      : false,
  });
  const repository = new Repository(pool);
  const simulatorEnabled = options.simulatorEnabled === true;
  const secureCookies = options.secureCookies === true;
  await app.register(cookie);
  await app.register(rateLimit, { max: 300, timeWindow: '1 minute' });
  app.addHook('onRequest', async (request, reply) => {
    reply.header('Cache-Control', 'no-store');
    reply.header('X-Content-Type-Options', 'nosniff');
    reply.header('Referrer-Policy', 'same-origin');
    reply.header(
      'Content-Security-Policy',
      "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    );
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method)) {
      const origin = request.headers.origin;
      const allowedOrigin =
        options.publicOrigin ?? `${secureCookies ? 'https' : 'http'}://${request.headers.host}`;
      if (
        request.headers['sec-fetch-site'] === 'cross-site' ||
        (origin && origin !== allowedOrigin)
      ) {
        throw new DomainError('ORIGIN_REJECTED', 'مصدر الطلب غير مسموح', 403);
      }
      if (!request.headers['content-type']?.startsWith('application/json'))
        throw new DomainError('JSON_REQUIRED', 'يجب إرسال JSON', 415);
    }
  });
  app.setErrorHandler((error, _request, reply) => {
    if (error instanceof DomainError)
      return reply.code(error.status).send({ code: error.code, message: error.message });
    if (error instanceof ZodError)
      return reply
        .code(400)
        .send({ code: 'INVALID_INPUT', message: 'تحقق من الحقول المطلوبة وصحة القيم' });
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === '23505')
      return reply.code(409).send({ code: 'ALREADY_EXISTS', message: 'السجل موجود بالفعل' });
    if (
      typeof error === 'object' &&
      error !== null &&
      'statusCode' in error &&
      typeof error.statusCode === 'number' &&
      error.statusCode < 500
    )
      return reply
        .code(error.statusCode)
        .send({ code: 'REQUEST_REJECTED', message: 'تعذر قبول الطلب' });
    app.log.error(error);
    return reply
      .code(500)
      .send({ code: 'INTERNAL_ERROR', message: 'حدث خطأ داخلي. أعد المحاولة.' });
  });
  async function authenticate(request: FastifyRequest): Promise<Identity> {
    const token = request.cookies.rr_session;
    if (!token || token.length > 100)
      throw new DomainError('UNAUTHENTICATED', 'سجّل الدخول أولًا', 401);
    const result = await pool.query(
      `SELECT u.id,u.name FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=$1 AND s.expires_at>now()`,
      [hashToken(token)],
    );
    if (!result.rowCount)
      throw new DomainError('UNAUTHENTICATED', 'انتهت الجلسة. سجّل الدخول مجددًا', 401);
    return result.rows[0];
  }
  async function authorize(request: FastifyRequest, restaurantId: string) {
    const user = await authenticate(request);
    const result = await pool.query(
      'SELECT role FROM memberships WHERE restaurant_id=$1 AND user_id=$2',
      [restaurantId, user.id],
    );
    if (!result.rowCount) throw new DomainError('FORBIDDEN', 'لا تملك صلاحية لهذا المطعم', 403);
    return { ...user, role: result.rows[0].role as Role };
  }
  async function members(restaurantId: string): Promise<Member[]> {
    const result = await pool.query(
      'SELECT u.id,u.name,m.role FROM memberships m JOIN users u ON u.id=m.user_id WHERE m.restaurant_id=$1 ORDER BY u.name',
      [restaurantId],
    );
    return result.rows;
  }
  app.get('/api/health', async () => {
    await pool.query('SELECT 1');
    return { status: 'ok', mode: 'LOCAL_ONLY', simulatorEnabled };
  });
  // The dummy hash equalizes expensive password verification for unknown accounts.
  const dummyHash = await hashPassword(newToken());
  app.post(
    '/api/auth/login',
    { config: { rateLimit: { max: 10, timeWindow: '1 minute' } } },
    async (request, reply) => {
      const body = credentials.parse(request.body);
      const result = await pool.query('SELECT id,password_hash FROM users WHERE email=$1', [
        body.email,
      ]);
      const user = result.rows[0];
      const valid = await verifyPassword(body.password, user?.password_hash ?? dummyHash);
      if (!user || !valid)
        throw new DomainError('INVALID_CREDENTIALS', 'البريد أو كلمة المرور غير صحيحة', 401);
      const token = newToken();
      await pool.query(
        "INSERT INTO auth_sessions(token_hash,user_id,expires_at) VALUES($1,$2,now()+interval '12 hours')",
        [hashToken(token), user.id],
      );
      reply.setCookie('rr_session', token, {
        path: '/',
        httpOnly: true,
        sameSite: 'strict',
        secure: secureCookies,
        maxAge: 43200,
      });
      return { ok: true };
    },
  );
  app.post('/api/auth/logout', async (request, reply) => {
    if (request.cookies.rr_session)
      await pool.query('DELETE FROM auth_sessions WHERE token_hash=$1', [
        hashToken(request.cookies.rr_session),
      ]);
    reply.clearCookie('rr_session', {
      path: '/',
      httpOnly: true,
      sameSite: 'strict',
      secure: secureCookies,
    });
    return { ok: true };
  });
  app.get('/api/me', async (request) => {
    const user = await authenticate(request);
    const result = await pool.query(
      'SELECT r.id,r.name,m.role FROM memberships m JOIN restaurants r ON r.id=m.restaurant_id WHERE m.user_id=$1 ORDER BY r.name',
      [user.id],
    );
    return { user, restaurants: result.rows };
  });
  app.get<{ Params: { restaurantId: string } }>(
    '/api/restaurants/:restaurantId/dashboard',
    async (request) => {
      const restaurantId = uuid.parse(request.params.restaurantId);
      const actor = await authorize(request, restaurantId);
      return dashboard(
        await repository.read(restaurantId),
        await members(restaurantId),
        actor.id,
        actor.role,
        simulatorEnabled,
      );
    },
  );
  app.post<{ Params: { restaurantId: string } }>(
    '/api/restaurants/:restaurantId/commands',
    async (request) => {
      const restaurantId = uuid.parse(request.params.restaurantId);
      const actor = await authorize(request, restaurantId);
      const { commandId, command } = commandEnvelopeSchema.parse(request.body);
      const fingerprint = createHash('sha256')
        .update(JSON.stringify({ actorId: actor.id, command }))
        .digest('hex');
      return repository.update(
        restaurantId,
        async (s, c) => {
          const waiters = await c.query(
            "SELECT user_id FROM memberships WHERE restaurant_id=$1 AND role='WAITER'",
            [restaurantId],
          );
          return executeCommand(s, command, {
            actor,
            waiterIds: waiters.rows.map((r) => r.user_id),
            now: new Date().toISOString(),
            simulatorEnabled,
          });
        },
        { commandId, fingerprint },
      );
    },
  );
  app.post<{ Params: { restaurantId: string } }>(
    '/api/restaurants/:restaurantId/users',
    async (request) => {
      const restaurantId = uuid.parse(request.params.restaurantId);
      const actor = await authorize(request, restaurantId);
      if (actor.role !== 'OWNER')
        throw new DomainError('FORBIDDEN', 'إدارة الحسابات للمالك فقط', 403);
      const body = newUser.parse(request.body),
        id = randomUUID();
      const passwordHash = await hashPassword(body.password);
      await repository.update(restaurantId, async (previous, c) => {
        await c.query('INSERT INTO users(id,email,name,password_hash) VALUES($1,$2,$3,$4)', [
          id,
          body.email,
          body.name,
          passwordHash,
        ]);
        await c.query('INSERT INTO memberships(restaurant_id,user_id,role) VALUES($1,$2,$3)', [
          restaurantId,
          id,
          body.role,
        ]);
        const next = structuredClone(previous);
        next.version++;
        next.audit.push({
          id: randomUUID(),
          actorId: actor.id,
          action: 'user.created',
          at: new Date().toISOString(),
          context: { userId: id, role: body.role },
        });
        return next;
      });
      return { id, name: body.name, role: body.role };
    },
  );
  const webRoot = fileURLToPath(new URL('../../web/dist/', import.meta.url));
  if (existsSync(webRoot)) {
    await app.register(staticFiles, { root: webRoot });
    app.setNotFoundHandler((request, reply) => {
      if (request.method === 'GET' && !request.url.startsWith('/api/'))
        return reply.sendFile('index.html');
      return reply.code(404).send({ code: 'NOT_FOUND', message: 'المسار غير موجود' });
    });
  }
  let worker: ReturnType<typeof setInterval> | undefined;
  let leader: Client | undefined;
  let runningTick: Promise<void> | undefined;
  app.addHook('onClose', async () => {
    if (worker) clearInterval(worker);
    await runningTick;
    if (leader) {
      await leader.query(
        "SELECT pg_advisory_unlock(hashtextextended(current_schema() || ':restoradar:edge', 0))",
      );
      leader.release();
      leader = undefined;
    }
  });
  if (options.workers) {
    leader = await pool.connect();
    const lock = await leader.query(
      "SELECT pg_try_advisory_lock(hashtextextended(current_schema() || ':restoradar:edge', 0)) AS acquired",
    );
    if (!lock.rows[0].acquired) {
      leader.release();
      leader = undefined;
      throw new Error('An edge worker is already running for this database.');
    }
    const recovered = await pool.query('SELECT id,last_heartbeat FROM restaurants');
    for (const r of recovered.rows) {
      await repository.update(r.id, (s) => {
        // A new/legacy restaurant may restart before its first periodic heartbeat.
        const lastKnown =
          r.last_heartbeat ??
          s.events.at(-1)?.at ??
          s.audit.reduce((latest, entry) => (entry.at > latest ? entry.at : latest), '');
        if (!lastKnown) return s;
        const now = new Date();
        return markMonitoringGap(
          s,
          new Date(Math.min(new Date(lastKnown).getTime(), now.getTime())).toISOString(),
          now.toISOString(),
        );
      });
    }
    // No overlapping worker ticks. Deadlines are derived from durable timestamps.
    const tick = async () => {
      try {
        const restaurants = await pool.query('SELECT id FROM restaurants');
        for (const r of restaurants.rows) {
          await repository.update(r.id, (s) => processDeadlines(s, new Date().toISOString()));
          await pool.query('UPDATE restaurants SET last_heartbeat=now() WHERE id=$1', [r.id]);
        }
        await pool.query('DELETE FROM auth_sessions WHERE expires_at<=now()');
      } catch (e) {
        app.log.error(e);
      }
    };
    const scheduleTick = () => {
      if (!runningTick)
        runningTick = tick().finally(() => {
          runningTick = undefined;
        });
      return runningTick;
    };
    await scheduleTick();
    worker = setInterval(() => {
      void scheduleTick();
    }, 1000);
    worker.unref();
  }
  return app;
}
