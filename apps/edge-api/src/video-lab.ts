import type { FastifyInstance, FastifyRequest } from 'fastify';
import { Readable } from 'node:stream';
import { z } from 'zod';
import { videoTrialSchema } from '@restoradar/contracts';
import { DomainError } from '@restoradar/domain';

export async function registerVideoLab(
  app: FastifyInstance,
  baseUrl: string | undefined,
  requireOwner: (request: FastifyRequest, restaurantId: string) => Promise<void>,
  tableIds: (restaurantId: string) => Promise<Set<string>>,
) {
  type Params = { restaurantId: string; jobId?: string };
  const root = '/api/restaurants/:restaurantId/video-lab';
  async function owner(request: FastifyRequest<{ Params: Params }>) {
    const id = z.uuid().parse(request.params.restaurantId);
    await requireOwner(request, id);
    return id;
  }
  async function upstream(path: string, body?: unknown, range?: string) {
    if (!baseUrl)
      throw new DomainError('VIDEO_DISABLED', 'ميزة الفيديو لم تُفعّل على هذا الخادم', 503);
    let response: Response;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      response = await fetch(new URL(path, baseUrl), {
        method: body === undefined ? 'GET' : 'POST',
        headers: {
          ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
          ...(range ? { Range: range } : {}),
        },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        signal: controller.signal,
        redirect: 'error',
      });
    } catch {
      throw new DomainError('VIDEO_UNAVAILABLE', 'خدمة تجربة الفيديو غير متاحة حاليًا', 503);
    } finally {
      // Video may legitimately stream for longer; bound connection/header latency only.
      clearTimeout(timer);
    }
    if (!response.ok && response.status !== 416) {
      const result = await response.json().catch(() => ({}));
      throw new DomainError(
        'VIDEO_REQUEST_FAILED',
        typeof result.detail === 'string' ? result.detail : 'تعذر تنفيذ طلب الفيديو',
        response.status >= 400 && response.status < 500 ? response.status : 503,
      );
    }
    return response;
  }
  app.get<{ Params: Params }>(root, async (request) => {
    await owner(request);
    if (!baseUrl) return { enabled: false, models: 'ERROR', source: null };
    return (await upstream('/status')).json();
  });
  app.get<{ Params: Params }>(`${root}/video`, async (request, reply) => {
    await owner(request);
    const range = request.headers.range;
    // Single byte ranges only. Never proxy arbitrary headers, URLs or paths.
    if (range && !/^bytes=(?:\d{1,12}-\d{0,12}|-\d{1,12})$/.test(range))
      throw new DomainError('INVALID_RANGE', 'نطاق الفيديو غير صالح', 416);
    const response = await upstream('/video', undefined, range);
    reply.code(response.status);
    for (const header of ['content-type', 'content-length', 'content-range', 'accept-ranges']) {
      const value = response.headers.get(header);
      if (value) reply.header(header, value);
    }
    return response.body ? reply.send(Readable.fromWeb(response.body as never)) : reply.send();
  });
  app.post<{ Params: Params }>(`${root}/jobs`, async (request, reply) => {
    const id = await owner(request);
    const trial = videoTrialSchema.parse(request.body);
    const allowed = await tableIds(id);
    if (trial.regions.some((r) => !allowed.has(r.tableId)))
      throw new DomainError('INVALID_TABLE', 'اختر طاولات موجودة في هذا المطعم', 400);
    const response = await upstream(`/restaurants/${id}/jobs`, trial);
    return reply.code(response.status).send(await response.json());
  });
  app.get<{ Params: Params }>(`${root}/jobs/latest`, async (request) => {
    const id = await owner(request);
    return (await upstream(`/restaurants/${id}/jobs/latest`)).json();
  });
  app.get<{ Params: Params }>(`${root}/jobs/:jobId`, async (request) => {
    const id = await owner(request);
    const jobId = z.uuid().parse(request.params.jobId);
    return (await upstream(`/restaurants/${id}/jobs/${jobId}`)).json();
  });
  app.post<{ Params: Params }>(`${root}/models/retry`, async (request) => {
    await owner(request);
    return (await upstream('/models/retry', {})).json();
  });
}
