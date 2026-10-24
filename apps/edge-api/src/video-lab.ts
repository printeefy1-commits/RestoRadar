import type { FastifyInstance, FastifyRequest } from 'fastify';
import { Readable } from 'node:stream';
import { z } from 'zod';
import { MAX_VIDEO_UPLOAD_BYTES, videoTrialSchema } from '@restoradar/contracts';
import { DomainError } from '@restoradar/domain';

declare module 'fastify' {
  interface FastifyContextConfig {
    videoUpload?: boolean;
  }
}

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
    const id = await owner(request);
    if (!baseUrl) return { enabled: false, models: 'ERROR', source: null };
    return (await upstream(`/restaurants/${id}/status`)).json();
  });
  app.get<{ Params: Params; Querystring: { source?: string } }>(
    `${root}/video`,
    async (request, reply) => {
      const id = await owner(request);
      const source = request.query.source;
      if (source !== undefined && !/^[a-f0-9]{64}$/.test(source))
        throw new DomainError('INVALID_SOURCE', 'معرف الفيديو غير صالح', 400);
      const range = request.headers.range;
      // Single byte ranges only. Never proxy arbitrary headers, URLs or paths.
      if (range && !/^bytes=(?:\d{1,12}-\d{0,12}|-\d{1,12})$/.test(range))
        throw new DomainError('INVALID_RANGE', 'نطاق الفيديو غير صالح', 416);
      const response = await upstream(
        `/restaurants/${id}/video${source ? `?source=${source}` : ''}`,
        undefined,
        range,
      );
      reply.code(response.status);
      for (const header of ['content-type', 'content-length', 'content-range', 'accept-ranges']) {
        const value = response.headers.get(header);
        if (value) reply.header(header, value);
      }
      return response.body ? reply.send(Readable.fromWeb(response.body as never)) : reply.send();
    },
  );
  await app.register(async (uploadApp) => {
    // Stream binary content only on this route; keep the normal JSON body limit.
    uploadApp.addContentTypeParser('video/mp4', (_request, stream, done) => done(null, stream));
    uploadApp.post<{ Params: Params }>(
      `${root}/video`,
      {
        config: { videoUpload: true, rateLimit: { max: 6, timeWindow: '1 minute' } },
        bodyLimit: MAX_VIDEO_UPLOAD_BYTES,
        onRequest: async (request) => {
          await owner(request);
          if (request.headers['content-type'] !== 'video/mp4')
            throw new DomainError('MP4_REQUIRED', 'استخدم فيديو MP4', 415);
          const length = request.headers['content-length'];
          if (length && (!/^\d+$/.test(length) || Number(length) > MAX_VIDEO_UPLOAD_BYTES))
            throw new DomainError('VIDEO_TOO_LARGE', 'الحد الأقصى للفيديو 100 ميغابايت', 413);
        },
      },
      async (request, reply) => {
        const id = z.uuid().parse(request.params.restaurantId);
        if (!baseUrl)
          throw new DomainError('VIDEO_DISABLED', 'ميزة الفيديو لم تُفعّل على هذا الخادم', 503);
        const declared = request.headers['content-length'];
        const controller = new AbortController();
        const abort = () => controller.abort();
        request.raw.once('aborted', abort);
        const closed = () => {
          if (!reply.raw.writableFinished) abort();
        };
        reply.raw.once('close', closed);
        const timer = setTimeout(abort, 900000);
        let streamError: Error | undefined;
        let bytes = 0;
        const body = Readable.from(
          (async function* () {
            try {
              for await (const chunk of request.body as Readable) {
                bytes += chunk.length;
                if (bytes > MAX_VIDEO_UPLOAD_BYTES)
                  throw new DomainError('VIDEO_TOO_LARGE', 'الحد الأقصى للفيديو 100 ميغابايت', 413);
                yield chunk;
              }
              if (!bytes || (declared && bytes !== Number(declared)))
                throw new DomainError('INCOMPLETE_UPLOAD', 'لم يصل ملف فيديو كامل', 400);
            } catch (error) {
              streamError = error as Error;
              throw error;
            }
          })(),
        );
        try {
          const response = await fetch(new URL(`/restaurants/${id}/video`, baseUrl), {
            method: 'POST',
            headers: {
              'Content-Type': 'video/mp4',
              ...(declared ? { 'Content-Length': declared } : {}),
              'X-Video-Name':
                typeof request.headers['x-video-name'] === 'string'
                  ? request.headers['x-video-name'].slice(0, 2048)
                  : 'video.mp4',
            },
            body: body as never,
            duplex: 'half',
            signal: controller.signal,
            redirect: 'error',
          } as RequestInit & { duplex: 'half' });
          const result = await response.json();
          if (!response.ok)
            throw new DomainError(
              'VIDEO_UPLOAD_FAILED',
              typeof result.detail === 'string' ? result.detail : 'تعذر رفع الفيديو',
              response.status < 500 ? response.status : 503,
            );
          return reply.code(201).send(result);
        } catch (error) {
          if (streamError instanceof DomainError) throw streamError;
          if (error instanceof DomainError) throw error;
          throw new DomainError(
            'VIDEO_UPLOAD_FAILED',
            'لم يكتمل رفع الفيديو؛ تحقق من الاتصال وحاول مجددًا',
            503,
          );
        } finally {
          clearTimeout(timer);
          request.raw.off('aborted', abort);
          reply.raw.off('close', closed);
          body.destroy();
        }
      },
    );
  });
  app.post<{ Params: Params }>(`${root}/video/delete`, async (request) => {
    const id = await owner(request);
    const body = z
      .object({ sourceId: z.string().regex(/^[a-f0-9]{64}$/) })
      .strict()
      .parse(request.body);
    return (await upstream(`/restaurants/${id}/video/delete`, body)).json();
  });
  app.post<{ Params: Params }>(`${root}/jobs`, async (request, reply) => {
    const id = await owner(request);
    const trial = videoTrialSchema.parse(request.body);
    const allowed = await tableIds(id);
    if ([...trial.regions, ...(trial.tabletopRegions ?? [])].some((r) => !allowed.has(r.tableId)))
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
