import { createPool } from '@restoradar/database';
import { buildApp } from './app.ts';
const pool = createPool();
const app = await buildApp(pool, {
  simulatorEnabled: process.env.ENABLE_SIMULATOR === 'true',
  secureCookies: process.env.COOKIE_SECURE === 'true',
  publicOrigin: process.env.PUBLIC_ORIGIN,
  workers: true,
  logger: true,
});
// Fastify runs onClose hooks in reverse order. End the pool only after the
// worker's close hook has returned its dedicated advisory-lock connection.
let closing: Promise<void> | undefined;
function shutdown() {
  closing ??= (async () => {
    try {
      await app.close();
    } finally {
      await pool.end();
    }
  })();
  return closing;
}
try {
  await app.listen({
    host: process.env.HOST ?? '127.0.0.1',
    port: Number(process.env.PORT ?? 3001),
  });
} catch (error) {
  await shutdown();
  throw error;
}
for (const signal of ['SIGINT', 'SIGTERM'] as const)
  process.on(signal, () => {
    void shutdown().catch((error) => {
      app.log.error(error);
      process.exitCode = 1;
    });
  });
