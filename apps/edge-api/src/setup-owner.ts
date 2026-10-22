import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { createPool } from '@restoradar/database';
import { hashPassword } from './auth.ts';
// Supply the password securely through the process environment, never command arguments.
const input = z
  .object({
    RESTAURANT_NAME: z.string().trim().min(1).max(100),
    OWNER_NAME: z.string().trim().min(1).max(80),
    OWNER_EMAIL: z
      .email()
      .max(254)
      .transform((s) => s.toLowerCase()),
    OWNER_PASSWORD: z.string().min(12).max(256),
  })
  .safeParse(process.env);
if (!input.success) {
  console.error(
    'Set RESTAURANT_NAME, OWNER_NAME, OWNER_EMAIL and OWNER_PASSWORD (12–256 characters). No accounts have default credentials.',
  );
  process.exit(1);
}
const pool = createPool(),
  c = await pool.connect();
try {
  await c.query('BEGIN');
  await c.query('SELECT pg_advisory_xact_lock(108294602)');
  const exists = await c.query('SELECT 1 FROM restaurants LIMIT 1');
  if (exists.rowCount)
    throw new Error('Already initialized. Use the owner dashboard to add accounts.');
  const restaurantId = randomUUID(),
    userId = randomUUID();
  await c.query('INSERT INTO restaurants(id,name,last_heartbeat) VALUES($1,$2,now())', [
    restaurantId,
    input.data.RESTAURANT_NAME,
  ]);
  await c.query('INSERT INTO users(id,email,name,password_hash) VALUES($1,$2,$3,$4)', [
    userId,
    input.data.OWNER_EMAIL,
    input.data.OWNER_NAME,
    await hashPassword(input.data.OWNER_PASSWORD),
  ]);
  await c.query("INSERT INTO memberships(restaurant_id,user_id,role) VALUES($1,$2,'OWNER')", [
    restaurantId,
    userId,
  ]);
  const auditId = randomUUID();
  await c.query('INSERT INTO audit_log(id,restaurant_id,data) VALUES($1,$2,$3)', [
    auditId,
    restaurantId,
    JSON.stringify({
      id: auditId,
      actorId: userId,
      action: 'restaurant.initialized',
      at: new Date().toISOString(),
      context: {},
    }),
  ]);
  await c.query('COMMIT');
  console.log('Owner created. Sign in, enter timeout policy, and add tables and staff.');
} catch (e) {
  await c.query('ROLLBACK');
  console.error(e instanceof Error ? e.message : 'Initialization failed');
  process.exitCode = 1;
} finally {
  c.release();
  await pool.end();
}
