import { createPool, migrate } from './index.ts';
const pool = createPool();
try {
  await migrate(pool);
  console.log('Database migrations applied.');
} finally {
  await pool.end();
}
