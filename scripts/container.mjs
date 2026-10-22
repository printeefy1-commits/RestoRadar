import { readFile } from 'node:fs/promises';

// Both migrations and the service get the same URL without printing its password.
if (process.env.DATABASE_PASSWORD_FILE) {
  const password = (await readFile(process.env.DATABASE_PASSWORD_FILE, 'utf8')).trimEnd();
  if (!password) throw new Error('Database password file is empty.');
  const url = new URL('postgres://localhost/restoradar');
  url.hostname = process.env.DATABASE_HOST ?? 'postgres';
  url.port = '5432';
  url.username = 'restoradar';
  url.password = password;
  process.env.DATABASE_URL = url.toString();
}
if (!process.env.DATABASE_URL) throw new Error('A database connection is required.');
switch (process.argv[2] ?? 'serve') {
  case 'serve':
    await import('../packages/database/src/migrate.ts');
    await import('../apps/edge-api/src/main.ts');
    break;
  case 'migrate':
    await import('../packages/database/src/migrate.ts');
    break;
  case 'setup-owner':
    await import('../apps/edge-api/src/setup-owner.ts');
    break;
  default:
    throw new Error('Supported commands: serve, migrate, setup-owner.');
}
