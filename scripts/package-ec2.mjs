import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const files = execFileSync(
  'git',
  ['ls-files', '--cached', '--others', '--exclude-standard', '-z'],
  { cwd: root, encoding: 'utf8' },
)
  .split('\0')
  .filter(Boolean)
  .filter((file) => {
    const parts = file.split('/');
    const rootFiles = new Set([
      'package.json',
      'package-lock.json',
      'tsconfig.json',
      'vitest.config.ts',
      'vitest.integration.config.ts',
      'playwright.config.ts',
      'README.md',
      'REQUIREMENTS.md',
      '.gitignore',
      '.dockerignore',
      '.nvmrc',
      '.prettierrc.json',
      '.prettierignore',
      '.env.example',
    ]);
    if (
      parts.length === 1
        ? !rootFiles.has(file)
        : !['apps', 'packages', 'infra', 'scripts', 'docs', 'tests', '.github'].includes(parts[0])
    )
      return false;
    if (
      parts.some((p) =>
        [
          '.git',
          '.secrets',
          'node_modules',
          'dist',
          'artifacts',
          'recordings',
          'videos',
          'test-results',
          'playwright-report',
          'coverage',
        ].includes(p),
      )
    )
      return false;
    const name = parts.at(-1);
    if (['.npmrc', 'credentials', 'id_rsa', 'id_ed25519'].includes(name)) return false;
    if (
      file !== 'tests/fixtures/trial-preview.mp4' &&
      /\.(mp4|mov|avi|mkv|webm|m4v|mpeg|mpg|pem|key|p12|pfx)$/i.test(name)
    )
      return false;
    return (
      name === '.env.example' ||
      (!name.startsWith('.env') && !name.endsWith('.env') && !name.endsWith('.log'))
    );
  });
for (const file of files) {
  if ((await lstat(resolve(root, file))).isSymbolicLink())
    throw new Error(`Refusing symlink in deployment archive: ${file}`);
}
await mkdir(resolve(root, 'artifacts'), { recursive: true });
const archive = resolve(root, 'artifacts/restoradar-ec2.tar.gz');
execFileSync(
  'tar',
  ['--create', '--gzip', '--file', archive, '--null', '--verbatim-files-from', '--files-from', '-'],
  { cwd: root, input: files.join('\0') + '\0' },
);
const checksum = createHash('sha256')
  .update(await readFile(archive))
  .digest('hex');
await writeFile(`${archive}.sha256`, `${checksum}  restoradar-ec2.tar.gz\n`);
console.log(
  `Packaged ${files.length} source files into artifacts/restoradar-ec2.tar.gz; checksum saved alongside it.`,
);
