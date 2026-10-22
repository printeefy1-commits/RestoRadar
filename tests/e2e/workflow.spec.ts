import { test, expect, type BrowserContext, type Page } from '@playwright/test';
import { randomBytes, randomUUID } from 'node:crypto';
import { Pool, migrate, type DatabasePool } from '@restoradar/database';
import { buildApp } from '../../apps/edge-api/src/app.ts';
import { hashPassword } from '../../apps/edge-api/src/auth.ts';

if (!process.env.TEST_DATABASE_URL) throw new Error('TEST_DATABASE_URL is required.');
const schema = `e2e_${randomUUID().replaceAll('-', '')}`;
const admin = new Pool({ connectionString: process.env.TEST_DATABASE_URL });
const password = randomBytes(24).toString('hex');
const restaurantId = randomUUID(),
  ownerId = randomUUID(),
  waiterId = randomUUID();
let pool: DatabasePool, app: Awaited<ReturnType<typeof buildApp>>, address: string;
test.beforeAll(async () => {
  await admin.query(`CREATE SCHEMA ${schema}`);
  pool = new Pool({
    connectionString: process.env.TEST_DATABASE_URL,
    options: `-c search_path=${schema},public`,
  });
  await migrate(pool);
  await pool.query('INSERT INTO restaurants(id,name) VALUES($1,$2)', [
    restaurantId,
    'مطعم التجربة',
  ]);
  const hash = await hashPassword(password);
  for (const u of [
    { id: ownerId, email: 'owner@e2e.invalid', name: 'مالك التجربة', role: 'OWNER' },
    { id: waiterId, email: 'waiter@e2e.invalid', name: 'نادل التجربة', role: 'WAITER' },
  ]) {
    await pool.query('INSERT INTO users(id,email,name,password_hash) VALUES($1,$2,$3,$4)', [
      u.id,
      u.email,
      u.name,
      hash,
    ]);
    await pool.query('INSERT INTO memberships(restaurant_id,user_id,role) VALUES($1,$2,$3)', [
      restaurantId,
      u.id,
      u.role,
    ]);
  }
  app = await buildApp(pool, { simulatorEnabled: true });
  address = await app.listen({ host: '127.0.0.1', port: 0 });
});
test.afterAll(async () => {
  if (app) await app.close();
  if (pool) await pool.end();
  await admin.query(`DROP SCHEMA IF EXISTS ${schema} CASCADE`);
  await admin.end();
});
async function login(page: Page, email: string) {
  await page.goto(address);
  await page.getByLabel('البريد الإلكتروني').fill(email);
  await page.getByLabel('كلمة المرور', { exact: true }).fill(password);
  await page.getByRole('button', { name: 'دخول إلى المطعم' }).click();
  await expect(page.getByRole('heading', { name: 'الطاولات', exact: true })).toBeVisible();
}
async function noOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
}

test('Android-sized owner/waiter workflow: explicit setup, arrival, reception, departure, cleaning and manager readiness', async ({
  page,
  browser,
}, testInfo) => {
  const browserErrors: string[] = [];
  page.on('pageerror', (e) => browserErrors.push(e.message));
  await login(page, 'owner@e2e.invalid');
  await expect(page.getByText('مهلات المهام لم تُضبط بعد.', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'الإعدادات', exact: true }).click();
  const policy = page
    .locator('section.panel')
    .filter({ has: page.getByRole('heading', { name: 'مهلات الفريق' }) });
  for (const input of await policy.locator('input').all())
    expect(await input.inputValue()).toBe('');
  const reception = policy
    .locator('fieldset')
    .filter({ has: page.locator('legend', { hasText: 'استقبال الضيوف' }) });
  const cleaning = policy
    .locator('fieldset')
    .filter({ has: page.locator('legend', { hasText: 'تنظيف الطاولة' }) });
  await reception.getByLabel('مهلة القبول', { exact: true }).fill('60');
  await reception.getByLabel('مهلة الإكمال بعد القبول').fill('180');
  await cleaning.getByLabel('مهلة القبول', { exact: true }).fill('120');
  await cleaning.getByLabel('مهلة الإكمال بعد القبول').fill('420');
  await policy.getByRole('button', { name: 'حفظ مهلات المهام' }).click();
  await expect(page.getByRole('status')).toHaveText('حُفظت المهلات للتعيينات الجديدة');
  const tableForm = page
    .locator('section.panel')
    .filter({ has: page.getByRole('heading', { name: 'إضافة طاولة', exact: true }) });
  await tableForm.getByLabel('اسم الطاولة').fill('طاولة الحديقة');
  await tableForm.getByRole('button', { name: 'إضافة الطاولة', exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('أُضيفت الطاولة');
  await page.getByRole('button', { name: 'الطاولات', exact: true }).click();
  const card = page.locator('.table-card').filter({ hasText: 'طاولة الحديقة' });
  await expect(card.getByText('فارغة · جاهزة', { exact: true })).toBeVisible();
  await card.getByText('محاكاة أحداث الطاولة', { exact: true }).click();
  await card.getByRole('button', { name: 'جلوس متحقق 5 ث' }).click();
  await expect(card.getByText('مشغولة', { exact: true })).toBeVisible();
  await expect(page.locator('.arrival-banner')).toBeVisible();
  await page.locator('.arrival-banner').getByRole('button', { name: 'تعيين نادل' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('النادل', { exact: true }).selectOption(waiterId);
  await dialog.getByRole('button', { name: 'تأكيد التعيين' }).click();
  await expect(page.getByRole('status')).toHaveText('حُفظ تعيين المهمة');
  await page.locator('.arrival-banner').getByRole('button', { name: 'إقرار الاطلاع' }).click();
  await expect(page.locator('.arrival-banner')).toHaveCount(0);
  await noOverflow(page);
  await page.screenshot({ path: testInfo.outputPath('android-tables.png'), fullPage: true });

  let waiterContext: BrowserContext | undefined;
  try {
    waiterContext = await browser.newContext({
      viewport: { width: 393, height: 851 },
      isMobile: true,
      hasTouch: true,
    });
    const waiterPage = await waiterContext.newPage();
    await login(waiterPage, 'waiter@e2e.invalid');
    await expect(waiterPage.getByRole('button', { name: 'الإعدادات', exact: true })).toHaveCount(0);
    await expect(waiterPage.getByRole('button', { name: 'التنبيهات', exact: true })).toHaveCount(0);
    await waiterPage.getByRole('button', { name: 'مهامي', exact: true }).click();
    await waiterPage.getByRole('button', { name: 'قبول المهمة', exact: true }).click();
    await waiterPage.getByRole('button', { name: 'إكمال المهمة', exact: true }).click();
    await expect(waiterPage.getByRole('status')).toContainText('اكتملت المهمة');
    await noOverflow(waiterPage);

    await card.getByRole('button', { name: 'تصحيح الإشغال' }).click();
    await dialog.getByLabel('الإشغال المعتمد').selectOption('EMPTY');
    await dialog.getByLabel('السبب المطلوب').fill('تأكد المدير من مغادرة الضيوف');
    await dialog.getByRole('button', { name: 'حفظ التصحيح' }).click();
    await expect(card.getByText('فارغة · غير جاهزة', { exact: true })).toBeVisible();
    await card.getByRole('button', { name: 'تعيين تنظيف' }).click();
    await dialog.getByLabel('النادل', { exact: true }).selectOption(waiterId);
    await dialog.getByRole('button', { name: 'تأكيد التعيين' }).click();
    await waiterPage.getByRole('button', { name: 'قبول المهمة', exact: true }).click();
    await waiterPage.getByRole('button', { name: 'إكمال المهمة', exact: true }).click();
    await page.getByRole('button', { name: 'تحديث', exact: false }).click();
    await expect(card.getByText('فارغة · غير جاهزة', { exact: true })).toBeVisible();
    await expect(
      waiterPage.getByRole('button', { name: 'تأكيد الجاهزية', exact: false }),
    ).toHaveCount(0);
    await card.getByRole('button', { name: 'تأكيد الجاهزية', exact: false }).click();
    await expect(card.getByText('فارغة · جاهزة', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'السجل', exact: true }).click();
    await expect(page.locator('.session-record')).toHaveCount(1);
    await expect(page.getByText('تأكد المدير من مغادرة الضيوف', { exact: true })).toBeVisible();
    expect(browserErrors).toEqual([]);
  } finally {
    await waiterContext?.close();
  }
});

test('desktop login is responsive and data is visibly stale after a failed refresh', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await login(page, 'owner@e2e.invalid');
  await noOverflow(page);
  const created = await page.request.post(`${address}/api/restaurants/${restaurantId}/commands`, {
    data: { commandId: randomUUID(), command: { type: 'table.create', name: 'طاولة الاتصال' } },
  });
  expect(created.ok()).toBe(true);
  await page.getByRole('button', { name: 'تحديث', exact: false }).click();
  await expect(page.locator('.table-card').filter({ hasText: 'طاولة الاتصال' })).toBeVisible();
  await page.route('**/api/restaurants/*/dashboard', (route) => route.abort());
  await page.getByRole('button', { name: 'تحديث', exact: false }).click();
  await expect(page.getByText('الاتصال غير متاح', { exact: true })).toBeVisible();
  await expect(page.getByText('البيانات قديمة أو غير متاحة.', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'تصحيح الإشغال' }).first()).toBeDisabled();
});
