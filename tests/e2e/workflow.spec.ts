import { test, expect, type BrowserContext, type Page } from '@playwright/test';
import { randomBytes, randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import type { TrialReport, VideoTrial } from '@restoradar/contracts';
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

test('owner lists all roles and deletes an account with confirmation while self remains protected', async ({
  page,
}) => {
  await login(page, 'owner@e2e.invalid');
  const name = 'مدير للحذف';
  expect(
    (
      await page.request.post(`${address}/api/restaurants/${restaurantId}/users`, {
        data: { name, email: 'delete@e2e.invalid', password, role: 'MANAGER' },
      })
    ).ok(),
  ).toBe(true);
  await page.getByRole('button', { name: 'تحديث', exact: false }).click();
  await page.getByRole('button', { name: 'الإعدادات', exact: true }).click();
  await expect(page.getByRole('button', { name: 'حسابك الحالي محمي' })).toBeDisabled();
  const accounts = page
    .locator('section.panel')
    .filter({ has: page.getByRole('heading', { name: 'جميع حسابات المطعم', exact: true }) });
  for (const email of ['owner@e2e.invalid', 'waiter@e2e.invalid', 'delete@e2e.invalid'])
    await expect(accounts.getByText(email, { exact: true })).toBeVisible();
  for (const role of ['المالك', 'المدير', 'النادل'])
    await expect(accounts.getByText(role, { exact: true })).toBeVisible();
  await noOverflow(page);
  await expect(
    page.getByRole('button', { name: 'حذف حساب نادل التجربة', exact: true }),
  ).toBeVisible();
  page.once('dialog', (dialog) => dialog.dismiss());
  await page.getByRole('button', { name: `حذف حساب ${name}`, exact: true }).click();
  await expect(page.getByRole('button', { name: `حذف حساب ${name}`, exact: true })).toBeVisible();
  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: `حذف حساب ${name}`, exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('حُذف الحساب من المطعم مع حفظ السجل');
  await expect(page.getByRole('button', { name: `حذف حساب ${name}`, exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'السجل', exact: true }).click();
  await expect(page.getByText('حذف حساب من المطعم', { exact: true })).toBeVisible();
});

test('deleting a waiter cancels an accepted task and allows manual reassignment with history intact', async ({
  page,
}) => {
  await login(page, 'owner@e2e.invalid');
  const base = `${address}/api/restaurants/${restaurantId}`;
  const created = await page.request.post(`${base}/users`, {
    data: { name: 'نادل مؤقت', email: 'temporary@e2e.invalid', password, role: 'WAITER' },
  });
  expect(created.ok()).toBe(true);
  const temporaryId = (await created.json()).id;
  const send = async (command: unknown) => {
    expect(
      (
        await page.request.post(`${base}/commands`, { data: { commandId: randomUUID(), command } })
      ).ok(),
    ).toBe(true);
  };
  await send({ type: 'table.create', name: 'طاولة حذف الحساب' });
  const snapshot = async () => (await page.request.get(`${base}/dashboard`)).json();
  const table = (await snapshot()).tables.find(
    (t: { name: string }) => t.name === 'طاولة حذف الحساب',
  );
  const confirmed = Date.now() - 100;
  await send({
    type: 'simulation.transition',
    tableId: table.id,
    occupancy: 'OCCUPIED',
    observedAt: new Date(confirmed - 5000).toISOString(),
    confirmedAt: new Date(confirmed).toISOString(),
  });
  const state = await snapshot();
  const sourceId = state.assignments.find(
    (a: { tableId: string }) => a.tableId === table.id,
  ).sessionId;
  await send({
    type: 'task.assign',
    kind: 'CUSTOMER_RECEPTION',
    sourceId,
    waiterId: temporaryId,
    expectedVersion: 0,
  });
  const task = (await snapshot()).tasks.find((t: { sourceId: string }) => t.sourceId === sourceId);
  const auth = await app.inject({
    method: 'POST',
    url: '/api/auth/login',
    payload: { email: 'temporary@e2e.invalid', password },
  });
  expect(auth.statusCode).toBe(200);
  expect(
    (
      await app.inject({
        method: 'POST',
        url: `/api/restaurants/${restaurantId}/commands`,
        headers: { cookie: String(auth.headers['set-cookie']).split(';')[0] },
        payload: {
          commandId: randomUUID(),
          command: { type: 'task.accept', taskId: task.id, expectedVersion: task.version },
        },
      })
    ).statusCode,
  ).toBe(200);
  await page.getByRole('button', { name: 'تحديث', exact: false }).click();
  await page.getByRole('button', { name: 'الإعدادات', exact: true }).click();
  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'حذف حساب نادل مؤقت', exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('حُذف الحساب من المطعم مع حفظ السجل');
  await page.getByRole('button', { name: 'مهام الفريق', exact: true }).click();
  const row = page.locator('article.list-row').filter({ hasText: 'طاولة حذف الحساب' });
  await expect(row).toContainText('نادل مؤقت');
  await expect(
    row.getByText('أُلغيت لحذف حساب النادل؛ السجل محفوظ ويمكن إعادة الإسناد.', { exact: true }),
  ).toBeVisible();
  await row.getByRole('button', { name: 'إعادة التعيين', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog.locator('option').filter({ hasText: 'نادل مؤقت' })).toHaveCount(0);
  await dialog.getByLabel('النادل', { exact: true }).selectOption(waiterId);
  await dialog.getByRole('button', { name: 'تأكيد التعيين' }).click();
  await expect(page.getByRole('status')).toHaveText('حُفظ تعيين المهمة');
  const after = await snapshot();
  expect(after.attempts.find((a: { id: string }) => a.id === task.currentAttemptId).status).toBe(
    'CANCELLED',
  );
  expect(after.tasks.find((t: { id: string }) => t.id === task.id).status).toBe('ASSIGNED');
  await noOverflow(page);
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

test('Android video trial: authenticated preview, seat mapping, report and no live changes', async ({
  page,
}, testInfo) => {
  await login(page, 'owner@e2e.invalid');
  const sourceId = 'b'.repeat(64),
    jobId = randomUUID();
  const video = await readFile(new URL('../fixtures/trial-preview.mp4', import.meta.url));
  let submitted: VideoTrial | undefined;
  const browserErrors: string[] = [];
  page.on('pageerror', (e) => browserErrors.push(e.message));
  await page.route('**/api/restaurants/*/video-lab**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith('/video')) {
      const range = route.request().headers().range;
      const match = range?.match(/^bytes=(\d+)-(\d*)$/);
      const start = match ? Number(match[1]) : 0;
      const end = match?.[2] ? Math.min(Number(match[2]), video.length - 1) : video.length - 1;
      return route.fulfill({
        status: match ? 206 : 200,
        contentType: 'video/mp4',
        headers: {
          'Accept-Ranges': 'bytes',
          'Content-Length': String(end - start + 1),
          ...(match ? { 'Content-Range': `bytes ${start}-${end}/${video.length}` } : {}),
        },
        body: video.subarray(start, end + 1),
      });
    }
    if (path.endsWith('/jobs/latest')) return route.fulfill({ json: null });
    if (path.endsWith('/jobs') && route.request().method() === 'POST') {
      submitted = route.request().postDataJSON() as VideoTrial;
      return route.fulfill({
        status: 202,
        json: {
          id: jobId,
          status: 'RUNNING',
          progress: 0,
          message: 'جارٍ تجهيز نموذج تقدير الجلوس؛ قد يتضمن تنزيله لأول مرة.',
        },
      });
    }
    if (path.endsWith(`/jobs/${jobId}`)) {
      const report: TrialReport = {
        mode: 'RECORDED_VIDEO_DRY_RUN',
        sourceId,
        startSeconds: 0,
        endSeconds: 6,
        models: {
          detector: 'synthetic test graph',
          pose: 'synthetic test graph',
          tracker: 'ByteTrack',
          artifacts: {},
        },
        analyzedSamples: 13,
        processingSeconds: 2,
        samplesPerProcessingSecond: 6.5,
        departureEvaluated: false,
        detectionMode: submitted!.detectionMode,
        poseModel: submitted!.poseModel,
        wristTrial: submitted!.wristTrial,
        tabletopRegions: submitted!.tabletopRegions,
        wristPolicy: {
          version: 'WRIST_SEAT_DWELL_V1',
          seconds: 5,
          minKeypointScore: 0.4,
          maxDriftBoxWidths: 0.25,
          maxDriftBoxHeights: 0.15,
        },
        regions: submitted!.regions,
        events: [
          {
            tableId: submitted!.regions[0].tableId,
            kind: 'SEATING_CONFIRMED',
            observedSeconds: 0,
            confirmedSeconds: 5,
          },
        ],
        samples: [
          {
            seconds: 5,
            people: [
              {
                box: [0.1, 0.1, 0.4, 0.8],
                trackId: 1,
                posture: 'SEATED',
                tableId: submitted!.regions[0].tableId,
                score: 0.9,
              },
              {
                box: [0.55, 0.1, 0.8, 0.8],
                trackId: 39,
                posture: 'SEATED',
                tableId: submitted!.regions[0].tableId,
                score: 0.1659,
              },
            ],
            tables: { [submitted!.regions[0].tableId]: 'OCCUPIED' },
          },
          {
            seconds: 5.5,
            detectedBoxes: 6,
            people: [
              {
                box: [0.55, 0.1, 0.8, 0.8],
                trackId: 39,
                posture: 'SEATED',
                tableId: submitted!.regions[0].tableId,
                score: 0.1659,
                evidence: 'LOW_DETECTION_SCORE',
              },
            ],
            tables: { [submitted!.regions[0].tableId]: 'UNKNOWN' },
          },
          {
            seconds: 6,
            people: [
              {
                box: [0.1, 0.1, 0.4, 0.8],
                trackId: 77,
                posture: 'UNCERTAIN',
                tableId: null,
                score: 0.9,
                evidence: 'POSE_UNCERTAIN',
                wristEvidence: {
                  state: 'PROBABLE',
                  reason: 'PROBABLE',
                  stableSeconds: 5,
                  tableId: submitted!.regions[0].tableId,
                  seatRegionId: submitted!.regions[0].id,
                  bodyAnchor: [0.2, 0.3],
                  anchorSource: 'BOX_CENTER',
                },
                posePoints: [{ part: 'LEFT_WRIST', point: [0.55, 0.35], score: 0.9 }],
              },
            ],
            tables: { [submitted!.regions[0].tableId]: 'UNKNOWN' },
            wristTables: { [submitted!.regions[0].tableId]: 'PROBABLE' },
          },
        ],
      };
      return route.fulfill({ json: { id: jobId, status: 'COMPLETED', progress: 1, report } });
    }
    return route.fulfill({
      json: {
        enabled: true,
        models: 'READY',
        source: { id: sourceId, duration: 8, width: 320, height: 240, bytes: video.length },
      },
    });
  });
  const response = await page.request.post(`${address}/api/restaurants/${restaurantId}/commands`, {
    data: { commandId: randomUUID(), command: { type: 'table.create', name: 'طاولة الفيديو' } },
  });
  expect(response.ok()).toBe(true);
  await page.getByRole('button', { name: 'تحديث', exact: false }).click();
  const before = (
    await page.request.get(`${address}/api/restaurants/${restaurantId}/dashboard`)
  ).json();
  await page.getByRole('button', { name: 'تجربة الفيديو', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'فيديو التجربة', exact: true })).toBeVisible();
  await expect
    .poll(() => page.locator('video').evaluate((element: HTMLVideoElement) => element.readyState))
    .toBeGreaterThanOrEqual(1);
  await page.getByLabel('الطاولة المرتبطة').selectOption({ label: 'طاولة الفيديو' });
  await page.getByText('إضافة منطقة بالأرقام بدل الرسم', { exact: true }).click();
  await page.getByLabel('البداية الأفقية %').fill('10');
  await page.getByLabel('البداية الرأسية %').fill('20');
  await page.getByLabel('العرض %').fill('20');
  await page.getByLabel('الارتفاع %').fill('20');
  await page.getByRole('button', { name: 'إضافة المنطقة', exact: true }).click();
  await expect(
    page.getByRole('checkbox', { name: 'تجربة إشارة المعصمين', exact: true }),
  ).not.toBeChecked();
  await page.getByRole('checkbox', { name: 'تجربة إشارة المعصمين', exact: true }).check();
  await expect(
    page.getByRole('button', { name: 'تحليل الفيديو على CPU', exact: true }),
  ).toBeDisabled();
  await page.getByRole('combobox', { name: 'نوع المنطقة', exact: true }).selectOption('TABLETOP');
  await page.getByLabel('البداية الأفقية %').fill('50');
  await page.getByLabel('البداية الرأسية %').fill('30');
  await page.getByLabel('العرض %').fill('40');
  await page.getByLabel('الارتفاع %').fill('30');
  await page.getByRole('button', { name: 'إضافة المنطقة', exact: true }).click();
  await expect(page.getByRole('button', { name: 'حذف سطح الطاولة 1', exact: true })).toBeVisible();
  await expect(page.getByLabel('طريقة رصد الأشخاص')).toHaveValue('FULL_FRAME');
  await expect(page.getByRole('combobox', { name: 'نموذج تقدير الجلوس', exact: true })).toHaveValue(
    'YOLO26_NANO_POSE',
  );
  await page.getByLabel('طريقة رصد الأشخاص').selectOption('SEAT_FOCUS');
  await page
    .getByRole('combobox', { name: 'نموذج تقدير الجلوس', exact: true })
    .selectOption('RTMPOSE_MEDIUM');
  await page.getByLabel('مدة الاختبار بالثواني').fill('6');
  await page.getByRole('button', { name: 'تحليل الفيديو على CPU', exact: true }).click();
  await expect(
    page.getByText('جارٍ تجهيز نموذج تقدير الجلوس؛ قد يتضمن تنزيله لأول مرة.', { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: 'نتيجة التجربة', exact: true })).toBeVisible();
  expect(submitted?.detectionMode).toBe('SEAT_FOCUS');
  expect(submitted?.poseModel).toBe('RTMPOSE_MEDIUM');
  expect(submitted?.wristTrial).toBe(true);
  expect(submitted?.tabletopRegions?.[0]).toMatchObject({
    x: 0.5,
    y: 0.3,
    width: 0.4,
    height: 0.3,
  });
  await expect(
    page.getByText('طريقة الرصد في التقرير: الصورة ومناطق المقاعد', { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText('نموذج تقدير الجلوس في التقرير: النموذج الأكبر', { exact: true }),
  ).toBeVisible();
  await page
    .getByRole('combobox', { name: 'نموذج تقدير الجلوس', exact: true })
    .selectOption('RTMPOSE_TINY');
  await expect(
    page.getByText('نموذج تقدير الجلوس في التقرير: النموذج الأكبر', { exact: true }),
  ).toBeVisible();
  expect(submitted?.regions[0]).toMatchObject({ x: 0.1, y: 0.2, width: 0.2, height: 0.2 });
  await page.getByRole('button', { name: 'عرض اللحظة', exact: true }).click();
  await expect
    .poll(() => page.locator('video').evaluate((element: HTMLVideoElement) => element.currentTime))
    .toBeCloseTo(5, 1);
  const diagnostics = page.getByRole('region', { name: 'تشخيص العينة الحالية', exact: true });
  await expect(diagnostics).toContainText(
    'أشخاص مرصودون: 2 · بوضعية جلوس: 2 · أدلة مقبولة للتحقق: 1',
  );
  await diagnostics.getByText('أسباب قبول الرصد واستبعاده', { exact: true }).click();
  await expect(diagnostics.getByText(/#39 · جلوس مستبعد: رصد ضعيف/)).toBeVisible();
  // Inspect the actual canvas: blue seat pixels must disappear even with a retained report.
  const colorPixels = (red: number, green: number, blue: number) =>
    page.locator('canvas').evaluate(
      (canvas: HTMLCanvasElement, rgb: number[]) => {
        const pixels = canvas
          .getContext('2d')!
          .getImageData(0, 0, canvas.width, canvas.height).data;
        let count = 0;
        for (let i = 0; i < pixels.length; i += 4)
          if (
            pixels[i] === rgb[0] &&
            pixels[i + 1] === rgb[1] &&
            pixels[i + 2] === rgb[2] &&
            pixels[i + 3] > 0
          )
            count++;
        return count;
      },
      [red, green, blue],
    );
  const bluePixels = () => colorPixels(66, 133, 255);
  const purplePixels = () => colorPixels(167, 139, 250);
  const greenPixels = () => colorPixels(35, 211, 152);
  await expect.poll(bluePixels).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'حذف المنطقة 1', exact: true }).click();
  await expect.poll(bluePixels).toBe(0);
  await page.locator('video').evaluate((element: HTMLVideoElement) => {
    element.currentTime = 5.5;
  });
  await expect(diagnostics).toContainText(
    'أشخاص مرصودون: 1 · بوضعية جلوس: 1 · أدلة مقبولة للتحقق: 0',
  );
  await expect(diagnostics).toContainText('صناديق الأشخاص قبل التتبع: 6');
  await expect(diagnostics.getByText(/الإشغال غير محسوم في هذه العينة/)).toBeVisible();
  await expect.poll(greenPixels).toBe(0);
  await expect.poll(purplePixels).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'حذف سطح الطاولة 1', exact: true }).click();
  await expect.poll(purplePixels).toBe(0);
  await page.locator('video').evaluate((element: HTMLVideoElement) => {
    element.currentTime = 6;
  });
  const wristDiagnostics = page.getByRole('region', {
    name: 'إشارة المعصمين التجريبية',
    exact: true,
  });
  await expect(wristDiagnostics).toContainText('إشغال مرجّح بإشارة المعصمين: 1');
  await expect(wristDiagnostics).toContainText('ليس تأكيد جلوس');
  await expect(diagnostics).toContainText('أدلة مقبولة للتحقق: 0');
  await expect.poll(greenPixels).toBe(0);
  await page.getByRole('button', { name: 'استخدام مناطق التقرير السابق', exact: true }).click();
  await expect.poll(bluePixels).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'حذف المنطقة 1', exact: true }).click();
  await expect.poll(bluePixels).toBe(0);
  const importReport = page.getByLabel('استيراد مناطق من تقرير JSON');
  await importReport.setInputFiles({
    name: 'wrong-source.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ sourceId: 'c'.repeat(64), regions: submitted!.regions })),
  });
  await expect(page.getByRole('alert')).toContainText('التقرير يخص ملف فيديو مختلفًا');
  await expect.poll(bluePixels).toBe(0);
  await importReport.setInputFiles({
    name: 'baseline.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ sourceId, regions: submitted!.regions })),
  });
  await expect.poll(bluePixels).toBeGreaterThan(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'حذف سطح الطاولة 1', exact: true })).toHaveCount(0);
  await importReport.setInputFiles({
    name: 'wrist-regions.json',
    mimeType: 'application/json',
    buffer: Buffer.from(
      JSON.stringify({
        sourceId,
        regions: submitted!.regions,
        tabletopRegions: submitted!.tabletopRegions,
      }),
    ),
  });
  await expect(page.getByRole('button', { name: 'حذف سطح الطاولة 1', exact: true })).toBeVisible();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'تنزيل تقرير التجربة JSON', exact: true }).click();
  expect((await downloadPromise).suggestedFilename()).toBe('restoradar-video-trial.json');
  await page.getByLabel('طريقة رصد الأشخاص').selectOption('FULL_FRAME');
  await page
    .getByRole('combobox', { name: 'نموذج تقدير الجلوس', exact: true })
    .selectOption('YOLO26_NANO_POSE');
  await page.getByRole('button', { name: 'تحليل الفيديو على CPU', exact: true }).click();
  await expect(
    page.getByText('نموذج تقدير الجلوس في التقرير: YOLO26n-Pose', { exact: true }),
  ).toBeVisible();
  expect(submitted?.poseModel).toBe('YOLO26_NANO_POSE');
  expect(submitted?.detectionMode).toBe('FULL_FRAME');
  await expect(page.getByText(/سرعة معالجة المقطع: 3\.00×/)).toBeVisible();
  await importReport.setInputFiles({
    name: 'overlapping-seats.json',
    mimeType: 'application/json',
    buffer: Buffer.from(
      JSON.stringify({
        sourceId,
        regions: [submitted!.regions[0], { ...submitted!.regions[0], id: randomUUID() }],
      }),
    ),
  });
  const overlapNotice = page.getByText(/توجد مناطق مقاعد متداخلة/);
  await expect(overlapNotice).toBeVisible();
  await page.getByRole('button', { name: 'حذف المنطقة 2', exact: true }).click();
  await expect(overlapNotice).toHaveCount(0);
  const after = await (
    await page.request.get(`${address}/api/restaurants/${restaurantId}/dashboard`)
  ).json();
  const original = await before;
  expect(after.sessions).toEqual(original.sessions);
  expect(after.alerts).toEqual(original.alerts);
  await noOverflow(page);
  await page.screenshot({ path: testInfo.outputPath('android-video-trial.png'), fullPage: true });
  expect(browserErrors).toEqual([]);
});
