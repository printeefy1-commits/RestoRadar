import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import type {
  Command,
  Dashboard,
  Role,
  Table,
  Task,
  TaskKind,
  TimeoutPolicy,
} from '@restoradar/contracts';
import './styles.css';
import { VideoLab } from './VideoLab.tsx';

type Me = {
  user: { id: string; name: string };
  restaurants: { id: string; name: string; role: Role }[];
};
type Tab = 'tables' | 'alerts' | 'tasks' | 'history' | 'settings' | 'video';
const roleNames: Record<Role, string> = { OWNER: 'المالك', MANAGER: 'المدير', WAITER: 'النادل' };
const kindNames = { CUSTOMER_RECEPTION: 'استقبال الضيوف', TABLE_CLEANING: 'تنظيف الطاولة' };
const alertNames = {
  NEW_CUSTOMERS: 'ضيوف جدد',
  CLEANING_REQUIRED: 'تنظيف مطلوب',
  ACCEPTANCE_OVERDUE: 'تأخر قبول المهمة',
  COMPLETION_OVERDUE: 'تأخر إكمال المهمة',
};
const taskNames = {
  ASSIGNED: 'بانتظار القبول',
  ACCEPTED: 'قيد التنفيذ',
  COMPLETED: 'مكتملة',
  CANCELLED: 'ملغاة',
};
const visionNames = {
  EMPTY: 'فارغة',
  OCCUPIED: 'مشغولة',
  UNKNOWN: 'غير معروفة',
  VISION_UNAVAILABLE: 'الرؤية غير متاحة',
};
const auditNames: Record<string, string> = {
  'table.create': 'إضافة طاولة',
  'table.correct': 'تصحيح الإشغال',
  'table.ready': 'تأكيد الجاهزية',
  'policy.configure': 'تعديل مهلات المهام',
  'task.assign': 'إسناد مهمة',
  'task.accept': 'قبول مهمة',
  'task.complete': 'إكمال مهمة',
  'alert.acknowledge': 'إقرار تنبيه',
  'session.adjust': 'تعديل وقت جلسة',
  'user.created': 'إضافة مستخدم',
  'restaurant.initialized': 'إعداد المطعم',
  'simulation.transition': 'محاكاة انتقال إشغال',
  'simulation.vision': 'محاكاة فقد الرؤية',
};
class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    credentials: 'same-origin',
    cache: 'no-store',
    ...(body !== undefined
      ? {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }
      : {}),
  });
  const result = await response.json();
  if (!response.ok) throw new ApiError(result.message ?? 'تعذر تنفيذ الطلب', response.status);
  return result;
}
const duration = (since: string, now: number) => {
  const seconds = Math.max(0, Math.floor((now - Date.parse(since)) / 1000));
  return `${Math.floor(seconds / 3600)
    .toString()
    .padStart(2, '0')}:${Math.floor((seconds / 60) % 60)
    .toString()
    .padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
};
const dateLabel = (date: string | null) =>
  date
    ? new Date(date).toLocaleString('ar', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    : '—';
function Mark() {
  return <img src="/radar.svg" width="40" height="40" alt="" />;
}
function App() {
  const [me, setMe] = useState<Me | null>(null),
    [authLoading, setAuthLoading] = useState(true);
  const [restaurantId, setRestaurantId] = useState(''),
    [data, setData] = useState<Dashboard | null>(null);
  const [tab, setTab] = useState<Tab>('tables'),
    [error, setError] = useState(''),
    [notice, setNotice] = useState('');
  const [online, setOnline] = useState(false),
    [lastSync, setLastSync] = useState<number | null>(null),
    [busy, setBusy] = useState(false);
  const [clock, setClock] = useState(Date.now()),
    offset = useRef(0),
    refreshBusy = useRef(false);
  const [correcting, setCorrecting] = useState<Table | null>(null),
    [assigning, setAssigning] = useState<{ kind: TaskKind; sourceId: string } | null>(null);
  const selectedRestaurant = useRef(restaurantId);
  selectedRestaurant.current = restaurantId;
  const restaurant = me?.restaurants.find((r) => r.id === restaurantId);
  const role = restaurant?.role ?? 'WAITER',
    management = role !== 'WAITER';
  const loadMe = useCallback(async () => {
    try {
      const result = await api<Me>('/me');
      setMe(result);
      setRestaurantId(result.restaurants[0]?.id ?? '');
    } catch (e) {
      if (!(e instanceof ApiError && e.status === 401)) setError((e as Error).message);
    } finally {
      setAuthLoading(false);
    }
  }, []);
  useEffect(() => {
    void loadMe();
  }, [loadMe]);
  const refresh = useCallback(async () => {
    if (!restaurantId || refreshBusy.current) return;
    refreshBusy.current = true;
    try {
      const next = await api<Dashboard>(`/restaurants/${restaurantId}/dashboard`);
      if (selectedRestaurant.current !== restaurantId) return;
      offset.current = Date.parse(next.serverTime) - Date.now();
      setData(next);
      setOnline(true);
      setLastSync(Date.now());
    } catch (e) {
      if (selectedRestaurant.current !== restaurantId) return;
      setOnline(false);
      if (e instanceof ApiError && e.status === 401) {
        setMe(null);
        setData(null);
      } else setError((e as Error).message);
    } finally {
      refreshBusy.current = false;
    }
  }, [restaurantId]);
  useEffect(() => {
    setData(null);
    void refresh();
    const interval = setInterval(() => {
      void refresh();
    }, 3000);
    return () => clearInterval(interval);
  }, [refresh]);
  useEffect(() => {
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const now = clock + offset.current;
  async function action(fn: () => Promise<unknown>, success: string) {
    if (busy) return false;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      await fn();
      setNotice(success);
      await refresh();
      return true;
    } catch (e) {
      setError((e as Error).message);
      await refresh();
      return false;
    } finally {
      setBusy(false);
    }
  }
  const send = (command: Command, success: string) =>
    action(
      () =>
        api(`/restaurants/${restaurantId}/commands`, { commandId: crypto.randomUUID(), command }),
      success,
    );
  const memberName = (id: string | null) =>
    data?.members.find((m) => m.id === id)?.name ?? 'مستخدم سابق';
  const activeAssignment = (tableId: string) =>
    data?.assignments.find((a) => a.tableId === tableId && !a.endedAt);
  const openCycle = (tableId: string) =>
    data?.cleaningCycles.find((c) => c.tableId === tableId && !c.readyAt);
  const activeTask = (kind: TaskKind, sourceId: string) =>
    data?.tasks.find((t) => t.kind === kind && t.sourceId === sourceId);
  function simulation(table: Table, occupancy: 'EMPTY' | 'OCCUPIED') {
    const end = new Date(now).toISOString();
    return send(
      {
        type: 'simulation.transition',
        tableId: table.id,
        occupancy,
        confirmedAt: end,
        observedAt: new Date(now - (occupancy === 'OCCUPIED' ? 5000 : 30000)).toISOString(),
      },
      'أُرسلت محاكاة انتقال متحقق منه',
    );
  }
  if (authLoading)
    return (
      <main className="loading">
        <Mark />
        <p>جارٍ فتح RestoRadar…</p>
      </main>
    );
  if (!me)
    return (
      <main className="login-layout">
        <section className="login-story">
          <div className="brand">
            <Mark />
            <span dir="ltr">RestoRadar</span>
          </div>
          <span className="eyebrow">كل طاولة. في وقتها.</span>
          <h1>
            صالة واضحة.
            <br />
            خدمة أكثر تنظيمًا.
          </h1>
          <p>تابع إشغال الطاولات، نسّق مهام الفريق، وأبقِ تفاصيل الخدمة في مكان واحد.</p>
          <div className="story-table">
            <span className="pulse" /> رؤية محلية · قرارات موثّقة · فريق متصل
          </div>
          <small>الإصدار الأولي · أحداث محاكاة، دون اتصال بالكاميرات</small>
        </section>
        <section className="login-form">
          <span className="eyebrow">أهلًا بعودتك</span>
          <h2>تسجيل الدخول</h2>
          <p className="muted">استخدم حساب المطعم المخصص لك.</p>
          {error && (
            <p className="message error" role="alert">
              {error}
            </p>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const f = new FormData(e.currentTarget);
              void action(async () => {
                await api('/auth/login', { email: f.get('email'), password: f.get('password') });
                await loadMe();
              }, '');
            }}
          >
            <label>
              البريد الإلكتروني
              <input name="email" type="email" autoComplete="username" dir="ltr" required />
            </label>
            <label>
              كلمة المرور
              <input
                name="password"
                type="password"
                autoComplete="current-password"
                required
                maxLength={256}
              />
            </label>
            <button className="primary" disabled={busy}>
              {busy ? 'جارٍ الدخول…' : 'دخول إلى المطعم ←'}
            </button>
          </form>
          <p className="fineprint">ليس لديك حساب؟ اطلب من مالك المطعم إضافتك.</p>
        </section>
      </main>
    );
  const nav: { key: Tab; name: string; icon: string }[] = [
    { key: 'tables', name: 'الطاولات', icon: '▦' },
    ...(management ? [{ key: 'alerts' as Tab, name: 'التنبيهات', icon: '◉' }] : []),
    { key: 'tasks', name: management ? 'مهام الفريق' : 'مهامي', icon: '☷' },
    ...(management ? [{ key: 'history' as Tab, name: 'السجل', icon: '◷' }] : []),
    ...(role === 'OWNER'
      ? [
          { key: 'video' as Tab, name: 'تجربة الفيديو', icon: '▷' },
          { key: 'settings' as Tab, name: 'الإعدادات', icon: '⚙' },
        ]
      : []),
  ];
  const pending = data?.alerts.filter((a) => a.status === 'PENDING') ?? [];
  const arrivals = pending.filter((a) => a.kind === 'NEW_CUSTOMERS');
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="RestoRadar">
          <Mark />
          <span dir="ltr">
            RestoRadar<small>إدارة صالة المطعم</small>
          </span>
        </a>
        <div className="restaurant-name">
          <span className="eyebrow">مساحة العمل</span>
          <strong>{restaurant?.name}</strong>
          {me.restaurants.length > 1 && (
            <select
              aria-label="المطعم"
              value={restaurantId}
              onChange={(e) => {
                setData(null);
                setRestaurantId(e.target.value);
                setTab('tables');
              }}
            >
              {me.restaurants.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          )}
        </div>
        <nav>
          {nav.map((n) => (
            <button
              key={n.key}
              className={tab === n.key ? 'nav-item active' : 'nav-item'}
              onClick={() => setTab(n.key)}
            >
              <span aria-hidden="true">{n.icon}</span>
              {n.name}
              {n.key === 'alerts' && pending.length > 0 && <b>{pending.length}</b>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy">
            ◈ معالجة الفيديو على خادمك<small>لا يوجد بث إلى خدمة تحليل خارجية</small>
          </div>
          <div className="profile">
            <span className="avatar">{me.user.name.slice(0, 1)}</span>
            <div>
              <strong>{me.user.name}</strong>
              <small>{roleNames[role]}</small>
            </div>
            <button
              title="تسجيل الخروج"
              aria-label="تسجيل الخروج"
              onClick={() => {
                void action(async () => {
                  await api('/auth/logout', {});
                  setMe(null);
                  setData(null);
                  setRestaurantId('');
                  setTab('tables');
                }, '');
              }}
            >
              ↪
            </button>
          </div>
        </div>
      </aside>
      <main className="workspace">
        <header className="topbar">
          <span>
            {new Date(clock).toLocaleDateString('ar', {
              weekday: 'long',
              day: 'numeric',
              month: 'long',
            })}
          </span>
          <div className={online ? 'connection' : 'connection disconnected'}>
            <i />
            <span>{online ? 'متصل بالخادم المحلي' : 'الاتصال غير متاح'}</span>
            <span className="local-tag">محلي فقط</span>
          </div>
        </header>
        <div className="page-heading">
          <div>
            <span className="eyebrow">{management ? 'نظرة على الصالة' : 'مساحة فريق الخدمة'}</span>
            <h1>{nav.find((n) => n.key === tab)?.name}</h1>
            <p className="muted">
              {tab === 'tables'
                ? 'حالة كل طاولة، والخطوة التالية لخدمتها.'
                : tab === 'tasks'
                  ? 'كل مهمة لها مسؤول وخطوة تالية واضحة.'
                  : tab === 'alerts'
                    ? 'إقرار التنبيه مستقل عن إنجاز المهمة.'
                    : tab === 'video'
                      ? 'اختبر رصد الجلوس على فيديو مسجل وراجع النتائج.'
                      : tab === 'history'
                        ? 'تسلسل واضح للجلسات والتعديلات والقرارات.'
                        : 'أعدّ مطعمك وفريقك قبل بدء التشغيل.'}
            </p>
          </div>
          <button
            className="secondary"
            onClick={() => {
              void refresh();
            }}
          >
            ↻ تحديث
          </button>
        </div>
        {!online && (
          <p className="message warning" role="alert">
            البيانات قديمة أو غير متاحة.{' '}
            {lastSync ? `آخر اتصال: ${dateLabel(new Date(lastSync).toISOString())}.` : ''} لا يمكن
            اعتماد إجراءات جديدة حتى يعود الاتصال.
          </p>
        )}
        {error && (
          <p className="message error" role="alert">
            {error}
            <button className="text-button" onClick={() => setError('')}>
              إغلاق
            </button>
          </p>
        )}
        {notice && (
          <p className="message success" role="status">
            {notice}
          </p>
        )}
        {!data ? (
          <section className="empty-state">
            <h2>{restaurantId ? 'جارٍ تحميل المطعم…' : 'لا يوجد مطعم مرتبط بالحساب'}</h2>
          </section>
        ) : (
          <>
            {data.simulatorEnabled && (
              <div className="simulation-label">
                وضع المحاكاة مفعّل · الأحداث تجريبية وليست رصدًا من كاميرات حقيقية
              </div>
            )}
            {!data.policy && management && (
              <div className="message warning">
                مهلات المهام لم تُضبط بعد. يجب أن يحددها المالك قبل إسناد أي مهمة.{' '}
                {role === 'OWNER' && (
                  <button className="text-button" onClick={() => setTab('settings')}>
                    ضبط المهلات ←
                  </button>
                )}
              </div>
            )}
            {management && arrivals.length > 0 && tab !== 'alerts' && (
              <section className="arrival-banner" aria-label="ضيوف جدد">
                <div>
                  <span className="eyebrow">حان وقت الترحيب</span>
                  <strong>
                    ضيوف جدد عند {data.tables.find((t) => t.id === arrivals[0].tableId)?.name}
                  </strong>
                  <small>
                    {arrivals.length > 1
                      ? `و${arrivals.length - 1} تنبيهات وصول أخرى`
                      : 'اختر نادلًا لاستقبالهم'}
                  </small>
                </div>
                <div className="actions">
                  <button
                    disabled={busy || !online}
                    className="secondary"
                    onClick={() => {
                      void send(
                        { type: 'alert.acknowledge', alertId: arrivals[0].id },
                        'تم إقرار التنبيه',
                      );
                    }}
                  >
                    إقرار الاطلاع
                  </button>
                  <button
                    disabled={busy || !online || !data.policy}
                    className="primary"
                    onClick={() =>
                      setAssigning({ kind: 'CUSTOMER_RECEPTION', sourceId: arrivals[0].sourceId })
                    }
                  >
                    تعيين نادل
                  </button>
                </div>
              </section>
            )}
            {tab === 'tables' && (
              <>
                <section className="stats">
                  <div>
                    <span>إجمالي الطاولات</span>
                    <strong>{data.tables.length.toLocaleString('ar')}</strong>
                    <small>ضمن المطعم</small>
                  </div>
                  <div>
                    <span>مشغولة الآن</span>
                    <strong>
                      {data.tables
                        .filter((t) => t.occupancy === 'OCCUPIED')
                        .length.toLocaleString('ar')}
                    </strong>
                    <small>جلسات جارية</small>
                  </div>
                  <div>
                    <span>فارغة وجاهزة</span>
                    <strong>
                      {data.tables
                        .filter((t) => t.occupancy === 'EMPTY' && !openCycle(t.id))
                        .length.toLocaleString('ar')}
                    </strong>
                    <small>بانتظار الضيوف</small>
                  </div>
                  <div>
                    <span>بانتظار الجاهزية</span>
                    <strong>
                      {data.cleaningCycles.filter((c) => !c.readyAt).length.toLocaleString('ar')}
                    </strong>
                    <small>تحتاج تأكيد المدير</small>
                  </div>
                </section>
                <div className="section-heading">
                  <h2>
                    خريطة الطاولات <span className="count">{data.tables.length}</span>
                  </h2>
                  <span className="muted">تُحدّث كل 3 ثوانٍ</span>
                </div>
                <section className="table-grid">
                  {data.tables.map((t) => {
                    const assignment = activeAssignment(t.id),
                      session = data.sessions.find((s) => s.id === assignment?.sessionId),
                      cycle = openCycle(t.id);
                    return (
                      <article
                        className={`table-card ${t.occupancy === 'OCCUPIED' ? 'occupied' : cycle ? 'needs-cleaning' : 'ready'}`}
                        key={t.id}
                      >
                        <div className="card-top">
                          <span className="table-symbol" aria-hidden="true">
                            ▤
                          </span>
                          <span className="badge">
                            {t.occupancy === 'OCCUPIED'
                              ? 'مشغولة'
                              : cycle
                                ? 'فارغة · غير جاهزة'
                                : 'فارغة · جاهزة'}
                          </span>
                        </div>
                        <h3>{t.name}</h3>
                        <div className="table-time">
                          <small>
                            {session
                              ? 'الوقت منذ الجلوس'
                              : cycle
                                ? 'انتظار تأكيد الجاهزية'
                                : 'جاهزة لاستقبال الضيوف'}
                          </small>
                          <strong dir="ltr">
                            {session
                              ? duration(session.startedAt, now)
                              : cycle
                                ? duration(cycle.startedAt, now)
                                : '—'}
                          </strong>
                        </div>
                        {cycle && t.occupancy === 'OCCUPIED' && (
                          <p className="readiness-warning">⚠ جلس الضيوف قبل تأكيد الجاهزية</p>
                        )}
                        <div className="vision-status">
                          الرؤية: {visionNames[t.aiObservedState]}
                          {t.override && <span> · تصحيح يدوي محفوظ</span>}
                        </div>
                        {management && (
                          <div className="card-actions">
                            <button disabled={busy || !online} onClick={() => setCorrecting(t)}>
                              تصحيح الإشغال
                            </button>
                            {session && (
                              <button
                                disabled={
                                  busy ||
                                  !online ||
                                  !data.policy ||
                                  activeTask('CUSTOMER_RECEPTION', session.id)?.status ===
                                    'COMPLETED'
                                }
                                onClick={() =>
                                  setAssigning({ kind: 'CUSTOMER_RECEPTION', sourceId: session.id })
                                }
                              >
                                تعيين استقبال
                              </button>
                            )}
                            {cycle && t.occupancy === 'EMPTY' && (
                              <>
                                <button
                                  disabled={
                                    busy ||
                                    !online ||
                                    !data.policy ||
                                    activeTask('TABLE_CLEANING', cycle.id)?.status === 'COMPLETED'
                                  }
                                  onClick={() =>
                                    setAssigning({ kind: 'TABLE_CLEANING', sourceId: cycle.id })
                                  }
                                >
                                  تعيين تنظيف
                                </button>
                                <button
                                  className="ready-button"
                                  disabled={busy || !online}
                                  onClick={() => {
                                    void send(
                                      {
                                        type: 'table.ready',
                                        tableId: t.id,
                                        expectedVersion: t.version,
                                      },
                                      'أكد المدير جاهزية الطاولة',
                                    );
                                  }}
                                >
                                  تأكيد الجاهزية ✓
                                </button>
                              </>
                            )}
                          </div>
                        )}
                        {data.simulatorEnabled && (
                          <details className="simulator">
                            <summary>محاكاة أحداث الطاولة</summary>
                            <div className="actions">
                              <button
                                disabled={busy || !online}
                                onClick={() => {
                                  void simulation(t, 'OCCUPIED');
                                }}
                              >
                                جلوس متحقق 5 ث
                              </button>
                              <button
                                disabled={busy || !online}
                                onClick={() => {
                                  void simulation(t, 'EMPTY');
                                }}
                              >
                                مغادرة متحققة 30 ث
                              </button>
                              <button
                                disabled={busy || !online}
                                onClick={() => {
                                  void send(
                                    {
                                      type: 'simulation.vision',
                                      tableId: t.id,
                                      state: 'VISION_UNAVAILABLE',
                                    },
                                    'الرؤية غير متاحة؛ الإشغال محفوظ',
                                  );
                                }}
                              >
                                فقد الرؤية
                              </button>
                            </div>
                          </details>
                        )}
                      </article>
                    );
                  })}
                </section>
                {data.tables.length === 0 && (
                  <section className="empty-state">
                    <span className="empty-icon">▦</span>
                    <h2>ابدأ بطاولتك الأولى</h2>
                    <p>ستظهر هنا حالات الطاولات وجلساتها ومهام الخدمة.</p>
                    {role === 'OWNER' && (
                      <button className="primary" onClick={() => setTab('settings')}>
                        إضافة الطاولات
                      </button>
                    )}
                  </section>
                )}
              </>
            )}
            {tab === 'alerts' && management && (
              <section className="panel">
                <div className="section-heading">
                  <h2>مركز التنبيهات</h2>
                  <span className="badge">{pending.length} بانتظار الاطلاع</span>
                </div>
                {data.alerts.length === 0 && (
                  <Empty
                    title="لا توجد تنبيهات"
                    description="ستظهر هنا أحداث الوصول والتنظيف وتأخر المهام."
                  />
                )}
                {data.alerts.map((a) => (
                  <article className="list-row" key={a.id}>
                    <span className={`event-icon ${a.kind === 'NEW_CUSTOMERS' ? 'arrival' : ''}`}>
                      ◉
                    </span>
                    <div className="row-main">
                      <strong>
                        {alertNames[a.kind]} · {data.tables.find((t) => t.id === a.tableId)?.name}
                      </strong>
                      <small>{dateLabel(a.createdAt)}</small>
                      {a.acknowledgedBy && (
                        <small>
                          أقرّه {memberName(a.acknowledgedBy)} · {dateLabel(a.acknowledgedAt)}
                        </small>
                      )}
                      {a.obsoleteReason && <small>انتهت قابلية الإجراء؛ السجل محفوظ.</small>}
                    </div>
                    <span className="badge">
                      {a.status === 'PENDING'
                        ? 'بانتظار الاطلاع'
                        : a.status === 'ACKNOWLEDGED'
                          ? 'تم الاطلاع'
                          : 'لم يعد قابلًا للإجراء'}
                    </span>
                    {a.status === 'PENDING' && (
                      <div className="actions">
                        <button
                          disabled={busy || !online}
                          className="secondary"
                          onClick={() => {
                            void send(
                              { type: 'alert.acknowledge', alertId: a.id },
                              'حُفظ إقرار الاطلاع',
                            );
                          }}
                        >
                          إقرار
                        </button>
                        {['NEW_CUSTOMERS', 'CLEANING_REQUIRED'].includes(a.kind) && (
                          <button
                            className="primary"
                            disabled={busy || !online || !data.policy}
                            onClick={() =>
                              setAssigning({
                                kind:
                                  a.kind === 'NEW_CUSTOMERS'
                                    ? 'CUSTOMER_RECEPTION'
                                    : 'TABLE_CLEANING',
                                sourceId: a.sourceId,
                              })
                            }
                          >
                            تعيين نادل
                          </button>
                        )}
                      </div>
                    )}
                  </article>
                ))}
              </section>
            )}
            {tab === 'tasks' && (
              <>
                <section className="stats compact">
                  {(['ASSIGNED', 'ACCEPTED', 'COMPLETED'] as const).map((status) => (
                    <div key={status}>
                      <span>{taskNames[status]}</span>
                      <strong>
                        {data.tasks.filter((t) => t.status === status).length.toLocaleString('ar')}
                      </strong>
                    </div>
                  ))}
                </section>
                {management && (
                  <section className="workload">
                    {data.members
                      .filter((m) => m.role === 'WAITER')
                      .map((w) => (
                        <div key={w.id}>
                          <span className="avatar">{w.name[0]}</span>
                          <div>
                            <strong>{w.name}</strong>
                            <small>
                              {
                                data.tasks.filter(
                                  (t) =>
                                    ['ASSIGNED', 'ACCEPTED'].includes(t.status) &&
                                    data.attempts.find((a) => a.id === t.currentAttemptId)
                                      ?.waiterId === w.id,
                                ).length
                              }{' '}
                              مهام مفتوحة · حالة الاتصال غير مقاسة
                            </small>
                          </div>
                        </div>
                      ))}
                  </section>
                )}
                <section className="panel">
                  {data.tasks.length === 0 && (
                    <Empty
                      title="لا توجد مهام بعد"
                      description={
                        management
                          ? 'عيّن نادلًا من بطاقة الطاولة أو مركز التنبيهات.'
                          : 'ستظهر المهام التي يسندها المدير إليك هنا.'
                      }
                    />
                  )}
                  {(['ASSIGNED', 'ACCEPTED', 'COMPLETED', 'CANCELLED'] as const).map((status) => {
                    const tasks = data.tasks.filter((t) => t.status === status);
                    return (
                      tasks.length > 0 && (
                        <section key={status} className="task-group">
                          <h2>
                            {taskNames[status]} <span className="count">{tasks.length}</span>
                          </h2>
                          {tasks.map((t) => {
                            const attempt = data.attempts.find((a) => a.id === t.currentAttemptId)!;
                            const due =
                              t.status === 'ASSIGNED'
                                ? attempt.acceptanceDueAt
                                : t.status === 'ACCEPTED'
                                  ? attempt.completionDueAt
                                  : null;
                            return (
                              <article className="list-row" key={t.id}>
                                <span className="event-icon">
                                  {t.kind === 'CUSTOMER_RECEPTION' ? '↗' : '✧'}
                                </span>
                                <div className="row-main">
                                  <strong>
                                    {kindNames[t.kind]} ·{' '}
                                    {data.tables.find((table) => table.id === t.tableId)?.name}
                                  </strong>
                                  <small>
                                    {memberName(attempt.waiterId)} · أُسندت{' '}
                                    {dateLabel(attempt.assignedAt)}
                                  </small>
                                  {due && (
                                    <small className={Date.parse(due) <= now ? 'late' : ''}>
                                      {Date.parse(due) <= now ? 'تجاوزت المهلة' : 'الموعد المحدد'}:{' '}
                                      {dateLabel(due)}
                                    </small>
                                  )}
                                  {t.cancelledReason && (
                                    <small>أُلغيت لتغيّر حالة الطاولة؛ السجل محفوظ.</small>
                                  )}
                                </div>
                                {role === 'WAITER' &&
                                  ['ASSIGNED', 'ACCEPTED'].includes(t.status) && (
                                    <button
                                      className="primary"
                                      disabled={busy || !online}
                                      onClick={() => {
                                        void send(
                                          {
                                            type:
                                              t.status === 'ASSIGNED'
                                                ? 'task.accept'
                                                : 'task.complete',
                                            taskId: t.id,
                                            expectedVersion: t.version,
                                          },
                                          t.status === 'ASSIGNED'
                                            ? 'قُبلت المهمة'
                                            : 'اكتملت المهمة؛ جاهزية الطاولة يحددها المدير',
                                        );
                                      }}
                                    >
                                      {t.status === 'ASSIGNED' ? 'قبول المهمة' : 'إكمال المهمة'}
                                    </button>
                                  )}
                                {management && ['ASSIGNED', 'ACCEPTED'].includes(t.status) && (
                                  <button
                                    className="secondary"
                                    disabled={busy || !online}
                                    onClick={() =>
                                      setAssigning({ kind: t.kind, sourceId: t.sourceId })
                                    }
                                  >
                                    إعادة التعيين
                                  </button>
                                )}
                              </article>
                            );
                          })}
                        </section>
                      )
                    );
                  })}
                </section>
              </>
            )}
            {tab === 'video' && role === 'OWNER' && (
              <VideoLab key={restaurantId} restaurantId={restaurantId} tables={data.tables} />
            )}
            {tab === 'history' && management && (
              <>
                <section className="panel">
                  <h2>تاريخ الجلسات</h2>
                  {data.sessions.length === 0 && (
                    <Empty
                      title="لا توجد جلسات مسجلة"
                      description="الوقت الأصلي محفوظ مستقلًا عن تعديلات المدير."
                    />
                  )}
                  {[...data.sessions]
                    .sort((a, b) => b.startedAt.localeCompare(a.startedAt))
                    .map((s) => (
                      <article className="session-record" key={s.id}>
                        <div className="section-heading">
                          <strong>
                            {
                              data.tables.find((t) =>
                                data.assignments.some(
                                  (a) => a.sessionId === s.id && a.tableId === t.id,
                                ),
                              )?.name
                            }
                          </strong>
                          <span className="badge">
                            {s.source === 'MANUAL' ? 'يدوية' : 'محاكاة'} ·{' '}
                            {s.endedAt ? 'منتهية' : 'نشطة'}
                          </span>
                        </div>
                        <p>
                          البداية: {dateLabel(s.startedAt)} · النهاية: {dateLabel(s.endedAt)}
                        </p>
                        <small className="muted">
                          الرصد الأصلي: {dateLabel(s.originalStartedAt)} ←{' '}
                          {dateLabel(s.originalEndedAt)}
                        </small>
                        <details>
                          <summary>تعديل وقت إلى الخلف</summary>
                          <form
                            className="inline-form"
                            onSubmit={(e) => {
                              e.preventDefault();
                              const f = new FormData(e.currentTarget);
                              const at = new Date(String(f.get('at')));
                              if (!Number.isFinite(at.getTime())) return;
                              void send(
                                {
                                  type: 'session.adjust',
                                  sessionId: s.id,
                                  expectedVersion: s.version,
                                  field: f.get('field') as 'startedAt' | 'endedAt',
                                  at: at.toISOString(),
                                  reason: String(f.get('reason')),
                                },
                                'حُفظ تعديل الوقت مع السبب',
                              );
                            }}
                          >
                            <label>
                              الحقل
                              <select name="field">
                                <option value="startedAt">البداية</option>
                                {s.endedAt && <option value="endedAt">النهاية</option>}
                              </select>
                            </label>
                            <label>
                              الوقت الجديد
                              <input type="datetime-local" name="at" step="1" required />
                            </label>
                            <label>
                              سبب التعديل
                              <input name="reason" minLength={3} maxLength={1000} required />
                            </label>
                            <button className="secondary" disabled={busy || !online}>
                              حفظ التعديل
                            </button>
                          </form>
                        </details>
                        {data.adjustments
                          .filter((a) => a.sessionId === s.id)
                          .map((a) => (
                            <p className="fineprint" key={a.id}>
                              {memberName(a.actorId)}: {a.reason} · {dateLabel(a.previous)} ←{' '}
                              {dateLabel(a.next)}
                            </p>
                          ))}
                      </article>
                    ))}
                </section>
                <section className="panel">
                  <h2>فجوات المراقبة</h2>
                  {data.monitoringGaps.length === 0 ? (
                    <p className="muted">لا توجد فجوات إعادة تشغيل مسجلة.</p>
                  ) : (
                    data.monitoringGaps.map((g) => (
                      <p key={g.id}>
                        {dateLabel(g.startedAt)} — {dateLabel(g.endedAt)} · فترة غير متحقق منها
                      </p>
                    ))
                  )}
                </section>
                <section className="panel">
                  <h2>سجل التدقيق</h2>
                  {data.audit.slice(0, 100).map((a) => (
                    <div className="audit-row" key={a.id}>
                      <strong>{auditNames[a.action] ?? a.action}</strong>
                      <span>{memberName(a.actorId)}</span>
                      <small>{dateLabel(a.at)}</small>
                      {typeof a.context.reason === 'string' && <p>{a.context.reason}</p>}
                    </div>
                  ))}
                </section>
              </>
            )}
            {tab === 'settings' && role === 'OWNER' && (
              <Settings
                data={data}
                busy={busy || !online}
                send={send}
                addUser={(body) =>
                  action(
                    () => api(`/restaurants/${restaurantId}/users`, body),
                    'أُضيف الحساب بنجاح',
                  )
                }
              />
            )}
          </>
        )}
        <footer>
          RestoRadar <span>إصدار تأسيسي · تشغيل محلي · لا مزامنة سحابية أو إشعارات دفع بعد</span>
        </footer>
      </main>
      {correcting && (
        <Modal title={`تصحيح إشغال ${correcting.name}`} close={() => setCorrecting(null)}>
          <p className="muted">يُحفظ التصحيح وسببه، ويُحمى من الاستبدال الآلي لمدة 60 ثانية.</p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const f = new FormData(e.currentTarget);
              const command: Command = {
                type: 'table.correct',
                tableId: correcting.id,
                expectedVersion: correcting.version,
                occupancy: f.get('occupancy') as 'EMPTY' | 'OCCUPIED',
                reason: String(f.get('reason')),
              };
              setCorrecting(null);
              void send(command, 'حُفظ التصحيح اليدوي');
            }}
          >
            <label>
              الإشغال المعتمد
              <select name="occupancy" defaultValue={correcting.occupancy}>
                <option value="EMPTY">فارغة</option>
                <option value="OCCUPIED">مشغولة</option>
              </select>
            </label>
            <label>
              السبب المطلوب
              <textarea name="reason" required minLength={3} maxLength={1000} />
            </label>
            <button className="primary" disabled={busy || !online}>
              حفظ التصحيح
            </button>
          </form>
        </Modal>
      )}
      {assigning && data && (
        <Modal title={kindNames[assigning.kind]} close={() => setAssigning(null)}>
          <p className="muted">اختر النادل بنفسك. إعادة التعيين تسحب صلاحية النادل السابق.</p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const f = new FormData(e.currentTarget);
              const t = activeTask(assigning.kind, assigning.sourceId);
              const command: Command = {
                type: 'task.assign',
                ...assigning,
                waiterId: String(f.get('waiterId')),
                expectedVersion: t?.version ?? 0,
              };
              setAssigning(null);
              void send(command, 'حُفظ تعيين المهمة');
            }}
          >
            <label>
              النادل
              <select name="waiterId" aria-label="النادل" required defaultValue="">
                <option value="" disabled>
                  اختر نادلًا
                </option>
                {data.members
                  .filter((m) => m.role === 'WAITER')
                  .map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
              </select>
            </label>
            {!data.members.some((m) => m.role === 'WAITER') && (
              <p className="message warning">يجب أن يضيف المالك حساب نادل أولًا.</p>
            )}
            <button className="primary" disabled={busy || !online || !data.policy}>
              تأكيد التعيين
            </button>
          </form>
        </Modal>
      )}
    </div>
  );
}
function Empty({ title, description }: { title: string; description: string }) {
  return (
    <div className="empty-state">
      <span className="empty-icon">◌</span>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}
function Modal({
  title,
  close,
  children,
}: {
  title: string;
  close: () => void;
  children: React.ReactNode;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog ref={dialog} aria-labelledby={titleId} onCancel={close} className="modal">
      <div className="section-heading">
        <h2 id={titleId}>{title}</h2>
        <button type="button" aria-label="إغلاق" onClick={close}>
          ✕
        </button>
      </div>
      {children}
    </dialog>
  );
}
function Settings({
  data,
  busy,
  send,
  addUser,
}: {
  data: Dashboard;
  busy: boolean;
  send: (command: Command, success: string) => Promise<boolean>;
  addUser: (body: unknown) => Promise<boolean>;
}) {
  return (
    <div className="settings-grid">
      <section className="panel">
        <span className="eyebrow">قبل تعيين المهام</span>
        <h2>مهلات الفريق</h2>
        <p className="muted">
          جميع القيم بالثواني ويُدخلها المالك. التغييرات تخص التعيينات الجديدة فقط.
        </p>
        <form
          key={JSON.stringify(data.policy)}
          onSubmit={(e) => {
            e.preventDefault();
            const f = new FormData(e.currentTarget);
            const policy = Object.fromEntries(
              (['CUSTOMER_RECEPTION', 'TABLE_CLEANING'] as const).map((k) => [
                k,
                {
                  acceptanceSeconds: Number(f.get(`${k}.acceptance`)),
                  completionSeconds: Number(f.get(`${k}.completion`)),
                },
              ]),
            ) as unknown as TimeoutPolicy;
            void send({ type: 'policy.configure', policy }, 'حُفظت المهلات للتعيينات الجديدة');
          }}
        >
          {(['CUSTOMER_RECEPTION', 'TABLE_CLEANING'] as const).map((k) => (
            <fieldset key={k}>
              <legend>{kindNames[k]}</legend>
              <div className="two-fields">
                <label>
                  مهلة القبول
                  <input
                    type="number"
                    min={1}
                    max={86400}
                    name={`${k}.acceptance`}
                    defaultValue={data.policy?.[k].acceptanceSeconds ?? ''}
                    required
                  />
                </label>
                <label>
                  مهلة الإكمال بعد القبول
                  <input
                    type="number"
                    min={1}
                    max={86400}
                    name={`${k}.completion`}
                    defaultValue={data.policy?.[k].completionSeconds ?? ''}
                    required
                  />
                </label>
              </div>
            </fieldset>
          ))}
          <button className="primary" disabled={busy}>
            حفظ مهلات المهام
          </button>
        </form>
      </section>
      <section className="panel">
        <h2>إضافة طاولة</h2>
        <p className="muted">
          إنشاء طاولة لا يفعّل مراقبة كاميرا. الربط والرصد الحقيقي يأتيان في مرحلة لاحقة.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const form = e.currentTarget,
              f = new FormData(form);
            void send({ type: 'table.create', name: String(f.get('name')) }, 'أُضيفت الطاولة').then(
              (ok) => {
                if (ok) form.reset();
              },
            );
          }}
        >
          <label>
            اسم الطاولة
            <input name="name" required maxLength={80} />
          </label>
          <button className="primary" disabled={busy}>
            إضافة الطاولة
          </button>
        </form>
        <h3>أعضاء الفريق</h3>
        {data.members.map((m) => (
          <div className="member-row" key={m.id}>
            <strong>{m.name}</strong>
            <span className="badge">{roleNames[m.role]}</span>
          </div>
        ))}
      </section>
      <section className="panel">
        <h2>إضافة حساب</h2>
        <p className="muted">لكل شخص حساب وصلاحية منفصلان، دون كلمات مرور افتراضية.</p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const form = e.currentTarget,
              f = new FormData(form);
            void addUser(Object.fromEntries(f)).then((ok) => {
              if (ok) form.reset();
            });
          }}
        >
          <label>
            الاسم
            <input name="name" required maxLength={80} />
          </label>
          <label>
            البريد
            <input type="email" name="email" dir="ltr" autoComplete="off" required />
          </label>
          <label>
            الدور
            <select name="role" defaultValue="WAITER">
              <option value="WAITER">نادل</option>
              <option value="MANAGER">مدير</option>
              <option value="OWNER">مالك</option>
            </select>
          </label>
          <label>
            كلمة مرور أولية
            <input
              name="password"
              type="password"
              minLength={12}
              maxLength={256}
              autoComplete="new-password"
              required
            />
          </label>
          <small className="muted">12 حرفًا على الأقل. شارك بيانات الدخول عبر وسيلة خاصة.</small>
          <button className="primary" disabled={busy}>
            إنشاء الحساب
          </button>
        </form>
      </section>
      <section className="panel">
        <h2>حالة التكاملات</h2>
        <ul className="integration-list">
          <li>
            <span>الكاميرات الحية</span>
            <b>لم تُربط بعد</b>
          </li>
          <li>
            <span>المزامنة السحابية</span>
            <b>مرحلة لاحقة</b>
          </li>
          <li>
            <span>إشعارات Android</span>
            <b>مرحلة لاحقة</b>
          </li>
          <li>
            <span>رصد اتصال النوادل</span>
            <b>غير مفعّل</b>
          </li>
        </ul>
        <p className="fineprint">
          أحداث لوحة الطاولات محاكاة معلنة. تجربة الفيديو منفصلة ولا تغير الجلسات أو التنبيهات، ولا
          تتضمن التعرف على الوجوه.
        </p>
      </section>
    </div>
  );
}
createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
