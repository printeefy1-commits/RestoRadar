import { useCallback, useEffect, useRef, useState } from 'react';
import type { SeatRegion, Table, TrialJob, VideoLabStatus } from '@restoradar/contracts';
import { seatRegionSchema } from '@restoradar/contracts';
import { evidenceLabels, trialEvidence } from './video-evidence.ts';

async function request<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    credentials: 'same-origin',
    cache: 'no-store',
    ...(body === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.message ?? 'تعذر تنفيذ طلب الفيديو');
  return result;
}
const timeLabel = (seconds: number) =>
  `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, '0')}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0')}`;

export function VideoLab({ restaurantId, tables }: { restaurantId: string; tables: Table[] }) {
  const base = `/api/restaurants/${restaurantId}/video-lab`;
  const [status, setStatus] = useState<VideoLabStatus | null>(null);
  const [error, setError] = useState('');
  const [regions, setRegions] = useState<SeatRegion[]>([]);
  const [tableId, setTableId] = useState(tables[0]?.id ?? '');
  const [job, setJob] = useState<TrialJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<{ x: number; y: number; endX: number; endY: number } | null>(
    null,
  );
  const [startSeconds, setStartSeconds] = useState(0);
  const [durationSeconds, setDurationSeconds] = useState(60);
  const [detectionMode, setDetectionMode] = useState<'FULL_FRAME' | 'SEAT_FOCUS'>('FULL_FRAME');
  const [poseModel, setPoseModel] = useState<'RTMPOSE_TINY' | 'RTMPOSE_MEDIUM'>('RTMPOSE_TINY');
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const sourceId = useRef<string | null>(null);
  const validTables = new Set(tables.map((t) => t.id));
  const running = busy || job?.status === 'RUNNING';
  const tableName = (id: string) => tables.find((t) => t.id === id)?.name ?? 'طاولة محذوفة';

  const loadStatus = useCallback(async () => {
    const next = await request<VideoLabStatus>(base);
    if (sourceId.current && sourceId.current !== next.source?.id) {
      setRegions([]);
      setJob(null);
      setError('تغير ملف الفيديو. أعد تحديد مناطق المقاعد.');
    }
    sourceId.current = next.source?.id ?? null;
    setStatus(next);
  }, [base]);
  useEffect(() => {
    let live = true;
    const load = async () => {
      try {
        if (!live) return;
        await loadStatus();
      } catch (e) {
        if (live) setError((e as Error).message);
      }
    };
    void load();
    void request<TrialJob | null>(`${base}/jobs/latest`)
      .then((latest) => {
        if (live) setJob(latest);
      })
      .catch(() => undefined);
    const timer = setInterval(() => {
      void load();
    }, 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [base, loadStatus]);
  useEffect(() => {
    if (!job || job.status !== 'RUNNING') return;
    let live = true;
    const timer = setInterval(() => {
      void request<TrialJob>(`${base}/jobs/${job.id}`)
        .then((next) => {
          if (live) setJob(next);
        })
        .catch((e: Error) => {
          if (live) setError(e.message);
        });
    }, 2000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [base, job?.id, job?.status]);

  const report = job?.report?.sourceId === status?.source?.id ? job?.report : undefined;
  const sample = report?.samples.find((s) => Math.abs(s.seconds - seconds) < 0.26);
  const minDetectionScore = report?.evidencePolicy?.minDetectionScore ?? 0.35;
  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d');
    if (!element || !context) return;
    const w = element.width,
      h = element.height;
    context.clearRect(0, 0, w, h);
    context.font = `${Math.max(14, w / 70)}px sans-serif`;
    context.lineWidth = Math.max(2, w / 500);
    for (const region of regions) {
      context.strokeStyle = '#4285ff';
      context.fillStyle = '#4285ff22';
      context.fillRect(region.x * w, region.y * h, region.width * w, region.height * h);
      context.strokeRect(region.x * w, region.y * h, region.width * w, region.height * h);
      context.fillStyle = '#ffffff';
      context.fillText(tableName(region.tableId), region.x * w, Math.max(18, region.y * h - 5));
    }
    if (draft) {
      context.strokeStyle = '#ffffff';
      context.strokeRect(
        draft.x * w,
        draft.y * h,
        (draft.endX - draft.x) * w,
        (draft.endY - draft.y) * h,
      );
    }
    for (const person of sample?.people ?? []) {
      const [x1, y1, x2, y2] = person.box;
      const evidence = trialEvidence(person, minDetectionScore);
      context.strokeStyle = evidence === 'ACCEPTED' ? '#23d398' : '#ffd166';
      context.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
      context.fillStyle = context.strokeStyle;
      context.fillText(
        `#${person.trackId ?? '?'} · ${evidenceLabels[evidence]}`,
        x1 * w,
        Math.max(18, y1 * h - 5),
      );
    }
  }, [regions, sample, draft, tables, minDetectionScore]);

  function seek(value: number) {
    if (video.current) video.current.currentTime = value;
    setSeconds(value);
  }
  function point(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
    };
  }
  function download(value: unknown) {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }),
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = 'restoradar-video-trial.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function start() {
    if (!status?.source || running) return;
    setDrawing(false);
    setDraft(null);
    setError('');
    setBusy(true);
    try {
      setJob(
        await request<TrialJob>(`${base}/jobs`, {
          sourceId: status.source.id,
          startSeconds,
          durationSeconds,
          detectionMode,
          poseModel,
          regions,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="video-lab">
      <div className="message warning">
        تجربة فيديو مسجّل · النتائج للمراجعة فقط ولا تنشئ جلسات أو تنبيهات تشغيلية. المغادرة غير
        مفعّلة في هذه التجربة، واختفاء شخص لا يعني خلو الطاولة.
      </div>
      {error && (
        <p className="message error" role="alert">
          {error}
        </p>
      )}
      {!status ? (
        <p>جارٍ الاتصال بخدمة الفيديو…</p>
      ) : !status.enabled ? (
        <section className="panel">
          <h2>تجربة الفيديو غير مفعّلة</h2>
          <p>يلزم تفعيل خدمة الفيديو على هذا الخادم أولًا.</p>
        </section>
      ) : (
        <>
          <section className="panel">
            <div className="section-heading">
              <h2>فيديو التجربة</h2>
              <span className="badge">
                {status.models === 'READY'
                  ? 'النماذج جاهزة'
                  : status.models === 'DOWNLOADING'
                    ? 'جارٍ تحميل النماذج…'
                    : 'تعذر تجهيز النماذج'}
              </span>
            </div>
            {status.message && <p className="message warning">{status.message}</p>}
            {status.models === 'ERROR' && (
              <button
                className="secondary"
                onClick={() => {
                  setError('');
                  void request(`${base}/models/retry`, {})
                    .then(loadStatus)
                    .catch((e: Error) => setError(e.message));
                }}
              >
                إعادة تجهيز النماذج
              </button>
            )}
            {!status.source ? (
              <p>لم يوجد ملف فيديو التجربة على الخادم.</p>
            ) : (
              <>
                <p className="muted">
                  {status.source.width} × {status.source.height} ·{' '}
                  {timeLabel(status.source.duration)} ·{' '}
                  {(status.source.bytes / 1024 / 1024).toFixed(1)} MB
                </p>
                <div className="trial-video-frame">
                  <video
                    ref={video}
                    src={`${base}/video?source=${status.source.id}`}
                    preload="metadata"
                    playsInline
                    muted
                    onTimeUpdate={() => setSeconds(video.current?.currentTime ?? 0)}
                    onPlay={() => setPlaying(true)}
                    onPause={() => setPlaying(false)}
                    onError={() =>
                      setError(
                        'تعذر تشغيل الفيديو في المتصفح. قد يحتاج إلى تحويله إلى MP4 بترميز H.264.',
                      )
                    }
                  />
                  <canvas
                    ref={canvas}
                    width={status.source.width}
                    height={status.source.height}
                    aria-label="رسم منطقة المقعد على الفيديو"
                    className={drawing ? 'seat-canvas drawing' : 'seat-canvas'}
                    onPointerDown={(event) => {
                      if (!drawing || !tableId || running) return;
                      video.current?.pause();
                      event.currentTarget.setPointerCapture(event.pointerId);
                      const p = point(event);
                      setDraft({ ...p, endX: p.x, endY: p.y });
                    }}
                    onPointerMove={(event) => {
                      if (draft) {
                        const p = point(event);
                        setDraft({ ...draft, endX: p.x, endY: p.y });
                      }
                    }}
                    onPointerCancel={() => setDraft(null)}
                    onPointerUp={(event) => {
                      if (!draft) return;
                      const p = point(event);
                      const region = {
                        id: crypto.randomUUID(),
                        tableId,
                        x: Math.min(draft.x, p.x),
                        y: Math.min(draft.y, p.y),
                        width: Math.abs(p.x - draft.x),
                        height: Math.abs(p.y - draft.y),
                      };
                      if (region.width >= 0.01 && region.height >= 0.01)
                        setRegions((old) => [...old, region]);
                      setDraft(null);
                      setDrawing(false);
                    }}
                  />
                </div>
                <div className="trial-playback">
                  <button
                    className="secondary"
                    onClick={() => {
                      if (video.current?.paused) {
                        setDrawing(false);
                        void video.current.play().catch(() => setError('تعذر تشغيل الفيديو'));
                      } else video.current?.pause();
                    }}
                  >
                    {playing ? 'إيقاف مؤقت' : 'تشغيل الفيديو'}
                  </button>
                  <label>
                    موضع الفيديو
                    <input
                      type="range"
                      min={0}
                      max={status.source.duration}
                      step={0.1}
                      value={seconds}
                      onChange={(e) => seek(Number(e.target.value))}
                      dir="ltr"
                    />
                  </label>
                  <output dir="ltr">{timeLabel(seconds)}</output>
                </div>
              </>
            )}
          </section>
          {status.source && (
            <div className="settings-grid">
              <section className="panel">
                <h2>مناطق المقاعد</h2>
                <label>
                  استيراد مناطق من تقرير JSON
                  <input
                    type="file"
                    accept="application/json,.json"
                    disabled={running}
                    onChange={async (event) => {
                      const file = event.currentTarget.files?.[0];
                      event.currentTarget.value = '';
                      if (!file) return;
                      try {
                        if (file.size > 10 * 1024 * 1024)
                          throw new Error('حجم التقرير يتجاوز 10 ميغابايت');
                        const previous = JSON.parse(await file.text());
                        if (previous?.sourceId !== sourceId.current)
                          throw new Error('التقرير يخص ملف فيديو مختلفًا؛ لم تُستورد المناطق');
                        const restored = seatRegionSchema
                          .array()
                          .min(1)
                          .max(40)
                          .safeParse(previous.regions);
                        if (
                          !restored.success ||
                          new Set(restored.data.map((r) => r.id)).size !== restored.data.length
                        )
                          throw new Error('مناطق التقرير غير صالحة');
                        if (restored.data.some((r) => !validTables.has(r.tableId)))
                          throw new Error('بعض طاولات التقرير غير موجودة في هذا المطعم');
                        setRegions(restored.data);
                        setDraft(null);
                        setDrawing(false);
                        setError('');
                      } catch (error) {
                        setError(
                          error instanceof SyntaxError
                            ? 'تعذر قراءة تقرير JSON'
                            : (error as Error).message,
                        );
                      }
                    }}
                  />
                </label>
                <p className="muted">
                  استورد المناطق نفسها للمقارنة بعد التحديث. يستورد هذا الخيار مناطق المقاعد فقط.
                </p>
                <p className="muted">
                  أوقف الفيديو عند ظهور ضيف جالس. اختر الطاولة وارسم مستطيلًا حول موضع حوضه المتوقع
                  على المقعد، لا حول سطح الطاولة. أضف منطقة لكل مقعد ظاهر.
                </p>
                {!tables.length ? (
                  <p>أضف طاولة من الإعدادات أولًا.</p>
                ) : (
                  <>
                    <label>
                      الطاولة المرتبطة
                      <select
                        value={tableId}
                        disabled={running}
                        onChange={(e) => setTableId(e.target.value)}
                      >
                        {tables.map((t) => (
                          <option key={t.id} value={t.id}>
                            {t.name}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      className="secondary"
                      disabled={running || !tableId || regions.length >= 40}
                      onClick={() => {
                        video.current?.pause();
                        setDrawing(!drawing);
                      }}
                    >
                      {drawing ? 'إلغاء الرسم' : 'رسم منطقة مقعد'}
                    </button>
                    <details>
                      <summary>إضافة منطقة بالأرقام بدل الرسم</summary>
                      <form
                        onSubmit={(event) => {
                          event.preventDefault();
                          const fields = new FormData(event.currentTarget);
                          const x = Number(fields.get('x')) / 100,
                            y = Number(fields.get('y')) / 100;
                          const width = Number(fields.get('width')) / 100,
                            height = Number(fields.get('height')) / 100;
                          if (x + width > 1 || y + height > 1) {
                            setError('يجب أن تبقى المنطقة داخل حدود الصورة');
                            return;
                          }
                          setRegions((old) => [
                            ...old,
                            { id: crypto.randomUUID(), tableId, x, y, width, height },
                          ]);
                        }}
                      >
                        <p className="muted">النسب من الحافة اليسرى والعليا للصورة.</p>
                        {(
                          [
                            ['x', 'البداية الأفقية %', 0],
                            ['y', 'البداية الرأسية %', 0],
                            ['width', 'العرض %', 1],
                            ['height', 'الارتفاع %', 1],
                          ] as const
                        ).map(([name, label, min]) => (
                          <label key={name}>
                            {label}
                            <input
                              name={name}
                              type="number"
                              min={min}
                              max={100}
                              step={0.1}
                              required
                            />
                          </label>
                        ))}
                        <button disabled={running || !tableId || regions.length >= 40}>
                          إضافة المنطقة
                        </button>
                      </form>
                    </details>
                  </>
                )}
                {regions.map((r, i) => (
                  <div className="member-row" key={r.id}>
                    <span>
                      {tableName(r.tableId)} · مقعد {i + 1}
                    </span>
                    <button
                      disabled={running}
                      onClick={() => setRegions((old) => old.filter((v) => v.id !== r.id))}
                    >
                      حذف المنطقة {i + 1}
                    </button>
                  </div>
                ))}
                {report && !regions.length && (
                  <button
                    onClick={() =>
                      setRegions(report.regions.filter((r) => validTables.has(r.tableId)))
                    }
                  >
                    استخدام مناطق التقرير السابق
                  </button>
                )}
                <small className="muted">
                  مناطق التجربة لا تفعّل مراقبة الطاولات. احتفظ بالتقرير قبل مغادرة الصفحة؛ النتائج
                  مؤقتة لمدة ساعة وتُفقد عند إعادة تشغيل الخدمة.
                </small>
              </section>
              <section className="panel">
                <h2>تشغيل التحليل</h2>
                <p className="muted">
                  نعالج إطارين في الثانية، ونتحقق من دليل جلوس مستمر لمدة 5 ثوانٍ. ضعف وضوح مفاصل
                  الجسم قد يمنع التأكيد.
                </p>
                <label>
                  طريقة رصد الأشخاص
                  <select
                    disabled={running}
                    value={detectionMode}
                    onChange={(event) =>
                      setDetectionMode(event.target.value as typeof detectionMode)
                    }
                  >
                    <option value="FULL_FRAME">الصورة كاملة · الرصد الحالي</option>
                    <option value="SEAT_FOCUS">الصورة ومناطق المقاعد · تجربة أبطأ</option>
                  </select>
                </label>
                {detectionMode === 'SEAT_FOCUS' && (
                  <p className="muted">
                    نفحص مناطق المقاعد بتكبير أعلى لتحسين فرص رصد الأشخاص الصغار في الصورة. التجربة
                    تدعم أربع طاولات كحد أقصى، ولا تضمن تأكيد الجلوس عند حجب الجسم.
                  </p>
                )}
                <label>
                  نموذج تقدير الجلوس
                  <select
                    disabled={running}
                    value={poseModel}
                    onChange={(event) => setPoseModel(event.target.value as typeof poseModel)}
                  >
                    <option value="RTMPOSE_TINY">النموذج الخفيف الحالي</option>
                    <option value="RTMPOSE_MEDIUM">نموذج أكبر · تجربة أبطأ</option>
                  </select>
                </label>
                {poseModel === 'RTMPOSE_MEDIUM' && (
                  <p className="muted">
                    يُنزّل النموذج الأكبر عند أول تشغيل له، وقد يستغرق التنزيل والتحليل وقتًا
                    إضافيًا. نستخدمه للمقارنة؛ لا يضمن تأكيد الجلوس عند حجب الجسم.
                  </p>
                )}
                <label>
                  بداية المقطع بالثواني
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, status.source.duration - 5)}
                    step={0.1}
                    value={startSeconds}
                    disabled={running}
                    onChange={(e) => setStartSeconds(Number(e.target.value))}
                  />
                </label>
                <label>
                  مدة الاختبار بالثواني
                  <input
                    type="number"
                    min={6}
                    max={120}
                    value={durationSeconds}
                    disabled={running}
                    onChange={(e) => setDurationSeconds(Number(e.target.value))}
                  />
                </label>
                <button
                  className="primary"
                  disabled={
                    running ||
                    status.models !== 'READY' ||
                    !regions.length ||
                    regions.some((r) => !validTables.has(r.tableId)) ||
                    (detectionMode === 'SEAT_FOCUS' &&
                      new Set(regions.map((r) => r.tableId)).size > 4) ||
                    durationSeconds < 6 ||
                    durationSeconds > 120
                  }
                  onClick={() => {
                    void start();
                  }}
                >
                  {running ? 'جارٍ التحليل…' : 'تحليل الفيديو على CPU'}
                </button>
                {job?.status === 'RUNNING' && (
                  <>
                    <label>
                      تقدم التحليل
                      <progress max={1} value={job.progress} />
                    </label>
                    {job.message && <p className="muted">{job.message}</p>}
                  </>
                )}
                {job?.status === 'FAILED' && (
                  <p className="message error" role="alert">
                    {job.message}
                  </p>
                )}
                {report && (
                  <>
                    <h3>نتيجة التجربة</h3>
                    <p>
                      طريقة الرصد في التقرير:{' '}
                      {report.detectionMode === 'SEAT_FOCUS'
                        ? 'الصورة ومناطق المقاعد'
                        : 'الصورة كاملة'}
                    </p>
                    <p>
                      نموذج تقدير الجلوس في التقرير:{' '}
                      {report.poseModel === 'RTMPOSE_MEDIUM' ? 'النموذج الأكبر' : 'النموذج الخفيف'}
                    </p>
                    <p>
                      عولجت {report.analyzedSamples} عينة خلال {report.processingSeconds.toFixed(1)}{' '}
                      ثانية.
                    </p>
                    <p>
                      {report.events.length} تأكيد إشغال طاولة، وليس عدد الأشخاص.{' '}
                      {report.events.length === 0 && 'غياب التأكيد لا يعني أن الطاولات فارغة.'}
                    </p>
                    {report.events.map((event) => (
                      <div
                        className="member-row"
                        key={`${event.tableId}-${event.confirmedSeconds}`}
                      >
                        <span>
                          {tableName(event.tableId)} · تأكيد عند {timeLabel(event.confirmedSeconds)}
                        </span>
                        <button onClick={() => seek(event.confirmedSeconds)}>عرض اللحظة</button>
                      </div>
                    ))}
                    <p className="muted">
                      الأخضر دليل جلوس مقبول للتحقق، والأصفر رصد لم يستوفِ شروط التأكيد. تأكيد
                      الطاولة يتطلب دليلًا متواصلًا لخمس ثوانٍ. درجات الرصد ليست قياسًا معتمدًا
                      للدقة. الجلوس الموجود منذ بداية المقطع لا يثبت وقت الوصول الفعلي.
                    </p>
                    {sample && (
                      <section aria-label="تشخيص العينة الحالية">
                        {sample.detectedBoxes !== undefined && (
                          <p>
                            صناديق الأشخاص قبل التتبع: {sample.detectedBoxes}. هذا ليس عددًا مؤكدًا
                            للأشخاص.
                          </p>
                        )}
                        <p>
                          العينة الحالية · أشخاص مرصودون: {sample.people.length} · بوضعية جلوس:{' '}
                          {sample.people.filter((p) => p.posture === 'SEATED').length} · أدلة مقبولة
                          للتحقق:{' '}
                          {
                            sample.people.filter(
                              (p) => trialEvidence(p, minDetectionScore) === 'ACCEPTED',
                            ).length
                          }
                        </p>
                        <details>
                          <summary>أسباب قبول الرصد واستبعاده</summary>
                          {sample.people.map((person, index) => (
                            <p key={`${person.trackId}-${index}`}>
                              #{person.trackId ?? '?'} ·{' '}
                              {evidenceLabels[trialEvidence(person, minDetectionScore)]}
                              {person.tableId && ` · ${tableName(person.tableId)}`} · درجة الرصد{' '}
                              <bdi>{person.score.toFixed(3)}</bdi>
                            </p>
                          ))}
                          {Object.entries(sample.tables).map(([id, state]) => (
                            <p key={id}>
                              {tableName(id)} ·{' '}
                              {state === 'OCCUPIED'
                                ? 'إشغال متحقق في هذه العينة'
                                : 'الإشغال غير محسوم في هذه العينة'}
                            </p>
                          ))}
                        </details>
                      </section>
                    )}
                    <details>
                      <summary>النماذج المستخدمة</summary>
                      <p dir="ltr">
                        {report.models.detector}
                        <br />
                        {report.models.pose}
                        <br />
                        {report.models.tracker}
                      </p>
                    </details>
                    <button className="secondary" onClick={() => download(report)}>
                      تنزيل تقرير التجربة JSON
                    </button>
                  </>
                )}
              </section>
            </div>
          )}
        </>
      )}
    </div>
  );
}
