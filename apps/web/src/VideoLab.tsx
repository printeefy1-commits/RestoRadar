import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  SeatRegion,
  Table,
  TrialJob,
  VideoLabStatus,
  VideoTrial,
} from '@restoradar/contracts';
import { MAX_VIDEO_UPLOAD_BYTES, seatRegionSchema } from '@restoradar/contracts';
import { makeReviewImage, reviewMoments } from './video-review.ts';
import {
  evidenceLabels,
  trialEvidence,
  wristReasonLabels,
  wristResetLabels,
  wristMotionLabels,
  seatReasonLabels,
} from './video-evidence.ts';

async function request<T>(url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, {
    credentials: 'same-origin',
    cache: 'no-store',
    ...(signal ? { signal } : {}),
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
  const [notice, setNotice] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [mediaWorking, setMediaWorking] = useState(false);
  const uploadRequest = useRef<XMLHttpRequest | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const statusVersion = useRef(0);
  const [regions, setRegions] = useState<SeatRegion[]>([]);
  const [tabletopRegions, setTabletopRegions] = useState<SeatRegion[]>([]);
  const [wristTrial, setWristTrial] = useState(false);
  const [seatTrial, setSeatTrial] = useState(false);
  const [regionKind, setRegionKind] = useState<'SEAT' | 'TABLETOP'>('SEAT');
  const [tableId, setTableId] = useState(tables[0]?.id ?? '');
  const [job, setJob] = useState<TrialJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [reviewProgress, setReviewProgress] = useState<[number, number] | null>(null);
  const reviewAbort = useRef<AbortController | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<{ x: number; y: number; endX: number; endY: number } | null>(
    null,
  );
  const [startSeconds, setStartSeconds] = useState(0);
  const [durationSeconds, setDurationSeconds] = useState(60);
  const [detectionMode, setDetectionMode] = useState<'FULL_FRAME' | 'SEAT_FOCUS'>('FULL_FRAME');
  const [poseModel, setPoseModel] =
    useState<NonNullable<VideoTrial['poseModel']>>('YOLO26_NANO_POSE');
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const sourceId = useRef<string | null>(null);
  useEffect(() => () => reviewAbort.current?.abort(), [base, status?.source?.id]);
  useEffect(() => () => uploadRequest.current?.abort(), [base]);
  const validTables = new Set(tables.map((t) => t.id));
  const running = busy || mediaWorking || status?.mediaBusy === true || job?.status === 'RUNNING';
  const tabletopDrawing = wristTrial && regionKind === 'TABLETOP';
  const drawingRegions = tabletopDrawing ? tabletopRegions : regions;
  const overlappingSeats = regions.some((a, index) =>
    regions
      .slice(index + 1)
      .some(
        (b) =>
          Math.max(a.x, b.x) < Math.min(a.x + a.width, b.x + b.width) &&
          Math.max(a.y, b.y) < Math.min(a.y + a.height, b.y + b.height),
      ),
  );
  const appendRegion = (region: SeatRegion) => {
    if (tabletopDrawing) setTabletopRegions((old) => [...old, region]);
    else setRegions((old) => [...old, region]);
  };
  const tableName = (id: string) => tables.find((t) => t.id === id)?.name ?? 'طاولة محذوفة';

  const loadStatus = useCallback(async () => {
    const version = ++statusVersion.current;
    const next = await request<VideoLabStatus>(base);
    if (version !== statusVersion.current) return;
    if (sourceId.current && sourceId.current !== next.source?.id) {
      setRegions([]);
      setTabletopRegions([]);
      setDraft(null);
      setDrawing(false);
      setJob(null);
      setSeconds(0);
      setStartSeconds(0);
      setPlaying(false);
      setError('تغير ملف الفيديو. أعد تحديد مناطق المقاعد.');
    }
    sourceId.current = next.source?.id ?? null;
    setStatus(next);
  }, [base]);

  async function refreshMedia(message: string) {
    reviewAbort.current?.abort();
    setRegions([]);
    setTabletopRegions([]);
    setDraft(null);
    setDrawing(false);
    setJob(null);
    setSeconds(0);
    setStartSeconds(0);
    setPlaying(false);
    await loadStatus();
    setError('');
    setNotice(message);
    setUploadFile(null);
    if (fileInput.current) fileInput.current.value = '';
  }
  async function uploadVideo() {
    if (!uploadFile || running || reviewProgress) return;
    const limit = status?.maxUploadBytes ?? MAX_VIDEO_UPLOAD_BYTES;
    if (!/\.mp4$/i.test(uploadFile.name) || !uploadFile.size || uploadFile.size > limit) {
      setError('اختر ملف MP4 غير فارغ بحجم لا يتجاوز 100 ميغابايت.');
      return;
    }
    setMediaWorking(true);
    setUploadProgress(0);
    setError('');
    setNotice('');
    const xhr = new XMLHttpRequest();
    uploadRequest.current = xhr;
    try {
      await new Promise<void>((resolve, reject) => {
        xhr.open('POST', `${base}/video`);
        xhr.timeout = 180000;
        xhr.setRequestHeader('Content-Type', 'video/mp4');
        xhr.setRequestHeader('X-Video-Name', encodeURIComponent(uploadFile.name));
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable)
            setUploadProgress(Math.round((event.loaded * 100) / event.total));
        };
        xhr.onload = () => {
          let result: { message?: string } = {};
          try {
            result = JSON.parse(xhr.responseText);
          } catch {
            /* A proxy may return non-JSON. */
          }
          if (xhr.status === 201) resolve();
          else reject(new Error(result.message ?? 'تعذر رفع الفيديو؛ حاول مجددًا.'));
        };
        xhr.onerror = () => reject(new Error('انقطع الاتصال أثناء الرفع؛ حاول مجددًا.'));
        xhr.ontimeout = () => reject(new Error('انتهت مهلة الرفع؛ جرّب ملفًا أصغر.'));
        xhr.onabort = () => reject(new Error('أُلغي رفع الفيديو.'));
        xhr.send(uploadFile);
      });
      await refreshMedia('تم رفع الفيديو. ارسم مناطق المقاعد لهذا المقطع قبل التحليل.');
    } catch (e) {
      setError((e as Error).message);
      // The server may have completed just before a response was interrupted.
      void loadStatus().catch(() => undefined);
    } finally {
      uploadRequest.current = null;
      setUploadProgress(null);
      setMediaWorking(false);
    }
  }
  async function deleteVideo() {
    if (!status?.source?.uploaded || running || reviewProgress) return;
    if (!window.confirm('حذف الفيديو المرفوع؟ سيعود فيديو التجربة الأصلي إن كان متوفرًا.')) return;
    setMediaWorking(true);
    setError('');
    setNotice('');
    try {
      await request(`${base}/video/delete`, { sourceId: status.source.id });
      await refreshMedia('حُذف الفيديو المرفوع. سيظهر فيديو التجربة الأصلي إن كان متوفرًا.');
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setMediaWorking(false);
    }
  }
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
    if (wristTrial)
      for (const region of tabletopRegions) {
        context.strokeStyle = '#a78bfa';
        context.fillStyle = '#a78bfa22';
        context.fillRect(region.x * w, region.y * h, region.width * w, region.height * h);
        context.strokeRect(region.x * w, region.y * h, region.width * w, region.height * h);
        context.fillStyle = '#ffffff';
        context.fillText(
          `${tableName(region.tableId)} · سطح الطاولة`,
          region.x * w,
          Math.max(18, region.y * h - 5),
        );
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
      if (wristTrial && report?.wristTrial)
        for (const point of person.posePoints ?? []) {
          if (
            !['LEFT_WRIST', 'RIGHT_WRIST'].includes(point.part) ||
            point.score < (report.wristPolicy?.minKeypointScore ?? 0.4)
          )
            continue;
          context.fillStyle = '#a78bfa';
          context.beginPath();
          context.arc(point.point[0] * w, point.point[1] * h, Math.max(3, w / 250), 0, Math.PI * 2);
          context.fill();
        }
    }
  }, [
    regions,
    tabletopRegions,
    wristTrial,
    report?.wristTrial,
    report?.wristPolicy,
    sample,
    draft,
    tables,
    minDetectionScore,
  ]);

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
  async function downloadReview() {
    if (!report || !status?.source || reviewProgress || running) return;
    if (report.sourceId !== status.source.id) {
      setError('التقرير يخص ملف فيديو مختلفًا.');
      return;
    }
    const controller = new AbortController();
    reviewAbort.current = controller;
    setError('');
    setReviewProgress([0, 0]);
    try {
      const blob = await makeReviewImage(
        `${base}/video?source=${report.sourceId}`,
        report,
        controller.signal,
        (done, total) => setReviewProgress([done, total]),
      );
      const current = await request<VideoLabStatus>(base, undefined, controller.signal);
      controller.signal.throwIfAborted();
      if (current.source?.id !== report.sourceId)
        throw new Error('تغير ملف الفيديو أثناء تجهيز اللقطات. أعد فتح الصفحة.');
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'restoradar-review.jpg';
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      reviewAbort.current = null;
      setReviewProgress(null);
    }
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
          ...(wristTrial ? { wristTrial: true, tabletopRegions } : {}),
          ...(seatTrial ? { seatTrial: true } : {}),
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
      {notice && (
        <p className="message" role="status">
          {notice}
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
            <section aria-label="إدارة فيديو التجربة">
              <p className="muted">
                ارفع فيديو MP4 بترميز H.264، حتى 100 ميغابايت ومدته ست ثوانٍ على الأقل. يُرسل إلى
                الخادم لتحليله ويبقى حتى تحذفه أو تستبدله. الرفع الجديد يستبدل الفيديو المرفوع
                السابق بعد فحصه، ويُبقي فيديو التجربة الأصلي محفوظًا.
              </p>
              <label>
                اختيار فيديو من الجهاز
                <input
                  ref={fileInput}
                  type="file"
                  accept="video/mp4,.mp4"
                  disabled={running || !!reviewProgress}
                  onChange={(event) => {
                    setUploadFile(event.target.files?.[0] ?? null);
                    setNotice('');
                  }}
                />
              </label>
              <button
                disabled={!uploadFile || running || !!reviewProgress}
                onClick={() => void uploadVideo()}
              >
                {status.source?.uploaded ? 'رفع فيديو بديل' : 'رفع الفيديو'}
              </button>
              {uploadProgress !== null && (
                <>
                  <p role="status">
                    {uploadProgress < 100
                      ? `جارٍ رفع الفيديو: ${uploadProgress}%`
                      : 'اكتمل النقل؛ جارٍ فحص الفيديو…'}
                  </p>
                  <progress aria-label="تقدم رفع الفيديو" value={uploadProgress} max={100} />
                  <button className="secondary" onClick={() => uploadRequest.current?.abort()}>
                    إلغاء الرفع
                  </button>
                </>
              )}
              {status.source?.uploaded && (
                <>
                  <p>الفيديو المرفوع: {status.source.name ?? 'video.mp4'}</p>
                  <button
                    className="secondary"
                    disabled={running || !!reviewProgress}
                    onClick={() => void deleteVideo()}
                  >
                    حذف الفيديو المرفوع
                  </button>
                </>
              )}
            </section>
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
                    aria-label={
                      tabletopDrawing
                        ? 'رسم منطقة سطح الطاولة على الفيديو'
                        : 'رسم منطقة المقعد على الفيديو'
                    }
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
                      if (region.width >= 0.01 && region.height >= 0.01) appendRegion(region);
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
                        const restoredTops = seatRegionSchema
                          .array()
                          .max(40)
                          .safeParse(previous.tabletopRegions ?? []);
                        if (
                          !restored.success ||
                          !restoredTops.success ||
                          new Set([...restored.data, ...restoredTops.data].map((r) => r.id))
                            .size !==
                            restored.data.length + restoredTops.data.length
                        )
                          throw new Error('مناطق التقرير غير صالحة');
                        if (
                          [...restored.data, ...restoredTops.data].some(
                            (r) => !validTables.has(r.tableId),
                          ) ||
                          restoredTops.data.some(
                            (r) => !restored.data.some((seat) => seat.tableId === r.tableId),
                          )
                        )
                          throw new Error('بعض طاولات التقرير غير موجودة في هذا المطعم');
                        setRegions(restored.data);
                        setTabletopRegions(restoredTops.data);
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
                  {tabletopDrawing
                    ? 'اختر الطاولة وارسم مستطيلًا حول سطحها الذي قد تظهر فوقه المعاصم.'
                    : 'أوقف الفيديو عند ظهور ضيف جالس. اختر الطاولة وارسم مستطيلًا حول موضع حوضه المتوقع على المقعد، لا حول سطح الطاولة. أضف منطقة لكل مقعد ظاهر.'}
                </p>
                <label>
                  <input
                    type="checkbox"
                    checked={wristTrial}
                    disabled={running}
                    onChange={(event) => {
                      setWristTrial(event.target.checked);
                      setRegionKind('SEAT');
                      setDrawing(false);
                      setDraft(null);
                    }}
                  />
                  تجربة إشارة المعصمين
                </label>
                {wristTrial && (
                  <>
                    <p className="muted">
                      ارسم موضع كل مقعد بالأزرق، وسطح الطاولة بالبنفسجي. نبحث عن معصم داخل سطح
                      الطاولة وجسم ثابت قرب المقعد لخمس ثوانٍ. النتيجة إشغال مرجّح؛ لا تثبت ملامسة
                      الطاولة أو الجلوس، وقد تشمل واقفًا محجوبًا.
                    </p>
                    {overlappingSeats && (
                      <p role="status">
                        توجد مناطق مقاعد متداخلة. حاول فصلها حول مواضع الأجسام؛ قد تبقى إشارة
                        المعصمين غير محسومة عند التداخل.
                      </p>
                    )}
                    <label>
                      نوع المنطقة
                      <select
                        value={regionKind}
                        disabled={running}
                        onChange={(event) => {
                          setRegionKind(event.target.value as typeof regionKind);
                          setDrawing(false);
                          setDraft(null);
                        }}
                      >
                        <option value="SEAT">موضع المقعد</option>
                        <option value="TABLETOP">سطح الطاولة للمعصمين</option>
                      </select>
                    </label>
                  </>
                )}
                <label>
                  <input
                    type="checkbox"
                    checked={seatTrial}
                    disabled={running}
                    onChange={(event) => setSeatTrial(event.target.checked)}
                  />
                  تجربة تثبيت الجلوس لكل مقعد
                </label>
                {seatTrial && (
                  <p className="muted">
                    نتحقق أولًا من دليل قوي لخمس ثوانٍ للشخص والمقعد نفسيهما. تستخدم التجربة
                    الوضعية، وتضيف دليل المعصمين عند تفعيله. عند انقطاع قصير تبقى الحالة غير محسومة
                    حتى يعود رصد قوي ووضعية جلوس مقبولة خلال ثانيتين. النتائج تجريبية مستقلة.
                  </p>
                )}
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
                      disabled={running || !tableId || drawingRegions.length >= 40}
                      onClick={() => {
                        video.current?.pause();
                        setDrawing(!drawing);
                      }}
                    >
                      {drawing
                        ? 'إلغاء الرسم'
                        : tabletopDrawing
                          ? 'رسم سطح الطاولة'
                          : 'رسم منطقة مقعد'}
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
                          appendRegion({ id: crypto.randomUUID(), tableId, x, y, width, height });
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
                        <button disabled={running || !tableId || drawingRegions.length >= 40}>
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
                {wristTrial &&
                  tabletopRegions.map((r, i) => (
                    <div className="member-row" key={r.id}>
                      <span>
                        {tableName(r.tableId)} · سطح الطاولة {i + 1}
                      </span>
                      <button
                        disabled={running}
                        onClick={() =>
                          setTabletopRegions((old) => old.filter((v) => v.id !== r.id))
                        }
                      >
                        حذف سطح الطاولة {i + 1}
                      </button>
                    </div>
                  ))}
                {report &&
                  (!regions.length ||
                    (wristTrial &&
                      !tabletopRegions.length &&
                      !!report.tabletopRegions?.length)) && (
                    <button
                      onClick={() => {
                        setRegions(report.regions.filter((r) => validTables.has(r.tableId)));
                        setTabletopRegions(
                          (report.tabletopRegions ?? []).filter((r) => validTables.has(r.tableId)),
                        );
                        setDrawing(false);
                        setDraft(null);
                      }}
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
                  كاشف الأشخاص:{' '}
                  {poseModel === 'YOLO26_SMALL_POSE'
                    ? 'YOLO26s-Pose'
                    : poseModel === 'YOLO26_NANO_POSE'
                      ? 'YOLO26n-Pose'
                      : 'YOLO11 Large'}
                  . نعالج إطارين من زمن الفيديو في الثانية، ونتحقق من دليل جلوس مستمر لمدة 5 ثوانٍ.
                  ضعف وضوح مفاصل الجسم قد يمنع التأكيد.
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
                    <option value="YOLO26_NANO_POSE">
                      YOLO26n-Pose · كشف ومفاصل معًا · تجربة خفيفة
                    </option>
                    <option value="YOLO26_SMALL_POSE">
                      YOLO26s-Pose · أكبر من Nano · تجربة مقارنة
                    </option>
                    <option value="RTMPOSE_TINY">YOLO11 Large + RTMPose الخفيف</option>
                    <option value="RTMPOSE_MEDIUM">نموذج أكبر · تجربة أبطأ</option>
                  </select>
                </label>
                {(poseModel === 'YOLO26_NANO_POSE' || poseModel === 'YOLO26_SMALL_POSE') && (
                  <p className="muted">
                    يكشف الأشخاص ومفاصلهم بتمرير واحد لتقليل زمن المعالجة. ابدأ بالصورة الكاملة
                    وقارن النتيجة بالتقرير السابق. هذه تجربة مسجلة؛ لا تفعّل بث الكاميرا الحي، ولا
                    تضمن سرعة آنية أو دقة الجلوس.
                  </p>
                )}
                {poseModel === 'YOLO26_SMALL_POSE' && (
                  <p className="muted">
                    نموذج Small أكبر من Nano وقد يستغرق وقتًا أطول. قارنه بالمناطق والفترة
                    والإعدادات نفسها؛ لا يضمن رصدًا أفضل للأشخاص المحجوبين أو سرعة آنية.
                  </p>
                )}
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
                    (wristTrial &&
                      (!tabletopRegions.length ||
                        tabletopRegions.some(
                          (r) =>
                            !validTables.has(r.tableId) ||
                            !regions.some((seat) => seat.tableId === r.tableId),
                        ))) ||
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
                    {report.wristTrial && (
                      <p>
                        تجربة المعصمين مفعّلة في هذا التقرير. الإشغال المرجّح منفصل عن تأكيدات
                        الجلوس.
                      </p>
                    )}
                    <p>
                      طريقة الرصد في التقرير:{' '}
                      {report.detectionMode === 'SEAT_FOCUS'
                        ? 'الصورة ومناطق المقاعد'
                        : 'الصورة كاملة'}
                    </p>
                    <p>
                      نموذج تقدير الجلوس في التقرير:{' '}
                      {report.poseModel === 'YOLO26_SMALL_POSE'
                        ? 'YOLO26s-Pose'
                        : report.poseModel === 'YOLO26_NANO_POSE'
                          ? 'YOLO26n-Pose'
                          : report.poseModel === 'RTMPOSE_MEDIUM'
                            ? 'النموذج الأكبر'
                            : 'النموذج الخفيف'}
                    </p>
                    {report.processingSeconds > 0 && report.endSeconds > report.startSeconds && (
                      <p>
                        سرعة معالجة المقطع:{' '}
                        {(
                          (report.endSeconds - report.startSeconds) /
                          report.processingSeconds
                        ).toFixed(2)}
                        × من الزمن الحقيقي. قيمة 1 أو أكثر تعني أن معالجة هذه التجربة استغرقت زمنًا
                        لا يزيد على مدة المقطع؛ يشمل القياس تجهيز النموذج وفك الفيديو.
                      </p>
                    )}
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
                        {report.wristTrial && (
                          <section aria-label="إشارة المعصمين التجريبية">
                            <p>
                              إشغال مرجّح بإشارة المعصمين:{' '}
                              {
                                sample.people.filter((p) => p.wristEvidence?.state === 'PROBABLE')
                                  .length
                              }{' '}
                              · ليس عدد جلوس مؤكدًا.
                            </p>
                            {sample.people.map(
                              (person, index) =>
                                person.wristEvidence && (
                                  <p key={`${person.trackId}-${index}`}>
                                    #{person.trackId ?? '?'} ·{' '}
                                    {wristReasonLabels[person.wristEvidence.reason]}
                                    {person.wristEvidence.tableId &&
                                      ` · ${tableName(person.wristEvidence.tableId)}`}{' '}
                                    · ثبات {person.wristEvidence.stableSeconds.toFixed(1)} ثانية
                                    {person.wristEvidence.scoreContinuation &&
                                      ' · استمرار مؤقت عند انخفاض درجة الرصد'}
                                    {person.wristEvidence.seatResolution === 'CONTINUITY' &&
                                      ' · احتُفظ بالمقعد السابق مع ثبات الجسم'}
                                    {person.wristEvidence.anchorSource === 'BOX_CENTER' &&
                                      ' · موضع الجسم من مركز الصندوق بسبب ضعف الحوض'}
                                  </p>
                                ),
                            )}
                            {Object.entries(sample.wristTables ?? {}).map(([id, state]) => (
                              <p key={id}>
                                {tableName(id)} ·{' '}
                                {state === 'PROBABLE'
                                  ? 'إشغال مرجّح فقط'
                                  : 'إشارة المعصمين غير محسومة'}
                              </p>
                            ))}
                            {!!sample.wristResets?.length && (
                              <details>
                                <summary>أسباب تصفير مؤقتات المعصمين في هذه العينة</summary>
                                {sample.wristResets.map((reset, index) => (
                                  <p key={`${reset.trackId}-${reset.seatRegionId}-${index}`}>
                                    #{reset.trackId} · {tableName(reset.tableId)} · انقطع المؤقت بعد{' '}
                                    {reset.stableSeconds.toFixed(1)} ثانية ·{' '}
                                    {wristResetLabels[reset.reason]}
                                    {reset.detectionScore !== null && (
                                      <>
                                        {' '}
                                        · درجة الرصد <bdi>{reset.detectionScore.toFixed(5)}</bdi>
                                      </>
                                    )}
                                    {reset.motionFailure && (
                                      <>
                                        {' '}
                                        · {wristMotionLabels[reset.motionFailure.check]}
                                        {reset.motionFailure.observed !== null && (
                                          <>
                                            {' '}
                                            · القيمة{' '}
                                            <bdi>{reset.motionFailure.observed.toFixed(4)}</bdi> ·
                                            المجال{' '}
                                            <bdi>
                                              {reset.motionFailure.minimum?.toFixed(4)}–
                                              {reset.motionFailure.maximum?.toFixed(4)}
                                            </bdi>
                                          </>
                                        )}
                                      </>
                                    )}
                                  </p>
                                ))}
                                <small className="muted">
                                  الإزاحات نسب إلى أبعاد الصورة، وتغيّر الحجم نسبة إلى الصندوق
                                  المرجعي. هذا تشخيص للرصد ولا يثبت حركة فعلية أو مغادرة.
                                </small>
                              </details>
                            )}
                          </section>
                        )}
                        {report.seatTrial && (
                          <section aria-label="تثبيت الجلوس لكل مقعد">
                            <p>
                              جلوس مرجّح لكل المقاعد:{' '}
                              {Object.values(sample.seatCounts ?? {}).reduce(
                                (sum, n) => sum + n,
                                0,
                              )}{' '}
                              · أدلة حالية وليست جلسات تشغيلية.
                            </p>
                            {Object.entries(sample.seatCounts ?? {}).map(([id, count]) => (
                              <p key={id}>
                                {tableName(id)} · {count} أشخاص بأدلة مرجّحة
                              </p>
                            ))}
                            {sample.people.map(
                              (person, index) =>
                                person.seatEvidence && (
                                  <p key={`${person.trackId}-${index}`}>
                                    #{person.trackId ?? '?'} ·{' '}
                                    {seatReasonLabels[person.seatEvidence.reason]}
                                    {person.seatEvidence.tableId &&
                                      ` · ${tableName(person.seatEvidence.tableId)}`}{' '}
                                    · مدة الوضعية الحالية{' '}
                                    {person.seatEvidence.poseSeconds.toFixed(1)} ثانية
                                    {person.seatEvidence.qualifiedAtSeconds !== null &&
                                      ` · تحقق أول عند ${timeLabel(person.seatEvidence.qualifiedAtSeconds)}`}
                                    {person.seatEvidence.resumed &&
                                      ` · إعادة تحقق بعد ${person.seatEvidence.gapSeconds.toFixed(1)} ثانية`}
                                    {person.seatEvidence.revalidationPending &&
                                      ' · تاريخ محفوظ مؤقتًا؛ غير محسوب حاليًا'}
                                  </p>
                                ),
                            )}
                            <small className="muted">
                              كل مقعد يُحسب مرة واحدة. الصفر لا يثبت فراغ المقاعد أو المغادرة. تبقى
                              إشارات المعصمين وتأكيدات الطاولات منفصلة للمقارنة.
                            </small>
                          </section>
                        )}
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
                    <button
                      className="secondary"
                      disabled={running || !!reviewProgress || !reviewMoments(report).length}
                      onClick={() => {
                        void downloadReview();
                      }}
                    >
                      {reviewProgress
                        ? `جارٍ تجهيز اللقطات (${reviewProgress[0]}/${reviewProgress[1]})`
                        : 'تنزيل لقطات مراجعة الانقطاع'}
                    </button>
                    {reviewProgress && (
                      <button onClick={() => reviewAbort.current?.abort()}>
                        إلغاء تجهيز اللقطات
                      </button>
                    )}
                    <small className="muted">
                      صورة واحدة لآخر ثماني لحظات مناسبة، تضم قبل الانقطاع وعنده وبعده، مع تكبير
                      الشخص وإطار الرصد. تُجهز من الفيديو في متصفحك وتُنزّل على جهازك عند الطلب.
                      تتطلب تقريرًا يحتوي أسباب تصفير المؤقتات.
                    </small>
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
