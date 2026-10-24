import type { TrialPerson, TrialReport, WristReset } from '@restoradar/contracts';
import { wristMotionLabels, wristResetLabels } from './video-evidence.ts';

type Sample = TrialReport['samples'][number];
export interface ReviewMoment {
  seconds: number;
  reset: WristReset;
  samples: [Sample, Sample, Sample];
  crop: [number, number, number, number];
}

/** A fixed crop across the three views makes box and body motion comparable. */
export function reviewMoments(report: TrialReport): ReviewMoment[] {
  if (!report.wristTrial) return [];
  const moments: ReviewMoment[] = [];
  report.samples.forEach((sample, index, samples) => {
    for (const reset of sample.wristResets ?? []) {
      if (reset.reason !== 'LOW_DETECTION_SCORE' && reset.stableSeconds < 1) continue;
      const views: ReviewMoment['samples'] = [
        samples[index - 1] ?? sample,
        sample,
        samples[index + 1] ?? sample,
      ];
      const boxes = views
        .flatMap((view) => view.people.filter((p) => p.trackId === reset.trackId).map((p) => p.box))
        .filter(
          (b) =>
            b.every(Number.isFinite) &&
            b[2] > b[0] &&
            b[3] > b[1] &&
            b[0] < 1 &&
            b[1] < 1 &&
            b[2] > 0 &&
            b[3] > 0,
        );
      const seat = report.regions.find((r) => r.id === reset.seatRegionId);
      const bounds: [number, number, number, number] = boxes.length
        ? [
            Math.min(...boxes.map((b) => b[0])),
            Math.min(...boxes.map((b) => b[1])),
            Math.max(...boxes.map((b) => b[2])),
            Math.max(...boxes.map((b) => b[3])),
          ]
        : seat
          ? [seat.x, seat.y, seat.x + seat.width, seat.y + seat.height]
          : [0, 0, 1, 1];
      const padX = Math.max(0.025, (bounds[2] - bounds[0]) * 0.3);
      const padY = Math.max(0.025, (bounds[3] - bounds[1]) * 0.2);
      moments.push({
        seconds: sample.seconds,
        reset,
        samples: views,
        crop: [
          Math.max(0, bounds[0] - padX),
          Math.max(0, bounds[1] - padY),
          Math.min(1, bounds[2] + padX),
          Math.min(1, bounds[3] + padY),
        ],
      });
    }
  });
  // Bound canvas memory and seek work on phones; retain the latest failures.
  return moments.slice(-8);
}

function mediaEvent(
  video: HTMLVideoElement,
  event: string,
  signal: AbortSignal,
  begin?: () => void,
) {
  return new Promise<void>((resolve, reject) => {
    let timer: ReturnType<typeof setTimeout>;
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener(event, ready);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
    };
    const ready = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(new Error('تعذر قراءة لقطات الفيديو؛ حاول مرة أخرى.'));
    };
    const aborted = () => {
      cleanup();
      reject(new DOMException('أُلغي تجهيز اللقطات', 'AbortError'));
    };
    if (signal.aborted) {
      aborted();
      return;
    }
    video.addEventListener(event, ready, { once: true });
    video.addEventListener('error', failed, { once: true });
    signal.addEventListener('abort', aborted, { once: true });
    timer = setTimeout(failed, 15000);
    try {
      begin?.();
    } catch {
      failed();
    }
  });
}

export async function makeReviewImage(
  source: string,
  report: TrialReport,
  signal: AbortSignal,
  progress: (done: number, total: number) => void,
): Promise<Blob> {
  const moments = reviewMoments(report);
  if (!moments.length) throw new Error('لا توجد انقطاعات مناسبة في هذا التقرير.');
  const reader = document.createElement('video');
  reader.preload = 'metadata';
  reader.muted = true;
  reader.playsInline = true;
  const canvas = document.createElement('canvas');
  const width = 480,
    height = 288,
    caption = 30,
    rowHeader = 58,
    header = 76;
  canvas.width = width * 3;
  canvas.height = header + moments.length * (height + caption + rowHeader);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('تعذر تجهيز صورة المراجعة في هذا المتصفح.');
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const line = (value: string, x: number, y: number, size = 18) => {
    ctx.font = `${size}px Arial, sans-serif`;
    ctx.textAlign = 'right';
    ctx.direction = 'rtl';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(value, x, y, canvas.width - 24);
  };
  line(
    'مراجعة انقطاع مؤقتات المعصمين · من اليسار: قبل الانقطاع، عنده، بعده',
    canvas.width - 12,
    28,
    22,
  );
  line(
    `النموذج: \u2066${report.models.pose}\u2069 · بصمة الفيديو: \u2066${report.sourceId.slice(0, 12)}\u2069 · عدد الحالات: ${moments.length}`,
    canvas.width - 12,
    58,
  );
  let done = 0;
  const deadline = performance.now() + 60000;
  progress(done, moments.length * 3);
  try {
    await mediaEvent(reader, 'loadedmetadata', signal, () => {
      reader.src = source;
      reader.load();
    });
    if (reader.readyState < 2)
      await mediaEvent(reader, 'loadeddata', signal, () => {
        reader.preload = 'auto';
        reader.load();
      });
    if (
      !Number.isFinite(reader.duration) ||
      reader.duration < report.endSeconds - 0.1 ||
      reader.videoWidth <= 0 ||
      reader.videoHeight <= 0
    )
      throw new Error('مدة الفيديو لا تطابق التقرير.');
    for (const [row, moment] of moments.entries()) {
      const top = header + row * (height + caption + rowHeader);
      line(
        `#${moment.reset.trackId} · ${moment.seconds.toFixed(2)} ثانية · ${wristResetLabels[moment.reset.reason]}`,
        canvas.width - 12,
        top + 23,
        20,
      );
      const detail = moment.reset.motionFailure;
      line(
        `الثبات قبل الانقطاع: ${moment.reset.stableSeconds.toFixed(1)} ثانية${detail ? ` · ${wristMotionLabels[detail.check]}` : ''}`,
        canvas.width - 12,
        top + 48,
      );
      const [x1, y1, x2, y2] = moment.crop;
      const cropWidth = (x2 - x1) * reader.videoWidth,
        cropHeight = (y2 - y1) * reader.videoHeight;
      const scale = Math.min(width / cropWidth, height / cropHeight);
      for (const [column, sample] of moment.samples.entries()) {
        signal.throwIfAborted();
        if (performance.now() > deadline)
          throw new Error('استغرق تجهيز اللقطات وقتًا طويلًا؛ حاول مرة أخرى.');
        if (done === 0 || Math.abs(reader.currentTime - sample.seconds) > 0.0001) {
          await mediaEvent(reader, 'seeked', signal, () => {
            reader.currentTime = sample.seconds;
          });
        }
        if (reader.readyState < 2) await mediaEvent(reader, 'loadeddata', signal);
        // Allow the decoded frame to reach the canvas source, including time 0
        // where loadeddata may precede the first compositor update.
        await new Promise<void>((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
        );
        signal.throwIfAborted();
        const left = column * width + (width - cropWidth * scale) / 2;
        const imageTop = top + rowHeader + (height - cropHeight * scale) / 2;
        ctx.drawImage(
          reader,
          x1 * reader.videoWidth,
          y1 * reader.videoHeight,
          cropWidth,
          cropHeight,
          left,
          imageTop,
          cropWidth * scale,
          cropHeight * scale,
        );
        const person: TrialPerson | undefined = sample.people.find(
          (p) => p.trackId === moment.reset.trackId,
        );
        if (person) {
          const bx = left + (person.box[0] - x1) * reader.videoWidth * scale;
          const by = imageTop + (person.box[1] - y1) * reader.videoHeight * scale;
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 2;
          ctx.strokeRect(
            bx,
            by,
            (person.box[2] - person.box[0]) * reader.videoWidth * scale,
            (person.box[3] - person.box[1]) * reader.videoHeight * scale,
          );
        }
        line(
          `${['قبل', 'عند', 'بعد'][column]} · ${sample.seconds.toFixed(2)} ثانية · ${person ? `درجة ${person.score.toFixed(3)}` : 'المعرّف غير ظاهر'}`,
          (column + 1) * width - 10,
          top + rowHeader + height + 23,
          17,
        );
        progress(++done, moments.length * 3);
      }
    }
    signal.throwIfAborted();
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.9),
    );
    if (!blob) throw new Error('تعذر حفظ صورة المراجعة.');
    signal.throwIfAborted();
    return blob;
  } finally {
    reader.pause();
    reader.removeAttribute('src');
    reader.load();
  }
}
