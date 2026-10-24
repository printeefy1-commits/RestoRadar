import { describe, expect, it } from 'vitest';
import type { TrialReport, WristReset } from '@restoradar/contracts';
import { reviewMoments } from '../../apps/web/src/video-review.ts';

function trial(): TrialReport {
  return {
    mode: 'RECORDED_VIDEO_DRY_RUN',
    sourceId: 'a'.repeat(64),
    models: { detector: 'synthetic', pose: 'synthetic', tracker: 'synthetic', artifacts: {} },
    startSeconds: 0,
    endSeconds: 6,
    processingSeconds: 2,
    samplesPerProcessingSecond: 2,
    analyzedSamples: 12,
    departureEvaluated: false,
    wristTrial: true,
    regions: [{ id: 'seat', tableId: 'table', x: 0.2, y: 0.2, width: 0.2, height: 0.3 }],
    events: [],
    samples: Array.from({ length: 12 }, (_, i) => ({
      seconds: i / 2,
      tables: { table: 'UNKNOWN' as const },
      people: [
        {
          trackId: 1,
          box: [0.1 + i * 0.01, 0.1, 0.3 + i * 0.01, 0.6],
          score: 0.8,
          posture: 'UNCERTAIN' as const,
          tableId: null,
        },
      ],
    })),
  };
}
const reset = (extra: Partial<WristReset> = {}): WristReset => ({
  trackId: 1,
  seatRegionId: 'seat',
  tableId: 'table',
  reason: 'MOVING',
  startedSeconds: 0,
  lastObservedSeconds: 1,
  stableSeconds: 1,
  wasProbable: false,
  detectionScore: 0.8,
  ...extra,
});

describe('visual review moments', () => {
  it('bounds phone memory and keeps the latest failures with adjacent source samples', () => {
    const report = trial();
    report.samples.forEach((s) => {
      s.wristResets = [reset()];
    });
    const original = JSON.stringify(report);
    const moments = reviewMoments(report);
    expect(moments).toHaveLength(8);
    expect(moments[0]!.samples.map((s) => s.seconds)).toEqual([1.5, 2, 2.5]);
    expect(moments.at(-1)!.samples.map((s) => s.seconds)).toEqual([5, 5.5, 5.5]);
    expect(JSON.stringify(report)).toBe(original);
  });
  it('omits trivial cold motion while retaining short confidence interruptions', () => {
    const report = trial();
    report.samples[0]!.wristResets = [reset({ stableSeconds: 0 })];
    report.samples[1]!.wristResets = [reset({ stableSeconds: 0.5, reason: 'AMBIGUOUS_SEAT' })];
    report.samples[2]!.wristResets = [reset({ stableSeconds: 0, reason: 'LOW_DETECTION_SCORE' })];
    report.samples[3]!.wristResets = [reset()];
    expect(reviewMoments(report).map((m) => m.seconds)).toEqual([1, 1.5]);
  });
  it('uses the same tracked person across a missing view without borrowing another person', () => {
    const report = trial();
    report.samples = report.samples.slice(0, 3);
    report.samples[0]!.people[0]!.box = [0.1, 0.1, 0.2, 0.5];
    report.samples[1]!.people[0] = {
      ...report.samples[1]!.people[0]!,
      trackId: 2,
      box: [0, 0, 1, 1],
    };
    report.samples[2]!.people[0]!.box = [0.15, 0.2, 0.25, 0.55];
    report.samples[1]!.wristResets = [reset()];
    const crop = reviewMoments(report)[0]!.crop;
    expect(crop[0]).toBeCloseTo(0.055);
    expect(crop[2]).toBeCloseTo(0.295);
    expect(crop[1]).toBeCloseTo(0.01);
    expect(crop[3]).toBeCloseTo(0.64);
  });
  it('falls back to the recorded seat when the track is absent in all three views', () => {
    const report = trial();
    report.samples.forEach((s) => {
      s.people = [];
    });
    report.samples[0]!.wristResets = [reset({ reason: 'MISSING_TRACK' })];
    const moment = reviewMoments(report)[0]!;
    expect(moment.samples.map((s) => s.seconds)).toEqual([0, 0, 0.5]);
    expect(moment.crop[0]).toBeCloseTo(0.14);
    expect(moment.crop[2]).toBeCloseTo(0.46);
    expect(moment.crop[1]).toBeCloseTo(0.14);
    expect(moment.crop[3]).toBeCloseTo(0.56);
  });
  it('clamps crops to the frame and leaves historical reports without resets valid', () => {
    const report = trial();
    expect(reviewMoments(report)).toEqual([]);
    report.samples[1]!.wristResets = [reset()];
    report.samples.slice(0, 3).forEach((s) => {
      s.people[0]!.box = [0, 0, 1, 1];
    });
    expect(reviewMoments(report)[0]!.crop).toEqual([0, 0, 1, 1]);
    report.wristTrial = false;
    expect(reviewMoments(report)).toEqual([]);
  });
});
