import { describe, expect, it } from 'vitest';
import { videoTrialSchema, type TrialPerson } from '@restoradar/contracts';
import { randomUUID } from 'node:crypto';
import { trialEvidence } from '../../apps/web/src/video-evidence.ts';

const seated: TrialPerson = {
  box: [0.1, 0.1, 0.3, 0.8],
  trackId: 39,
  posture: 'SEATED',
  tableId: 'table-2',
  score: 0.1659,
};

describe('video evidence in historical and current reports', () => {
  it('rejects the low-score seated track from the reported 43-second sample', () => {
    expect(trialEvidence(seated)).toBe('LOW_DETECTION_SCORE');
  });
  it('does not equate a person, a seated posture, or a table mapping with accepted evidence', () => {
    expect(trialEvidence({ ...seated, score: 0.9, posture: 'UNCERTAIN' })).toBe('POSE_UNCERTAIN');
    expect(trialEvidence({ ...seated, score: 0.9, tableId: null })).toBe('UNMAPPED');
    expect(trialEvidence({ ...seated, score: 0.35 })).toBe('ACCEPTED');
  });
  it('uses explicit worker diagnostics and the report cutoff when supplied', () => {
    expect(trialEvidence({ ...seated, tableId: null, evidence: 'AMBIGUOUS_REGIONS' })).toBe(
      'AMBIGUOUS_REGIONS',
    );
    expect(trialEvidence({ ...seated, score: 0.5 }, 0.6)).toBe('LOW_DETECTION_SCORE');
    expect(trialEvidence({ ...seated, score: Number.NaN })).toBe('LOW_DETECTION_SCORE');
  });
});

describe('focused video trial contract', () => {
  const region = {
    id: randomUUID(),
    tableId: randomUUID(),
    x: 0.1,
    y: 0.2,
    width: 0.2,
    height: 0.3,
  };
  const trial = {
    sourceId: 'a'.repeat(64),
    startSeconds: 0,
    durationSeconds: 10,
    regions: [region],
  };
  it('keeps old requests valid and accepts focused mode explicitly', () => {
    expect(videoTrialSchema.parse(trial).detectionMode).toBeUndefined();
    expect(videoTrialSchema.parse({ ...trial, detectionMode: 'SEAT_FOCUS' }).detectionMode).toBe(
      'SEAT_FOCUS',
    );
    expect(videoTrialSchema.safeParse({ ...trial, detectionMode: 'unexpected' }).success).toBe(
      false,
    );
  });
  it('bounds focused CPU work by table count without limiting the baseline', () => {
    const regions = Array.from({ length: 5 }, () => ({
      ...region,
      id: randomUUID(),
      tableId: randomUUID(),
    }));
    expect(
      videoTrialSchema.safeParse({ ...trial, regions, detectionMode: 'SEAT_FOCUS' }).success,
    ).toBe(false);
    expect(videoTrialSchema.safeParse({ ...trial, regions }).success).toBe(true);
    expect(
      videoTrialSchema.safeParse({
        ...trial,
        regions: regions.map((r) => ({ ...r, tableId: region.tableId })),
        detectionMode: 'SEAT_FOCUS',
      }).success,
    ).toBe(true);
  });
});
