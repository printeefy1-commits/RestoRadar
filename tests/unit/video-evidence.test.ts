import { describe, expect, it } from 'vitest';
import type { TrialPerson } from '@restoradar/contracts';
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
