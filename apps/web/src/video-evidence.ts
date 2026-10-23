import type { TrialEvidence, TrialPerson } from '@restoradar/contracts';

export const evidenceLabels: Record<TrialEvidence, string> = {
  ACCEPTED: 'دليل جلوس مقبول للتحقق',
  POSE_UNCERTAIN: 'وضعية جلوس غير محسومة',
  UNMAPPED: 'جلوس خارج مناطق المقاعد أو ربط غير محسوم',
  AMBIGUOUS_REGIONS: 'تعارض بين مناطق طاولات مختلفة',
  LOW_DETECTION_SCORE: 'جلوس مستبعد: رصد ضعيف',
};

export function trialEvidence(person: TrialPerson, minDetectionScore = 0.35): TrialEvidence {
  if (person.evidence) return person.evidence;
  // Reports generated before evidence diagnostics used the same 0.35 cutoff.
  if (person.posture !== 'SEATED') return 'POSE_UNCERTAIN';
  if (!person.tableId) return 'UNMAPPED';
  if (!Number.isFinite(person.score) || person.score < minDetectionScore)
    return 'LOW_DETECTION_SCORE';
  return 'ACCEPTED';
}
