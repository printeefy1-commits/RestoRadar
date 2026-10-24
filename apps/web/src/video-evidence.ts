import type {
  TrialEvidence,
  TrialPerson,
  WristEvidence,
  WristMotionFailure,
  WristReset,
} from '@restoradar/contracts';

export const wristReasonLabels: Record<WristEvidence['reason'], string> = {
  NO_TRACK: 'لا يوجد تتبع صالح',
  LOW_DETECTION_SCORE: 'رصد الشخص ضعيف',
  STANDING: 'وقوف ظاهر؛ إشارة المعصمين مستبعدة',
  WRIST_UNCERTAIN: 'موضع المعصمين غير موثوق',
  NO_TABLETOP_MATCH: 'المعصمان خارج مناطق سطح الطاولة',
  AMBIGUOUS_TABLETOP: 'تعارض مناطق سطح الطاولة',
  NO_SEAT_MATCH: 'الجسم خارج مناطق المقاعد',
  AMBIGUOUS_SEAT: 'الجسم داخل أكثر من منطقة مقعد',
  TABLE_MISMATCH: 'الجسم والمعصم مرتبطان بطاولتين مختلفتين',
  SEAT_CONFLICT: 'تعارض تتبع أشخاص في المقعد نفسه',
  MOVING: 'تجاوز حد الثبات في الصورة',
  DWELLING: 'بانتظار خمس ثوانٍ متواصلة',
  PROBABLE: 'إشغال مرجّح بإشارة المعصمين؛ ليس تأكيد جلوس',
};

export const wristResetLabels: Record<WristReset['reason'], string> = {
  ...wristReasonLabels,
  MISSING_TRACK: 'لم يظهر معرّف الشخص في هذه العينة',
  TIMESTAMP_GAP: 'فجوة زمنية تتجاوز حد الاستمرارية',
  MOTION_REFERENCE_CHANGED: 'تغيّر مرجع قياس الحركة',
  SEAT_CHANGED: 'تغيّر المقعد أو الطاولة المرتبطان بالشخص',
};
export const wristMotionLabels: Record<WristMotionFailure['check'], string> = {
  REFERENCE_CHANGED: 'تغيّر مرجع قياس الحركة',
  INVALID_GEOMETRY: 'إحداثيات أو أبعاد غير صالحة',
  ANCHOR_X: 'إزاحة مرجع الجسم أفقيًا',
  ANCHOR_Y: 'إزاحة مرجع الجسم رأسيًا',
  BOX_CENTER_X: 'إزاحة مركز الصندوق أفقيًا',
  BOX_WIDTH_RATIO: 'تغيّر عرض الصندوق',
  BOX_HEIGHT_RATIO: 'تغيّر ارتفاع الصندوق',
  BOX_TOP_Y: 'إزاحة أعلى الصندوق',
};

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
