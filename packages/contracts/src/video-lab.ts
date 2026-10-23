import { z } from 'zod';

// Normalized rectangles enclose the expected seated hips, not the tabletop.
export const seatRegionSchema = z
  .object({
    id: z.uuid(),
    tableId: z.uuid(),
    x: z.number().min(0).max(1),
    y: z.number().min(0).max(1),
    width: z.number().min(0.01).max(1),
    height: z.number().min(0.01).max(1),
  })
  .strict()
  .refine((r) => r.x + r.width <= 1.000001 && r.y + r.height <= 1.000001);
export const videoTrialSchema = z
  .object({
    sourceId: z.string().regex(/^[a-f0-9]{64}$/),
    startSeconds: z.number().min(0).max(86400),
    durationSeconds: z.number().int().min(6).max(120),
    detectionMode: z.enum(['FULL_FRAME', 'SEAT_FOCUS']).optional(),
    poseModel: z.enum(['RTMPOSE_TINY', 'RTMPOSE_MEDIUM', 'YOLO26_NANO_POSE']).optional(),
    wristTrial: z.boolean().optional(),
    tabletopRegions: z.array(seatRegionSchema).max(40).optional(),
    regions: z.array(seatRegionSchema).min(1).max(40),
  })
  .strict()
  .superRefine((v, ctx) => {
    const tops = v.tabletopRegions ?? [];
    const all = [...v.regions, ...tops];
    if (new Set(all.map((r) => r.id)).size !== all.length)
      ctx.addIssue({ code: 'custom', message: 'Repeated region identifiers' });
    if (v.wristTrial && !tops.length)
      ctx.addIssue({ code: 'custom', message: 'ارسم منطقة سطح الطاولة لتجربة المعصمين' });
    if (!v.wristTrial && tops.length)
      ctx.addIssue({ code: 'custom', message: 'فعّل تجربة المعصمين لاستخدام مناطق سطح الطاولة' });
    if (tops.some((r) => !v.regions.some((seat) => seat.tableId === r.tableId)))
      ctx.addIssue({ code: 'custom', message: 'اربط سطح الطاولة بمنطقة مقعد للطاولة نفسها' });
    if (v.detectionMode === 'SEAT_FOCUS' && new Set(v.regions.map((r) => r.tableId)).size > 4)
      ctx.addIssue({
        code: 'custom',
        message: 'الرصد المركّز يدعم أربع طاولات كحد أقصى في التجربة الواحدة',
      });
  });
export type SeatRegion = z.infer<typeof seatRegionSchema>;
export type VideoTrial = z.infer<typeof videoTrialSchema>;
export interface VideoLabStatus {
  enabled: boolean;
  models: 'DOWNLOADING' | 'READY' | 'ERROR';
  message?: string;
  source: { id: string; duration: number; width: number; height: number; bytes: number } | null;
}
export type TrialEvidence =
  'ACCEPTED' | 'POSE_UNCERTAIN' | 'UNMAPPED' | 'AMBIGUOUS_REGIONS' | 'LOW_DETECTION_SCORE';
export interface WristEvidence {
  state: 'UNKNOWN' | 'CANDIDATE' | 'PROBABLE';
  reason:
    | 'NO_TRACK'
    | 'LOW_DETECTION_SCORE'
    | 'STANDING'
    | 'WRIST_UNCERTAIN'
    | 'NO_TABLETOP_MATCH'
    | 'AMBIGUOUS_TABLETOP'
    | 'NO_SEAT_MATCH'
    | 'AMBIGUOUS_SEAT'
    | 'TABLE_MISMATCH'
    | 'SEAT_CONFLICT'
    | 'MOVING'
    | 'DWELLING'
    | 'PROBABLE';
  tableId: string | null;
  seatRegionId: string | null;
  bodyAnchor: [number, number] | null;
  anchorSource: 'HIP' | 'BOX_CENTER' | null;
  motionAnchor?: [number, number] | null;
  motionSource?: 'SHOULDERS' | 'BOX_CENTER' | null;
  seatRegionCandidates?: string[];
  seatResolution?: 'DIRECT' | 'CONTINUITY' | null;
  stableSeconds: number;
  detectionScore?: number | null;
  scoreContinuation?: boolean;
  lowScoreSeconds?: number;
}
export interface TrialPerson {
  box: [number, number, number, number];
  trackId: number | null;
  posture: 'SEATED' | 'UNCERTAIN';
  tableId: string | null;
  score: number;
  evidence?: TrialEvidence;
  seatedAnchor?: [number, number] | null;
  wristEvidence?: WristEvidence;
  posePoints?: {
    part:
      | 'LEFT_SHOULDER'
      | 'RIGHT_SHOULDER'
      | 'LEFT_HIP'
      | 'RIGHT_HIP'
      | 'LEFT_KNEE'
      | 'RIGHT_KNEE'
      | 'LEFT_WRIST'
      | 'RIGHT_WRIST';
    point: [number, number];
    score: number;
  }[];
}
export interface TrialReport {
  mode: 'RECORDED_VIDEO_DRY_RUN';
  detectionMode?: 'FULL_FRAME' | 'SEAT_FOCUS';
  poseModel?: 'RTMPOSE_TINY' | 'RTMPOSE_MEDIUM' | 'YOLO26_NANO_POSE';
  focusMergePolicy?: 'ALIGNED_CONTAINMENT_V1';
  sourceId: string;
  detectorModel?: 'YOLO11_LARGE' | 'YOLO26_NANO_POSE';
  personMergePolicy?: 'STRICT_UPPER_BODY_ALIGNMENT_V1';
  models: { detector: string; pose: string; tracker: string; artifacts: Record<string, string> };
  startSeconds: number;
  endSeconds: number;
  analyzedSamples: number;
  processingSeconds: number;
  samplesPerProcessingSecond: number;
  departureEvaluated: false;
  evidencePolicy?: { minDetectionScore: number; minKeypointScore: number; seatingSeconds: number };
  regions: SeatRegion[];
  wristTrial?: boolean;
  tabletopRegions?: SeatRegion[];
  wristPolicy?: {
    version: 'WRIST_SEAT_DWELL_V1' | 'WRIST_SEAT_DWELL_V2' | 'WRIST_SEAT_DWELL_V3';
    startDetectionScore?: number;
    continueDetectionScore?: number;
    maxLowScoreSeconds?: number;
    motionReference?: 'SHOULDERS_THEN_BOX_CENTER';
    overlapResolution?: 'PREVIOUS_SEAT_WHILE_STATIONARY';
    seconds: number;
    minKeypointScore: number;
    maxDriftBoxWidths: number;
    maxDriftBoxHeights: number;
  };
  events: {
    tableId: string;
    kind: 'SEATING_CONFIRMED';
    observedSeconds: number;
    confirmedSeconds: number;
  }[];
  samples: {
    seconds: number;
    /** Detector boxes before ByteTrack filtering; not a unique person count. */
    detectedBoxes?: number;
    focusCrops?: number;
    people: TrialPerson[];
    tables: Record<string, 'OCCUPIED' | 'UNKNOWN'>;
    wristTables?: Record<string, 'PROBABLE' | 'UNKNOWN'>;
  }[];
}
export interface TrialJob {
  id: string;
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  progress: number;
  message?: string;
  report?: TrialReport;
}
