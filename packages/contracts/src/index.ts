import { z } from 'zod';

export const roleSchema = z.enum(['OWNER', 'MANAGER', 'WAITER']);
export type Role = z.infer<typeof roleSchema>;
export type Occupancy = 'EMPTY' | 'OCCUPIED';
export type VisionState = Occupancy | 'UNKNOWN' | 'VISION_UNAVAILABLE';
export const taskKindSchema = z.enum(['CUSTOMER_RECEPTION', 'TABLE_CLEANING']);
export type TaskKind = z.infer<typeof taskKindSchema>;
const seconds = z.number().int().min(1).max(86400);
export const policySchema = z
  .object({
    CUSTOMER_RECEPTION: z
      .object({ acceptanceSeconds: seconds, completionSeconds: seconds })
      .strict(),
    TABLE_CLEANING: z.object({ acceptanceSeconds: seconds, completionSeconds: seconds }).strict(),
  })
  .strict();
export type TimeoutPolicy = z.infer<typeof policySchema>;
const id = z.uuid();
const version = z.number().int().nonnegative();
const reason = z.string().trim().min(3).max(1000);
const date = z.iso.datetime().transform((value) => new Date(value).toISOString());
// Confirmed transition fixtures exercise the domain. This is not a camera detector API.
export const commandSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('policy.configure'), policy: policySchema }),
  z.object({ type: z.literal('table.create'), name: z.string().trim().min(1).max(80) }),
  z.object({
    type: z.literal('table.correct'),
    tableId: id,
    expectedVersion: version,
    occupancy: z.enum(['EMPTY', 'OCCUPIED']),
    reason,
  }),
  z.object({ type: z.literal('table.ready'), tableId: id, expectedVersion: version }),
  z.object({
    type: z.literal('simulation.transition'),
    tableId: id,
    occupancy: z.enum(['EMPTY', 'OCCUPIED']),
    observedAt: date,
    confirmedAt: date,
  }),
  z.object({
    type: z.literal('simulation.vision'),
    tableId: id,
    state: z.enum(['UNKNOWN', 'VISION_UNAVAILABLE']),
  }),
  z.object({
    type: z.literal('session.adjust'),
    sessionId: id,
    expectedVersion: version,
    field: z.enum(['startedAt', 'endedAt']),
    at: date,
    reason,
  }),
  z.object({ type: z.literal('alert.acknowledge'), alertId: id }),
  z.object({
    type: z.literal('task.assign'),
    kind: taskKindSchema,
    sourceId: id,
    waiterId: id,
    expectedVersion: version,
  }),
  z.object({ type: z.literal('task.accept'), taskId: id, expectedVersion: version }),
  z.object({ type: z.literal('task.complete'), taskId: id, expectedVersion: version }),
]);
export type Command = z.infer<typeof commandSchema>;
export const commandEnvelopeSchema = z.object({ commandId: id, command: commandSchema }).strict();
export type CommandEnvelope = z.infer<typeof commandEnvelopeSchema>;
export interface Actor {
  id: string;
  role: Role;
}
export interface Table {
  id: string;
  name: string;
  version: number;
  occupancy: Occupancy;
  aiObservedState: VisionState;
  lastVerifiedAiState: Occupancy | null;
  override: { actorId: string; at: string; protectedUntil: string; reason: string } | null;
}
export interface Session {
  id: string;
  version: number;
  source: 'AI_SIMULATION' | 'MANUAL';
  startedAt: string;
  endedAt: string | null;
  confirmedAt: string;
  originalStartedAt: string;
  originalEndedAt: string | null;
}
export interface SessionTableAssignment {
  id: string;
  sessionId: string;
  tableId: string;
  startedAt: string;
  endedAt: string | null;
}
export interface CleaningCycle {
  id: string;
  tableId: string;
  startedAt: string;
  readyAt: string | null;
  readyBy: string | null;
}
export interface Task {
  id: string;
  version: number;
  kind: TaskKind;
  sourceId: string;
  tableId: string;
  status: 'ASSIGNED' | 'ACCEPTED' | 'COMPLETED' | 'CANCELLED';
  currentAttemptId: string;
  createdAt: string;
  cancelledReason: string | null;
}
export interface AssignmentAttempt {
  id: string;
  taskId: string;
  waiterId: string;
  assignedAt: string;
  acceptanceDueAt: string;
  acceptedAt: string | null;
  completionDueAt: string | null;
  completedAt: string | null;
  completionSeconds: number;
  status: Task['status'] | 'REASSIGNED';
}
export type AlertKind =
  'NEW_CUSTOMERS' | 'CLEANING_REQUIRED' | 'ACCEPTANCE_OVERDUE' | 'COMPLETION_OVERDUE';
export interface Alert {
  id: string;
  kind: AlertKind;
  dedupKey: string;
  tableId: string;
  sourceId: string;
  createdAt: string;
  status: 'PENDING' | 'ACKNOWLEDGED' | 'OBSOLETE';
  acknowledgedBy: string | null;
  acknowledgedAt: string | null;
  obsoleteReason: string | null;
}
export interface DomainEvent {
  id: string;
  sequence: number;
  type: string;
  at: string;
  data: Record<string, unknown>;
}
export interface AuditEntry {
  id: string;
  actorId: string;
  action: string;
  at: string;
  context: Record<string, unknown>;
}
export interface Adjustment {
  id: string;
  sessionId: string;
  field: 'startedAt' | 'endedAt';
  previous: string;
  next: string;
  actorId: string;
  at: string;
  reason: string;
}
export interface MonitoringGap {
  id: string;
  startedAt: string;
  endedAt: string;
  reason: 'PROCESS_RESTART';
}
export interface RestaurantState {
  id: string;
  name: string;
  version: number;
  policy: TimeoutPolicy | null;
  tables: Table[];
  sessions: Session[];
  assignments: SessionTableAssignment[];
  cleaningCycles: CleaningCycle[];
  tasks: Task[];
  attempts: AssignmentAttempt[];
  alerts: Alert[];
  events: DomainEvent[];
  audit: AuditEntry[];
  adjustments: Adjustment[];
  monitoringGaps: MonitoringGap[];
}
export interface Member {
  id: string;
  name: string;
  role: Role;
}
export interface Dashboard extends Omit<RestaurantState, 'events'> {
  members: Member[];
  serverTime: string;
  simulatorEnabled: boolean;
  sync: 'LOCAL_ONLY';
}
