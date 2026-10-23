import { randomUUID } from 'node:crypto';
import type {
  Actor,
  Alert,
  AlertKind,
  Command,
  Member,
  Occupancy,
  RestaurantState,
  Table,
  Task,
  TaskKind,
} from '@restoradar/contracts';
export type { RestaurantState } from '@restoradar/contracts';

export class DomainError extends Error {
  constructor(
    public code: string,
    message: string,
    public status = 409,
  ) {
    super(message);
  }
}
function ensure(
  condition: unknown,
  code: string,
  message: string,
  status = 409,
): asserts condition {
  if (!condition) throw new DomainError(code, message, status);
}
function find<T extends { id: string }>(rows: T[], id: string): T {
  const row = rows.find((r) => r.id === id);
  ensure(row, 'NOT_FOUND', 'العنصر غير موجود', 404);
  return row;
}
function checkVersion(actual: number, expected: number) {
  ensure(actual === expected, 'VERSION_CONFLICT', 'تغيرت البيانات. حدّث الصفحة وأعد المحاولة.');
}
const addSeconds = (at: string, seconds: number) =>
  new Date(Date.parse(at) + seconds * 1000).toISOString();
export function initialState(id: string, name: string): RestaurantState {
  return {
    id,
    name,
    version: 0,
    policy: null,
    tables: [],
    sessions: [],
    assignments: [],
    cleaningCycles: [],
    tasks: [],
    attempts: [],
    alerts: [],
    events: [],
    audit: [],
    adjustments: [],
    monitoringGaps: [],
  };
}
function emit(s: RestaurantState, type: string, at: string, data: Record<string, unknown>) {
  s.events.push({ id: randomUUID(), sequence: s.events.length + 1, type, at, data });
}
function audit(
  s: RestaurantState,
  actorId: string,
  action: string,
  at: string,
  context: Record<string, unknown>,
) {
  s.audit.push({ id: randomUUID(), actorId, action, at, context });
}
function alert(s: RestaurantState, kind: AlertKind, sourceId: string, tableId: string, at: string) {
  const dedupKey = `${kind}:${sourceId}`;
  if (s.alerts.some((a) => a.dedupKey === dedupKey)) return;
  const a: Alert = {
    id: randomUUID(),
    kind,
    sourceId,
    tableId,
    dedupKey,
    createdAt: at,
    status: 'PENDING',
    acknowledgedBy: null,
    acknowledgedAt: null,
    obsoleteReason: null,
  };
  s.alerts.push(a);
  emit(s, 'alert.created', at, { alertId: a.id, kind, dedupKey });
}
function obsolete(s: RestaurantState, sourceIds: string[], reason: string) {
  for (const a of s.alerts)
    if (sourceIds.includes(a.sourceId) && a.status === 'PENDING') {
      a.status = 'OBSOLETE';
      a.obsoleteReason = reason;
    }
}
function cancelTasks(
  s: RestaurantState,
  kind: TaskKind,
  sourceId: string,
  reason: string,
  at: string,
) {
  for (const t of s.tasks.filter(
    (t) =>
      t.kind === kind && t.sourceId === sourceId && ['ASSIGNED', 'ACCEPTED'].includes(t.status),
  )) {
    t.status = 'CANCELLED';
    t.cancelledReason = reason;
    t.version++;
    find(s.attempts, t.currentAttemptId).status = 'CANCELLED';
    obsolete(
      s,
      s.attempts.filter((a) => a.taskId === t.id).map((a) => a.id),
      reason,
    );
    emit(s, 'task.cancelled', at, { taskId: t.id, reason });
  }
}

export function removeMember(
  previous: RestaurantState,
  member: Member,
  actor: Actor,
  at: string,
): RestaurantState {
  ensure(actor.role === 'OWNER', 'FORBIDDEN', 'إدارة الحسابات للمالك فقط', 403);
  ensure(actor.id !== member.id, 'CURRENT_OWNER', 'لا يمكن حذف حساب المالك الحالي');
  const s = structuredClone(previous);
  const cancelledTaskIds: string[] = [];
  for (const task of s.tasks) {
    const attempt = find(s.attempts, task.currentAttemptId);
    if (attempt.waiterId !== member.id || !['ASSIGNED', 'ACCEPTED'].includes(task.status)) continue;
    cancelTasks(s, task.kind, task.sourceId, 'USER_DELETED', at);
    cancelledTaskIds.push(task.id);
  }
  audit(s, actor.id, 'user.deleted', at, {
    userId: member.id,
    name: member.name,
    role: member.role,
    cancelledTaskIds,
  });
  s.version++;
  return s;
}
function changeOccupancy(
  s: RestaurantState,
  table: Table,
  occupancy: Occupancy,
  observedAt: string,
  confirmedAt: string,
  source: 'MANUAL' | 'AI_SIMULATION',
) {
  if (table.occupancy === occupancy) return;
  if (occupancy === 'OCCUPIED') {
    // A backdated observation must never introduce overlapping table intervals.
    ensure(
      !s.assignments.some(
        (a) => a.tableId === table.id && (a.endedAt === null || a.endedAt > observedAt),
      ),
      'SESSION_OVERLAP',
      'وقت الجلوس يتداخل مع جلسة سابقة',
    );
    const session = {
      id: randomUUID(),
      version: 0,
      source,
      startedAt: observedAt,
      endedAt: null,
      originalStartedAt: observedAt,
      originalEndedAt: null,
      confirmedAt,
    };
    s.sessions.push(session);
    s.assignments.push({
      id: randomUUID(),
      sessionId: session.id,
      tableId: table.id,
      startedAt: observedAt,
      endedAt: null,
    });
    for (const cycle of s.cleaningCycles.filter((c) => c.tableId === table.id && !c.readyAt)) {
      cancelTasks(s, 'TABLE_CLEANING', cycle.id, 'NEW_GUESTS_SEATED', confirmedAt);
      obsolete(s, [cycle.id], 'NEW_GUESTS_SEATED');
    }
    emit(s, 'session.started', confirmedAt, {
      sessionId: session.id,
      tableId: table.id,
      observedAt,
      source,
    });
    if (source === 'AI_SIMULATION') alert(s, 'NEW_CUSTOMERS', session.id, table.id, confirmedAt);
  } else {
    const assignment = s.assignments.find((a) => a.tableId === table.id && a.endedAt === null);
    ensure(assignment, 'MISSING_SESSION', 'لا توجد جلسة نشطة');
    const session = find(s.sessions, assignment.sessionId);
    ensure(observedAt >= session.startedAt, 'INVALID_SESSION_TIME', 'نهاية الجلسة تسبق بدايتها');
    session.endedAt = observedAt;
    session.originalEndedAt = observedAt;
    session.version++;
    assignment.endedAt = observedAt;
    cancelTasks(s, 'CUSTOMER_RECEPTION', session.id, 'SESSION_ENDED', confirmedAt);
    obsolete(s, [session.id], 'SESSION_ENDED');
    // An unresolved cycle survives reseating; it is not duplicated on the next departure.
    if (!s.cleaningCycles.some((c) => c.tableId === table.id && !c.readyAt)) {
      const cycle = {
        id: randomUUID(),
        tableId: table.id,
        startedAt: confirmedAt,
        readyAt: null,
        readyBy: null,
      };
      s.cleaningCycles.push(cycle);
      alert(s, 'CLEANING_REQUIRED', cycle.id, table.id, confirmedAt);
      emit(s, 'cleaning.required', confirmedAt, { cycleId: cycle.id, tableId: table.id });
    }
    emit(s, 'session.ended', confirmedAt, { sessionId: session.id, tableId: table.id, observedAt });
  }
  table.occupancy = occupancy;
}

export interface CommandContext {
  actor: Actor;
  now: string;
  waiterIds: string[];
  simulatorEnabled: boolean;
}
export function executeCommand(
  previous: RestaurantState,
  command: Command,
  ctx: CommandContext,
): RestaurantState {
  const { actor, now } = ctx;
  const ownerOnly = [
    'policy.configure',
    'table.create',
    'simulation.transition',
    'simulation.vision',
  ];
  if (ownerOnly.includes(command.type))
    ensure(actor.role === 'OWNER', 'FORBIDDEN', 'هذه العملية للمالك فقط', 403);
  else if (!['task.accept', 'task.complete'].includes(command.type))
    ensure(actor.role !== 'WAITER', 'FORBIDDEN', 'هذه العملية للمدير أو المالك', 403);
  const s = structuredClone(previous);
  let auditContext: Record<string, unknown> = { ...command };
  switch (command.type) {
    case 'policy.configure':
      s.policy = command.policy;
      break;
    case 'table.create':
      ensure(
        !s.tables.some((t) => t.name.toLocaleLowerCase() === command.name.toLocaleLowerCase()),
        'DUPLICATE_TABLE',
        'اسم الطاولة مستخدم',
      );
      s.tables.push({
        id: randomUUID(),
        name: command.name,
        version: 0,
        occupancy: 'EMPTY',
        aiObservedState: 'UNKNOWN',
        lastVerifiedAiState: null,
        override: null,
      });
      break;
    case 'table.correct': {
      const t = find(s.tables, command.tableId);
      checkVersion(t.version, command.expectedVersion);
      ensure(command.reason.trim().length >= 3, 'REASON_REQUIRED', 'سبب التصحيح مطلوب', 400);
      const before = t.occupancy;
      auditContext = {
        ...command,
        previousState: before,
        newState: command.occupancy,
        aiObservedState: t.aiObservedState,
      };
      changeOccupancy(s, t, command.occupancy, now, now, 'MANUAL');
      t.override = {
        actorId: actor.id,
        at: now,
        protectedUntil: addSeconds(now, 60),
        reason: command.reason,
      };
      t.version++;
      emit(s, 'table.corrected', now, {
        tableId: t.id,
        before,
        after: t.occupancy,
        aiObservedState: t.aiObservedState,
        reason: command.reason,
      });
      break;
    }
    case 'simulation.transition': {
      ensure(ctx.simulatorEnabled, 'SIMULATOR_DISABLED', 'محاكي الأحداث غير مفعّل', 403);
      const t = find(s.tables, command.tableId);
      const observed = Date.parse(command.observedAt),
        confirmed = Date.parse(command.confirmedAt);
      ensure(
        Number.isFinite(observed) &&
          Number.isFinite(confirmed) &&
          confirmed <= Date.parse(now) &&
          confirmed - observed >= (command.occupancy === 'OCCUPIED' ? 5000 : 30000),
        'INVALID_EVIDENCE',
        'فترة التحقق غير كافية أو الوقت في المستقبل',
        400,
      );
      // Out-of-order evidence cannot revise newer decisions or bypass an override.
      const latest = s.events
        .filter((e) => e.type === 'simulation.verified' && e.data.tableId === t.id)
        .at(-1);
      ensure(
        !latest || command.confirmedAt >= latest.at,
        'STALE_OBSERVATION',
        'الملاحظة أقدم من آخر قرار',
      );
      const newTransition =
        t.lastVerifiedAiState !== null && t.lastVerifiedAiState !== command.occupancy;
      t.aiObservedState = command.occupancy;
      t.lastVerifiedAiState = command.occupancy;
      if (!t.override || (newTransition && command.observedAt >= t.override.protectedUntil)) {
        t.override = null;
        changeOccupancy(
          s,
          t,
          command.occupancy,
          command.observedAt,
          command.confirmedAt,
          'AI_SIMULATION',
        );
      }
      t.version++;
      emit(s, 'simulation.verified', command.confirmedAt, {
        tableId: t.id,
        occupancy: command.occupancy,
        observedAt: command.observedAt,
      });
      break;
    }
    case 'simulation.vision': {
      ensure(ctx.simulatorEnabled, 'SIMULATOR_DISABLED', 'محاكي الأحداث غير مفعّل', 403);
      const t = find(s.tables, command.tableId);
      t.aiObservedState = command.state;
      t.version++;
      break;
    }
    case 'table.ready': {
      const t = find(s.tables, command.tableId);
      checkVersion(t.version, command.expectedVersion);
      ensure(t.occupancy === 'EMPTY', 'TABLE_OCCUPIED', 'لا يمكن تأكيد جاهزية طاولة مشغولة');
      const cycle = s.cleaningCycles.find((c) => c.tableId === t.id && !c.readyAt);
      ensure(cycle, 'ALREADY_READY', 'الطاولة جاهزة بالفعل');
      cycle.readyAt = now;
      cycle.readyBy = actor.id;
      // Manager readiness supersedes outstanding cleaning tasks without pretending completion.
      cancelTasks(s, 'TABLE_CLEANING', cycle.id, 'MANAGER_CONFIRMED_READY', now);
      obsolete(s, [cycle.id], 'MANAGER_CONFIRMED_READY');
      t.version++;
      emit(s, 'table.ready', now, { tableId: t.id, cycleId: cycle.id });
      break;
    }
    case 'session.adjust': {
      ensure(command.reason.trim().length >= 3, 'REASON_REQUIRED', 'سبب تعديل الوقت مطلوب', 400);
      const session = find(s.sessions, command.sessionId);
      checkVersion(session.version, command.expectedVersion);
      const previousTime = session[command.field];
      ensure(
        previousTime && command.at < previousTime && command.at <= now,
        'INVALID_ADJUSTMENT',
        'يسمح بتعديل الوقت إلى الخلف فقط',
      );
      const startedAt = command.field === 'startedAt' ? command.at : session.startedAt;
      const endedAt = command.field === 'endedAt' ? command.at : session.endedAt;
      ensure(!endedAt || endedAt >= startedAt, 'INVALID_SESSION_TIME', 'نهاية الجلسة تسبق بدايتها');
      const assignment = s.assignments.find((a) => a.sessionId === session.id);
      ensure(assignment, 'MISSING_ASSIGNMENT', 'ربط الجلسة غير موجود');
      for (const other of s.assignments.filter(
        (a) => a.tableId === assignment.tableId && a.sessionId !== session.id,
      )) {
        ensure(
          (endedAt !== null && endedAt <= other.startedAt) ||
            (other.endedAt !== null && other.endedAt <= startedAt),
          'SESSION_OVERLAP',
          'التعديل يتداخل مع جلسة أخرى',
        );
      }
      session[command.field] = command.at;
      assignment[command.field] = command.at;
      session.version++;
      s.adjustments.push({
        id: randomUUID(),
        sessionId: session.id,
        field: command.field,
        previous: previousTime,
        next: command.at,
        actorId: actor.id,
        at: now,
        reason: command.reason,
      });
      break;
    }
    case 'alert.acknowledge': {
      const a = find(s.alerts, command.alertId);
      ensure(a.status !== 'OBSOLETE', 'ALERT_OBSOLETE', 'هذا التنبيه لم يعد قابلًا للإجراء');
      if (a.status === 'ACKNOWLEDGED') return previous;
      a.status = 'ACKNOWLEDGED';
      a.acknowledgedBy = actor.id;
      a.acknowledgedAt = now;
      break;
    }
    case 'task.assign': {
      ensure(s.policy, 'POLICY_REQUIRED', 'يجب أن يُدخل المالك مهلات المهام أولًا');
      ensure(
        ctx.waiterIds.includes(command.waiterId),
        'INVALID_WAITER',
        'النادل غير موجود في هذا المطعم',
        400,
      );
      let tableId: string;
      if (command.kind === 'CUSTOMER_RECEPTION') {
        const session = find(s.sessions, command.sourceId);
        ensure(session.endedAt === null, 'SOURCE_OBSOLETE', 'انتهت جلسة الاستقبال');
        tableId = s.assignments.find((a) => a.sessionId === session.id)!.tableId;
      } else {
        const cycle = find(s.cleaningCycles, command.sourceId);
        ensure(
          !cycle.readyAt && find(s.tables, cycle.tableId).occupancy === 'EMPTY',
          'SOURCE_OBSOLETE',
          'التنظيف غير قابل للإسناد الآن',
        );
        tableId = cycle.tableId;
      }
      let task = s.tasks.find((t) => t.kind === command.kind && t.sourceId === command.sourceId);
      checkVersion(task?.version ?? 0, command.expectedVersion);
      if (task) {
        ensure(task.status !== 'COMPLETED', 'TASK_COMPLETED', 'المهمة مكتملة');
        const oldAttempt = find(s.attempts, task.currentAttemptId);
        if (['ASSIGNED', 'ACCEPTED'].includes(oldAttempt.status)) oldAttempt.status = 'REASSIGNED';
        obsolete(s, [oldAttempt.id], 'TASK_REASSIGNED');
        task.version++;
        task.status = 'ASSIGNED';
        task.cancelledReason = null;
      } else {
        task = {
          id: randomUUID(),
          version: 1,
          kind: command.kind,
          sourceId: command.sourceId,
          tableId,
          status: 'ASSIGNED',
          currentAttemptId: '',
          createdAt: now,
          cancelledReason: null,
        };
        s.tasks.push(task);
      }
      const policy = s.policy[command.kind];
      const attempt = {
        id: randomUUID(),
        taskId: task.id,
        waiterId: command.waiterId,
        assignedAt: now,
        acceptanceDueAt: addSeconds(now, policy.acceptanceSeconds),
        completionSeconds: policy.completionSeconds,
        acceptedAt: null,
        completionDueAt: null,
        completedAt: null,
        status: 'ASSIGNED' as const,
      };
      s.attempts.push(attempt);
      task.currentAttemptId = attempt.id;
      emit(s, 'task.assigned', now, {
        taskId: task.id,
        attemptId: attempt.id,
        waiterId: command.waiterId,
      });
      break;
    }
    case 'task.accept':
    case 'task.complete': {
      const t = find(s.tasks, command.taskId),
        a = find(s.attempts, t.currentAttemptId);
      ensure(
        actor.role === 'WAITER' && actor.id === a.waiterId,
        'FORBIDDEN',
        'هذه المهمة ليست مسندة إليك',
        403,
      );
      checkVersion(t.version, command.expectedVersion);
      if (command.type === 'task.accept') {
        ensure(t.status === 'ASSIGNED', 'INVALID_TASK_STATE', 'المهمة ليست بانتظار القبول');
        t.status = 'ACCEPTED';
        a.status = 'ACCEPTED';
        a.acceptedAt = now;
        a.completionDueAt = addSeconds(now, a.completionSeconds);
        // Preserve a single late condition even when the periodic worker has not run yet.
        if (now >= a.acceptanceDueAt)
          alert(s, 'ACCEPTANCE_OVERDUE', a.id, t.tableId, a.acceptanceDueAt);
        obsolete(
          s,
          s.alerts
            .filter((x) => x.kind === 'ACCEPTANCE_OVERDUE' && x.sourceId === a.id)
            .map((x) => x.sourceId),
          'TASK_ACCEPTED',
        );
      } else {
        ensure(t.status === 'ACCEPTED', 'INVALID_TASK_STATE', 'اقبل المهمة قبل إكمالها');
        if (a.completionDueAt && now >= a.completionDueAt)
          alert(s, 'COMPLETION_OVERDUE', a.id, t.tableId, a.completionDueAt);
        t.status = 'COMPLETED';
        a.status = 'COMPLETED';
        a.completedAt = now;
        obsolete(s, [a.id], 'TASK_COMPLETED');
      }
      t.version++;
      emit(s, command.type, now, { taskId: t.id, attemptId: a.id });
      break;
    }
  }
  s.version++;
  audit(s, actor.id, command.type, now, auditContext);
  emit(s, command.type, now, { actorId: actor.id });
  return s;
}

export function processDeadlines(previous: RestaurantState, now: string): RestaurantState {
  const s = structuredClone(previous);
  for (const task of s.tasks) {
    const a = find(s.attempts, task.currentAttemptId);
    if (task.status === 'ASSIGNED' && now >= a.acceptanceDueAt)
      alert(s, 'ACCEPTANCE_OVERDUE', a.id, task.tableId, a.acceptanceDueAt);
    if (task.status === 'ACCEPTED' && a.completionDueAt && now >= a.completionDueAt)
      alert(s, 'COMPLETION_OVERDUE', a.id, task.tableId, a.completionDueAt);
  }
  if (s.events.length === previous.events.length) return previous;
  s.version++;
  return s;
}

export function markMonitoringGap(
  previous: RestaurantState,
  startedAt: string,
  endedAt: string,
): RestaurantState {
  const s = structuredClone(previous);
  s.monitoringGaps.push({ id: randomUUID(), startedAt, endedAt, reason: 'PROCESS_RESTART' });
  for (const t of s.tables) {
    t.aiObservedState = 'UNKNOWN';
    t.version++;
  }
  s.version++;
  emit(s, 'monitoring.gap', endedAt, { startedAt, endedAt });
  return s;
}
