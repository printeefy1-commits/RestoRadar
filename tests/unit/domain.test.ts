import { randomUUID } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { commandSchema } from '@restoradar/contracts';
import type { Actor, Command, RestaurantState, TimeoutPolicy } from '@restoradar/contracts';
import {
  executeCommand,
  initialState,
  markMonitoringGap,
  processDeadlines,
} from '@restoradar/domain';

const owner: Actor = { id: randomUUID(), role: 'OWNER' };
const manager: Actor = { id: randomUUID(), role: 'MANAGER' };
const waiter: Actor = { id: randomUUID(), role: 'WAITER' };
const waiter2: Actor = { id: randomUUID(), role: 'WAITER' };
const start = Date.parse('2026-10-22T10:00:00.000Z');
const at = (seconds: number) => new Date(start + seconds * 1000).toISOString();
// Explicit test fixtures, not application defaults.
const policy: TimeoutPolicy = {
  CUSTOMER_RECEPTION: { acceptanceSeconds: 10, completionSeconds: 20 },
  TABLE_CLEANING: { acceptanceSeconds: 15, completionSeconds: 30 },
};
function fixture(configure = true) {
  let state = initialState(randomUUID(), 'Test restaurant');
  function run(command: Command, seconds = 0, actor = owner) {
    state = executeCommand(state, command, {
      actor,
      now: at(seconds),
      waiterIds: [waiter.id, waiter2.id],
      simulatorEnabled: true,
    });
    return state;
  }
  run({ type: 'table.create', name: 'T1' });
  if (configure) run({ type: 'policy.configure', policy });
  const tableId = state.tables[0].id;
  function transition(
    occupancy: 'EMPTY' | 'OCCUPIED',
    observed: number,
    confirmed = observed + (occupancy === 'OCCUPIED' ? 5 : 30),
  ) {
    return run(
      {
        type: 'simulation.transition',
        tableId,
        occupancy,
        observedAt: at(observed),
        confirmedAt: at(confirmed),
      },
      confirmed,
    );
  }
  function assign(
    kind: 'CUSTOMER_RECEPTION' | 'TABLE_CLEANING' = 'CUSTOMER_RECEPTION',
    seconds = 6,
    id = waiter.id,
  ) {
    const sourceId =
      kind === 'CUSTOMER_RECEPTION' ? state.sessions.at(-1)!.id : state.cleaningCycles.at(-1)!.id;
    const task = state.tasks.find((t) => t.sourceId === sourceId && t.kind === kind);
    return run(
      { type: 'task.assign', kind, sourceId, waiterId: id, expectedVersion: task?.version ?? 0 },
      seconds,
      manager,
    );
  }
  return {
    get s() {
      return state;
    },
    run,
    transition,
    assign,
    tableId,
  };
}

describe('sessions and independent readiness', () => {
  it('starts one session/arrival from the initial observation; camera reconfirmation preserves time', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.transition('OCCUPIED', 2, 7);
    expect(f.s.sessions).toHaveLength(1);
    expect(f.s.assignments).toHaveLength(1);
    expect(f.s.sessions[0].startedAt).toBe(at(0));
    expect(f.s.alerts.filter((a) => a.kind === 'NEW_CUSTOMERS')).toHaveLength(1);
  });
  it('refuses unverified short seating or departure intervals and future evidence', () => {
    const f = fixture();
    expect(() =>
      f.run(
        {
          type: 'simulation.transition',
          tableId: f.tableId,
          occupancy: 'OCCUPIED',
          observedAt: at(0),
          confirmedAt: at(4),
        },
        4,
      ),
    ).toThrow();
    expect(() =>
      f.run(
        {
          type: 'simulation.transition',
          tableId: f.tableId,
          occupancy: 'EMPTY',
          observedAt: at(0),
          confirmedAt: at(29),
        },
        29,
      ),
    ).toThrow();
    expect(() =>
      f.run(
        {
          type: 'simulation.transition',
          tableId: f.tableId,
          occupancy: 'OCCUPIED',
          observedAt: at(0),
          confirmedAt: at(5),
        },
        4,
      ),
    ).toThrow();
  });
  it('closes at observed departure but starts cleaning wait at confirmation', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.transition('EMPTY', 60);
    expect(f.s.sessions[0].endedAt).toBe(at(60));
    expect(f.s.cleaningCycles[0].startedAt).toBe(at(90));
    expect(f.s.alerts.filter((a) => a.kind === 'CLEANING_REQUIRED')).toHaveLength(1);
    expect(f.s.alerts.find((a) => a.kind === 'NEW_CUSTOMERS')?.status).toBe('OBSOLETE');
  });
  it('does not treat camera loss as vacancy or reset a session', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.run({ type: 'simulation.vision', tableId: f.tableId, state: 'VISION_UNAVAILABLE' }, 10);
    expect(f.s.tables[0].occupancy).toBe('OCCUPIED');
    expect(f.s.sessions[0].endedAt).toBeNull();
    expect(f.s.tables[0].aiObservedState).toBe('VISION_UNAVAILABLE');
  });
  it('acknowledging cleaning does not confirm readiness or finish a task', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.transition('EMPTY', 60);
    f.assign('TABLE_CLEANING', 91);
    const a = f.s.alerts.find((a) => a.kind === 'CLEANING_REQUIRED')!;
    f.run({ type: 'alert.acknowledge', alertId: a.id }, 92, manager);
    expect(f.s.cleaningCycles[0].readyAt).toBeNull();
    expect(f.s.tasks[0].status).toBe('ASSIGNED');
  });
  it('waiter completion leaves readiness for manager confirmation', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.transition('EMPTY', 60);
    f.assign('TABLE_CLEANING', 91);
    f.run({ type: 'task.accept', taskId: f.s.tasks[0].id, expectedVersion: 1 }, 92, waiter);
    f.run({ type: 'task.complete', taskId: f.s.tasks[0].id, expectedVersion: 2 }, 93, waiter);
    expect(f.s.cleaningCycles[0].readyAt).toBeNull();
    expect(() =>
      f.run(
        { type: 'table.ready', tableId: f.tableId, expectedVersion: f.s.tables[0].version },
        94,
        waiter,
      ),
    ).toThrow();
    f.run(
      { type: 'table.ready', tableId: f.tableId, expectedVersion: f.s.tables[0].version },
      94,
      manager,
    );
    expect(f.s.cleaningCycles[0].readyAt).toBe(at(94));
  });
  it('cancels reception on departure, cleaning on reseating, retaining a single unresolved cleaning cycle', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.assign();
    f.transition('EMPTY', 60);
    expect(f.s.tasks[0].cancelledReason).toBe('SESSION_ENDED');
    f.assign('TABLE_CLEANING', 91);
    f.transition('OCCUPIED', 100);
    expect(f.s.tasks[1].cancelledReason).toBe('NEW_GUESTS_SEATED');
    expect(f.s.cleaningCycles[0].readyAt).toBeNull();
    f.transition('EMPTY', 160);
    expect(f.s.cleaningCycles).toHaveLength(1);
    expect(f.s.alerts.filter((a) => a.kind === 'CLEANING_REQUIRED')).toHaveLength(1);
    f.assign('TABLE_CLEANING', 191);
    expect(f.s.tasks).toHaveLength(2);
    expect(f.s.tasks[1].status).toBe('ASSIGNED');
    expect(f.s.attempts).toHaveLength(3);
  });
  it('blocks ready confirmation on occupied tables', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    expect(() =>
      f.run(
        { type: 'table.ready', tableId: f.tableId, expectedVersion: f.s.tables[0].version },
        10,
      ),
    ).toThrow();
  });
});

describe('manual correction and session history', () => {
  it('manual seating requires a reason, starts at correction time, and creates no arrival alert', () => {
    const f = fixture();
    expect(() =>
      f.run({
        type: 'table.correct',
        tableId: f.tableId,
        expectedVersion: 0,
        occupancy: 'OCCUPIED',
        reason: '',
      }),
    ).toThrow();
    f.run(
      {
        type: 'table.correct',
        tableId: f.tableId,
        expectedVersion: 0,
        occupancy: 'OCCUPIED',
        reason: 'Confirmed by manager',
      },
      10,
      manager,
    );
    expect(f.s.sessions[0].startedAt).toBe(at(10));
    expect(f.s.alerts).toHaveLength(0);
    expect(f.s.audit.at(-1)?.context.reason).toBe('Confirmed by manager');
    expect(f.s.audit.at(-1)?.context.previousState).toBe('EMPTY');
    expect(f.s.audit.at(-1)?.context.newState).toBe('OCCUPIED');
  });
  it('protects manual state for 60 seconds and requires a NEW transition afterward', () => {
    const f = fixture();
    f.transition('EMPTY', 0);
    f.run(
      {
        type: 'table.correct',
        tableId: f.tableId,
        expectedVersion: f.s.tables[0].version,
        occupancy: 'OCCUPIED',
        reason: 'Guests hidden',
      },
      40,
      manager,
    );
    f.transition('EMPTY', 50, 80);
    expect(f.s.tables[0].occupancy).toBe('OCCUPIED');
    f.transition('EMPTY', 110, 140);
    expect(f.s.tables[0].occupancy).toBe('OCCUPIED'); // Persistent empty evidence is not a new transition.
    f.transition('OCCUPIED', 145);
    f.transition('EMPTY', 160);
    expect(f.s.tables[0].occupancy).toBe('EMPTY');
    expect(f.s.tables[0].override).toBeNull();
  });
  it('does not cancel an override on disconnect/restart or mistake regained visibility for a transition', () => {
    const f = fixture();
    f.transition('EMPTY', 0);
    f.run(
      {
        type: 'table.correct',
        tableId: f.tableId,
        expectedVersion: f.s.tables[0].version,
        occupancy: 'OCCUPIED',
        reason: 'Visual occlusion',
      },
      40,
      manager,
    );
    const recovered = markMonitoringGap(f.s, at(50), at(120));
    expect(recovered.tables[0].override).not.toBeNull();
    expect(recovered.sessions[0].startedAt).toBe(at(40));
    expect(recovered.tables[0].aiObservedState).toBe('UNKNOWN');
    expect(recovered.monitoringGaps).toHaveLength(1);
    const next = executeCommand(
      recovered,
      {
        type: 'simulation.transition',
        tableId: f.tableId,
        occupancy: 'EMPTY',
        observedAt: at(130),
        confirmedAt: at(160),
      },
      { actor: owner, now: at(160), waiterIds: [], simulatorEnabled: true },
    );
    expect(next.tables[0].occupancy).toBe('OCCUPIED');
  });
  it('rejects stale concurrent manual edits', () => {
    const f = fixture();
    f.run({
      type: 'table.correct',
      tableId: f.tableId,
      expectedVersion: 0,
      occupancy: 'OCCUPIED',
      reason: 'Observed guests',
    });
    expect(() =>
      f.run({
        type: 'table.correct',
        tableId: f.tableId,
        expectedVersion: 0,
        occupancy: 'EMPTY',
        reason: 'Stale device',
      }),
    ).toThrow();
  });
  it('preserves originals during backward adjustments and rejects invalid time changes', () => {
    const f = fixture();
    f.transition('OCCUPIED', 10);
    f.transition('EMPTY', 60);
    const id = f.s.sessions[0].id;
    f.run(
      {
        type: 'session.adjust',
        sessionId: id,
        expectedVersion: 1,
        field: 'startedAt',
        at: at(5),
        reason: 'Correct arrival',
      },
      100,
      manager,
    );
    expect(f.s.sessions[0].originalStartedAt).toBe(at(10));
    expect(f.s.assignments[0].startedAt).toBe(at(5));
    expect(f.s.adjustments).toHaveLength(1);
    expect(() =>
      f.run(
        {
          type: 'session.adjust',
          sessionId: id,
          expectedVersion: 2,
          field: 'endedAt',
          at: at(4),
          reason: 'Invalid end',
        },
        100,
      ),
    ).toThrow();
    expect(() =>
      f.run(
        {
          type: 'session.adjust',
          sessionId: id,
          expectedVersion: 2,
          field: 'startedAt',
          at: at(110),
          reason: 'Future start',
        },
        100,
      ),
    ).toThrow();
  });
  it('rejects a backward correction that overlaps an earlier session', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.transition('EMPTY', 60);
    f.transition('OCCUPIED', 100);
    expect(() =>
      f.run(
        {
          type: 'session.adjust',
          sessionId: f.s.sessions[1].id,
          expectedVersion: 0,
          field: 'startedAt',
          at: at(50),
          reason: 'Overlapping period',
        },
        110,
      ),
    ).toThrow();
  });
});

describe('tasks, deadlines and authorization', () => {
  it('has no timeout defaults and blocks assignments until owner configuration', () => {
    const f = fixture(false);
    f.transition('OCCUPIED', 0);
    expect(f.s.policy).toBeNull();
    expect(() => f.assign()).toThrow();
    expect(() => f.run({ type: 'policy.configure', policy }, 10, manager)).toThrow();
  });
  it('keeps policy snapshots and starts completion deadline at acceptance', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.assign();
    f.run(
      {
        type: 'policy.configure',
        policy: {
          ...policy,
          CUSTOMER_RECEPTION: { acceptanceSeconds: 500, completionSeconds: 600 },
        },
      },
      7,
    );
    f.run({ type: 'task.accept', taskId: f.s.tasks[0].id, expectedVersion: 1 }, 8, waiter);
    expect(f.s.attempts[0].acceptanceDueAt).toBe(at(16));
    expect(f.s.attempts[0].completionDueAt).toBe(at(28));
    expect(f.s.sessions[0].startedAt).toBe(at(0));
  });
  it('generates one timeout per attempt, without recurring reminders', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.assign();
    let s = processDeadlines(f.s, at(20));
    s = processDeadlines(s, at(25));
    expect(s.alerts.filter((a) => a.kind === 'ACCEPTANCE_OVERDUE')).toHaveLength(1);
    const t = s.tasks[0];
    s = executeCommand(
      s,
      {
        type: 'task.assign',
        kind: t.kind,
        sourceId: t.sourceId,
        waiterId: waiter2.id,
        expectedVersion: t.version,
      },
      { actor: manager, now: at(30), waiterIds: [waiter2.id], simulatorEnabled: true },
    );
    s = processDeadlines(s, at(45));
    expect(s.alerts.filter((a) => a.kind === 'ACCEPTANCE_OVERDUE')).toHaveLength(2);
    expect(s.attempts[0].status).toBe('REASSIGNED');
  });
  it('records completion lateness once and does not restart the occupancy timer', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.assign();
    f.run({ type: 'task.accept', taskId: f.s.tasks[0].id, expectedVersion: 1 }, 7, waiter);
    f.run({ type: 'task.complete', taskId: f.s.tasks[0].id, expectedVersion: 2 }, 30, waiter);
    expect(f.s.alerts.filter((a) => a.kind === 'COMPLETION_OVERDUE')).toHaveLength(1);
    expect(f.s.attempts[0].completedAt).toBe(at(30));
    expect(f.s.sessions[0].endedAt).toBeNull();
    expect(processDeadlines(f.s, at(300)).alerts).toHaveLength(f.s.alerts.length);
  });
  it('revokes former assignee authorization and rejects stale task versions', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.assign();
    f.assign('CUSTOMER_RECEPTION', 8, waiter2.id);
    const t = f.s.tasks[0];
    expect(() =>
      f.run({ type: 'task.accept', taskId: t.id, expectedVersion: t.version }, 9, waiter),
    ).toThrow();
    expect(() =>
      f.run({ type: 'task.accept', taskId: t.id, expectedVersion: 1 }, 9, waiter2),
    ).toThrow();
    f.run({ type: 'task.accept', taskId: t.id, expectedVersion: t.version }, 9, waiter2);
    expect(f.s.tasks[0].status).toBe('ACCEPTED');
  });
  it('allows several concurrent tasks for the same waiter', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    f.assign();
    f.run({ type: 'table.create', name: 'T2' }, 10);
    f.run(
      {
        type: 'simulation.transition',
        tableId: f.s.tables[1].id,
        occupancy: 'OCCUPIED',
        observedAt: at(10),
        confirmedAt: at(15),
      },
      15,
    );
    f.assign('CUSTOMER_RECEPTION', 16);
    expect(f.s.tasks).toHaveLength(2);
    expect(f.s.attempts.every((a) => a.waiterId === waiter.id)).toBe(true);
  });
  it('first acknowledgement wins without acknowledging an obsolete alert', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    const id = f.s.alerts[0].id;
    f.run({ type: 'alert.acknowledge', alertId: id }, 6, manager);
    f.run({ type: 'alert.acknowledge', alertId: id }, 7, owner);
    expect(f.s.alerts[0].acknowledgedBy).toBe(manager.id);
    expect(f.s.alerts[0].acknowledgedAt).toBe(at(6));
    f.transition('EMPTY', 60);
    f.transition('OCCUPIED', 100);
    const obsolete = f.s.alerts.find((a) => a.kind === 'CLEANING_REQUIRED')!;
    expect(() => f.run({ type: 'alert.acknowledge', alertId: obsolete.id }, 110)).toThrow();
  });
  it('prevents waiters from managing tables, viewing alerts through commands or configuring simulation', () => {
    const f = fixture();
    f.transition('OCCUPIED', 0);
    for (const command of [
      { type: 'table.create', name: 'Unauthorized' },
      { type: 'alert.acknowledge', alertId: f.s.alerts[0].id },
      {
        type: 'table.correct',
        tableId: f.tableId,
        expectedVersion: f.s.tables[0].version,
        occupancy: 'EMPTY',
        reason: 'Unauthorized',
      },
    ] as Command[])
      expect(() => f.run(command, 10, waiter)).toThrow();
    expect(() =>
      executeCommand(
        f.s,
        { type: 'simulation.vision', tableId: f.tableId, state: 'UNKNOWN' },
        { actor: owner, now: at(10), waiterIds: [], simulatorEnabled: false },
      ),
    ).toThrow();
  });
  it('validates policy fields and normalizes timestamps at the API boundary', () => {
    expect(commandSchema.safeParse({ type: 'policy.configure', policy: {} }).success).toBe(false);
    expect(
      commandSchema.safeParse({
        type: 'policy.configure',
        policy: { ...policy, TABLE_CLEANING: { acceptanceSeconds: 0, completionSeconds: 30 } },
      }).success,
    ).toBe(false);
    const cmd = commandSchema.parse({
      type: 'simulation.transition',
      tableId: randomUUID(),
      occupancy: 'EMPTY',
      observedAt: '2026-10-22T10:00:00Z',
      confirmedAt: '2026-10-22T10:00:30Z',
    });
    expect('observedAt' in cmd && cmd.observedAt).toBe(at(0));
  });
});
