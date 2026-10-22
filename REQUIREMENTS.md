# Restaurant AI — Product Requirements Document

**Document:** `REQUIREMENTS.md`  
**Status:** Product-design baseline for Codex review; implementation not yet authorized  
**Version:** 0.1  
**Date:** 2026-10-22  
**Language:** English (project communication with owner in Arabic)

## 1. Product vision

Restaurant AI is a hybrid, camera-assisted restaurant table monitoring system. It detects when guests sit at a table, starts a table-session timer, alerts managers to send a waiter, detects when all guests leave, and alerts managers to arrange cleaning. A manager confirms that the table is ready for the next guests. Owner, Manager, and Waiter have separate accounts and permissions.

The first release focuses on **table occupancy, timing, notifications, cleaning readiness, and manually dispatched waiter tasks**. It is **not** a point-of-sale, ordering, payments, facial recognition, or employee surveillance application.

## 2. Approved scope and design decisions

### 2.1 Camera and deployment

- Support compatible existing DVR/NVR systems and new IP cameras, potentially multiple video streams simultaneously.
- Investigate RTSP and ONVIF compatibility. Do not assume every recorder supplies an accessible video stream.
- Support deployment of the local inference service on either a general-purpose PC or a dedicated AI Box, subject to benchmarking.
- **Hybrid architecture:** analyze raw video locally; synchronize table/event/task state and notifications to cloud services for authorized remote access. Do not upload continuous raw video by default.
- Operate local detection and timers while internet is unavailable. Reconcile queued events when connectivity returns; show stale/unknown remote status appropriately.
- Several cameras may observe one table. Use a single globally unique table identity, one authoritative table state, and one session/arrival event per new seating, not one per camera.

### 2.2 Camera setup and seating regions

- AI-assisted discovery of tables and seating locations **with manual correction**.
- Support individual chairs, booth/sofa seating, and custom seating regions.
- Manager/Owner review of discovered table geometry, seat zones, and camera mapping before activation.
- Each table may have a primary camera and optional secondary camera(s).
- Do not treat loss of camera visibility as evidence of an empty table.
- Prefer incremental activation of *validated* table mappings as an implementation suggestion; the final activation policy has **not yet been approved**.

### 2.3 AI seating/departure decision policy

- Seat confirmation: **5 seconds**, initially configurable only if explicitly supported by product settings.
- Departure confirmation: **30 seconds** with reliable visibility that **all** guests have left.
- High/Medium/Low confidence categories, with initial illustrative thresholds High >= 85%, Medium >= 60% and < 85%, Low < 60%. These are *provisional*, not calibrated production thresholds.
- High-confidence, temporally supported seating triggers a session and a single arrival event. Medium-confidence signals require further verification; low-confidence signals never directly create a seating event.
- Fuse posture, relationship to table/seat region, temporal persistence and multi-camera evidence. Avoid counting passing people or standing waiters as seated guests.
- Count time from the initial reliably observed seating instant once the 5-second verification succeeds; for departure, record the initial reliably observed vacancy instant after the 30-second verification succeeds.
- Allow brief visual occlusions without blindly resetting a valid observation, but do not count complete loss of visibility as positive evidence.
- Track `UNKNOWN` or `VISION_UNAVAILABLE` separately from `EMPTY`/`OCCUPIED`.
- Distinguish detection confidence from the actual evaluated performance of the model. Validate in real restaurant footage and lighting conditions.

### 2.4 Sessions and timers

- Each table has at most one active occupancy session in MVP v1.
- The timer is derived from persisted authoritative start/end timestamps, not a fragile continuously incremented in-memory counter.
- Sessions can begin by confirmed AI seating or by an authorized manager's manual `Empty -> Occupied` correction.
- On manual session creation, start time defaults to correction time. An authorized manager may adjust the start time **backward**, with a mandatory reason and immutable audit entry.
- Authorized managers may also adjust the session end time **backward**, with a mandatory reason and audit entry.
- Reject future session start times, end-before-start times and overlapping per-table occupancy periods. Preserve initial observations as distinct from corrected/authoritative timestamps.
- Changing detection source or reconfirming occupancy must not reset a live timer.
- Restart/recovery must preserve time records; mark monitoring gaps as unverified, rather than claiming observation during downtime.
- MVP v2 will add **session transfer between tables** and **temporary grouping/splitting of tables**. Design the schema to allow a session-to-table assignment/history entity rather than hard-coding a permanent `session.table_id` assumption.

### 2.5 Smart manual override

- Owner and Manager may correct effective table occupancy with a **required reason**. Every correction records actor, time, previous/new state and evidence context if available.
- Separate `ai_observed_state` from `effective_table_state`.
- A newly applied manual correction is protected for **60 seconds**: AI must not immediately overwrite it.
- After 60 seconds, AI regains control only on a **new, reliably verified transition**; persistent old evidence does not override the manual correction. Otherwise retain the manual state until review.
- Camera disconnect does not cancel an active override. Conflicting edits must use concurrency/version checks.
- A manual `Empty -> Occupied` starts a session but, by the currently proposed policy, **does not automatically generate a new-arrival push alert** to managers who already know about it. Confirm this subtle rule before implementation.

### 2.6 Table readiness and cleaning

A table can be **empty but not ready**. Represent occupancy and cleaning/readiness distinctly rather than conflating `EMPTY` and `AVAILABLE`.

- On confirmed departure of **all** guests, close the occupancy session and generate **one cleaning-needed event** and manager notification.
- Cleaning wait duration begins when departure is confirmed and remains visible until readiness is confirmed; do not stop it on acknowledgement of an alert.
- Owners/Managers, **not Waiters**, make the final manual `Table Ready` confirmation.
- A Waiter may mark an assigned cleaning task `Completed`, but this **does not** automatically set `Ready`.
- If new guests sit before readiness has been confirmed, retain a visible unresolved readiness warning, still process the genuine new occupancy, and avoid duplicating the old cleaning event.
- Whether the optional `Start Cleaning` state/button belongs in MVP v1 remains **unapproved**; the simple flow is preferable by default: cleaning alert -> acknowledgement/task handling -> manager `Table Ready`.

### 2.7 Roles and authorization

Exactly three role types:

| Permission | Owner | Manager | Waiter |
|---|:---:|:---:|:---:|
| View relevant table statuses | Yes | Yes | Yes |
| Receive and acknowledge shared AI/operational manager alerts | Yes | Yes | No |
| Assign/reassign waiter tasks | Yes | Yes | No |
| View/operate own assigned tasks (`Accept`, `Complete`) | Optional oversight | Optional oversight | Yes |
| Manual occupancy corrections | Yes | Yes | No |
| Adjust session start/end timestamps (with reason) | Yes | Yes | No |
| Confirm cleaned table is Ready | Yes | Yes | No |
| Configure cameras, AI mapping and settings | Yes | No | No |
| Manage user accounts/permissions | Yes | No | No |
| Review full audit/event history | Yes | Yes | No |

Waiters must not see administrative settings or other waiters' private task details by default. Apply server-side authorization, not just UI hiding.

### 2.8 Manager alerts and shared acknowledgement

Alert types in MVP v1:
1. **New Customers** — when a fresh seating session is confirmed.
2. **Cleaning Required** — when departure is confirmed.
3. **Acceptance Overdue** — when an assigned waiter does not accept a task by its deadline.
4. **Completion Overdue** — when an accepted task is not completed by its deadline.

- Send manager alerts to multiple Owner/Manager accounts; **first successful acknowledgement wins** and closes the receipt state for all, preserving `acknowledged_by` and `acknowledged_at`.
- Acknowledgement means the alert was seen, **not** that service or cleaning is complete.
- Do **not** send recurring reminder notifications. Each original event or timeout condition produces one alert under its deduplication key.
- Prominent **non-blocking in-app banner** for newly seated customers, with quick `Acknowledge` and `Assign Waiter` actions, and a push notification where supported.
- Priority then time ordering of manager inbox; arrival alerts come first. Proposed initial sort: Arrival > Acceptance Overdue > Completion Overdue > Cleaning Required. Within each class, prioritize oldest still-actionable alerts. This order is a design default.
- Show status categories (`Pending`, `Acknowledged`, `Closed/No longer actionable`) and keep event history.
- Alert acknowledgement and task assignment are independent and may occur in either order.
- Replays/reconnections must not create duplicate tasks, sessions or alerts. Use event IDs/idempotency keys and atomic updates.
- If a task or session is superseded, mark related outstanding alerts obsolete/no longer actionable without pretending they were acknowledged.

### 2.9 Waiter tasks

Two task types in MVP v1:
- `CUSTOMER_RECEPTION`: send a waiter to welcome newly seated guests.
- `TABLE_CLEANING`: clean/prepare table after departure.

- A Manager/Owner explicitly chooses and assigns the waiter. **No automatic waiter assignment.**
- Waiters see tasks assigned to them and use workflow **Assigned -> Accepted -> Completed**; `Cancelled`/`Reassigned` are exceptional workflow outcomes.
- Allow **multiple concurrent tasks** per waiter. Show open-task counts and distinguish `Online/Offline` from inferred `Available/Busy`.
- The assigned waiter accepts their task, and completion updates the management dashboard. Reception completion does not affect the occupied session timer.
- Shared manager visibility prevents double assignment. Reassignment revokes the previous assignment and creates a new assignment attempt/history without losing the overarching task identity.
- Acceptance timeout: one alert to managers **per assignment attempt** if a waiter has not accepted by that assignment's deadline; no repeated reminders. A new assignment attempt may legitimately have its own single timeout alert.
- Completion timeout: one alert to managers for an accepted, still-incomplete task after its completion deadline; no repeated reminders.
- **Owner** configures timeout policy, with potentially separate thresholds for reception and cleaning. Example values floated in discussion (NOT approved): reception acceptance 60 s, cleaning acceptance 120 s, reception completion 3 min, cleaning completion 7 min.
- Timer for acceptance starts at assignment; completion timer starts at acceptance. Later policy changes should ordinarily apply to new assignments, not retrospectively to active ones.
- Accept/Complete updates must be idempotent and validate the currently authorized assignee/task version. An old assignee cannot accept a reassigned task.
- Task delay is an operational signal, not a basis for automatic employee penalties or inferences about worker behavior.

## 3. Required UI surfaces

1. **Owner dashboard:** cameras/configuration, mapping setup, users, timeout policies, live tables, manager inbox, audit history.
2. **Manager dashboard:** live table grid/status/timers, prioritized unified alert center, non-blocking arrival banner, assignment controls, waiter workload and task statuses, manual correction and `Table Ready` actions.
3. **Waiter dashboard:** own tasks separated by `Assigned`, `Accepted` and `Completed`; Accept/Complete interactions and status/connectivity.
4. **Camera/table setup:** video-preview and editable AI-suggested table/seat areas, primary/secondary camera bindings, per-table verification/activation.
5. **Session history:** original vs adjusted start/end, source, affected table assignment periods, correcting actor/reason, monitoring gaps.

Ensure accessible statuses beyond color alone and optimize waiter/manager workflows for mobile screens.

## 4. Suggested conceptual data entities (not physical schema)

- `Restaurant`
- `User`, `Role`, `Membership`
- `Camera`, `StreamEndpoint`, `CameraHealth`
- `Table`, `SeatRegion`, `CameraTableMapping`, `TableMonitoringConfiguration`
- `DetectionObservation`, `TableDecisionState`, `ManualOverride`
- `OccupancySession`, `SessionTableAssignment`, `SessionTimeAdjustment`
- `TableReadinessState`, `CleaningCycle`
- `DomainEvent`, `ManagerAlert`, `AlertAcknowledgement`
- `WaiterTask`, `TaskAssignmentAttempt`, `TaskStatusHistory`
- `AuditLog`, `SynchronizationOutbox`

Design for versioned writes, durable event identifiers, proper tenant isolation per restaurant, and reliable cloud/local reconciliation. Avoid unnecessary storage of personally identifying visual information.

## 5. MVP v2 (explicitly deferred)

- Transfer an existing group/session from one table to another while preserving session continuity and per-table occupancy intervals.
- Temporarily group/ungroup tables to serve one group under one linked session/timer, with appropriate audit and conflict checks.

Do not implement these workflows in MVP v1. Do not add ordering, payment, POS, face identification, or automated waiter assignment without separate approval.

## 6. Non-functional and safety requirements

- Privacy: process raw camera footage locally by default; avoid facial identification; minimize unnecessary storage; support notice/consent and local legal compliance as applicable.
- Security: authenticated accounts, authorization checks at API level, least privilege, secure video credentials, encrypted remote communications, protected audit logs and backups.
- Reliability: local operation during WAN outage, persisted timestamp-based timers, camera health/unknown state, idempotent state transitions, eventual reconciliation without duplicate manager notifications.
- Scalability: multiple feeds and tables; benchmark actual supported feed counts/frame rates on target hardware rather than promising arbitrary limits.
- Observability: inspect camera health, per-table detector confidence, decision history, sync backlog and operational alert delivery status.
- Testing: unit/integration tests of every state transition, concurrency/race tests for shared acknowledgement/reassignments, restart/offline recovery, simulated video events, and real footage validation with appropriate privacy precautions.
- AI quality: measure seating precision/recall, false alerts per table-hour, detection latency, and performance across occlusions, night/poor lighting, standing waiters, booth seating and multiple cameras.

## 7. Decisions still open — do not silently assume finalized policy

1. Table activation: incremental activation of validated tables vs whole-restaurant activation.
2. Cleaning workflow: simple `Cleaning Required -> Manager Ready` vs adding a separate `Start Cleaning` stage. Waiter task completion must still remain separate from final readiness.
3. Exact acceptance and completion timeout durations; Owner can configure them.
4. Actual AI confidence calibration and threshold values after testing.
5. Whether to generate arrival alerts for manual session creation (proposed no by default).
6. Policy for explicitly cancelling, expiring, or resolving tasks that become obsolete because a session ends or new guests arrive.
7. Final technical stack, deployment model, support matrix (OS, cameras, mobile/web), database/cloud sync details and push provider — choose after inspecting the repository and considering operational constraints.
8. How to handle delayed offline acknowledgements and manager actions so the local source of truth and cloud replicas never both independently finalize a conflicting action.
9. Retention policies, evidence/debug-frame handling and applicable local privacy requirements.

These open choices do **not** prevent implementation planning or creation of mocks/tests. Mark them as assumptions or seek approval before irreversible product decisions.

## 8. Suggested Codex delivery plan

**Phase A — Repository inspection and architecture (no coding until approval):**
- Inspect current repository layout, tools, existing app and CI, and deployment constraints.
- Propose component boundaries, system data flow, local vs cloud source-of-truth strategy, security model and milestones.
- Turn open decisions into a short decision log.

**Phase B — Domain and API foundations:**
- Tables, sessions, task/alert model, RBAC, immutable audit log and key state-machine tests.
- Simulated events before connecting real cameras.

**Phase C — Manager / Waiter / Owner UI:**
- Live table/timer dashboard, shared alert inbox, non-blocking arrival banner, task assignment and Accept/Complete workflow, ready confirmation.

**Phase D — Video integration and AI:**
- Camera compatibility proof of concept, assisted mapping, 5s/30s temporal verification, multi-camera deduplication and confidence calibration.

**Phase E — Hybrid hardening and acceptance:**
- Local outage handling, replay/idempotency, push delivery, performance and field trials on PC/AI Box.

## 9. Acceptance scenarios for MVP v1

1. Guest sits at a mapped ready table: after 5 seconds of reliable evidence, exactly one session and one arrival alert exist; timer reflects first observed seated time.
2. Two cameras see the same guest/table: still one session, one event, one alert.
3. Two managers see the same alert: first atomic acknowledgement wins, other device displays acknowledged identity/time.
4. Manager assigns reception to a chosen waiter: waiter accepts then completes; occupancy timer continues uninterrupted.
5. A waiter receives multiple tasks: correct counts and per-task independent state; busy status is advisory.
6. Acceptance threshold exceeded: one manager timeout alert for this assignment; no repeated notification. Reassignment creates a new attempt and revokes previous access.
7. Completion threshold exceeded: one alert for the overdue accepted task; completion after timeout preserves the late record.
8. Last guest leaves: after 30 seconds of reliable vacancy, close session using initial observed departure time; send one cleaning alert; mark not ready.
9. Waiter completes cleaning task: table remains not ready until an authorized Manager/Owner confirms `Table Ready`.
10. Manual occupancy correction: required reason, protected 60 seconds, no immediate AI reversion; restoration requires a genuinely new verified transition.
11. Manager retroactively adjusts start or end: preserve original timestamps and audit history; reject overlapping sessions.
12. Restart or WAN outage: timestamps and sessions recover, uncertain camera visibility is not treated as vacancy, and eventual sync does not multiply events.
13. A dismissed/acknowledged/obsolete alert cannot be misinterpreted as completing the underlying waiter task or confirming table cleanliness.

## 10. Instructions to Codex

**Before writing code:** inspect the repository, compare it with this requirements file, then produce:
- A concise architecture recommendation and technical stack rationale.
- A proposed repository layout and data model.
- Explicit risks and unapproved product decisions from Section 7.
- A phased implementation plan with testable deliverables.

Ask for approval before coding. **Communicate with the project owner in Arabic.** This document is the current product baseline; do not assume it grants permission to change existing repository files beyond documentation review.
