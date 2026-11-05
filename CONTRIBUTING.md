# Contributing to RestoRadar

Thank you for your interest in improving RestoRadar. The project is at an early foundation stage, and its existing code, approved product decisions, and operational documentation should be reviewed before suggesting changes.

> **Licensing status:** This repository is publicly visible, but an open-source license has not yet been selected and published. Public visibility alone does not grant permission to reuse or redistribute code. The maintainer must decide and publish an appropriate license before representing the repository as an open-source project. Discuss proposed contributions with the maintainer while that decision remains pending.

## Where to start

- [README.md](README.md): current capabilities, limitations, setup, and test commands.
- [REQUIREMENTS.md](REQUIREMENTS.md): original requirements.
- [Approved product decisions](docs/decisions/001-approved-baseline.md): decisions to preserve.
- [Architecture](docs/architecture/foundation.md): system boundaries and core design.
- [Implementation plan](docs/implementation-plan.md): planned work.

Useful contribution topics include reproducible bugs, tests, clear documentation, accessibility and Arabic UI improvements, and performance measurements backed by reproducible test cases. Propose major architectural changes in an issue before implementing them.

## Local development

Use Node.js 24, npm, and Docker Compose (or a suitable local PostgreSQL 17 instance). Follow the README to configure the local database and create a development owner account. Never use production credentials or real customer data in development.

```bash
npm ci
docker compose -f infra/compose/compose.yml up -d --wait
test -f .env || cp .env.example .env
npm run db:migrate
npm run dev
```

Run the local checks before opening a pull request:

```bash
npm run check
npm run format:check
```

Integration and browser tests have additional prerequisites; see the README for the dedicated PostgreSQL test database and Playwright setup.

## Pull request process

1. Open an issue first for significant changes, especially permission changes, AI inference, video processing, new dependencies, or breaking APIs.
2. Make a focused change on a separate branch. Keep unrelated refactors out of the pull request.
3. Explain the problem, proposed solution, tests run, and any security/privacy implications.
4. Include screenshots only for non-sensitive example data. Do not upload real restaurant footage, patron images, tokens, passwords, or customer information.
5. Expect review and changes before merging. A pull request or issue is not a commitment to accept the contribution.

## Security and privacy

RestoRadar is intended to process restaurant occupancy information, potentially derived from cameras. Never add secrets, identifiable customer footage, or sensitive operational records to issues, commits, fixtures, or bug reports. Use synthetic fixtures and anonymized examples. Contact the maintainer privately before disclosing a suspected vulnerability publicly.

## Design constraints

Preserve the existing owner/manager/waiter roles, manual correction audit trail, timer/session invariants, and explicit table-readiness workflow unless the maintainers approve a design change. AI-detected events must not silently override protected manual decisions.

The project's goals are useful contributions and transparent development, not artificial stars, forks, issue traffic, or activity metrics.
