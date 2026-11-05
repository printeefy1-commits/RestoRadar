# Open-source readiness plan

**Status: work in progress.** This checklist is for making RestoRadar legitimately reusable by outside developers. It does **not** claim that RestoRadar is already licensed as open source or widely adopted.

## Current technical foundation

RestoRadar has TypeScript workspaces for its application and domain/database packages, a PostgreSQL-backed backend, role-based staff workflows, automated tests, and a recorded-video evaluation path. As described in the README, it is not yet a complete live-camera/cloud deployment product. See the README for an authoritative list of current capabilities and exclusions.

## Before describing this repository as open source

- [ ] **Owner chooses an OSI-approved license** for code they have the right to license, after checking third-party dependencies, model weights, datasets, media, and any employer/client ownership restrictions.
- [ ] Add a root `LICENSE` file, plus any necessary third-party notices; confirm GitHub detects the intended license.
- [ ] Decide whether all workspaces can be licensed the same way, or if commercial/private pieces need to remain separate.
- [ ] Review tracked files and Git history for credentials, personal information, private footage, or inadvertently committed assets. Rotate any exposed secrets; deleting them from the latest commit is not sufficient.
- [ ] Publish a clear project description, supported environments, and reproducible quickstart; keep README claims aligned with working features.
- [ ] Verify CI passes, with clean installation, static checks, and tests on a fresh environment.
- [ ] Publish at least one tagged release and a change log once there is a stable, usable developer-facing capability.

## Make the project useful to external developers

- [ ] Identify a **genuinely reusable** and stable unit, for example the table occupancy/session state-machine or event contracts, separate from restaurant-specific deployment. This is a proposal, not an already published package.
- [ ] Publish a minimal working example and a documented API contract for that unit, after validating that reuse is permitted by the selected license.
- [ ] Make test fixtures synthetic and explicitly describe the limitations of occupancy/video inference.
- [ ] Add issue and pull request templates, a support channel, and a vulnerability reporting policy suited to available maintainer capacity.
- [ ] Invite meaningful external review, accept useful contributions when justified, and record their attribution accurately.
- [ ] Measure actual use: external contributors, GitHub dependents, registry downloads if a real package is published, and downstream projects. Never fabricate metrics or create artificial engagement.

## Project ownership and contribution attribution

The main repository is currently `printeefy1-commits/RestoRadar`. A developer who also owns `kabgeek0` can contribute through that account, but must actually sign into `kabgeek0` and author meaningful commits or pull requests. Adding an account to the repository or moving repository ownership does not by itself create historical contributions or prove maintainer impact.

For GitHub attribution, check the account used to create the pull request, the commit author email linked to that account, and whether any commits were co-authored. Do not rewrite authorship solely to manufacture contribution history.

## Claude for Open Source (informational, verified 2026-11-05)

Anthropic's official rules at https://www.anthropic.com/claude-for-oss-terms require, among other things, a GitHub account at least two years old, public OSS activity in the preceding 90 days, and contribution to at least one OSI-licensed project. Applicants must also qualify for a maintainer/contributor track or demonstrate meaningful ecosystem dependence under the discretionary impact track. A public repository and good documentation alone are **not sufficient**. The thresholds and availability can change, and eligibility is assessed at the time of application. A person must not claim adoption, contributors, dependency counts, or activity they cannot substantiate.

This plan is for building a genuinely beneficial project. It is not a guarantee of program eligibility or acceptance.
