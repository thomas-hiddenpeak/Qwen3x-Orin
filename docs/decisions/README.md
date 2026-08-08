---
q3x_document:
  id: q3x-decision-index
  class: procedure
  status: active
  owner: project-maintainers
  authority: navigation to accepted and superseded project decisions
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: architecture-decision index
  review_trigger: ADR addition, state change, or supersession
---

# Architecture decision records

Architecture decision records explain why a durable project choice was made.
They do not own dynamic status, current measurements, implementation details,
or experiment evidence.

## States

- `proposed`: under discussion; no implementation authority.
- `accepted`: controls new work within its stated scope.
- `superseded`: retained for lineage and linked to its successor.
- `rejected`: considered but never adopted.

An accepted ADR remains subordinate to the
[Engineering Constitution](../ENGINEERING_CONSTITUTION.md). Current
implementation truth remains in [Current Status](../CURRENT_STATUS.md).

## Records

- [ADR-0001: End-state-first leakage and controlled evolution](0001-end-state-first-leakage.md)
