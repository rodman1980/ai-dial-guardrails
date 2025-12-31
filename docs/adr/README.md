# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records documenting key design choices for the AI DIAL Guardrails project.

## What is an ADR?

An Architecture Decision Record (ADR) captures an important architectural decision along with its context and consequences. ADRs help teams understand:
- Why specific technologies or patterns were chosen
- What alternatives were considered
- What trade-offs were accepted
- When and by whom decisions were made

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-NNN: Title

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded by ADR-XXX]

## Context
What is the issue we're facing? What factors are driving this decision?

## Decision
What is the change we're making?

## Consequences
What becomes easier or harder as a result of this decision?

## Alternatives Considered
What other options did we evaluate and why were they rejected?
```

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](./ADR-001-llm-based-validation.md) | LLM-Based Validation Over Regex | Accepted | 2025-12-31 |
| [ADR-002](./ADR-002-streaming-architecture.md) | Streaming Guardrail Architecture | Accepted | 2025-12-31 |
| [ADR-003](./ADR-003-layered-defense.md) | Layered Defense Strategy | Accepted | 2025-12-31 |
| [ADR-004](./ADR-004-presidio-integration.md) | Presidio for NLP-Based PII Detection | Accepted | 2025-12-31 |

## Creating New ADRs

1. Copy template: `cp ADR-template.md ADR-NNN-title.md`
2. Assign next sequential number
3. Fill in all sections
4. Submit for review
5. Update index above

## References

- [ADR GitHub Organization](https://adr.github.io/)
- [Documenting Architecture Decisions - Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
