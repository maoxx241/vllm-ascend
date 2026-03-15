# Deployment acceptance reference

This deployment skill is accepted against behavior, not against legacy case IDs.

What matters:
- self-acquire before question-gate
- user-only blocker questions only
- blocked results do not emit scripts
- compatible/candidate results emit a bundle with reasoning + validation checklist + scripts
- default single-instance assumption
- A3 single-node defaults to 8 cards / 16 chips
- A2 still asks card count because single-node specs vary
