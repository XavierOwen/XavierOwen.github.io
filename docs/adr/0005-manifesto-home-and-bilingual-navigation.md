# ADR 0005: Use the Manifesto home and reader-intent primary navigation

- Status: accepted
- Date: 2026-07-21

## Context

The inherited navigation exposed Jekyll collection names before it explained
why the site's research, teaching, writing, faith, and creative work belong
together. Three throwaway home-page directions were prototyped; the Manifesto
direction was selected for production. The language controls also needed fixed
geometry on narrow screens rather than labels whose width changed by language.

## Decision

The production home begins with the question “How might the world become a
‘we’?”, presents the four reader paths with equal weight, and only then presents
the professional identity and CV layer. The old home-page questions remain as
an editorial “sources of the question” section instead of being discarded.

Primary navigation now follows the same four reader paths, then About and CV.
Collection and tag indexes remain available as secondary discovery surfaces.
Chinese and English home/profile routes share structured bilingual data and a
single layout. Language controls are fixed circles labeled `中` and `EN`.

## Consequences

- Public navigation describes reader intent rather than repository folders.
- The root Chinese home keeps its URL; English uses `/en/` and both pages emit
  reciprocal language metadata.
- The selected prototype is now production code in focused layouts and Sass;
  the throwaway prototype route is removed.
- Future home copy should be changed in `_data/home.yml`, not duplicated across
  route files or embedded in JavaScript.
