# ADR 0001: Keep a content-first Jekyll architecture

- Status: accepted
- Date: 2026-07-20

## Context

The site is a GitHub Pages Jekyll site with several custom content collections
and a growing set of personal knowledge-graph behaviors. Its durable value is
the Markdown corpus and its URLs, not the Academic Pages theme implementation.
Custom behavior had begun to accumulate directly in a shared layout, which
makes future changes hard to locate and easy to duplicate.

## Decision

Keep Jekyll and GitHub Pages as the publishing platform. Continue to make
Markdown plus front matter the primary authoring interface. Place article-wide
behavior behind focused includes and compose those includes in `single.html`.

The backlink include is captured once by the layout and reused both in the
rendered article and in table-of-contents generation. This gives the layout a
small interface while retaining the existing `[[Title]]` authoring syntax and
public output.

## Consequences

- Existing content and URLs remain compatible.
- Future build-time features have an obvious home in includes or plugins.
- A future dynamic search or a client application is possible, but must be a
  separate decision with a migration plan rather than an incremental layer.
- Backlink discovery still scans site documents at build time; that is
  acceptable at the present scale and should be measured before optimization.
