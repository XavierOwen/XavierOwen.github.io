# Project guide

This is a content-first Jekyll site deployed through GitHub Pages. Its custom
work is intentional: the site brings together academic work, teaching,
projects, creative writing, and spiritual notes. Treat the theme as an
implementation detail and the content collections as the durable asset.

## Before changing code

- Preserve existing, uncommitted changes unless the user explicitly asks to
  revise them. In particular, `.env`, `.env.example`, and `scripts/tavily-search.mjs`
  are a local research helper, not part of the published site.
- Read `CONTEXT.md` and `docs/architecture.md`; record durable, hard-to-reverse
  choices in `docs/adr/`.
- Prefer Markdown and front matter for new content. Do not add bespoke HTML or
  JavaScript to a content file when an existing collection, layout, or include
  can own that behavior.

## Site model

- `_notes`, `_spirits`, and `_projects` are the custom knowledge collections.
  Their public indexes live in `_pages/` and their categories are declared in
  `_config.yml`.
- `[[Title]]` and `[[Label::https://example.com]]` are published-note syntax.
  The `wiki-links` include resolves them on the client; `backlinks` resolves
  internal references during the Jekyll build.
- `single.html` is the integration seam for article pages. Put cross-cutting
  article behavior in a focused include, not inline in the layout.

## Verification

Run both checks after a site or JavaScript change:

```sh
bundle exec jekyll build
node --check assets/js/toc-scrollspy.js
```

Use `npm run build:js` only after Node dependencies are installed; it rewrites
the generated `assets/js/main.min.js` bundle.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in this repository's GitHub Issues. See
`docs/agents/issue-tracker.md`.

### Triage labels

Use the repository's five canonical GitHub label mappings. See
`docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository: read the root `CONTEXT.md` and relevant
`docs/adr/` records. See `docs/agents/domain.md`.
