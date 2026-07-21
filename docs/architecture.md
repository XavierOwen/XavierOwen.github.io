# Architecture

## Shape of the site

```text
Authored Markdown + front matter
  ├── _notes / _spirits / _projects ──> collection index pages
  ├── _publications / _teaching / _posts ──> template-provided indexes
  └── _pages ──> standalone pages and navigation
                    │
                    ▼
              Jekyll build (GitHub Pages)
                    │
                    ▼
               layouts and includes
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    server-rendered       progressive enhancement
    metadata, TOC,        theme, TOC scrollspy,
    backlinks             wiki-link previews
```

## Modules and seams

| Module | Interface | Implementation | Seam |
| --- | --- | --- | --- |
| Content collections | Markdown plus front matter | Jekyll collections in `_config.yml` | collection directories |
| Collection indexes | configured category + authored entries | `_pages/notes.html`, `spirits.html`, `projects.html` | index page templates |
| Article layout | page front matter and rendered Markdown | `_layouts/single.html` | article-page rendering |
| Backlinks | `[[Title]]` in source content | `_includes/backlinks.html` scans eligible site documents | captured output in `single.html` |
| Wiki links | `[[Title]]` / `[[Label::URL]]` in rendered content | `_includes/wiki-links.html` | browser enhancement after render |
| Article navigation | headings plus `toc: true` | Jekyll TOC include and `toc-scrollspy.js` | generated `.toc__left` links |

The desired direction is **depth**: callers should add content through a
small, stable interface (Markdown and front matter), while display complexity
stays in the module that owns it. For example, article pages should invoke a
single backlink module rather than each reimplementing detection logic.

## Change rules

1. A new content type begins with a product decision: decide whether it is a
   new collection, a category in an existing collection, or a standalone page.
2. A new article-wide behavior belongs in an include with one documented
   interface; `single.html` only composes it.
3. Keep published Markdown syntax backward compatible. Migrate source text
   explicitly if a syntax must change.
4. Avoid expanding the client-side content index without measuring payload and
   page behavior. Search is a separate design decision.
5. Verify Jekyll output after changing Liquid, Sass, layouts, or collection
   configuration.

## Known debt, deliberately not changed in this pass

- Some template sample content still remains in the site tree. Remove or
  replace it only as part of a deliberate editorial pass, so existing URLs do
  not disappear by accident.
- The client-side wiki-link index serializes excerpts from every custom
  collection on each relevant article page. It is fine for the current corpus,
  but should become a generated data asset if the corpus grows substantially.
- Some custom style rules and the Academic Pages theme are interleaved. A
  visual redesign should first extract an explicit site design layer rather
  than continue patching theme files.
