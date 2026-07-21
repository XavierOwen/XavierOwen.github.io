# Architecture

## Shape of the site

```text
Authored Markdown + front matter
  ├── _notes / _spirits / _projects
  ├── _publications / _teaching / _posts
  └── _pages
          │
          ├──> published-content audit
          ├──> generated content-index.json
          │       ├── reader paths
          │       ├── language versions
          │       └── wiki aliases + backlink graph
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
| Content contract | stable Content ID, bilingual discovery metadata, reader paths | `scripts/audit_published_content.rb` | Markdown front matter |
| Generated content graph | conceptual items, versions, aliases, backlinks | `scripts/build_content_index.rb` | `_data/content-index.json` |
| Reader paths | path key plus UI language | `_layouts/reader-path.html` | four bilingual route pairs |
| Manifesto home | bilingual thematic copy plus ordered reader paths | `_layouts/manifesto-home.html` and `_data/home.yml` | `/` and `/en/` |
| Primary navigation | reader paths, About, CV, and language controls | `_includes/masthead.html` and `_data/navigation.yml` | shared masthead |
| Language preference | explicit stored choice, then browser language | `assets/js/language-preference.js` | `data-language-context` |
| Article layout | page front matter and rendered Markdown | `_layouts/single.html` | article-page rendering |
| Backlinks | generated Content ID references | `_includes/backlinks.html` reads the content graph | captured output in `single.html` |
| Wiki links | `[[Title]]` / `[[Label::URL]]` in rendered content | `assets/js/wiki-links.js` resolves the generated alias index | browser enhancement after render |
| Article navigation | headings plus `toc: true` | Jekyll TOC include and `toc-scrollspy.js` | generated `.toc__left` links |
| Publication gate | `npm run verify:site` | `.github/workflows/pages.yml` | verified `_site` artifact before Pages deployment |

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
4. Keep `_data/content-index.json` deterministic and metadata-only. Avoid adding
   article bodies; search is a separate design decision with its own payload
   budget.
5. Run `npm run verify:site` after changing content metadata, Liquid, Sass,
   layouts, JavaScript, collection configuration, or deployment behavior.

## Known debt, deliberately not changed in this pass

- Some custom style rules and the Academic Pages theme are interleaved. A
  visual redesign should first extract an explicit site design layer rather
  than continue patching theme files.
