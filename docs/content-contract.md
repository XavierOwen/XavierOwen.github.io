# Published-content contract

Every public item in Notes, Spirits, Projects, Publications, Teaching, and
Posts must satisfy this contract before the bilingual reader-path launch.
Items marked `published: false` are excluded.

```yaml
content_id: "a-permanent-language-neutral-key"
language: zh # zh or en: the language of this document
original_language: zh # zh or en: the work's original language
reader_paths:
  - notes-writing
representative_paths: [] # optional; always a subset of reader_paths
title_zh: "中文标题"
title_en: "English title"
summary_zh: "用于列表和导航的中文简介。"
summary_en: "An English summary for listings and navigation."
```

`content_id` is permanent even when a title, filename, URL, or reader-path
assignment changes. A translation uses the same `content_id`, sets `language`
to the target language, keeps the pair's `original_language`, and adds:

```yaml
translation_reviewed: true
```

There is exactly one original document for a Content ID and at most one document
per supported language. The audit rejects an orphan translation, duplicate
language version, unsupported language/path key, missing bilingual discovery
metadata, or a representative-path designation outside `reader_paths`.

Existing originals retain their published URLs. A reviewed translation gets a
target-language URL (normally under `/en/` or `/zh/`) rather than replacing the
original. When both versions exist, each page emits a self canonical and
reciprocal `hreflang` links. When a version is absent, the page emits no false
alternate URL and the language control keeps the reader on the original with a
visible fallback notice.

The interface preference is stored in `localStorage` as
`xavier-site-language`. An explicit `zh` or `en` selection takes precedence;
otherwise the first supported browser language is used for that visit. This UI
preference never changes the article's truthful `<html lang>` content metadata.

Run `npm run audit:content` for a strict audit. During migration,
`npm run audit:content:report` reports all gaps without returning a failure;
that mode is intentionally temporary and is not the future release gate.
