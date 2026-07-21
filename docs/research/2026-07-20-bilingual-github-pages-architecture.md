# Bilingual Jekyll architecture on GitHub Pages

**Question.** Can the planned bilingual model be deployed on GitHub Pages
without depending on an unsupported Jekyll plugin, while preserving current
original URLs and making metadata validation a release gate?

## Finding

Yes. The smallest durable design is ordinary, paired Jekyll collection
documents plus a repository-owned pre-build validator/index generator. Keep
each existing original at its current URL; put an actual translation at an
explicit language-prefixed permalink such as `/en/note/example/`. The two
documents share `content_id`, but remain separately rendered static pages.

This uses Jekyll's documented custom front matter, collection documents and
permalinks; it does not require a multilingual Jekyll plugin. Jekyll exposes
custom front matter to Liquid, collection documents through `site.collections`
and their rendered `url`, and supports a front-matter `permalink` as the final
output URL. [Jekyll front matter](https://jekyllrb.com/docs/front-matter/),
[collections](https://jekyllrb.com/docs/collections/), and
[permalinks](https://jekyllrb.com/docs/permalinks/).

## Deployment constraint

The present GitHub Pages-compatible configuration must not rely on a custom
`_plugins` implementation for translation lookup or validation. GitHub Pages
cannot build a site with unsupported plugins; Jekyll also documents that Pages
builds in safe mode and disables non-whitelisted plugins. [GitHub Pages and
Jekyll](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/about-github-pages-and-jekyll),
[Jekyll plugin installation](https://jekyllrb.com/docs/plugins/installation/).

Two viable deployment paths follow:

1. **Branch build plus separate CI check.** Run a validator on pull requests
   and protected default-branch pushes. This keeps the built-in Pages flow but
   the validator is not part of the Pages Jekyll process; it is a hard gate
   only if branch protection requires the check.
2. **Recommended: custom GitHub Pages Action.** Select GitHub Actions as the
   Pages source. A build job runs the validator, then `bundle exec jekyll
   build`, then uploads `_site`; a deploy job depends on that build artifact.
   A nonzero validator exit therefore prevents deployment. GitHub documents
   this build/upload/deploy pattern and says Actions supports Pages sites built
   with any static-site generator. [Custom Pages
   workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
   [publishing source configuration](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

The second path gives the desired production build-time contract without
turning the validator into a Jekyll deployment plugin. It does not require an
unsupported Jekyll plugin; it merely runs repository code before Jekyll in a
normal Actions step.

## Recommended content and build model

### Content files

Keep every version as a normal file in its current collection. A translation
may live in a readable source subdirectory, but its front matter—not the
source directory—sets its public URL. For example:

```yaml
content_id: wolves-in-the-storm
language: en
is_original: false
permalink: /en/note/wolves-in-the-storm/
title_zh: 风暴狼吟
summary_zh: ...
title_en: Wolves in the Storm
summary_en: ...
reader_paths: [notes-writing]
featured_in: [notes-writing]
translation_reviewed: true
```

The Chinese original keeps its existing permalink, carries the same
`content_id`, and is marked `is_original: true`. A future English original can
follow the symmetric rule: its own existing-style URL is stable and a Chinese
translation, if created, gets the target-language prefix. The prefix describes
the translation URL; it does not rewrite the original corpus.

`content_id` is the pairing key, not a route or title. The validator should
require exactly one original per ID and at most one published version per
language. It should reject an unreviewed published translation, duplicate
output routes, and a `featured_in` value outside `reader_paths`.

### Validator and generated lookup data

Create a small, dependency-light repository script (Ruby using YAML or Node
with a pinned YAML parser) that scans the published content collections before
Jekyll. It should fail on:

- missing or duplicate `content_id` / invalid translation-pair shape;
- missing `language`, non-empty `reader_paths`, bilingual title, or bilingual
  short summary;
- unknown reader-path keys and invalid path-specific representative metadata;
- translation URLs that do not use the target-language prefix; and
- an unreviewed translation marked published.

The same script can emit one generated `_data/content-index.json`, keyed by
`content_id` and language, with each version's URL, localized discovery text,
and original flag. Jekyll natively reads JSON/YAML/CSV/TSV in `_data` and
exposes it as `site.data`, so Liquid includes can use this index for language
switching, reader-path listings and language-aware backlinks without a plugin.
[Jekyll data files](https://jekyllrb.com/docs/datafiles/).

Treat the generated index as a build input: either generate it deterministically
in CI and do not commit it, or commit it and have CI reject a stale copy. The
first is preferable if the local build command runs the generator first.

### Fallback and links

When a selected language lacks a paired version, keep the reader on the
original's real URL and label its language. Do **not** create `/en/...` pages
that repeat Chinese article bodies merely to make the switcher look complete.
The lookup index can make the language switcher, wiki links and backlinks first
select the requested language, then fall back to the original. Backlinks first
deduplicate by `content_id`, then choose the best displayed version.

This preserves the agreed distinction: a translation pair is one conceptual
work, while each published language version is a separate static document.

## Search metadata and `hreflang`

Use a self-referential canonical URL for every rendered article. A true English
translation is not a duplicate to canonicalize to the Chinese original;
canonicalization and language alternates are distinct signals. Google's
guidance says to use `rel="alternate" hreflang` for language variants rather
than trying to express alternates with canonical attributes. [Canonical URL
guidance](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls),
[localized-page guidance](https://developers.google.com/search/docs/specialty/international/localized-versions).

For each actual pair, render the same complete, reciprocal set inside the
`<head>` of both pages:

```html
<link rel="canonical" href="https://xavierowen.github.io/current-page/">
<link rel="alternate" hreflang="zh-Hans" href="https://xavierowen.github.io/current-page/">
<link rel="alternate" hreflang="en" href="https://xavierowen.github.io/en/current-page/">
```

Use the appropriate stable language tags selected for the site, and construct
absolute URLs from `site.url` plus `page.url`. Google requires every language
version to list itself and every alternate, and requires fully-qualified
alternate URLs; incomplete reciprocal links are ignored. [Google localized
versions requirements](https://developers.google.com/search/docs/specialty/international/localized-versions).

For an unpaired original, render only its self canonical and its actual content
language; do not claim an absent English alternate. The English UI fallback is
not an English full-text version. Google determines content language from the
visible page rather than `hreflang` or the HTML `lang` attribute, so a Chinese
fallback must not be presented to search engines as an English translation.
[Google localized versions requirements](https://developers.google.com/search/docs/specialty/international/localized-versions).

An `x-default` link is optional for a genuinely language-neutral selector or
landing page, not a substitute for absent article translations. Google
specifically presents it as a fallback option for unmatched language selectors
or automatic language-routing pages. [Google localized versions
requirements](https://developers.google.com/search/docs/specialty/international/localized-versions).

## Decision

Adopt the custom Pages Action plus pre-build validator/index generator. Keep
the current collection model and original URLs; add translated collection
documents with explicit `/en/...` permalinks. Extend the existing custom SEO
include to emit self canonical links and reciprocal `hreflang` only where a
reviewed pair exists. This satisfies the decided URL stability, `content_id`,
original-language fallback, bilingual discovery metadata, language-aware wiki
links/backlinks, and strict publication contract without adding a Pages-hostile
runtime plugin.

## Follow-up implementation questions

- Define the exact front-matter field names and the set of collections covered
  by the published-content contract.
- Decide whether the generated content index is ignored or committed, then
  make local and CI build commands follow the same sequence.
- Add rendered-output tests for canonical/`hreflang`, translation switch, and
  original-language fallback.
