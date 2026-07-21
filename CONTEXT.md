# Site context

## Purpose

This is Yuanxing Cheng's public, content-first personal site. It presents a
single person through several connected bodies of work rather than as separate
microsites:

- academic research, publications, CV, and teaching;
- mathematical learning notes and tutoring material;
- software, games, scraping, and art projects;
- creative writing, translations, songs, and spiritual study.

The site began with Academic Pages in August 2025. Subsequent commits show a
deliberate evolution from a conventional academic homepage into a personal
knowledge garden: custom collections appeared first, projects moved out of
notes, an article table of contents was added, and wiki links/backlinks joined
the collections later. The original template copy that remains in some pages
is historical residue, not the site's current product definition.

## Vocabulary

- **Content collection**: a Jekyll collection that stores authored material.
  The site's custom collections are `notes`, `spirits`, and `projects`.
- **Authoring structure**: the stable collection and folder arrangement used
  to create, store, and preserve content and its URLs. It is distinct from
  reader navigation.
- **Index page**: the public collection listing in `_pages/`.
- **Wiki link**: source text in the form `[[Title]]`, resolved to an internal
  page in the browser; `[[Label::URL]]` is an external link.
- **Backlink**: a build-time list of documents that contain a wiki link to the
  current document.
- **Article behavior**: reusable display behavior on an individual content
  page, such as its table of contents, backlinks, or metadata.
- **First-time visitor**: a reader who first encounters the site's unifying
  question, then needs a clear path to understand its owner and current work.
- **Deep reader**: a reader who intentionally explores the connected corpus of
  notes, projects, creative work, and spiritual study.
- **Layered entry**: the information-architecture model in which the home page
  first presents the site's unifying question, then offers the owner's
  professional identity and deliberate paths into the knowledge garden.
- **Thematic gateway**: the home page's role as a short, invitation-style
  manifesto. It introduces the unifying question and routes readers into the
  corpus; it is not a complete standalone essay.
- **Reader path**: a public route organized around a reader's intent, rather
  than the repository folder that stores an item. The four paths are Research
  and Teaching, Notes and Writing, Faith and Spirituality, and Projects and
  Creation.
- **Reader-path key**: the permanent machine-readable identifier of a reader
  path: `research-teaching`, `notes-writing`, `faith-spirituality`, or
  `projects-creation`. Display names are localized separately.
- **Reader navigation**: the public organization of content through reader
  paths. It may draw from more than one content collection without moving the
  underlying source files. Reader paths are the shared primary language of the
  home page and global navigation; collection indexes remain complete archives
  reached within a path.
- **Curated landing page**: a reader-path page that introduces the path,
  presents a small editorial selection of representative work, and links to
  the complete chronological archive. It is distinct from an automatic index.
- **Reader-path metadata**: structured front matter that assigns an item to
  one or more reader paths and can mark it as representative work. Curated
  landing pages aggregate this metadata rather than maintaining duplicate
  hand-written link lists. Representative work is ordered by date, newest
  first; the site has no permanent-pinning mechanism.
- **Path-specific representative work**: an item is representative only for
  selected reader paths that it already belongs to. A curated landing page
  reads that path-specific selection and orders it by date, newest first.
- **Complete classification**: a migration requirement that every existing
  published item receives reader-path metadata before the reader-path launch;
  every new item must include it from launch onward.
- **Multi-path membership**: an item may belong to more than one reader path
  without duplicating its source file. Reader paths describe valid ways to
  enter the corpus, not mutually exclusive content types.
- **Template residue**: inherited Academic Pages demonstrations, sample posts,
  or generic pages that are not authored site content. Template residue is
  removed without archival or reader-path classification; the README retains
  the attribution to Academic Pages.
- **Bilingual experience**: the site can switch its interface and, where
  available, display a corresponding Chinese or English version of content.
- **Original-language fallback**: when the selected interface language has no
  translation of the current content, the site displays that content in its
  original language rather than hiding it or substituting a machine-generated
  version.
- **Translation pair**: one original work and zero or one explicitly linked
  Chinese or English translation. The pair remains one conceptual item for
  reader paths, representative-work selection, and backlinks; its versions
  are not unrelated duplicate articles.
- **Content ID**: a language-neutral, permanent identifier shared by every
  version in a translation pair. It remains stable when titles, filenames,
  URLs, or reader-path membership change.
- **Language-aware wiki link**: existing wiki-link source syntax remains
  compatible, but its destination resolves to the reader's selected-language
  version of the linked content when available, otherwise to the original
  language.
- **Language-aware backlink**: backlinks are deduplicated by content ID and
  present the version matching the reader's language preference when available,
  otherwise the original-language version.
- **Published-content contract**: every published item must provide its content
  ID, reader-path metadata, and bilingual discovery metadata. Build-time
  validation reports and rejects any published item that violates this
  contract.
- **Deferred translation**: a translation may be published after its original
  work. The absence of a translation does not block publication and invokes
  original-language fallback when a reader selects the other interface
  language.
- **Language preference**: the selected interface language. On a first visit,
  it follows the browser's language preference; a reader's explicit selection
  persists for later visits.
- **Bilingual discovery metadata**: every published item provides Chinese and
  English titles and short summaries for listings and navigation, even when a
  full translation is deferred. A listing labels the original language when it
  must link to an untranslated work.
- **Reviewed translation**: a public translation that may have used AI for a
  draft but has been reviewed by the author. Unreviewed machine output is not
  a publishable content version.
- **Reader-path launch readiness**: every existing published item has complete
  reader-path classification and bilingual discovery metadata. Full-text
  translations remain optional and do not block the launch.
- **Migration defaults**: Publications and Teaching begin in Research and
  Teaching; Spirits begin in Faith and Spirituality; Projects begin in Projects
  and Creation. Notes require item-by-item reader-path classification. Defaults
  assist the migration but do not replace explicit metadata or multi-path
  membership.
- **Home-page hierarchy**: the thematic gateway appears first, followed by the
  four reader paths with equal visual weight; professional identity and CV are
  a subsequent, available layer rather than the initial framing.
- **Professional identity layer**: a concise, verifiable introduction to the
  site's owner with a CV route. It follows the reader paths on the home page
  while remaining directly reachable from global navigation.
- **Primary navigation**: the four reader paths, the professional identity
  layer, and language switching. Tags are a secondary discovery tool reached
  from content and path pages rather than a primary-navigation destination.
- **Editorial design layer**: a site-owned visual system that expresses the
  thematic gateway and reader paths while preserving the Jekyll site's
  responsive, readable article foundation. It evolves independently of the
  inherited Academic Pages theme.
- **Editorial visual tone**: restrained, text-first, and rhythm-led. Bilingual
  typography, spacing, hierarchy, and reader-path markers carry the identity;
  decorative utopian imagery does not substitute for the site's ideas or work.

## Product constraints

- Content durability and readable URLs matter more than matching upstream
  Academic Pages exactly.
- GitHub Pages/Jekyll keeps the public site static. New capabilities should
  usually compile at build time or progressively enhance the rendered page.
- The site is bilingual in practice. New labels and navigation should be
  intentionally bilingual or consistently scoped, rather than accidentally
  mixing languages.
- The front page is a personal statement; do not replace it with generic
  portfolio/template copy without an explicit editorial decision.

## Current priorities

1. Make custom behavior discoverable, localized, and verifiable.
2. Preserve source compatibility for existing Markdown and public URLs.
3. Modernize template residue only when a replacement is deliberate and can
   be checked locally.
4. Treat a future design refresh, search, or content reclassification as
   separate product decisions, not incidental cleanup.
