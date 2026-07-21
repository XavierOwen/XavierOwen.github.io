# ADR 0002: Preserve existing URLs when adding translations

Existing published URLs remain the stable address of their original work. A
corresponding translation receives a target-language-prefixed URL, such as
`/en/...` for an English translation of an existing Chinese item. This accepts
asymmetric language paths in exchange for preserving established links and
keeping the bilingual Jekyll migration incremental.

## Considered options

- Move all content to symmetric `/zh/...` and `/en/...` URLs: visually
  consistent, but requires broad redirects and changes the canonical address
  of the existing corpus.
- Render both languages at one URL: avoids extra paths, but makes static
  indexing, translation pairing, and direct linking less clear.
