# ADR 0004: Separate authoring structure from reader navigation

The existing Jekyll collections remain the durable authoring structure and
preserve their published URLs. The home page and primary navigation instead use
four reader paths—Research and Teaching, Notes and Writing, Faith and
Spirituality, and Projects and Creation—which may aggregate content from more
than one collection. This favors an intentional reader journey and cross-cutting
work over exposing repository folders as the site's public taxonomy.

## Consequences

- Every published work carries stable reader-path metadata and may belong to
  multiple paths without source duplication.
- Each reader path has a curated landing page, while collection indexes remain
  available as complete archives.
- Published-content validation protects the metadata contract; template
  demonstrations are removed rather than included in the migration.
