# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub Issues. Use `gh` inside this
clone.

## Conventions

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`
- List: `gh issue list --state open`
- Comment: `gh issue comment <number> --body "..."`
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close: `gh issue close <number> --comment "..."`

The repository is inferred from the `origin` remote:
`XavierOwen/XavierOwen.github.io`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill conventions

When a skill says "publish to the issue tracker", create a GitHub Issue.
When it says "fetch the relevant ticket", run:

```sh
gh issue view <number> --comments
```

## Wayfinding

A wayfinding map is one issue labelled `wayfinder:map`; its decision tickets
are child issues, labelled `wayfinder:research`, `wayfinder:prototype`,
`wayfinder:grilling`, or `wayfinder:task`.

Represent blocking work with GitHub native issue dependencies. If unavailable,
put `Blocked by: #<number>` at the top of the child issue. Claim work by
assigning it to `@me`; resolve it by posting the answer, closing the issue, and
recording the decision in the map.
