# 0006: Keep the family planner's records encrypted in the browser

Status: Accepted
Date: 2026-10-29

## Context

The pregnancy and newborn planner belongs to the existing Projects collection,
under Märchen. The site and its source repository are public. The requested
private checklist needs to retain dates, notes and measurements between visits
without adding an account service to GitHub Pages.

## Decision

Publish the general checklist and its source links as public content. Store
personal records only in an encrypted browser-local vault. The user creates a
family passphrase on their own device. PBKDF2-SHA256 derives an AES-GCM key.
Each write uses a fresh nonce. The page persists neither the passphrase nor the key.
Reloading the page requires unlocking the vault again.

Use a dedicated Jekyll layout, a data file and native browser controls. This
layout loads only local styles and scripts, keeping the article layout's remote
libraries out of the page that handles private records. It retains the site's
content metadata and language discovery contract.

Offer encrypted backup export and restore. Check the backup's structure and
decrypt it before replacing the current vault. When records already exist,
offer their encrypted backup and require confirmation before replacement.
Report storage failures and changes made in another tab instead of silently
overwriting records.

## Consequences

- The general guide, source links and project listing are public. The unlock
  screen does not provide server-side access control for that content.
- Records belong to one browser profile and origin. They do not synchronize
  between devices. Moving records requires an encrypted backup and its
  passphrase.
- Clearing browser storage removes the local copy. Losing the passphrase makes
  that copy and its backups unreadable. There is no password reset service.
- Encryption protects stored records. It does not protect an unlocked page
  from someone using that device or from compromised code on the same origin.
- The checklist estimates dates from user-entered anchors. It does not diagnose
  measurements or replace individual care instructions. Medical windows and
  local service information carry source links and a review date.
- Supplement settings share the encrypted vault. The calculator multiplies a
  user-entered, clinician-confirmed daily coefficient by the specified weight;
  it provides no default coefficients or automatic prescribing. Public nutrient
  references describe total intake and do not scale with weight. Older version-1
  records open with empty calculator settings.
