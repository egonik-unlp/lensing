---
name: upstream-sync
description: Sync upstream lensing framework changes into this bootstrapped instance, stepwise with approval gates, without clobbering instance-owned files (domain.toml, models.toml, experiments/, data/) or working customizations. Spawns the upstream-sync agent. Use when the user wants to upgrade the framework, pull the latest lensing into their instance, apply framework updates, or audit what upstream changed versus their instance. Accepts a path to an unpacked template dir, a .tar.gz, or a lensing git checkout/URL.
user-invocable: true
argument-hint: "[sync|audit] <template-dir|tarball|checkout|git-url> [extra instructions]"
allowed-tools:
  - Task
---
<!-- GENERATED from agents-src/skills/upstream-sync/SKILL.md by agents-src/render.py — edit the template (and domain.toml), not this file; then run `zig build render-agents`. -->

Upgrade this instance to a newer lensing framework — the upstream evolves
(crates, predictors, ui, agents-src templates, build tooling, docs) while
the instance owns its domain and empirical record. All the real work is
done by the `upstream-sync` agent; this skill spawns it, relays each gated
batch proposal, and forwards the user's approvals.

## Modes

| Mode | What it does | Gate |
|---|---|---|
| **sync** (default) | Classify framework files against upstream, apply in gated batches (build/tooling → rust → predictors → ui → agents-src + re-render → docs), one git commit per approved batch | every batch stops for approval |
| **audit** | Classify and report only — no file changes, no commits | read-only |

## Concepts

- **3-way classification**: the instance's `.lensing-upstream.json`
  provenance manifest (written by upstream's `tools/package.py`) records the
  sha256 of every framework file as shipped. Comparing it with the local
  file and the new upstream separates *safe updates* (changed only
  upstream) from *conflicts* (changed both — gated merge) and *local
  customizations* (left alone).
- **Degraded mode**: instances that predate the manifest get diff-driven
  classification — every differing file is treated as a conflict and shown
  in full; nothing auto-applies. The sync writes the manifest so the next
  upgrade is 3-way.
- The server is **never restarted**: the agent commits source changes and
  defers rebuild/restart to the user, after `GET /api/runs` shows no live
  runs.

## How to run it

1. Parse the argument: mode (`sync` if omitted), the upstream source spec
   (required — if missing, ask the user where the newer lensing lives:
   an unpacked template dir, a tarball, or a checkout/git URL), and any
   extra instructions.
2. Precondition: a clean git worktree — the agent refuses dirty trees.
   If `git status` shows uncommitted changes, tell the user to commit or
   stash before spawning.
3. Spawn the agent via the Task/Agent tool with
   `subagent_type: upstream-sync` and a prompt of the form:

   > Mode: <mode>. Upstream source: <spec>. <extra user instructions,
   > verbatim, if any>

   Do not re-explain the sync conventions in the prompt — the agent
   definition carries them.
4. The agent audits one batch, returns a **proposal** (per-file
   classification + diff hunks for conflicts), and STOPS. Present it to the
   user as-is.
5. On approval, continue the SAME agent (SendMessage) with `APPROVED`
   (plus any scoping the user added, e.g. "APPROVED except ui/src/…") — do
   not spawn a fresh one, it would lose the classification and temp-dir
   state. Repeat per batch until the agent reports the final summary.
6. Relay the final summary: per-batch commits, verification results, and
   the deferred actions (rebuild/restart once `/api/runs` is idle; any
   `domain.toml` follow-ups from a schema_version bump).

## Notes

- "Already current — nothing to sync" is a valid outcome; tell the user,
  don't re-spawn hoping for different results.
- Rollback is per batch: each approved batch is one commit; `git revert
  <sha>` undoes it.
- This skill's own rendered file may change mid-sync (the agents-src batch
  re-renders `.claude/.agents/.gemini`) — that is expected, not drift.
- Instance-owned files are never part of a sync: `domain.toml`,
  `models.toml`, `registry.toml`, `experiments/PROJECT-FACTS.md`, `experiments/`,
  `docs/experiments.tex`, `data/`.
