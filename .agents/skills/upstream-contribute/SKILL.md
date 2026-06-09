---
name: upstream-contribute
description: Upstream this instance's improvements to lensing framework files back to the mother repository as a pull request, with changes generalized (domain literals de-instantiated into render placeholders / domain.toml keys) so other instances can pull them via upstream-sync and re-adapt. Spawns the upstream-contribute agent. Use when the user wants to contribute a fix or improvement upstream, file a PR against the mother lensing repo, or see which local framework changes are worth upstreaming. Read-only scan by default; nothing is pushed without an explicit approval gate.
user-invocable: true
argument-hint: "[scan|contribute] [mother-repo-url-or-path] [what to upstream]"
allowed-tools:
  - Task
---
<!-- GENERATED from agents-src/skills/upstream-contribute/SKILL.md by agents-src/render.py — edit the template (and domain.toml), not this file; then run `zig build render-agents`. -->

Contribute this instance's framework improvements back to the mother
lensing repository — the inverse of `/upstream-sync`. The instance owns its
domain and empirical record; what flows upstream is framework work (better
predictors, sharper agent templates, build fixes, ui polish), generalized
so any other instance can pull it and re-adapt. All the real work is done
by the `upstream-contribute` agent; this skill spawns it, relays the gated
proposal, and forwards the user's approval.

## Modes

| Mode | What it does | Gate |
|---|---|---|
| **scan** (default) | Hash framework files against `.lensing-upstream.json`, report contribution candidates + how each would generalize — no clone, no push, no commits | read-only |
| **contribute** | Clone the mother into a temp dir, apply the generalized changes on a `contrib/…` branch, verify against the mother's neutral domain.toml, then STOP with the full proposal; push + `gh pr create` only after approval | every outward action stops for approval |

## Concepts

- **Candidate discovery**: the instance's `.lensing-upstream.json`
  provenance manifest records the sha256 of every framework file as
  shipped. A local file whose hash differs is an instance-modified
  framework file — an upstream candidate. Instance-owned files
  (`domain.toml`, `models.toml`, `registry.toml`, `experiments/PROJECT-FACTS.md`,
  `experiments/`, `data/`, the seeded docs) are never candidates.
- **Generalization**: the agent inverts the render placeholders against
  this instance's `domain.toml` and replaces every domain literal in the
  contribution (entity nouns, target field, collection names, metrics,
  URLs) with its placeholder or a `domain.toml`-routed value. Anything that
  can't be generalized (experiment results, dataset names, secrets) is a
  listed blocker, never shipped.
- **Rendered-edit redirection**: edits found in generated `.claude/`,
  `.agents/`, `.gemini/` outputs or `CLAUDE.md` are lifted back into their
  `agents-src/` template source — rendered files are never contributed
  directly.
- **Mother repo**: an explicit URL/path argument wins; else the manifest's
  `source_repo` field; else the agent asks. GitHub remotes get a full
  `gh pr create` flow (fork fallback included); other remotes get a pushed
  branch + manual instructions; a local path gets a prepared branch, no
  push.

## How to run it

1. Parse the argument: mode (`scan` if omitted), an optional mother repo
   spec, and what the user wants upstreamed (verbatim).
2. Precondition: a clean git worktree — the agent refuses dirty trees. If
   `git status` shows uncommitted changes, tell the user to commit or stash
   before spawning.
3. Spawn the agent via the Task/Agent tool with
   `subagent_type: upstream-contribute` and a prompt of the form:

   > Mode: <mode>. Mother repo: <spec, or "resolve from manifest">.
   > <what to upstream / extra user instructions, verbatim, if any>

   Do not re-explain the contribution conventions in the prompt — the
   agent definition carries them.
4. In `contribute` mode the agent prepares the branch, verifies it, returns
   a **proposal** (generalized diffs, literal→placeholder mapping, PR body
   preview), and STOPS. Present it to the user as-is.
5. On approval, continue the SAME agent (SendMessage) with `APPROVED`
   (plus any scoping the user added, e.g. "APPROVED change 1 only") — do
   not spawn a fresh one, it would lose the temp clone and classification
   state. Only then does it push and open the PR.
6. Relay the result: PR URL(s), or the pushed branch + manual
   instructions, plus any blockers that were dropped.

## Notes

- "Nothing framework-shaped has diverged — nothing to contribute" is a
  valid outcome; tell the user, don't re-spawn hoping for different
  results.
- An edit made only to a rendered file (template unchanged) is non-durable
  — the instance's own next `zig build render-agents` clobbers it. The
  agent reports these and offers to contribute the template equivalent.
- Instances packaged before `source_repo` existed in the manifest simply
  get asked for the mother repo — not an error.
- One PR per logical change is the default; ask for a single batch PR in
  the extra instructions if preferred.
