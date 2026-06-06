---
name: report-curator
description: Maintain the living scientific report in both its registers (docs/experiments.tex technical + docs/experiments-v2.tex plain-language + docs/figures/make_figures.py → docs/experiments.pdf + docs/experiments-v2.pdf) consolidating the {{report_dir}}/*.md campaign reports. Spawns the report-curator agent. Use when the user wants to update or sync the experiment PDFs, fold new experiment reports into the documents, add a figure (heatmaps for parameter scans), add or extend the theory/background section, refresh the leaderboard, polish the prose, or rebuild the report.
user-invocable: true
argument-hint: "[sync|polish|rebuild] [extra instructions]"
allowed-tools:
  - Task
---

Maintain the **living experiment synthesis** — one scientific story in two
registers (`docs/experiments.tex` technical, `docs/experiments-v2.tex`
plain-language; both always covering the same campaigns), updated and
improved over time, never regenerated per-experiment. All the real work is
done by the `report-curator` agent; this skill just spawns and relays it.

## Modes

| Mode | What it does | Gate |
|---|---|---|
| **sync** (default) | Detect `{{report_dir}}/*.md` reports not yet reflected in each document, fold each in additively into BOTH registers (section + transcribed tables + figures + leaderboard/abstract refresh), rebuild both PDFs | proceeds freely |
| **polish** | Quality pass, no new source material: prose, captions, cross-references, missing figures (heatmaps for 2-D parameter scans) | proceeds freely |
| **rebuild** | Full re-derivation of the document from the report corpus | always stops for approval |

Regardless of mode, **destructive changes stop for approval**: removing or
rewriting existing sections, deleting figures, rewriting the abstract's
thesis, or archiving/moving any `{{report_dir}}/*.md`. Additive work never
stops.

## How to run it

1. Parse the argument: mode (`sync` if omitted) plus any extra instructions
   (e.g. a specific figure or background topic the user asked for).
2. Spawn the agent via the Task/Agent tool with
   `subagent_type: report-curator` and a prompt of the form:

   > Mode: <mode>. <extra user instructions, verbatim, if any>

   Do not re-explain the document conventions in the prompt — the agent
   definition carries them.
3. Relay the agent's summary to the user: reports folded in, sections/figures
   added, leaderboard/abstract deltas, build result (tectonic exit, page and
   figure counts), sync status.
4. If the agent returns a **destructive proposal**, present it to the user
   as-is. On their approval, continue the SAME agent (SendMessage) with
   `APPROVED` (plus any scoping the user added) — do not spawn a fresh one,
   it would lose the prepared state.

## Notes

- If the agent reports "fully synced, nothing to fold in", that is a valid
  outcome — tell the user, don't re-spawn hoping for different results.
- The document's source of truth is `{{report_dir}}/*.md`; if the user wants a
  *new experiment* run and reported, that's the experiment-designer →
  experiment-runner agents / `/model-definitions experiment`, not this skill.
- Build requires `tectonic` (~/.local/bin) and python3 + matplotlib — the
  agent runs `bash docs/build.sh` and reports failures verbatim.
