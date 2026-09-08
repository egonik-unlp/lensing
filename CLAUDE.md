<!-- LENSING-TEMPLATE: bootstrap pending -->
# Lensing template — bootstrap pending

This is an unconfigured lensing instance with NO domain of its own yet.
The FIRST action in this repository is `/bootstrap` (or follow
BOOTSTRAP.md). This instance is fully independent: it must never read
from or write to any other lensing project's corpus, datasets, server,
or experiment record.

Until bootstrap completes:

- `domain.toml` is a neutral placeholder (placeholder fields, no real
  corpus bindings); the generated agent layer (`.claude/`, `.agents/`,
  `.gemini/`) renders from it and is equally unconfigured. A complete
  worked example of domain.toml lives at
  `crates/lensing-core/src/example-domain.toml`.
- `zig build render-agents` regenerates the agent layer from
  `domain.toml` but leaves this stub in place. Once `domain.toml`
  describes the new domain, bootstrap finishes with
  `rm CLAUDE.md && zig build render-agents`, which writes the real
  project instructions here.
- The empirical record (models.toml, experiments/PROJECT-FACTS.md, docs
  skeletons) is seeded empty; the first campaign writes it.

## The instantiation rules (enforced, not advisory)

`tools/lensing_guard.py` decides whether this tree is a valid, finished
instance; the hooks in `.claude/settings.json` enforce its verdict. Run
`zig build bootstrap-check` (or `python3 tools/lensing_guard.py check`)
to see it at any time.

1. **This instance owns its folder.** It must be a full copy produced by
   `zig build package` and unpacked somewhere of its own — never the
   lensing framework checkout, never a subdirectory of it, never a
   partial tree. The provenance manifest `.lensing-upstream.json` is what
   proves it, and every framework file it lists must exist. A tree that
   fails this is `INVALID` and no work may proceed in it.
2. **Bootstrap always finishes.** While the instance is unbootstrapped,
   `/bootstrap` is the only work permitted: writes outside bootstrap's own
   files are denied, and so is every API mutation. Once bootstrap starts
   (`python3 tools/lensing_guard.py bootstrap start`) the session may not
   end until the invariants pass — or until the user's decision to stop is
   recorded with `bootstrap abort --reason "<why>"`. Do not improvise on
   the placeholder domain, and do not read another lensing project's
   corpus, datasets, server or experiment record to fill the gaps.
3. **Explorations go through the agents.** After bootstrap, run launches
   and dataset builds require a run ticket, so an experiment cannot be
   started outside the experiment-designer → experiment-runner path that
   writes the report and reconciles experiments/PROJECT-FACTS.md. A single
   ad-hoc run the user explicitly asked for is the one exception
   (`ticket issue --kind oneshot --reason "<what they asked for>"`).
