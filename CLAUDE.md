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
