---
name: showcase
description: Publish a promoted {{target_noun}}-prediction model as a small, standalone, Cloudflare-Worker-compatible demo site — a showcase. This skill helps the implementer decide how to tell a compelling story with the model and data (which model, what framing, which example items, live-input vs. curated gallery, the look), assembles a showcase brief, and spawns the showcase-builder agent to do all the plumbing (ONNX export, featurizer wiring, embeddings, Worker scaffold, build, smoke test, deploy steps). Use when the user wants to demo, showcase, publish, or build a standalone/landing/marketing site or Worker for a model, "put a model on a site", or make an interactive prediction demo.
user-invocable: true
argument-hint: "[model-name] [extra framing instructions]"
allowed-tools:
  - Task
  - Read
  - Glob
  - Bash(curl *)
---

Turn a trained model into a **standalone showcase** — a small
Cloudflare-Worker-compatible app that publishes one promoted model to a live
site so people can see the algorithm work. **You own the story; the
`showcase-builder` agent owns the plumbing.** Your job here is to help the
implementer make the few decisions that shape a compelling demo, fold them into
a brief, and hand that brief to the agent verbatim — never to hand-wire ONNX,
embeddings, or wrangler config yourself.

## 1. Know what's available

Before deciding anything, see which models can actually be shown (the server
must be running, default `{{api_base_url}}`):

```sh
curl -s {{api_host}}/api/health
curl -s {{api_host}}/api/best-models   # the server's top-{{best_models_size}} by {{primary_metric}}
curl -s {{api_host}}/api/models        # all promoted models
```

The strongest demo is usually a **best-models** member (it leads on
{{primary_metric}}; see `{{facts_file}}` for the champion and its error band).
Note: not every family is ONNX-exportable — svm-moe, svm-quantile-moe and the
flux-* families can't be shipped standalone yet; the agent will reject them, so
steer toward an exportable one (nets, xgboost, lightgbm, random-forest, ridge,
svm, kernel-ridge, or an all-exportable blend).

## 2. Decide the story (with the implementer)

These are the choices that make a demo land. Settle them conversationally — the
user may have stated some already; ask only for what's missing and propose
sensible defaults rather than interrogating:

- **Which model** — a promoted model name. Default to the current champion.
- **Mode** — how visitors interact:
  - **curated** *(default)* — a fixed gallery of real example {{entity_noun_plural}}
    with their embeddings baked in. No API key, nothing to break, predicts
    instantly, and can show **predicted vs. actual** on the model's test split
    (the most honest, persuasive framing). Best for a public/marketing demo.
  - **live** — visitors type their own {{entity_noun}} text and get a
    {{target_noun}} prediction, embedded on the fly via OpenAI. More interactive,
    but the Worker needs an `OPENAI_API_KEY` secret and every visit costs an
    embedding call. Best for an internal or gated demo.
- **The framing** — title, the one-sentence narrative, and what comparison
  carries it: predicted-vs-actual, the consensus spread across models, or a
  per-feature "why" breakdown. This is the part only the implementer can
  author — pull it out of them ("what's the one thing a visitor should walk away
  understanding?").
- **Examples** (curated) — how many and which (default: 5–6 honest test-split
  items spanning the {{target_noun}} range). The user can name specific
  {{entity_noun_plural}} or a segment.
- **Inference site** — `browser` (fully static, model runs in the visitor's
  tab) or `worker` (runs in the Worker, model bytes stay server-side, clean
  `POST /api/predict`). Default browser for curated, worker for live. Most
  implementers don't care — pick the default and move on.
- **Look** — a clean functional default ships automatically. If the implementer
  wants a polished/branded site, plan to run `/impeccable` on the generated
  `public/` **after** the agent builds it (don't try to design and plumb in one
  step).

If the user already gave enough ("showcase the champion as a curated demo"),
don't over-ask — fill the rest with defaults and state them.

## 3. Assemble the brief and spawn the agent

Write a compact **showcase brief** capturing the decisions above and spawn the
agent via the Task/Agent tool with `subagent_type: showcase-builder`, brief
verbatim in the prompt:

> **Showcase brief**
> - model: `<name>`
> - mode: curated | live
> - inference site: browser | worker
> - examples: <count + any selection>  *(curated)*
> - framing: <title> — <one-sentence narrative> — show <comparison>
> - app name: `<kebab>` (default `{{project_name}}-demo`)
> - host: Cloudflare
> - look: default | will polish with /impeccable afterward

Don't re-explain export/featurize/wrangler mechanics in the prompt — the agent
definition carries all of it. Let the agent build, wire, and **smoke-test**; it
does not deploy.

## 4. Relay and (optionally) polish

Relay the agent's summary: app path, mode, model + family, the smoke-test
predicted-vs-actual numbers (and whether they matched the server), any warnings,
and the copy-pasteable deploy steps. Then:

- If the implementer wanted a polished look, run `/impeccable` on
  `clients/showcase/<app>/public/` now — the story is in place; this makes it
  beautiful.
- Deploying is the implementer's call (it publishes to *their* Cloudflare
  account). Hand them the `npx wrangler deploy` step and any secret to set; do
  not deploy for them.

## Notes

- The model is exported read-only and the showcase is fully self-contained under
  `clients/showcase/<app>/` — building one never touches `data/`, the corpus, or
  the running server's models. Re-run any time to refresh a demo after promoting
  a better model.
- One showcase = one model. To compare several models on one page, say so in the
  framing and the agent will use a blend or the consensus endpoint's shape; but
  the default is a focused single-model story.
- If the server isn't running or the chosen model isn't promoted/exportable, the
  agent stops and tells you why — fix that (promote a run, pick an exportable
  family) rather than re-spawning.
