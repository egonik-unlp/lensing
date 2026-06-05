**Report format** — `{{report_dir}}/<YYYY-MM-DD>-<slug>.md`, matching the
existing files:

```
# <predictor>: <axis> scan (<date>)

Goal: … (fixed baseline: dataset id, held hyperparams, reference champion
with its metrics and run id)

Outcome in one line: **<key finding> — <{{metric_columns_slash}}> —
definition `<name>`**

## Results (sorted by {{primary_metric}}; all on <ds-id>)
| config | {{metric_columns_md}} | run |
(bold the winner row and the best cell per metric; include failed runs)

## Findings
(interpretation: why the winner wins, trade-offs, failure modes, saturation)

## Best on record after this work
| model | {{metric_columns_md}} | run |   ← winner vs. previous champions

## Follow-ups
- next experiments worth running
```

Conventions: {{primary_metric}} in raw {{target_noun}} units with thousands
separators; percentage metrics rendered as %; a run id in every table row;
name the dataset id; reference prior experiments by filename when building
on them.
