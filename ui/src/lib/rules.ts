// Data-quality rule names → display labels. Shared between the build form's
// preflight preview and the dataset detail quality report.

import type { Domain } from './domain'

/** The domain's quality.rule_labels is authoritative; an unlabeled rule
 *  falls back to its raw key. */
export function ruleLabel(rule: string, domain: Domain): string {
  return domain.quality.rule_labels[rule] ?? rule
}
