// Data-quality rule names → display labels. Shared between the build form's
// preflight preview and the dataset detail quality report.

import type { Domain } from './domain'

/** Static fallback for call sites without a domain in scope. The domain's
 *  quality.rule_labels is authoritative; prefer ruleLabel(rule, domain). */
export const RULE_LABEL: Record<string, string> = {
  'nonpositive-price': 'price ≤ 0',
  'price-outlier': 'price outlier',
  'missing-fields': 'missing fields',
  'price-range': 'price out of range',
  'bedrooms-outlier': 'bedrooms outlier',
  'duplicate-content': 'duplicate listing',
  'short-content': 'thin description',
  'foreign-currency': 'foreign currency',
}

export function ruleLabel(rule: string, domain?: Domain): string {
  return domain?.quality.rule_labels[rule] ?? RULE_LABEL[rule] ?? rule
}
