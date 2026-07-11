/** Shared categorical line palette for the overlaid-comparison charts, so a
 *  model's line colour matches its swatch/topology wherever it appears. */
export const CHART_PALETTE = [
  '#2563eb',
  '#dc2626',
  '#16a34a',
  '#d97706',
  '#7c3aed',
  '#0891b2',
  '#db2777',
  '#65a30d',
]

export const paletteColor = (i: number): string => CHART_PALETTE[i % CHART_PALETTE.length]
