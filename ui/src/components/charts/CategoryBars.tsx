import type { CategoryStat } from '../../lib/stats'
import { fmtMoney, fmtPct } from '../../lib/format'
import './charts.css'

interface Props {
  data: CategoryStat[]
  ariaLabel: string
  /** Drill-down: select a category to inspect its items. */
  onSelect?: (value: string) => void
  selected?: string | null
}

/** Horizontal category bars rendered as an HTML list: label, count bar,
 *  count, share and median target, all keyboard-reachable when selectable. */
export default function CategoryBars({ data, ariaLabel, onSelect, selected }: Props) {
  if (data.length === 0) return <div className="chart-empty">No categories</div>
  const max = Math.max(...data.map((d) => d.count))

  return (
    <ul className="catbars" aria-label={ariaLabel}>
      {data.map((d) => {
        const row = (
          <>
            <span className="catbar-label" title={d.value}>
              {d.value}
            </span>
            <span className="catbar-track" aria-hidden>
              <span className="catbar-fill" style={{ width: `${(d.count / max) * 100}%` }} />
            </span>
            <span className="num catbar-count">{d.count.toLocaleString()}</span>
            <span className="num catbar-share">{fmtPct(d.share, 1)}</span>
            <span className="num catbar-median">{fmtMoney(d.medianTarget)} med</span>
          </>
        )
        return (
          <li key={d.value}>
            {onSelect ? (
              <button
                className={`catbar-row${selected === d.value ? ' is-selected' : ''}`}
                onClick={() => onSelect(d.value)}
                aria-pressed={selected === d.value}
                title={`Inspect ${d.value} items`}
              >
                {row}
              </button>
            ) : (
              <span className="catbar-row">{row}</span>
            )}
          </li>
        )
      })}
    </ul>
  )
}
