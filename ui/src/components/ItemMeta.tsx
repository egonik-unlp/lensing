import { useMemo } from 'react'
import type { Item } from '../api/types'
import { useDomain } from '../lib/DomainContext'
import { cappedNumericField, categoricalFields, fieldLabel } from '../lib/domain'

/** Domain-driven metadata list for one corpus item: the prominent
 *  categorical fields, the capped numeric, and the cluster label when
 *  present. Shared by the item-detail drilldowns. */
export default function ItemMeta({ item }: { item: Item }) {
  const domain = useDomain()
  const cats = useMemo(() => categoricalFields(domain).slice(0, 3), [domain])
  const capped = useMemo(() => cappedNumericField(domain), [domain])
  const str = (name: string) => {
    const v = item[name]
    return typeof v === 'string' ? v : v != null ? String(v) : ''
  }
  return (
    <dl className="item-meta">
      {cats.map((f) => (
        <div key={f.name}>
          <dt>{fieldLabel(f)}</dt>
          <dd>{str(f.name) || '—'}</dd>
        </div>
      ))}
      {capped && (
        <div>
          <dt>{fieldLabel(capped)}</dt>
          <dd className="num">{str(capped.name) || '—'}</dd>
        </div>
      )}
      {item.cluster_label && (
        <div>
          <dt>cluster</dt>
          <dd>{item.cluster_label}</dd>
        </div>
      )}
    </dl>
  )
}
