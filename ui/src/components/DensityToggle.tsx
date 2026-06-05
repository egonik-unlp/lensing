import type { Density } from '../hooks/useDensity'

export default function DensityToggle({
  density,
  onToggle,
}: {
  density: Density
  onToggle: () => void
}) {
  return (
    <button
      className="btn-inline"
      onClick={onToggle}
      aria-pressed={density === 'compact'}
      title="Toggle table row density"
    >
      {density === 'compact' ? 'comfortable rows' : 'compact rows'}
    </button>
  )
}
