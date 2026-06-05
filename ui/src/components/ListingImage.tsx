import { useState } from 'react'

/**
 * Hotlinked listing photo with a graceful no-photo fallback. Source pages
 * rot and CDNs block hotlinks; a broken image must degrade to the same
 * quiet placeholder as a listing that never had photos.
 */
export default function ListingImage({
  src,
  alt,
  className,
}: {
  src: string | undefined
  alt: string
  className?: string
}) {
  const [broken, setBroken] = useState(false)
  // A new URL (gallery swap) gets a fresh chance to load: derived-state
  // reset during render, not an effect.
  const [prevSrc, setPrevSrc] = useState(src)
  if (src !== prevSrc) {
    setPrevSrc(src)
    setBroken(false)
  }

  if (!src || broken) {
    return (
      <span
        className={`listing-noimg ${className ?? ''}`}
        role="img"
        aria-label={src ? 'Photo unavailable' : 'No photos'}
        title={src ? 'Photo unavailable' : 'No photos'}
      >
        <svg viewBox="0 0 24 24" aria-hidden width="20" height="20">
          <rect x="3" y="5" width="18" height="14" rx="1.5" fill="none" stroke="currentColor" />
          <circle cx="9" cy="10" r="1.6" fill="currentColor" />
          <path d="M3.5 17.5 9 13l3.5 3 4-4.5 4 6" fill="none" stroke="currentColor" />
        </svg>
      </span>
    )
  }
  return (
    <img
      className={className}
      src={src}
      alt={alt}
      loading="lazy"
      referrerPolicy="no-referrer"
      onError={() => setBroken(true)}
    />
  )
}
