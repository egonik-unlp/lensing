// App-wide domain config, fetched once at boot from GET /api/domain. The shell
// is gated behind a successful fetch so every view can read the domain
// synchronously via useDomain() without null checks.

import { createContext, useContext, type ReactNode } from 'react'
import { fetchDomain, type Domain } from './domain'
import { useAsync } from '../hooks/useAsync'

const DomainCtx = createContext<Domain | null>(null)

// eslint-disable-next-line react-refresh/only-export-components
export function useDomain(): Domain {
  const d = useContext(DomainCtx)
  if (!d) throw new Error('useDomain() called outside DomainProvider')
  return d
}

export function DomainProvider({ children }: { children: ReactNode }) {
  const domain = useAsync(() => fetchDomain(), [])

  if (domain.error) {
    return (
      <div className="domain-boot domain-boot-error" role="alert">
        Could not load domain config: {domain.error}{' '}
        <button className="btn" onClick={domain.reload}>
          Retry
        </button>
      </div>
    )
  }
  if (domain.loading || !domain.data) {
    return (
      <div className="domain-boot" role="status">
        loading…
      </div>
    )
  }

  return <DomainCtx.Provider value={domain.data}>{children}</DomainCtx.Provider>
}
