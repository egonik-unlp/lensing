import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { DatasetBuildForm } from '../components/DatasetBuildForm'
import ViewHeader from '../components/ViewHeader'

/** Dedicated build page: datasets are first-class, so creating one doesn't
 *  require setting up a run. On completion, lands on the new dataset. */
export default function NewDatasetView() {
  useEffect(() => {
    document.title = 'Build dataset · Price Guesser Models'
  }, [])
  const navigate = useNavigate()

  return (
    <section aria-label="Build a dataset">
      <ViewHeader
        glyph="ds"
        title="Build dataset"
        crumbs={[{ label: 'Datasets', to: '/datasets' }, { label: 'new' }]}
        meta={
          <span>
            Pulls sale listings from Qdrant, fits PCA on the train rows, encodes features and
            freezes a reproducible split.
          </span>
        }
      />
      <div className="view-body newrun">
        <DatasetBuildForm onBuilt={(id) => navigate(`/datasets/${id}`)} />
      </div>
    </section>
  )
}
