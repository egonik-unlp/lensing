import { useNavigate } from 'react-router-dom'
import ViewHeader from '../components/ViewHeader'
import { RepresentationBuildForm } from '../components/RepresentationBuildForm'
import { useDocTitle } from '../lib/DomainContext'

export default function NewRepresentationView() {
  useDocTitle('New representation')
  const navigate = useNavigate()
  return (
    <section aria-label="New representation">
      <ViewHeader
        glyph="rp"
        title="New representation"
        meta={
          <span>
            Compress a source collection into a dense latent — PCA, an autoencoder, or a sparse
            autoencoder. The latent is written to a new collection that datasets and predictors can
            ride.
          </span>
        }
      />
      <div className="view-body">
        <RepresentationBuildForm onBuilt={(id) => navigate(`/representations/${id}`)} />
      </div>
    </section>
  )
}
