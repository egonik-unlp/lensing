import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'

import '@fontsource-variable/inter/index.css'
import '@fontsource-variable/sora/index.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/600.css'
import './styles/tokens.css'
import './styles/base.css'

import { DomainProvider } from './lib/DomainContext'
import { CompareProvider } from './lib/CompareContext'
import Shell from './components/Shell'
import RunsView from './views/RunsView'
import NewRunView from './views/NewRunView'
import RunDetailView from './views/RunDetailView'
import CompareView from './views/CompareView'
import PredictorDetailView from './views/PredictorDetailView'
import ModelsView from './views/ModelsView'
import ModelDetailView from './views/ModelDetailView'
import DefinitionsView from './views/DefinitionsView'
import DefinitionDetailView from './views/DefinitionDetailView'
import DatasetsView from './views/DatasetsView'
import DatasetDetailView from './views/DatasetDetailView'
import NewDatasetView from './views/NewDatasetView'
import ListingsView from './views/ListingsView'
import NewListingView from './views/NewListingView'
import ListingDetailView from './views/ListingDetailView'
import InterpretabilityView from './views/InterpretabilityView'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Shell />,
    children: [
      { index: true, element: <RunsView /> },
      { path: 'new', element: <NewRunView /> },
      { path: 'runs/:runId', element: <RunDetailView /> },
      { path: 'compare', element: <CompareView /> },
      { path: 'predictors/:name', element: <PredictorDetailView /> },
      { path: 'datasets', element: <DatasetsView /> },
      { path: 'datasets/new', element: <NewDatasetView /> },
      { path: 'datasets/:id', element: <DatasetDetailView /> },
      { path: 'models', element: <ModelsView /> },
      { path: 'models/:name', element: <ModelDetailView /> },
      { path: 'definitions', element: <DefinitionsView /> },
      { path: 'definitions/:name', element: <DefinitionDetailView /> },
      { path: 'listings', element: <ListingsView /> },
      { path: 'listings/new', element: <NewListingView /> },
      { path: 'listings/:id', element: <ListingDetailView /> },
      { path: 'listings/:id/edit', element: <NewListingView /> },
      { path: 'interpretability', element: <InterpretabilityView /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DomainProvider>
      <CompareProvider>
        <RouterProvider router={router} />
      </CompareProvider>
    </DomainProvider>
  </StrictMode>,
)
