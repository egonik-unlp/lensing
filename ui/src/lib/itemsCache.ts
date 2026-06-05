// Per-dataset payload caches. items.json and split membership are immutable
// per dataset; cache the promises across component mounts.

import { api } from '../api/client'
import type { DatasetSplit, Items } from '../api/types'

const items = new Map<string, Promise<Items>>()

export function loadItems(datasetId: string): Promise<Items> {
  let p = items.get(datasetId)
  if (!p) {
    p = api.getItems(datasetId)
    items.set(datasetId, p)
  }
  return p
}

const splits = new Map<string, Promise<DatasetSplit>>()

export function loadSplit(datasetId: string): Promise<DatasetSplit> {
  let p = splits.get(datasetId)
  if (!p) {
    p = api.getDatasetSplit(datasetId)
    splits.set(datasetId, p)
  }
  return p
}
