
export type Status = 'loading' | 'ready' | 'error'

export type DrawBlockType = 'fill' | 'stroke'

export interface Point {
  x: number
  y: number
}

export interface PredictionRecord {
  id: number
  prediction: number
  timestamp: number
}