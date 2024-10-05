
export type Status = 'loading' | 'ready' | 'error'

export type RequestStatus = Status | 'init' | 'success'

export type DrawBlockType = 'fill' | 'stroke'

export interface Point {
  x: number
  y: number
}

export interface PredictionRecord {
  input: string
  id: number
  prediction_res: number
  timestamp: number
  proof?: string
}