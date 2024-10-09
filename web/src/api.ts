import { PredictionRecord } from "./types"

export const API_REQ_ERR_TIMEOUT_MS = 4000

export const API_URL = "http://localhost:5000"
export const API_ENDPOINTS = {
  GET_RECORDS: "/get_records",
  GET_PROOF: "/get_proof",
  GET_SRS: "/get_srs",
  GET_VK: "/get_vk",

  PREDICT: "/predict",
  VERIFY: "/verify",
}

export const sendPrediction = async (b64input: string, errCallback?: () => void) => {
  try {
    const url = API_URL + API_ENDPOINTS.PREDICT
    const res = await fetch(url, {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: b64input }),
    })
    return await res.json()
  } catch (err) {
    console.error(err)
    if (errCallback) errCallback()
  }
}

export const verifyPreidction = async (proof: File, errCallback?: () => void): Promise<boolean> => {
  try {
    const url = API_URL + API_ENDPOINTS.VERIFY
    const formData = new FormData()
    formData.append("file", proof)
    const res = await fetch(url, {
      method: "POST",
      body: formData,
    })
    const json = await res.json()
    return json.verified as boolean
    
  } catch (err) {
    console.error(err)
    if (errCallback) errCallback()
    return false
  }
}

export const fetchPredictionRecords = async (page?: number): Promise<PredictionRecord[]> => {
  try {
    const url = API_URL + API_ENDPOINTS.GET_RECORDS
    const res = await fetch(url + (page ? `?page=${page}` : ''))
    const json = await res.json()
    return json.records as PredictionRecord[]
  } catch (err) {
    console.error(err)
    return []
  }
}

export const downloadFile = async (endpoint: string, query: string = "") => {
  const url = API_URL + endpoint + query
  window.location.href = url
}