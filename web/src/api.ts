
const API_URL = "http://localhost:5000"
const API_ENDPOINTS = {
  GET_RECORDS: "/get_records",
  GET_PROOF: "/get_proof",

  PREDICT: "/predict",
  VERIFY: "/verify",
}

export const sendPrediction = async (b64input: string, errCallback?: () => void) => {
  try {
    const url = API_URL + API_ENDPOINTS.PREDICT
    const data = await fetch(url, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input: b64input })
    })
    return await data.json()
  } catch (err) {
    console.error(err)
    if (errCallback) errCallback()
  }
}

export const fetchPredictionRecords = async () => {}