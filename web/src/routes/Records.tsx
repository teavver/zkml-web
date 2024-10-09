import { Nav } from "../components/Nav";
import { Content } from "../components/Content";
import { useEffect, useRef, useState } from "react";
import { downloadFile, fetchPredictionRecords } from "../api";
import { PredictionRecord, RequestStatus } from "../types";
import { API_ENDPOINTS } from "../api";
import { routes } from "../router";
import proofIcon from "../assets/proof.svg"

export const Records = () => {

  const SECOND_IN_MS = 1000
  const BASE_64_PREFIX = "data:image/png;base64,"

  const [status, setStatus] = useState<RequestStatus>('init')
  const [records, setRecords] = useState<PredictionRecord[]>([])
  const [page, setPage] = useState<number>(0)
  const lastPage = useRef(false)

  useEffect(() => {
    handleFetchRecords()
  }, [])

  const handleFetchRecords = async (page?: number) => {
    setStatus('loading')
    const records = await fetchPredictionRecords(page)
    if (records.length > 0) {
      if (page !== undefined) setPage(page)
      lastPage.current = (records.some(r => r.id === 1)) ? true : false
      setStatus('ready')
      setRecords(records)
    } else setStatus('error')
  }

  // const downloadProof = async (recordId: number) => {
  //   const url = API_URL + API_ENDPOINTS.GET_PROOF + `?id=${recordId}`
  //   downloadFile(url)
  // }

  return (
    <Content>
      <Nav title={routes.records.title} />
      <div className="flex gap-2">
        <button disabled={status !== 'ready' || page === 0} onClick={() => handleFetchRecords(page - 1)}>Previous</button>
        <button disabled={status !== 'ready' || lastPage.current} onClick={() => handleFetchRecords(page + 1)}>Next</button>
      </div>
      {status !== 'ready' &&
        <span>{status}</span>
      }
      {records.length > 0 &&
        <div className="flex flex-col gap-2 w-1/2">
          {records.map((record, idx) => (
            <div key={idx} className="flex w-full items-center justify-between" style={{ borderBottom: "1px solid black"}}>
              <p>{record.id}.</p>

              <div className="flex items-center gap-1 h-16">
                <p>{"Input: "}</p>
                <img className="flex mx-2 w-14 h-14" src={BASE_64_PREFIX + record.input} />
                <p>{`Prediction:`}&nbsp;</p>
                <p className="text-3xl">{record.prediction_res}</p>
              </div>

              <div className="flex items-center gap-1">
                <img
                  onClick={() => downloadFile(API_ENDPOINTS.GET_PROOF,`?id=${record.id}`)}
                  className="flex w-6 h-6 cursor-pointer"
                  src={proofIcon}
                  alt="proof-icon"
                  title="Download proof of computation"
                />
                <p>{`
                  ${new Date(record.timestamp * SECOND_IN_MS).toLocaleDateString()}
                  ${new Date(record.timestamp * SECOND_IN_MS).toLocaleTimeString()}
                `}</p>
              </div>
            </div>
          ))}
        </div>
      }
    </Content>
  )
}