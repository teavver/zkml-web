import { Nav } from "../components/Nav";
import { Content } from "../components/Content";
import { useEffect, useRef, useState } from "react";
import { fetchPredictionRecords } from "../api";
import { PredictionRecord, RequestStatus } from "../types";
import { routes } from "../router";

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

  return (
    <Content>
      <Nav title={routes.records.title} />
      <div className="flex gap-2">
        <button disabled={status !== 'ready' || page === 0} onClick={() => handleFetchRecords(page - 1)}>Previous</button>
        <button disabled={status !== 'ready' || lastPage.current} onClick={() => handleFetchRecords(page + 1)}>Next</button>
      </div>
      {status !== 'ready' &&
        <p>{status}</p>
      }
      {records.length > 0 &&
        <div className="flex flex-col w-1/3">
          {records.map((record, idx) => (
            <div key={idx} className="flex w-full items-center justify-between px-2">
              <p>{record.id}.</p>
              <div className="flex items-center gap-1">
                <p>{"Input: "}</p>
                <img className="flex mx-2 w-16 h-16" src={BASE_64_PREFIX + record.input} />
                <p>{`Prediction:`}&nbsp;</p>
                <p className="text-3xl">{record.prediction_res}</p>
              </div>
              <p>{`
                ${new Date(record.timestamp * SECOND_IN_MS).toLocaleDateString()}
                ${new Date(record.timestamp * SECOND_IN_MS).toLocaleTimeString()}
              `}</p>
            </div>
          ))}
        </div>
      }
    </Content>
  )
}