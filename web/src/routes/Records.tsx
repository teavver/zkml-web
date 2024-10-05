import { Nav } from "../components/Nav";
import { Content } from "../components/Content";
import { useEffect, useState } from "react";
import { fetchPredictionRecords } from "../api";
import { PredictionRecord, RequestStatus } from "../types";
import { routes } from "../router";

export const Records = () => {

  const BASE_64_PREFIX = "data:image/png;base64,"

  const [status, setStatus] = useState<RequestStatus>('init')
  const [records, setRecords] = useState<PredictionRecord[]>([])

  useEffect(() => {
    (async () => {
      setStatus('loading')
      const records = await fetchPredictionRecords()
      if (records.length > 0) {
        setStatus('ready')
        setRecords(records)
      } else setStatus('error')
    })()
  }, [])

  return (
    <Content>
      <Nav title={routes.records.title} />
      {status !== 'ready' &&
        <p>{status}</p>
      }
      {records.length > 0 &&
        <div className="flex flex-col w-1/3">
          {records.map((record, idx) => (
            <div key={idx} className="flex w-full items-center justify-between p-2">
              <p>{record.id}.</p>
              <div className="flex items-center gap-1">
                <p>{"Input: "}</p>
                <img className="flex mx-2 w-16 h-16" src={BASE_64_PREFIX + record.input} />
                <p>{`Predicted:`}&nbsp;</p>
                <p className="text-3xl">{record.prediction_res}</p>
              </div>
              <p>{`
                ${new Date(record.timestamp * 1000).toLocaleDateString()}
                ${new Date(record.timestamp * 1000).toLocaleTimeString()}
              `}</p>
            </div>
          ))}
        </div>
      }
    </Content>
  )
}