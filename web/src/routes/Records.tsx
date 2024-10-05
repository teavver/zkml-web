import { Nav } from "../components/Nav";
import { Content } from "../components/Content";
import { useEffect, useState } from "react";
import { fetchPredictionRecords } from "../api";


export const Records = () => {

  const [records, setRecords] = useState(fetchPredictionRecords)

  return (
    <></>
  )
}