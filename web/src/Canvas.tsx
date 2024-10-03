import { useEffect, useRef, useState } from "react"
import { Point, Status, DrawBlockType, PredictionRecord } from "./types"
import { API_ENDPOINTS, API_URL } from "./utils"

const BLOCK_SIZE = 16
const CANVAS_SIZE = 28
const CANVAS_WIDTH = CANVAS_SIZE * BLOCK_SIZE
const CANVAS_HEIGHT = CANVAS_SIZE * BLOCK_SIZE

const Canvas = () => {

  const [status, setStatus] = useState<Status>('loading')
  const [res, setRes] = useState<PredictionRecord>()

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const state = useRef<Set<string>>(new Set())

  useEffect(() => {
    if (!canvasRef) return
    setStatus('ready')
  }, [canvasRef])

  const handlePredict = async () => {
    const canvas = document.getElementById("canvas") as HTMLCanvasElement
    const img = new Image()
    img.src = canvas.toDataURL("image/jpg")
    const url = API_URL + API_ENDPOINTS.PREDICT
    const data = await fetch(url, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input: img.src })
    })
    const res = await data.json()
    setRes(res as PredictionRecord)
  }

  const handleDownload = () => {
    const img = (document.getElementById("canvas") as HTMLCanvasElement).toDataURL("image/jpg")
    const link = document.createElement('a')
    link.download = 'zkml_web_drawing.jpg'
    link.href = img
    link.click()
    document.body.removeChild(link)
  }

  const handleMouseMove = (evt: MouseEvent) => {
    const ctx = canvasRef.current?.getContext("2d") as CanvasRenderingContext2D
    const { x, y } = mousePosToBlock(convertMousePos(evt))

    drawBaseCanvasFrame()

    if (evt.buttons === 1 && x >= 0 && x < CANVAS_SIZE && y >= 0 && y < CANVAS_SIZE) {
      state.current.add(`${x},${y}`)
      drawBlock(ctx, { x, y })
    } else {
      drawBlock(ctx, { x, y }, 'stroke')
    }
  }

  const drawBaseCanvasFrame = () => {
    const ctx = canvasRef.current?.getContext("2d") as CanvasRenderingContext2D
    // ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
    ctx.fillStyle = 'black'
    drawStateBlocks(ctx)
  }

  const drawStateBlocks = (ctx: CanvasRenderingContext2D) => {
    state.current.forEach(point => {
      const [px, py] = point.split(',').map(Number)
      drawBlock(ctx, { x: px, y: py })
    })
  }

  const convertMousePos = (evt: MouseEvent): Point => {
    //@ts-ignore
    let rect = evt.target.getBoundingClientRect()
    return { x: evt.clientX - rect.left, y: evt.clientY - rect.top }
  }

  const mousePosToBlock = (pos: Point): Point => {
    const round = (n: number) => Math.round((n + BLOCK_SIZE / 2) / BLOCK_SIZE) - 1
    return { x: round(pos.x), y: round(pos.y) }
  }

  const drawBlock = (ctx: CanvasRenderingContext2D, pos: Point, drawType: DrawBlockType = 'fill') => {
    if (drawType === 'stroke') return ctx.strokeRect(pos.x * BLOCK_SIZE, pos.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    return ctx.fillRect(pos.x * BLOCK_SIZE, pos.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
  }

  return (
    <div>
      {status === 'ready'
        ?
        <div className="flex flex-col">
          <canvas id="canvas" ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} className="outline outline-1"
            onMouseMove={(e) => handleMouseMove(e as unknown as MouseEvent)}
            onMouseLeave={() => drawBaseCanvasFrame()}
          />
          <div>
            <button onClick={handlePredict}>Predict!</button>
            <button onClick={handleDownload}>Download</button>
          </div>
          {res &&
            <p className="mt-2">{JSON.stringify(res, null, 2)}</p>
          }
        </div>
        : <p>...</p>
      }
    </div>
  )
}

export default Canvas