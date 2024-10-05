import { useEffect, useRef, useState } from "react"
import { Point, Status, RequestStatus, DrawBlockType, PredictionRecord } from "./types"
import { API_ENDPOINTS, API_URL } from "./utils"

const BLOCK_SIZE = 16
const CANVAS_SIZE = 28
const CANVAS_WIDTH = CANVAS_SIZE * BLOCK_SIZE
const CANVAS_HEIGHT = CANVAS_SIZE * BLOCK_SIZE
const REQ_STATUS_RESET_MS = 5000

const Canvas = () => {

  const [canvasStatus, setCanvasStatus] = useState<Status>('loading')
  const [reqStatus, setReqStatus] = useState<RequestStatus>('init')
  const [res, setRes] = useState<PredictionRecord>()

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const state = useRef<Set<string>>(new Set())

  useEffect(() => {
    if (!canvasRef) return
    setCanvasStatus('ready')
  }, [canvasRef])

  const handlePredict = async () => {
    setReqStatus('loading')
    const canvas = document.getElementById("canvas") as HTMLCanvasElement
    const img = new Image()
    img.src = canvas.toDataURL("image/jpg")
    const url = API_URL + API_ENDPOINTS.PREDICT
    try {
      const data = await fetch(url, {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input: img.src })
      })
      const res = await data.json()
      setRes(res as PredictionRecord)
      setReqStatus('success')
      setTimeout(() => setReqStatus('init'), REQ_STATUS_RESET_MS)
    } catch (err) {
      console.error((err as Error).message)
      setReqStatus('error')
    }
  }

  const handleDownload = () => {
    const img = (document.getElementById("canvas") as HTMLCanvasElement).toDataURL("image/jpg")
    const link = document.createElement('a')
    link.download = 'zkml_web_drawing.jpg'
    link.href = img
    link.click()
  }

  const handleClear = () => {
    state.current = new Set()
    drawBaseCanvasFrame()
  }

  const handleMouseMove = (evt: MouseEvent) => {
    const ctx = canvasRef.current?.getContext("2d") as CanvasRenderingContext2D
    const { x, y } = mousePosToBlock(convertMousePos(evt))

    drawBaseCanvasFrame()

    if (evt.buttons === 1 && x >= 0 && x < CANVAS_SIZE && y >= 0 && y < CANVAS_SIZE) {
      state.current.add(`${x},${y}`)
      drawBlock(ctx, { x, y })
    } else {
      drawBlock(ctx, { x, y }, 'stroke', 'magenta')
    }
  }

  const drawBaseCanvasFrame = () => {
    const ctx = canvasRef.current?.getContext("2d") as CanvasRenderingContext2D
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
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

  const drawBlock = (ctx: CanvasRenderingContext2D, pos: Point, drawType: DrawBlockType = 'fill', color: string = 'black') => {
    (drawType === 'fill') ? ctx.fillStyle = color : ctx.strokeStyle = color
    if (drawType === 'stroke') {
      ctx.strokeRect(pos.x * BLOCK_SIZE, pos.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    }
    else { ctx.fillRect(pos.x * BLOCK_SIZE, pos.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE) }
    (drawType === 'fill') ? ctx.fillStyle = 'black' : ctx.strokeStyle = 'black'
  }

  return (
    <div>
      {canvasStatus === 'ready'
        ?
        <div className="flex flex-col">
          <canvas id="canvas" ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} className="outline outline-1"
            onMouseMove={(e) => handleMouseMove(e as unknown as MouseEvent)}
            onMouseLeave={() => drawBaseCanvasFrame()}
          />
          <div className="flex gap-2 mt-2">
            <button onClick={handleClear}>
              {"clear"}
            </button>
            <button onClick={handleDownload}>
              {"download"}
            </button>
            <button disabled={reqStatus !== 'init'} onClick={handlePredict}>
              {reqStatus !== 'init' ? reqStatus : 'predict!'}
            </button>
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