import { routes } from "./router"
import { Link } from "react-router-dom"
import { useEffect, useRef, useState } from "react"
import { Point, Status, RequestStatus, DrawBlockType, PredictionRecord } from "./types"
import { API_REQ_ERR_TIMEOUT_MS, sendPrediction } from "./api"

const BLOCK_SIZE = 16 // px
const CANVAS_SIZE = 28 // tiles
const CANVAS_WIDTH = CANVAS_SIZE * BLOCK_SIZE
const CANVAS_HEIGHT = CANVAS_SIZE * BLOCK_SIZE

const Canvas = () => {

  const [canvasStatus, setCanvasStatus] = useState<Status>('loading')
  const [reqStatus, setReqStatus] = useState<RequestStatus>('init')
  const [res, setRes] = useState<PredictionRecord>()
  const [showGrid, setShowGrid] = useState<boolean>(false)
  const [cursorPos, setCursorPos] = useState<Point | null>(null)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const state = useRef<Set<string>>(new Set())

  useEffect(() => {
    if (!canvasRef) return
    setCanvasStatus('ready')
  }, [canvasRef])

  useEffect(() => {
    if (!canvasRef) return
    drawBaseCanvasFrame()
  }, [canvasRef, showGrid])

  const handlePredict = async () => {
    setReqStatus('loading')
    const canvas = document.getElementById("canvas") as HTMLCanvasElement
    const img = new Image()
    img.src = canvas.toDataURL("image/jpg")
    const res = await sendPrediction(img.src, () => setReqStatus('error'))
    setRes(res)
    setTimeout(() => {
      setReqStatus('init')
    }, API_REQ_ERR_TIMEOUT_MS)
  }

  const handleDownload = () => {
    const img = (document.getElementById("canvas") as HTMLCanvasElement).toDataURL("image/jpg")
    const link = document.createElement('a')
    link.download = 'zkml_web_drawing.jpg'
    link.href = img
    link.click()
    link.remove()
  }

  const handleClear = () => {
    state.current = new Set()
    drawBaseCanvasFrame()
  }

  const handleMouseMove = (evt: MouseEvent) => {
    const ctx = canvasRef.current?.getContext("2d") as CanvasRenderingContext2D
    if (!ctx) return

    const { x, y } = mousePosToBlock(convertMousePos(evt))
    setCursorPos({ x, y })
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
    if (!ctx) return
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
    if (showGrid) {
      for (let i = 0; i < CANVAS_SIZE; i++) {
        for (let j = 0; j < CANVAS_SIZE; j++) {
          drawBlock(ctx, { x: i, y: j }, 'stroke', 'rgba(0, 0, 0, 0.25)')
        }
      }
    }
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
    <div className="flex">
      {canvasStatus === 'ready'
        ?
        <div className="flex items-center flex-col">
          <div className="mb-2 text-md">
            <p>
              Classic&nbsp;
              <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/MNIST_database">
                MNIST
              </a>
              &nbsp;classifier.
            </p>
            <p>
              It's small (~3K parameters) and fast enough for ZKML on a&nbsp;
              <a target="_blank" rel="noopener noreferrer" href="https://www.parkytowers.me.uk/thin/hp/t520">
                HP t520
              </a>
              .
            </p>
          </div>
          <div className="flex">
            <canvas id="canvas" ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} className="outline outline-1"
              onMouseDown={(e) => handleMouseMove(e as unknown as MouseEvent)}
              onMouseMove={(e) => handleMouseMove(e as unknown as MouseEvent)}
              onMouseLeave={() => {
                drawBaseCanvasFrame()
                setCursorPos(null)
              }}
            />
          </div>
          <div className="flex items-center gap-2 mt-2">
            <button onClick={handleClear}>
              {"clear"}
            </button>
            <button onClick={handleDownload}>
              {"download"}
            </button>
            <button disabled={reqStatus !== 'init'} onClick={handlePredict}>
              {reqStatus !== 'init' ? reqStatus : 'predict!'}
            </button>
            <div className="flex gap-1">
              <span>Grid:</span>
              <input type="checkbox" defaultChecked={showGrid} name="options-grid" onClick={() => setShowGrid(!showGrid)} />
            </div>
          </div>
          {cursorPos !== null &&
            <p>{JSON.stringify(cursorPos, null, 2)}</p>
          }
          {res &&
            <div className="flex flex-col gap-2">
              <p className="mt-2">{JSON.stringify(res, null, 2)}</p>
              <Link to={routes.records.path}>{"Go to Records"}</Link>
            </div>
          }
        </div>
        : <p>...</p>
      }
    </div>
  )
}

export default Canvas