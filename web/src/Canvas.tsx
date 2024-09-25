import { useEffect, useRef, useState } from "react"
import { Point, Status } from "./types"

const BLOCK_SIZE = 16
const CANVAS_SIZE = 28
const CANVAS_WIDTH = CANVAS_SIZE * BLOCK_SIZE
const CANVAS_HEIGHT = CANVAS_SIZE * BLOCK_SIZE

const Canvas = () => {

  const [status, setStatus] = useState<Status>('loading')
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    const ctx = canvasRef.current.getContext("2d")
    ctx?.strokeRect(200, 200, 40, 50)
  }, [canvasRef])

  useEffect(() => {
    if (!canvasRef) return
    setStatus('ready')
  }, [canvasRef])

  const handleMouseMove = (evt: MouseEvent) => {
    if (!canvasRef.current) return
    const ctx = canvasRef.current.getContext("2d")
    if (!ctx) return
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
    drawBlock(ctx, mousePosToBlock(convertMousePos(evt)))
  }

  const convertMousePos = (evt: MouseEvent): Point => {
    //@ts-ignore
    let rect = evt.target.getBoundingClientRect()
    return { x: evt.clientX - rect.left, y: evt.clientY - rect.top }
  }

  const mousePosToBlock = (pos: Point): Point => {
    const round = (n: number) => n < BLOCK_SIZE ? 0 : (Math.round(n / BLOCK_SIZE) * BLOCK_SIZE) / BLOCK_SIZE
    return { x: round(pos.x), y: round(pos.y) }
  }

  const drawBlock = (ctx: CanvasRenderingContext2D, pos: Point) => {
    ctx.strokeRect(pos.x * BLOCK_SIZE, pos.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
  }

  return (
    <div>
      {status === 'ready'
        ?
          <canvas ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} className="border-2"
            onMouseMove={(e) => handleMouseMove(e as unknown as MouseEvent)}
          />
        : <p>...</p>
      }
    </div>
  )
}

export default Canvas