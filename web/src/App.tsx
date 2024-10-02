import Canvas from "./Canvas"

function App() {

  return (
    <div className="flex items-center flex-col gap-3 w-full h-full m-0 p-0">
        <p>ZKML web demo</p>
        <br />
      <div className="flex justify-center items-center">
        <Canvas />
      </div>
    </div>
  )
}

export default App
