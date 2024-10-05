
export const Content = (props: { children: React.ReactNode }) => {
  return (
    <div className="flex flex-nowrap flex-col justify-center items-center gap-2 w-full h-full m-0 p-0">{props.children}</div>
  )
}