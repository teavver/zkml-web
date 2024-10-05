import { routes } from "../router";
import { Link } from "react-router-dom";

export const Nav = (props: { title: string }) => {
  return (
    <div className="flex flex-col w-full items-center">
      <p>{props.title}</p>
      <div className="flex gap-1">
        {Object.entries(routes).map(([name, data], idx) => (
          <Link key={idx} to={data.path}>{name}</Link>
        ))}
      </div>
    </div>
  )
}