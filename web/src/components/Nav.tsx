import { Fragment } from "react/jsx-runtime";
import { routes } from "../router";
import { Link } from "react-router-dom";

export const Nav = (props: { title: string }) => {
  return (
    <div className="flex flex-col w-full items-center">
      <p>{props.title}</p>
      <div className="flex items-center gap-1">
        {Object.entries(routes).map(([name, data], idx, arr) => (
          <Fragment key={idx}>
            <Link to={data.path}>{name}</Link>
            {idx < arr.length - 1 && <p>&nbsp;|&nbsp;</p>}
          </Fragment>
        ))}
      </div>
    </div>
  )
}