import { Nav } from "../components/Nav"
import { Content } from "../components/Content"
import { routes } from "../router"
import { Link } from "react-router-dom"

export const Home = () => {
  return (
    <Content>
      <Nav title={routes.home.title} />
      <p>
        Testing the&nbsp;
        <Link target={"_blank"} to={"https://github.com/zkonduit/ezkl"}>EZKL</Link>
        &nbsp;library on light models and low-spec hardware.
      </p>
    </Content>
  )
}