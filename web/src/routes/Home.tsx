import { Nav } from "../components/Nav"
import { Content } from "../components/Content"
import { routes } from "../router"

export const Home = () => {
    return (
      <Content>
        <Nav title={routes.home.title} />
        <p>Homepage</p>
      </Content>
    )
}