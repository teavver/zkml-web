import { Nav } from "../components/Nav";
import { Content } from "../components/Content";
import Canvas from "../Canvas";
import { routes } from "../router";

export const Prediction = () => {
  return (
    <Content>
      <Nav title={routes.prediction.title} />
      <Canvas />
    </Content>
  )
}