import { createBrowserRouter, RouteObject } from "react-router-dom"
import { Home } from "./routes/Home"
import { Prediction } from "./routes/Prediction"
import { Records } from "./routes/Records"

export const routes: { [name: string]: { [k: string]: any } } = {
  home: {
    title: "ZKML-web mnist demo",
    path: "/",
    component: <Home />
  },
  prediction: {
    title: "Digit Prediction",
    path: "/prediction",
    component: <Prediction />
  },
  records: {
    title: "Prediction Records",
    path: "/records",
    component: <Records />
  }
}

export const router = createBrowserRouter(
  Object.entries(routes).map(([_, route]) => Object({ path: route.path, element: route.component })) as RouteObject[]
)