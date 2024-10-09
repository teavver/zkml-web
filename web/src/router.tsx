import { createBrowserRouter, RouteObject } from "react-router-dom"
import { Home } from "./routes/Home"
import { Prediction } from "./routes/Prediction"
import { Records } from "./routes/Records"
import { Verify } from "./routes/Verify"

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const routes: { [name: string]: { [k: string]: any } } = {
  home: {
    title: "ZKML-web MNIST demo",
    path: "/",
    component: <Home />
  },
  predict: {
    title: "Predict",
    path: "/predict",
    component: <Prediction />
  },
  records: {
    title: "Records",
    path: "/records",
    component: <Records />
  },
  verify: {
    title: "Verify",
    path: "/verify",
    component: <Verify />
  },
}

export const router = createBrowserRouter(
  Object.entries(routes).map(([, route]) => Object({ path: route.path, element: route.component })) as RouteObject[]
)