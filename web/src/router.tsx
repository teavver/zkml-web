import React from "react"
import { createBrowserRouter, RouteObject } from "react-router-dom"
import { Home } from "./routes/Home"
import { Prediction } from "./routes/Prediction"
import { Records } from "./routes/Records"
import { Verify } from "./routes/Verify"

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
  github: {
    title: "github repo",
    path: "https://github.com/teavver/zkml-web/tree/main",
    component: <React.Fragment />
  }
}

export const router = createBrowserRouter(
  Object.entries(routes).map(([_, route]) => Object({ path: route.path, element: route.component })) as RouteObject[]
)