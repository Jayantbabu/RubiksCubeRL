import type { CubeState, SolverMode, SolverResponse } from "./types";

export async function requestSolverSolution(state: CubeState, solverMode: SolverMode): Promise<SolverResponse> {
  const endpoint = solverMode === "kociemba" ? "/api/baseline" : "/api/solve";
  console.info("[SolverClient] request", { solverMode, endpoint, state });

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ state })
  });

  const payload = await response.json();
  console.info("[SolverClient] response", { solverMode, endpoint, payload });

  if (!response.ok) {
    throw new Error(typeof payload?.error === "string" ? payload.error : `${solverMode} solver request failed`);
  }

  return payload as SolverResponse;
}
