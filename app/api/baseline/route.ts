import { NextResponse } from "next/server";
import type { CubeState, Move, SolverResponse, SolverStep } from "@/lib/types";

function wrapBaseline(moves: Move[], solveTimeMs: number): SolverResponse {
  const steps: SolverStep[] = moves.map((move, index) => ({
    move,
    confidence: 1,
    valueEstimate: Number((index / Math.max(moves.length - 1, 1)).toFixed(3)),
    reward: index === moves.length - 1 ? 1 : -0.01
  }));

  return {
    moves,
    analytics: {
      moveCount: moves.length,
      solveTimeMs,
      baselineMoveCount: moves.length,
      baselineSolveTimeMs: solveTimeMs,
      finalValueEstimate: steps[steps.length - 1]?.valueEstimate ?? 1,
      averageConfidence: 1,
      totalReward: Number(steps.reduce((total, step) => total + step.reward, 0).toFixed(3)),
      steps
    }
  };
}

export async function POST(request: Request) {
  const body = (await request.json()) as { state?: CubeState };

  if (!body.state || body.state.length !== 54) {
    return NextResponse.json({ error: "Expected a 54-sticker cube state" }, { status: 400 });
  }

  if (!process.env.BACKEND_URL) {
    return NextResponse.json(
      { error: "BACKEND_URL is required for the Kociemba solver. Start FastAPI and set BACKEND_URL." },
      { status: 503 }
    );
  }

  const response = await fetch(`${process.env.BACKEND_URL}/baseline`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  const payload = (await response.json()) as { moves?: Move[]; solveTimeMs?: number };

  if (!response.ok) {
    return NextResponse.json(payload, { status: response.status });
  }

  return NextResponse.json(wrapBaseline(payload.moves ?? [], payload.solveTimeMs ?? 0));
}
