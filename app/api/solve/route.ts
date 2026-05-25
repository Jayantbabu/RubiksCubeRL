import { NextResponse } from "next/server";
import type { CubeState } from "@/lib/types";

export async function POST(request: Request) {
  const body = (await request.json()) as { state?: CubeState };

  if (!body.state || body.state.length !== 54) {
    return NextResponse.json({ error: "Expected a 54-sticker cube state" }, { status: 400 });
  }

  if (!process.env.BACKEND_URL) {
    return NextResponse.json(
      { error: "BACKEND_URL is required for solving. Start FastAPI and set BACKEND_URL." },
      { status: 503 }
    );
  }

  const response = await fetch(`${process.env.BACKEND_URL}/solve`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  const payload = await response.json();
  return NextResponse.json(payload, { status: response.status });
}
