import type { CubeState, Move } from "./types";
import { applyMoveToState } from "./cubePermutations";

type CubeRuntime = {
  applyMove?: (move: Move) => void | Promise<void>;
  applyMoves?: (moves: Move[]) => void | Promise<void>;
  setCubeState?: (state: CubeState) => void | Promise<void>;
  getCubeState?: () => CubeState | Promise<CubeState>;
  resetCube?: () => void | Promise<void>;
};

declare global {
  interface Window {
    rubiksCube?: CubeRuntime;
  }
}

const fillFace = (color: CubeState[number]) => Array<CubeState[number]>(9).fill(color);

export const solvedCubeState: CubeState = [
  ...fillFace("white"),
  ...fillFace("red"),
  ...fillFace("green"),
  ...fillFace("yellow"),
  ...fillFace("orange"),
  ...fillFace("blue")
];

let fallbackState: CubeState = [...solvedCubeState];

const runtime = () =>
  typeof window === "undefined" ? undefined : window.rubiksCube;

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

export async function applyMove(move: Move) {
  const cube = runtime();
  fallbackState = applyMoveToState(fallbackState, move);

  if (cube?.applyMove) {
    await cube.applyMove(move);
    return;
  }

  console.info("[CubeBridge] applyMove", move);
  await wait(460);
}

export async function applyMoves(moves: Move[]) {
  const cube = runtime();

  if (cube?.applyMoves) {
    await cube.applyMoves(moves);
    return;
  }

  for (const move of moves) {
    await applyMove(move);
  }
}

export async function setCubeState(state: CubeState) {
  fallbackState = [...state];
  const cube = runtime();
  await cube?.setCubeState?.(state);
}

export async function getCubeState(): Promise<CubeState> {
  const cube = runtime();
  const state = await cube?.getCubeState?.();
  return state ? [...state] : [...fallbackState];
}

export async function resetCube() {
  fallbackState = [...solvedCubeState];
  const cube = runtime();
  await cube?.resetCube?.();
}
