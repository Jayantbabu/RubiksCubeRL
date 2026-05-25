import type { Move } from "./types";

const moves: Move[] = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"];

const faceOf = (move: Move) => move[0];

export function generateScramble() {
  const length = 20 + Math.floor(Math.random() * 6);
  const scramble: Move[] = [];

  while (scramble.length < length) {
    const candidate = moves[Math.floor(Math.random() * moves.length)];
    const previous = scramble[scramble.length - 1];

    if (previous && faceOf(previous) === faceOf(candidate)) {
      continue;
    }

    scramble.push(candidate);
  }

  return scramble;
}
