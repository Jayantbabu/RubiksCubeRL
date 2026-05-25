export type CubeColor = "white" | "yellow" | "red" | "orange" | "blue" | "green";

export type CubeState = CubeColor[];

export type Move =
  | "U"
  | "U'"
  | "D"
  | "D'"
  | "L"
  | "L'"
  | "R"
  | "R'"
  | "F"
  | "F'"
  | "B"
  | "B'";

export type SolverStep = {
  move: Move;
  confidence: number;
  valueEstimate: number;
  reward: number;
};

export type SolverAnalytics = {
  moveCount: number;
  solveTimeMs: number;
  baselineMoveCount: number;
  baselineSolveTimeMs: number;
  finalValueEstimate: number;
  averageConfidence: number;
  totalReward: number;
  steps: SolverStep[];
};

export type SolverResponse = {
  moves: Move[];
  analytics: SolverAnalytics;
};

export type SolverMode = "rl" | "kociemba";
