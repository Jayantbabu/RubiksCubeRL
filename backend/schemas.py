from pydantic import BaseModel, Field


CubeColor = str
Move = str


class SolveRequest(BaseModel):
    state: list[CubeColor] = Field(min_length=54, max_length=54)


class SolverStep(BaseModel):
    move: Move
    confidence: float
    valueEstimate: float
    reward: float


class SolverAnalytics(BaseModel):
    moveCount: int
    solveTimeMs: int
    baselineMoveCount: int
    baselineSolveTimeMs: int
    finalValueEstimate: float
    averageConfidence: float
    totalReward: float
    steps: list[SolverStep]


class SolverResponse(BaseModel):
    moves: list[Move]
    analytics: SolverAnalytics


class ValidationResponse(BaseModel):
    valid: bool
    errors: list[str]
