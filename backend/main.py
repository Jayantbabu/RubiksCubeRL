from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from baseline_solver import KociembaBaselineSolver
from cube_state import validate_cube_state
from rl_solver import create_rl_solver
from schemas import SolveRequest, SolverResponse, ValidationResponse


load_dotenv()

app = FastAPI(title="Rubik's Cube RL Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3000",
        os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rl_solver = create_rl_solver()
baseline_solver = KociembaBaselineSolver()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/validate", response_model=ValidationResponse)
def validate(request: SolveRequest) -> ValidationResponse:
    errors = validate_cube_state(request.state)
    return ValidationResponse(valid=not errors, errors=errors)


@app.post("/baseline")
def baseline(request: SolveRequest) -> dict[str, int | list[str]]:
    errors = validate_cube_state(request.state)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    try:
        result = baseline_solver.solve(request.state)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Kociemba could not solve this cube state: {exc}") from exc
    print(f"[baseline] moves={result.moves}")
    return {
        "moves": result.moves,
        "moveCount": len(result.moves),
        "solveTimeMs": result.solve_time_ms,
    }


@app.post("/solve", response_model=SolverResponse)
def solve(request: SolveRequest) -> SolverResponse:
    errors = validate_cube_state(request.state)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    return rl_solver.solve(request.state)
