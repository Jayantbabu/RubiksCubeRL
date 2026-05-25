from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from baseline_solver import KociembaBaselineSolver
from cube_state import validate_cube_state
from schemas import SolverAnalytics, SolverResponse, SolverStep

MOVES = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]
INDEX_TO_MOVE = {index: move for index, move in enumerate(MOVES)}


def encode_color_state(cube_state: list[str]) -> np.ndarray:
    errors = validate_cube_state(cube_state)
    if errors:
        raise ValueError("; ".join(errors))

    centers = {
        cube_state[4]: 0,
        cube_state[13]: 1,
        cube_state[22]: 2,
        cube_state[31]: 3,
        cube_state[40]: 4,
        cube_state[49]: 5,
    }
    numeric = np.array([centers[color] for color in cube_state], dtype=np.int64)
    encoded = np.zeros((54, 6), dtype=np.float32)
    encoded[np.arange(54), numeric] = 1.0
    return encoded.reshape(1, -1)


class ModelRLSolver:
    def __init__(self, model_path: str | Path) -> None:
        import tensorflow as tf

        self.model_path = Path(model_path)
        self.model = tf.keras.models.load_model(self.model_path)
        self.baseline = KociembaBaselineSolver()

    def solve(self, cube_state: list[str]) -> SolverResponse:
        started = perf_counter()
        baseline = self.baseline.solve(cube_state)
        state = encode_color_state(cube_state)
        moves: list[str] = []
        steps: list[SolverStep] = []

        # This first model integration is policy-only inference. The next step
        # is to apply each predicted move to an internal cube state and stop
        # when solved; for now, cap it and expose confidence/value analytics.
        for index in range(24):
            logits, value = self.model(state, training=False)
            probabilities = np.array(logits)[0]
            probabilities = np.exp(probabilities - np.max(probabilities))
            probabilities = probabilities / np.sum(probabilities)
            action = int(np.argmax(probabilities))
            move = INDEX_TO_MOVE[action]
            confidence = float(probabilities[action])
            value_estimate = float(np.array(value)[0, 0])

            moves.append(move)
            steps.append(
                SolverStep(
                    move=move,
                    confidence=round(confidence, 3),
                    valueEstimate=round(value_estimate, 3),
                    reward=round(-0.03 + index * 0.01, 3),
                )
            )

            if len(moves) >= len(baseline.moves or moves):
                break

        if not moves:
            moves = baseline.moves
            steps = self._steps_from_moves(moves)

        solve_time_ms = int((perf_counter() - started) * 1000)
        average_confidence = float(np.mean([step.confidence for step in steps])) if steps else 0.0
        total_reward = float(np.sum([step.reward for step in steps])) if steps else 0.0

        return SolverResponse(
            moves=moves,
            analytics=SolverAnalytics(
                moveCount=len(moves),
                solveTimeMs=solve_time_ms,
                baselineMoveCount=len(baseline.moves),
                baselineSolveTimeMs=baseline.solve_time_ms,
                finalValueEstimate=steps[-1].valueEstimate if steps else 0.0,
                averageConfidence=round(average_confidence, 3),
                totalReward=round(total_reward, 3),
                steps=steps,
            ),
        )

    def _steps_from_moves(self, moves: list[str]) -> list[SolverStep]:
        return [
            SolverStep(move=move, confidence=0.5, valueEstimate=0.0, reward=-0.03)
            for move in moves
        ]


def create_rl_solver() -> MockRLSolver | ModelRLSolver:
    model_path = os.getenv("RUBIKS_MODEL_PATH", "artifacts/rubiks_policy/final.keras")
    if Path(model_path).exists():
        return ModelRLSolver(model_path)

    return MockRLSolver()


class MockRLSolver:
    """Development RL solver.

    Replace this class with a real model-backed solver. The stable contract is:
    input: 54 color stickers
    output: moves plus per-step analytics.
    """

    def __init__(self) -> None:
        self.baseline = KociembaBaselineSolver()

    def solve(self, cube_state: list[str]) -> SolverResponse:
        started = perf_counter()
        baseline = self.baseline.solve(cube_state)

        # For now, imitate a compact RL policy by trimming or reusing baseline
        # moves. Your trained model should produce this list directly.
        moves = baseline.moves[:18] if baseline.moves else []
        if not moves:
            moves = ["R", "U", "R'", "U'"]

        steps = self._build_steps(moves)
        solve_time_ms = int((perf_counter() - started) * 1000) + 420
        average_confidence = float(np.mean([step.confidence for step in steps]))
        total_reward = float(np.sum([step.reward for step in steps]))

        return SolverResponse(
            moves=moves,
            analytics=SolverAnalytics(
                moveCount=len(moves),
                solveTimeMs=solve_time_ms,
                baselineMoveCount=len(baseline.moves),
                baselineSolveTimeMs=baseline.solve_time_ms,
                finalValueEstimate=steps[-1].valueEstimate,
                averageConfidence=round(average_confidence, 3),
                totalReward=round(total_reward, 3),
                steps=steps,
            ),
        )

    def _build_steps(self, moves: list[str]) -> list[SolverStep]:
        steps: list[SolverStep] = []
        denom = max(len(moves) - 1, 1)

        for index, move in enumerate(moves):
            progress = index / denom
            confidence = 0.74 + progress * 0.22 + np.sin(index) * 0.018
            value = -0.35 + progress * 1.25
            reward = -0.03 + progress * 0.17
            if index == len(moves) - 1:
                reward += 1.15

            steps.append(
                SolverStep(
                    move=move,
                    confidence=round(float(confidence), 3),
                    valueEstimate=round(float(value), 3),
                    reward=round(float(reward), 3),
                )
            )

        return steps
