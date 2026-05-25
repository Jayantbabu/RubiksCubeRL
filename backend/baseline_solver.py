from time import perf_counter

from cube_state import to_kociemba_facelets

_rubik_solver_ready = False


def expand_double_turns(moves: list[str]) -> list[str]:
    expanded: list[str] = []

    for move in moves:
        if move.endswith("2"):
            expanded.extend([move[0], move[0]])
        else:
            expanded.append(move)

    return expanded


class BaselineSolveResult:
    def __init__(self, moves: list[str], solve_time_ms: int):
        self.moves = moves
        self.solve_time_ms = solve_time_ms


class KociembaBaselineSolver:
    def solve(self, cube_state: list[str]) -> BaselineSolveResult:
        started = perf_counter()

        facelets = to_kociemba_facelets(cube_state)
        solution = solve_facelets(facelets)
        moves = [] if solution == "Cube is already solved" else expand_double_turns(solution.split())
        if not verify_solution(facelets, moves):
            raise ValueError(f"solver returned non-solving moves for {facelets}: {' '.join(moves)}")

        elapsed_ms = int((perf_counter() - started) * 1000)
        return BaselineSolveResult(moves=moves, solve_time_ms=max(elapsed_ms, 1))


def solve_facelets(facelets: str) -> str:
    try:
        import kociemba

        return kociemba.solve(facelets)
    except ImportError:
        return solve_facelets_pure_python(facelets)


def solve_facelets_pure_python(facelets: str) -> str:
    global _rubik_solver_ready

    from rubik_solver import Cube, init_solver, solve

    if not _rubik_solver_ready:
        init_solver()
        _rubik_solver_ready = True

    solution = solve(Cube.from_string(facelets))
    return solution or "Cube is already solved"


def verify_solution(facelets: str, moves: list[str]) -> bool:
    from rubik_solver import Cube, init_solver

    global _rubik_solver_ready

    if not _rubik_solver_ready:
        init_solver()
        _rubik_solver_ready = True

    cube = Cube.from_string(facelets)
    if moves:
        cube.move(" ".join(moves))
    return cube.is_solved()
