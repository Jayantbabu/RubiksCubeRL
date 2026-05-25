# Rubik's Cube Solver Backend

FastAPI backend for the Next.js Rubik's Cube UI.

## Run Locally

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

For model training, use Python 3.11 or 3.12 and install:

```bash
pip install -r requirements-train.txt
```

Then set this in the frontend environment when you want Next.js to call the Python API:

```bash
BACKEND_URL=http://localhost:8000
```

## Endpoints

- `GET /health`
- `POST /solve`
- `POST /baseline`
- `POST /validate`

## Model Integration

Replace `MockRLSolver` in `rl_solver.py` with your trained model loader. Keep the public method:

```python
solve(cube_state: list[str]) -> SolverResponse
```

That lets the frontend stay unchanged.
