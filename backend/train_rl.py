"""Train a Rubik's Cube RL policy with Kociemba teacher pretraining.

This script is built for a practical workflow:

1. Generate scrambled cube states.
2. Ask Kociemba for the first expert move.
3. Train a policy/value model to imitate that teacher.
4. Fine-tune with curriculum RL from shallow to deeper scrambles.
5. Write checkpoints and metrics you can plug into the FastAPI backend later.

Start small first:

    python train_rl.py --teacher-samples 2000 --imitation-epochs 3 --rl-episodes 200

Then scale:

    python train_rl.py --teacher-samples 1000000 --batch-size 512 --rl-episodes 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from dotenv import load_dotenv

load_dotenv()

if os.getenv("RUBIKS_USE_GPU", "1").lower() in {"0", "false", "no", "off"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from baseline_solver import solve_facelets


MOVES = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]
MOVE_TO_INDEX = {move: index for index, move in enumerate(MOVES)}
INDEX_TO_MOVE = {index: move for move, index in MOVE_TO_INDEX.items()}
FACE_ORDER = "URFDLB"
SOLVED = tuple([0] * 9 + [1] * 9 + [2] * 9 + [3] * 9 + [4] * 9 + [5] * 9)


@dataclass
class TrainConfig:
    seed: int
    teacher_samples: int
    imitation_epochs: int
    batch_size: int
    rl_episodes: int
    eval_every: int
    max_depth: int
    output_dir: Path
    learning_rate: float
    gamma: float
    entropy_bonus: float
    teacher_mix: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def sticker_coordinates() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    stickers: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []

    # U face: y = 1, rows from back to front.
    for z in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            stickers.append(((x, 1, z), (0, 1, 0)))

    # R face: x = 1.
    for y in [1, 0, -1]:
        for z in [1, 0, -1]:
            stickers.append(((1, y, z), (1, 0, 0)))

    # F face: z = 1.
    for y in [1, 0, -1]:
        for x in [-1, 0, 1]:
            stickers.append(((x, y, 1), (0, 0, 1)))

    # D face: y = -1.
    for z in [1, 0, -1]:
        for x in [-1, 0, 1]:
            stickers.append(((x, -1, z), (0, -1, 0)))

    # L face: x = -1.
    for y in [1, 0, -1]:
        for z in [-1, 0, 1]:
            stickers.append(((-1, y, z), (-1, 0, 0)))

    # B face: z = -1.
    for y in [1, 0, -1]:
        for x in [1, 0, -1]:
            stickers.append(((x, y, -1), (0, 0, -1)))

    return stickers


STICKERS = sticker_coordinates()
STICKER_TO_INDEX = {sticker: index for index, sticker in enumerate(STICKERS)}


def rotate_vec(vec: tuple[int, int, int], axis: str, direction: int) -> tuple[int, int, int]:
    x, y, z = vec

    if axis == "x":
        return (x, -direction * z, direction * y)
    if axis == "y":
        return (direction * z, y, -direction * x)
    if axis == "z":
        return (-direction * y, direction * x, z)

    raise ValueError(f"Unknown axis: {axis}")


MOVE_ROTATIONS = {
    "U": ("y", 1, 1),
    "U'": ("y", 1, -1),
    "D": ("y", -1, -1),
    "D'": ("y", -1, 1),
    "R": ("x", 1, 1),
    "R'": ("x", 1, -1),
    "L": ("x", -1, -1),
    "L'": ("x", -1, 1),
    "F": ("z", 1, 1),
    "F'": ("z", 1, -1),
    "B": ("z", -1, -1),
    "B'": ("z", -1, 1),
}


def build_permutation(move: str) -> np.ndarray:
    axis, layer, direction = MOVE_ROTATIONS[move]
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    permutation = list(range(54))

    for old_index, (position, normal) in enumerate(STICKERS):
        if position[axis_index] != layer:
            continue

        new_position = rotate_vec(position, axis, direction)
        new_normal = rotate_vec(normal, axis, direction)
        new_index = STICKER_TO_INDEX[(new_position, new_normal)]
        permutation[new_index] = old_index

    return np.array(permutation, dtype=np.int64)


PERMUTATIONS = {move: build_permutation(move) for move in MOVES}


def apply_move(state: tuple[int, ...], move: str) -> tuple[int, ...]:
    array = np.array(state, dtype=np.int64)
    return tuple(array[PERMUTATIONS[move]].tolist())


def apply_moves(state: tuple[int, ...], moves: list[str]) -> tuple[int, ...]:
    for move in moves:
        state = apply_move(state, move)
    return state


def is_solved(state: tuple[int, ...]) -> bool:
    return all(len(set(state[index : index + 9])) == 1 for index in range(0, 54, 9))


def to_facelets(state: tuple[int, ...]) -> str:
    return "".join(FACE_ORDER[color] for color in state)


def random_scramble(depth: int) -> list[str]:
    scramble: list[str] = []

    while len(scramble) < depth:
        move = random.choice(MOVES)
        if scramble and scramble[-1][0] == move[0]:
            continue
        scramble.append(move)

    return scramble


def encode_state(state: tuple[int, ...]) -> np.ndarray:
    encoded = np.zeros((54, 6), dtype=np.float32)
    encoded[np.arange(54), np.array(state, dtype=np.int64)] = 1.0
    return encoded.reshape(-1)


def teacher_first_move(state: tuple[int, ...], fallback_scramble: list[str] | None = None) -> str:
    if is_solved(state):
        return "U"

    try:
        solution = solve_facelets(to_facelets(state))
        if solution and solution != "Cube is already solved":
            move = solution.split()[0]
            return move[0] if move.endswith("2") else move
    except Exception:
        pass

    if fallback_scramble:
        last = fallback_scramble[-1]
        return last[:-1] if last.endswith("'") else f"{last}'"

    return random.choice(MOVES)


def build_model(learning_rate: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(324,), name="cube_state")
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    policy = tf.keras.layers.Dense(len(MOVES), name="policy_logits")(x)
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(x)

    model = tf.keras.Model(inputs=inputs, outputs=[policy, value])
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return model


def generate_teacher_dataset(sample_count: int, max_depth: int) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((sample_count, 324), dtype=np.float32)
    labels = np.zeros((sample_count,), dtype=np.int64)
    started = perf_counter()

    for index in range(sample_count):
        depth = random.randint(1, max_depth)
        scramble = random_scramble(depth)
        state = apply_moves(SOLVED, scramble)
        move = teacher_first_move(state, scramble)
        states[index] = encode_state(state)
        labels[index] = MOVE_TO_INDEX[move]

        if (index + 1) % max(1, sample_count // 10) == 0:
            elapsed = perf_counter() - started
            print(f"teacher data {index + 1:,}/{sample_count:,} states in {elapsed:.1f}s")

    return states, labels


def imitation_train(model: tf.keras.Model, states: np.ndarray, labels: np.ndarray, config: TrainConfig) -> None:
    dataset = (
        tf.data.Dataset.from_tensor_slices((states, labels))
        .shuffle(min(len(states), 100_000), seed=config.seed)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    loss_metric = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(config.imitation_epochs):
        loss_metric.reset_state()
        accuracy_metric.reset_state()

        for batch_states, batch_labels in dataset:
            with tf.GradientTape() as tape:
                logits, values = model(batch_states, training=True)
                policy_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(batch_labels, logits, from_logits=True)
                )
                # Teacher states are on a path to solution, so value target is positive.
                value_loss = tf.reduce_mean(tf.square(tf.ones_like(values) * 0.65 - values))
                loss = policy_loss + 0.25 * value_loss

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_metric.update_state(loss)
            accuracy_metric.update_state(batch_labels, logits)

        print(
            f"imitation epoch {epoch + 1}/{config.imitation_epochs} "
            f"loss={loss_metric.result():.4f} acc={accuracy_metric.result():.3f}"
        )


def choose_action(logits: np.ndarray, epsilon: float) -> tuple[int, float]:
    if random.random() < epsilon:
        action = random.randrange(len(MOVES))
        return action, 1.0 / len(MOVES)

    probs = tf.nn.softmax(logits).numpy()
    action = int(np.random.choice(len(MOVES), p=probs))
    return action, float(probs[action])


def rl_episode(model: tf.keras.Model, depth: int, config: TrainConfig) -> dict[str, float | int | bool]:
    scramble = random_scramble(depth)
    state = apply_moves(SOLVED, scramble)
    max_steps = max(8, depth * 2 + 6)
    trajectory = []
    total_reward = 0.0
    solved = False

    for step in range(max_steps):
        encoded = encode_state(state)[None, :]
        logits, value = model(encoded, training=False)
        epsilon = max(0.02, 0.2 * (1 - depth / max(config.max_depth, 1)))
        action, confidence = choose_action(logits[0].numpy(), epsilon)
        move = INDEX_TO_MOVE[action]
        next_state = apply_move(state, move)

        reward = -0.03
        if is_solved(next_state):
            reward = 1.0
            solved = True

        trajectory.append((state, action, reward, float(value[0, 0]), confidence))
        total_reward += reward
        state = next_state

        if solved:
            break

    returns = []
    running = 0.0
    for _, _, reward, _, _ in reversed(trajectory):
        running = reward + config.gamma * running
        returns.append(running)
    returns.reverse()

    states = tf.convert_to_tensor(np.stack([encode_state(item[0]) for item in trajectory]), dtype=tf.float32)
    actions = tf.convert_to_tensor([item[1] for item in trajectory], dtype=tf.int64)
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

    with tf.GradientTape() as tape:
        logits, values = model(states, training=True)
        values = tf.squeeze(values, axis=1)
        advantages = returns_tensor - values
        action_losses = tf.keras.losses.sparse_categorical_crossentropy(actions, logits, from_logits=True)
        policy_loss = tf.reduce_mean(action_losses * tf.stop_gradient(advantages))
        value_loss = tf.reduce_mean(tf.square(advantages))
        probabilities = tf.nn.softmax(logits)
        entropy = -tf.reduce_mean(tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-8), axis=1))

        loss = policy_loss + 0.5 * value_loss - config.entropy_bonus * entropy

        if random.random() < config.teacher_mix:
            teacher = teacher_first_move(state, scramble)
            teacher_label = tf.convert_to_tensor([MOVE_TO_INDEX[teacher]], dtype=tf.int64)
            teacher_logits, _ = model(tf.convert_to_tensor(encode_state(state)[None, :]), training=True)
            teacher_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(teacher_label, teacher_logits, from_logits=True)
            )
            loss += 0.2 * teacher_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    confidences = [item[4] for item in trajectory]
    values_seen = [item[3] for item in trajectory]

    return {
        "solved": solved,
        "moves": len(trajectory),
        "reward": total_reward,
        "confidence": float(np.mean(confidences)) if confidences else 0.0,
        "value": float(values_seen[-1]) if values_seen else 0.0,
    }


def evaluate(model: tf.keras.Model, depths: list[int], episodes_per_depth: int) -> list[dict[str, float]]:
    rows = []

    for depth in depths:
        solved_count = 0
        move_counts = []
        rewards = []
        confidences = []
        values = []

        for _ in range(episodes_per_depth):
            scramble = random_scramble(depth)
            state = apply_moves(SOLVED, scramble)
            max_steps = max(8, depth * 2 + 6)
            total_reward = 0.0
            solved = False

            for step in range(max_steps):
                logits, value = model(encode_state(state)[None, :], training=False)
                probs = tf.nn.softmax(logits[0]).numpy()
                action = int(np.argmax(probs))
                state = apply_move(state, INDEX_TO_MOVE[action])
                total_reward -= 0.03
                confidences.append(float(probs[action]))
                values.append(float(value[0, 0]))

                if is_solved(state):
                    solved = True
                    total_reward += 1.0
                    move_counts.append(step + 1)
                    break

            if not solved:
                move_counts.append(max_steps)

            solved_count += int(solved)
            rewards.append(total_reward)

        rows.append(
            {
                "depth": depth,
                "solve_rate": solved_count / episodes_per_depth,
                "avg_moves": float(np.mean(move_counts)),
                "avg_reward": float(np.mean(rewards)),
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "avg_value": float(np.mean(values)) if values else 0.0,
            }
        )

    return rows


def curriculum_depth(episode: int, total_episodes: int, max_depth: int) -> int:
    schedule = [1, 2, 3, 5, 7, 10, 14, 18, 22, max_depth]
    schedule = [depth for depth in schedule if depth <= max_depth]
    bucket = min(len(schedule) - 1, math.floor((episode / max(total_episodes, 1)) * len(schedule)))
    return schedule[bucket]


def write_metrics(path: Path, rows: list[dict[str, float]], step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="") as file:
        fieldnames = ["step", "depth", "solve_rate", "avg_moves", "avg_reward", "avg_confidence", "avg_value"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"step": step, **row})


def rl_finetune(model: tf.keras.Model, config: TrainConfig) -> None:
    metrics_path = config.output_dir / "metrics.csv"
    eval_depths = [1, 2, 3, 5, min(10, config.max_depth), config.max_depth]
    eval_depths = sorted(set(depth for depth in eval_depths if depth >= 1))

    for episode in range(1, config.rl_episodes + 1):
        depth = curriculum_depth(episode - 1, config.rl_episodes, config.max_depth)
        result = rl_episode(model, depth, config)

        if episode % max(1, config.eval_every) == 0:
            rows = evaluate(model, eval_depths, episodes_per_depth=50)
            write_metrics(metrics_path, rows, episode)
            summary = " | ".join(
                f"d{row['depth']}: solve={row['solve_rate']:.2f} moves={row['avg_moves']:.1f}"
                for row in rows
            )
            print(
                f"episode {episode:,}/{config.rl_episodes:,} "
                f"last_depth={depth} last_solved={result['solved']} {summary}"
            )
            model.save(config.output_dir / "latest.keras")


def self_check() -> None:
    assert is_solved(SOLVED)
    for move in MOVES:
        moved = apply_move(SOLVED, move)
        assert not is_solved(moved)
        inverse = move[:-1] if move.endswith("'") else f"{move}'"
        assert apply_move(moved, inverse) == SOLVED

    solution = solve_facelets(to_facelets(apply_move(SOLVED, "R")))
    print(f"self-check ok; Kociemba solves R as: {solution}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-samples", type=int, default=20_000)
    parser.add_argument("--imitation-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rl-episodes", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rubiks_policy"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--teacher-mix", type=float, default=0.15)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        raise SystemExit(0)

    return TrainConfig(
        seed=args.seed,
        teacher_samples=args.teacher_samples,
        imitation_epochs=args.imitation_epochs,
        batch_size=args.batch_size,
        rl_episodes=args.rl_episodes,
        eval_every=args.eval_every,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        entropy_bonus=args.entropy_bonus,
        teacher_mix=args.teacher_mix,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"RUBIKS_USE_GPU={os.getenv('RUBIKS_USE_GPU', '1')}")
    print(f"TensorFlow devices: {[device.name for device in tf.config.list_physical_devices()]}")

    with (config.output_dir / "config.json").open("w") as file:
        json.dump({**config.__dict__, "output_dir": str(config.output_dir)}, file, indent=2)

    self_check()
    model = build_model(config.learning_rate)
    states, labels = generate_teacher_dataset(config.teacher_samples, config.max_depth)
    imitation_train(model, states, labels, config)
    model.save(config.output_dir / "imitation.keras")
    rl_finetune(model, config)
    model.save(config.output_dir / "final.keras")
    print(f"saved model artifacts to {config.output_dir}")


if __name__ == "__main__":
    main()
