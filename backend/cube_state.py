from collections import Counter


VALID_COLORS = {"white", "yellow", "red", "orange", "blue", "green"}
FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


def validate_cube_state(state: list[str]) -> list[str]:
    errors: list[str] = []

    if len(state) != 54:
        errors.append("Cube state must contain exactly 54 stickers.")
        return errors

    unknown = sorted(set(state) - VALID_COLORS)
    if unknown:
        errors.append(f"Unknown colors: {', '.join(unknown)}.")

    counts = Counter(state)
    for color in sorted(VALID_COLORS):
        if counts[color] != 9:
            errors.append(f"{color} must appear exactly 9 times, found {counts[color]}.")

    center_colors = [state[index] for index in [4, 13, 22, 31, 40, 49]]
    if len(set(center_colors)) != 6:
        errors.append("The six center stickers must all be different.")

    return errors


def to_kociemba_facelets(state: list[str]) -> str:
    """Convert frontend color state to Kociemba URFDLB facelet notation.

    The frontend stores six 3x3 faces in this order:
    U, R, F, D, L, B.
    Kociemba expects a 54-char string using the face labels URFDLB.
    We infer the color-to-face mapping from the center stickers.
    """
    errors = validate_cube_state(state)
    if errors:
        raise ValueError("; ".join(errors))

    centers = {
        state[4]: "U",
        state[13]: "R",
        state[22]: "F",
        state[31]: "D",
        state[40]: "L",
        state[49]: "B",
    }

    return "".join(centers[color] for color in state)


def solved_state() -> list[str]:
    return [
        *["white"] * 9,
        *["red"] * 9,
        *["green"] * 9,
        *["yellow"] * 9,
        *["orange"] * 9,
        *["blue"] * 9,
    ]
