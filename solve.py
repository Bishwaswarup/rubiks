"""
Solver wrapper, plus a rewriter that turns the solution into moves a pair of
hands can actually make.

WHY REWRITE THE SOLUTION
------------------------
A two-phase solver returns roughly 20 moves drawn from all six faces. B and D
turns are the problem: to make one, a beginner reaches behind or underneath
the cube, and in doing so almost always reorients it slightly without
noticing. From then on every remaining instruction is wrong, and the user has
no way to tell -- they just end up with a scrambled cube and no idea which
step broke it.

So `humanize` rewrites the solution to use only R, U and F turns, inserting
explicit whole-cube rotations whenever the face the solver wants is out of
reach. The move count goes up by roughly half, but every single instruction
becomes one of four unambiguous physical actions, and a whole-cube rotation is
much harder to do wrong than a hidden-face layer turn.

The rewrite is verified, not assumed: `humanize` checks that its own output
solves the cube (up to a whole-cube rotation) before returning, and raises if
it does not. selftest.py exercises that on random scrambles.
"""

import cube

# A solver move on a face sitting in one of these slots is reachable as-is.
REACHABLE = ("U", "R", "F")

# Any of these rotations brings a given awkward slot into reach; which one to
# use is chosen at rewrite time by short lookahead, because a rotation that
# also happens to park the NEXT few faces within reach saves a rotation later.
# Naively taking the first valid one produces sequences that thrash
# ("X' ... X ... X' ...") and run about 55% longer than the solver output.

MOVE_WORDS = {
    "U": "TOP", "D": "BOTTOM", "L": "LEFT",
    "R": "RIGHT", "F": "FRONT", "B": "BACK",
}
TURN_WORDS = {"": "clockwise 90\u00b0", "'": "anticlockwise 90\u00b0",
              "2": "a half turn (180\u00b0)"}

ROTATION_WORDS = {
    "X": "tip the WHOLE CUBE so the top face comes toward you",
    "X'": "tip the WHOLE CUBE so the bottom face comes toward you",
    "X2": "tip the WHOLE CUBE end over end (top face ends up on the bottom)",
    "Y": "spin the WHOLE CUBE so the right face comes toward you",
    "Y'": "spin the WHOLE CUBE so the left face comes toward you",
    "Y2": "spin the WHOLE CUBE half a turn (front face ends up at the back)",
    "Z": "roll the WHOLE CUBE clockwise, front face still facing you",
    "Z'": "roll the WHOLE CUBE anticlockwise, front face still facing you",
    "Z2": "roll the WHOLE CUBE upside down, front face still facing you",
}


class SolverUnavailable(RuntimeError):
    pass


class Unsolvable(ValueError):
    pass


def _centre_index(slot):
    return cube.FACE_ORDER.index(slot) * 9 + 4


def _slot_maps():
    """For each whole-cube rotation, {face that was here: slot it moves to}."""
    maps = {}
    for name in ("X", "Y", "Z"):
        for suffix in ("", "'", "2"):
            move = name + suffix
            perm = cube.ALL_TURNS[move]
            maps[move] = {
                cube.face_of(perm[_centre_index(slot)]): slot
                for slot in cube.FACE_ORDER
            }
    return maps


SLOT_MAPS = _slot_maps()

# The names in ROTATION_WORDS have to match what the permutations actually do.
assert SLOT_MAPS["X"]["U"] == "F", "X does not take the top face to the front"
assert SLOT_MAPS["Y"]["R"] == "F", "Y does not take the right face to the front"
assert SLOT_MAPS["Z"]["U"] == "R", "Z does not take the top face to the right"


def solve_state(state):
    """Return the solver's move list for a validated state string."""
    cube.validate(state)

    try:
        import kociemba
    except ImportError:
        try:
            import twophase.solver as _tp
        except ImportError:
            raise SolverUnavailable(
                "no solver installed. Run:  pip install kociemba\n"
                "(or 'pip install RubikTwoPhase' for the pure-Python one, "
                "which needs no compiler)"
            ) from None
        raw = _tp.solve(state)
        if raw.startswith("Error"):
            raise Unsolvable(_explain(raw))
        # RubikTwoPhase appends a "(Nf)" move count; drop it.
        return [m for m in raw.split() if not m.startswith("(")]

    try:
        raw = kociemba.solve(state)
    except Exception as exc:                     # its errors are plain strings
        raise Unsolvable(_explain(str(exc))) from None
    return raw.split()


def _explain(message):
    return (
        f"the solver rejected this cube state ({message.strip()}).\n"
        "That almost always means one of:\n"
        "  * a sticker was misread -- check the net printed above against the "
        "cube;\n"
        "  * the cube was turned the WRONG WAY during scanning, which records "
        "a mirror image of the real cube. Rescan, turning so the face that "
        "was on your RIGHT comes to the camera;\n"
        "  * the cube is physically disassembled/reassembled wrong, in which "
        "case no solution exists."
    )


def humanize(state, solution, target=None):
    """Rewrite `solution` using only R/U/F turns plus whole-cube rotations.

    `target` is where the sequence is supposed to end up, defaulting to a
    solved cube. Pattern routes finish somewhere else, and the check below
    would reject a perfectly good sequence without being told.

    Returns a list of steps, each a dict:
        {"kind": "turn"|"rotate", "move": "R'", "text": "...", "notation": "R'"}
    where "notation" is the ORIGINAL solver move for turns, so the user can
    still follow along in standard notation if they want to.
    """
    # orient[solver_face] = the slot that face currently occupies
    orient = {f: f for f in cube.FACE_ORDER}
    steps = []

    for index, move in enumerate(solution):
        face, suffix = move[0], move[1:]
        if face not in cube.FACE_ORDER:
            raise ValueError(f"unexpected move from solver: {move!r}")

        slot = orient[face]
        if slot not in REACHABLE:
            rot = _best_rotation(orient, slot, solution[index + 1:])
            mapping = SLOT_MAPS[rot]
            orient = {f: mapping[s] for f, s in orient.items()}
            steps.append({
                "kind": "rotate", "move": rot, "notation": rot,
                "text": ROTATION_WORDS[rot],
            })
            slot = orient[face]
            assert slot in REACHABLE, (move, slot)

        steps.append({
            "kind": "turn", "move": slot + suffix, "notation": move,
            "text": f"turn the {MOVE_WORDS[slot]} face {TURN_WORDS[suffix]}",
        })

    _verify(state, steps, target)
    return steps


def _best_rotation(orient, slot, upcoming, lookahead=6):
    """Pick the whole-cube rotation that brings `slot` into reach and keeps
    as many of the next few moves reachable as possible.

    Ties break toward a quarter turn about the vertical axis, which is the
    least disorienting thing to ask someone to do while holding a cube.
    """
    best, best_score = None, None
    for rot, mapping in SLOT_MAPS.items():
        if mapping[slot] not in REACHABLE:
            continue
        after = {f: mapping[s] for f, s in orient.items()}
        free = 0
        for nxt in upcoming[:lookahead]:
            if after[nxt[0]] in REACHABLE:
                free += 1
            else:
                break
        score = (free, rot.startswith("Y"), len(rot) == 1)
        if best_score is None or score > best_score:
            best, best_score = rot, score
    assert best is not None, f"no rotation brings {slot} into reach"
    return best


def _verify(state, steps, target=None):
    """Confirm the rewritten sequence really does arrive where it claims.

    The rewrite inserts whole-cube rotations, so the cube can finish correct
    but held a different way up. "Arrived" therefore means the result is the
    target in ANY of its 24 orientations, not that the strings match.
    """
    result = cube.apply_sequence(state, [s["move"] for s in steps])
    if result not in cube.orientations(target or cube.SOLVED):
        raise AssertionError(
            "internal error: the rewritten move sequence does not reach the "
            "target. Do not follow it. Please report this with the state: "
            f"{state}"
        )


def check_solution(state, solution):
    """Confirm the solver's own moves solve the cube. Cheap; always worth it."""
    return cube.apply_sequence(state, solution) == cube.SOLVED
