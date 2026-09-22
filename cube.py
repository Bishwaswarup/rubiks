"""
Cube geometry, facelet indexing, move tables and a facelet-level simulator.

Every move table in here is DERIVED FROM GEOMETRY at import time rather than
typed in by hand.  Hand-written facelet tables are the single most common
source of silent, maddening bugs in a project like this: an off-by-one or a
flipped column produces a cube state that looks plausible, validates fine, and
then yields a solution that does not solve the cube.  Deriving them from
coordinates means the only thing that can be wrong is the coordinate frame,
and selftest.py pins that down with round-trip tests against the real solver.

FACELET NUMBERING (the Kociemba convention)
-------------------------------------------
54 facelets in the order U, R, F, D, L, B; each face read row-major (left to
right, then top to bottom) as seen when that face is turned toward you:

                 +----------------+
                 |  0    1    2   |
                 |  3   U4    5   |
                 |  6    7    8   |
   +-------------+----------------+-------------+-------------+
   | 36  37  38  |  18  19  20    |  9  10  11  | 45  46  47  |
   | 39 L40  41  |  21  F22  23   | 12 R13  14  | 48 B49  50  |
   | 42  43  44  |  24  25  26    | 15  16  17  | 51  52  53  |
   +-------------+----------------+-------------+-------------+
                 | 27   28   29   |
                 | 30  D31   32   |
                 | 33   34   35   |
                 +----------------+

The two faces whose reading direction is easy to get wrong:

  U is read with B at the top and F at the bottom  (facelets 0,1,2 touch B).
  D is read with F at the top and B at the bottom  (facelets 27,28,29 touch F).

COORDINATE FRAME
----------------
Right-handed, from the solver's point of view:  +x = right, +y = up,
+z = toward the viewer (front).  Each facelet is identified by the pair
(outward normal, position), both integer vectors with components in {-1,0,1}.
That pair is unique: three facelets share a corner position but have three
different normals.

A face turn is a rotation about that face's outward normal by -90 degrees in
the right-hand sense, which is exactly "clockwise as seen from outside the
cube" -- the standard meaning of U, R, F, D, L, B.
"""

FACE_ORDER = "URFDLB"

# Outward normal of each face.
NORMAL = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}

# For each face: the direction you move in 3-space when you step DOWN one row,
# and when you step RIGHT one column, in that face's canonical reading order.
ROW_DIR = {
    "U": (0, 0, 1),    # rows run from the B edge toward the F edge
    "R": (0, -1, 0),
    "F": (0, -1, 0),
    "D": (0, 0, -1),   # rows run from the F edge toward the B edge
    "L": (0, -1, 0),
    "B": (0, -1, 0),
}
COL_DIR = {
    "U": (1, 0, 0),
    "R": (0, 0, -1),   # col 0 touches F, col 2 touches B
    "F": (1, 0, 0),
    "D": (1, 0, 0),
    "L": (0, 0, 1),    # col 0 touches B, col 2 touches F
    "B": (-1, 0, 0),   # col 0 touches R, col 2 touches L
}


def _add(*vs):
    return tuple(sum(c) for c in zip(*vs))


def _scale(v, k):
    return tuple(k * c for c in v)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rot_neg90(v, axis):
    """Rotate integer vector v by -90 degrees about unit `axis` (right-hand).

    With cos(-90)=0 and sin(-90)=-1, Rodrigues' formula collapses to
        v' = (v.a) a - (a x v)
    which is why this is three lines instead of a matrix.
    """
    return _add(_scale(axis, _dot(v, axis)), _scale(_cross(axis, v), -1))


# ---------------------------------------------------------------------------
# Facelet <-> (normal, position) tables
# ---------------------------------------------------------------------------

def _build_facelets():
    """index -> (normal, position), plus the reverse lookup."""
    by_index = []
    for face in FACE_ORDER:
        n, rd, cd = NORMAL[face], ROW_DIR[face], COL_DIR[face]
        for i in range(3):
            for j in range(3):
                pos = _add(n, _scale(rd, i - 1), _scale(cd, j - 1))
                by_index.append((n, pos))
    by_key = {key: idx for idx, key in enumerate(by_index)}
    assert len(by_key) == 54, "facelet keys are not unique -- frame is wrong"
    return by_index, by_key


FACELETS, FACELET_INDEX = _build_facelets()


def face_of(index):
    return FACE_ORDER[index // 9]


def _permutation(axis, layer_test):
    """Build perm[] where perm[dst] = src for a rotation about `axis`.

    layer_test(pos) decides which facelets move; everything else stays put.
    """
    perm = list(range(54))
    for src, (normal, pos) in enumerate(FACELETS):
        if not layer_test(pos):
            continue
        dst = FACELET_INDEX[(_rot_neg90(normal, axis), _rot_neg90(pos, axis))]
        perm[dst] = src
    return perm


def _build_moves():
    """Quarter-turn permutations for the six faces."""
    moves = {}
    for face in FACE_ORDER:
        n = NORMAL[face]
        # A facelet belongs to this face's layer iff it sits in the outer
        # slice along the normal, i.e. its position projects to +1 on n.
        moves[face] = _permutation(n, lambda pos, n=n: _dot(pos, n) == 1)
    return moves


MOVES = _build_moves()

# Whole-cube reorientations, used by the move humanizer so the solver's B and
# D turns can be replaced by turns the hands can actually reach.
#   X  = tip the cube so the top face comes toward you  (U -> F)
#   Y  = turn the cube so the right face comes toward you (R -> F)
#   Z  = roll the cube clockwise, front stays front      (U -> R)
ROTATIONS = {
    # NB the axis is -x, not +x: a -90 turn about +x would take U to B
    # (that is X'), whereas "tip the top toward you" is U -> F.
    "X": _permutation((-1, 0, 0), lambda pos: True),
    "Y": _permutation((0, 1, 0), lambda pos: True),
    "Z": _permutation((0, 0, 1), lambda pos: True),
}
# Sanity: check the three named effects above rather than trusting the sign.
assert _rot_neg90(NORMAL["U"], (-1, 0, 0)) == NORMAL["F"], "X is inverted"
assert _rot_neg90(NORMAL["R"], (0, 1, 0)) == NORMAL["F"], "Y is inverted"
assert _rot_neg90(NORMAL["U"], (0, 0, 1)) == NORMAL["R"], "Z is inverted"


# Slice moves. These exist ONLY so that the classic pattern sequences in
# patterns.py can be written the way the literature states them -- many of
# them are defined with M/E/S. They never appear in an instruction the user is
# asked to perform: patterns are turned into a target STATE, and the route to
# that state comes back from the solver as ordinary face turns.
#
#   M  the middle layer between L and R, turning the way L does
#   E  the equatorial layer between U and D, turning the way D does
#   S  the standing layer between F and B, turning the way F does
SLICES = {
    "M": _permutation(NORMAL["L"], lambda pos: pos[0] == 0),
    "E": _permutation(NORMAL["D"], lambda pos: pos[1] == 0),
    "S": _permutation(NORMAL["F"], lambda pos: pos[2] == 0),
}


def _invert(perm):
    out = [0] * 54
    for dst, src in enumerate(perm):
        out[src] = dst
    return out


def _compose(perm, times):
    result = list(range(54))
    for _ in range(times):
        result = [result[s] for s in perm]
    return result


ALL_TURNS = {}
for _name, _p in (list(MOVES.items()) + list(ROTATIONS.items())
                  + list(SLICES.items())):
    ALL_TURNS[_name] = _p
    ALL_TURNS[_name + "2"] = _compose(_p, 2)
    ALL_TURNS[_name + "'"] = _invert(_p)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

SOLVED = "".join(f * 9 for f in FACE_ORDER)


def apply_move(state, move):
    """Apply one move ('R', "U'", 'F2', or a rotation 'X'/'Y'/'Z') to a state."""
    try:
        perm = ALL_TURNS[move]
    except KeyError:
        raise ValueError(f"unknown move {move!r}") from None
    return "".join(state[perm[i]] for i in range(54))


def apply_sequence(state, moves):
    if isinstance(moves, str):
        moves = moves.split()
    for m in moves:
        state = apply_move(state, m)
    return state


def invert_sequence(moves):
    if isinstance(moves, str):
        moves = moves.split()
    flip = {"": "'", "'": "", "2": "2"}
    out = []
    for m in reversed(moves):
        base, suffix = m[0], m[1:]
        out.append(base + flip[suffix])
    return out


# ---------------------------------------------------------------------------
# Pieces, for validating a scan
# ---------------------------------------------------------------------------

def _piece_groups():
    """Group facelet indices by position: 3 per corner, 2 per edge."""
    groups = {}
    for idx, (_n, pos) in enumerate(FACELETS):
        groups.setdefault(pos, []).append(idx)
    corners = [tuple(v) for v in groups.values() if len(v) == 3]
    edges = [tuple(v) for v in groups.values() if len(v) == 2]
    assert len(corners) == 8 and len(edges) == 12
    return corners, edges


CORNERS, EDGES = _piece_groups()

def _sig(letters):
    """Order-independent signature of a piece: its face letters, sorted.

    Note this deliberately does NOT use frozensets. Sorting a list of
    frozensets sorts them by the subset partial order, which is not a total
    order, so sorted() returns something order-dependent and two equal
    collections can compare unequal. Sorted strings are a total order.
    """
    return "".join(sorted(letters))


_VALID_CORNERS = sorted(_sig(face_of(i) for i in c) for c in CORNERS)
_VALID_EDGES = sorted(_sig(face_of(i) for i in e) for e in EDGES)


class InvalidCube(ValueError):
    """A scanned state that cannot be a real cube. Message is user-facing."""


def validate(state):
    """Raise InvalidCube with a human-readable reason, or return None.

    Catching these here matters: fed a slightly misread state, the solver
    either raises something cryptic or searches for a very long time. A clear
    "rescan the orange face" beats both.
    """
    if len(state) != 54:
        raise InvalidCube(f"expected 54 facelets, got {len(state)}")

    centers = [state[4], state[13], state[22], state[31], state[40], state[49]]
    if len(set(centers)) != 6:
        dupes = sorted(c for c in set(centers) if centers.count(c) > 1)
        raise InvalidCube(
            "two faces were scanned with the same centre colour "
            f"({', '.join(dupes)}). A real cube has one centre of each colour, "
            "so at least one face was captured twice or misread."
        )

    for colour in sorted(set(state)):
        n = state.count(colour)
        if n != 9:
            raise InvalidCube(
                f"colour {colour!r} appears {n} times, not 9. "
                "At least one sticker was misread."
            )

    got_corners = sorted(_sig(state[i] for i in c) for c in CORNERS)
    if got_corners != _VALID_CORNERS:
        bad = [g for g in got_corners if g not in _VALID_CORNERS]
        raise InvalidCube(
            "the corner pieces are not all real cube pieces"
            + (f" (e.g. {bad[0]})" if bad else "")
            + ". Two stickers on one corner were probably swapped."
        )

    got_edges = sorted(_sig(state[i] for i in e) for e in EDGES)
    if got_edges != _VALID_EDGES:
        bad = [g for g in got_edges if g not in _VALID_EDGES]
        raise InvalidCube(
            "the edge pieces are not all real cube pieces"
            + (f" (e.g. {bad[0]})" if bad else "")
            + ". Two stickers on one edge were probably swapped."
        )


def orientations(state):
    """The 24 states you get by turning the whole cube without turning a layer.

    Needed whenever you have to ask "did we arrive?" after a sequence that
    contains whole-cube rotations: the cube can finish correct but held a
    different way up, and comparing state strings directly would call that a
    failure.
    """
    seen, frontier = {state}, [state]
    while frontier:
        cur = frontier.pop()
        for rot in ("X", "Y", "Z", "X'", "Y'", "Z'"):
            nxt = apply_move(cur, rot)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def pretty(state, colours=None):
    """Unfolded ASCII net, for eyeballing a scan before solving."""
    def cell(i):
        c = state[i]
        return (colours or {}).get(c, c)

    rows = []
    for i in range(3):
        rows.append("        " + " ".join(cell(0 + 3 * i + j) for j in range(3)))
    for i in range(3):
        rows.append(" ".join(
            " ".join(cell(off + 3 * i + j) for j in range(3))
            for off in (36, 18, 9, 45)))
    for i in range(3):
        rows.append("        " + " ".join(cell(27 + 3 * i + j) for j in range(3)))
    return "\n".join(rows)
