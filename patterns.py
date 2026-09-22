"""
Pattern catalogue: pretty cube positions, and how to reach one from wherever
your cube happens to be right now.

REACHING A PATTERN FROM A SCRAMBLED CUBE
----------------------------------------
A pattern is normally published as a move sequence you apply to a SOLVED cube.
That is useless if your cube is scrambled, and the obvious fix -- solve it
first, then apply the pattern -- works but costs about twenty wasted moves.

You do not need them. A pattern is really a target STATE, and a solver is a
machine for computing the moves between two states; it is only ever pointed at
"this state -> solved" by convention. The trick is to point it somewhere else.

Write `s . M` for applying move sequence M to state s. We want M with

    current . M = target

Take the naive route N = solve(current) + pattern_moves, which certainly works
but is long. Now build the state

    Y = solved . N⁻¹

Solving Y means finding a sequence that undoes N⁻¹ -- that is, any sequence
having exactly N's effect. Every solution of Y is the same cube-group element
as N, so it transforms `current` into `target` just as N does, but the solver
returns its own near-optimal sequence rather than your concatenation. Roughly
twenty moves instead of forty, from any starting position, for any pattern.

The same machinery also routes from one pattern straight to another, and the
pattern's own sequence length stops mattering -- which is why the discovered
shapes below can be stored with whatever sequence the search happened to find.

Patterns may be defined by moves (the classic ones, quoted as published, slice
moves and all) or by an explicit target state (the searched ones). Slice moves
never reach the user: the route always comes back as ordinary face turns.
"""

import cube

try:                                    # only needed for routing, not browsing
    import solve as _solve
except Exception:                       # pragma: no cover
    _solve = None


class Pattern:
    __slots__ = ("name", "category", "moves", "state", "note")

    def __init__(self, name, category, moves=None, note="", *, state=None):
        # `state` is keyword-only on purpose. With it fourth and positional,
        # a catalogue entry written Pattern(name, cat, moves, "description")
        # silently files the description as the target state, and every
        # pattern's target() then returns prose instead of a cube.
        if not moves and not state:
            raise ValueError(f"{name}: needs either moves or a state")
        if state is not None and len(state) != 54:
            raise ValueError(f"{name}: a state is 54 facelets, got {len(state)}")
        self.name = name
        self.category = category
        self.moves = moves.split() if isinstance(moves, str) else (moves or None)
        self.state = state
        self.note = note

    def target(self):
        """The 54-facelet state this pattern is, in solver face letters."""
        if self.state:
            return self.state
        return cube.apply_sequence(cube.SOLVED, self.moves)

    def __repr__(self):
        return f"<Pattern {self.name!r}>"


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------
# Classic patterns, quoted as they are published. Every one of these is
# rendered and eyeballed in the catalogue picker, so what you see is what the
# sequence actually does rather than what its name promises.

CATALOGUE = [
    # M2 E2 S2 and F2 B2 U2 D2 L2 R2 reach the SAME position -- it is one
    # pattern carrying two traditional names, not two patterns.
    Pattern("Checkerboard (Pons Asinorum)", "classic", "M2 E2 S2",
            "Every face a two-colour chequer. Also reached by "
            "F2 B2 U2 D2 L2 R2"),
    Pattern("Four Spots", "spots", "F2 B2 U D' R2 L2 U D'",
            "Four faces reduced to a single centre dot"),
    Pattern("Six Spots", "spots", "U D' R L' F B' U D'",
            "Every face a lone centre dot in a foreign colour"),
    Pattern("Twister", "classic", "F R' U L F' L' F U' R U L' U' L F'",
            "Looks like the cube has been wrung out"),
    Pattern("Tetris", "classic", "L R F B U' D' L' R'",
            "Interlocking blocks"),
    Pattern("Cube in a Cube", "nested", "F L F U' R U F2 L2 U' L' B D' B' L2 U",
            "A smaller cube apparently floating in the corner"),
    Pattern("Cube in a Cube in a Cube", "nested",
            "U' L' U' F' R2 B' R F U B2 U B' L U' F U R F'",
            "Three nested cubes"),
    Pattern("Superflip", "hard",
            "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2",
            "Every edge flipped in place -- one of the positions that needs "
            "all 20 moves to solve"),
    Pattern("Anaconda", "snakes", "L U B' U' R L' B R' F B' D R D' F'",
            "A snake winding round the cube"),
    Pattern("Python", "snakes", "F2 R' B' U R' L F' L F' B D' R B L2",
            "A fatter snake"),
    Pattern("Black Mamba", "snakes", "R D L F' R L' D R' U D' B U' R' D'",
            "A third snake"),
    Pattern("Green Mamba", "snakes", "R D R F R' F' B D R' U' B' U D2",
            "A fourth snake"),
    Pattern("Gift Box", "classic",
            "U B2 R2 B2 L2 F2 R2 D' F2 L2 B F' L F2 D U' R2 D R2 U'",
            "The cube wrapped up with a ribbon"),
    Pattern("Vertical Stripes", "stripes",
            "F U F R L2 B D' R D2 L D' B R2 L F U F",
            "Stripes running round the cube"),
]

# Shapes and letters found by searching every move sequence up to six moves
# deep and recording, for each of the 256 possible front-face masks, the
# sequence that leaves the REST of the cube tidiest. All 256 masks turn out to
# be reachable within six moves, so any 3x3 glyph can be drawn -- the limit is
# legibility, not the cube. The comment after each is the glyph, with '#' for
# the contrasting colour and '.' for the face's own colour.
#
# The centre sticker can never move, so it is always the face's own colour.
# A glyph whose middle cell is part of the stroke is therefore drawn IN the
# face colour on a contrasting ground; one whose middle cell is not is drawn
# in the contrasting colour instead. Both read fine, and it is why "Ring",
# "Dot" and "Letter O" are one and the same picture.
GLYPHS = [
    Pattern("Heart", "shapes", "R2 L2 U2 R2 L2",
            "A heart on the front face"),   # #.#/###/.#.
    # The shape search rediscovered the classic Plus Minus sequence exactly.
    Pattern("Cross (Plus Minus)", "shapes", "U2 R2 L2 U2 R2 L2",
            "A plus on each face, corners in a contrasting colour"),   # .#./###/.#.
    Pattern("X", "shapes", "U2 F2 R2 L2 F2 D2",
            "A diagonal cross on the front face"),   # #.#/.#./#.#
    Pattern("Ring", "shapes", "U2 R2 L2 D2",
            "A centre dot ringed by one colour (also reads as O)"),   # ###/#.#/###
    Pattern("Bar", "shapes", "U2 F2 B2 U2 F2 B2",
            "A horizontal bar across the front face"),   # .../###/...
    Pattern("Pillar", "shapes", "R2 F2 B2 R2 F2 B2",
            "A vertical bar down the front face"),   # .#./.#./.#.
    Pattern("Letter C", "letters", "U2 L2 D2",
            "The letter C on the front face"),   # ###/#../###
    Pattern("Letter F", "letters", "R' U2 L F2 L' U2",
            "The letter F on the front face"),   # ###/##./#..
    Pattern("Letter H", "letters", "U2 R2 L2 D2 R2 L2",
            "The letter H on the front face"),   # #.#/###/#.#
    Pattern("Letter I", "letters", "U2 R2 U2 D2 L2 D2",
            "The letter I on the front face"),   # ###/.#./###
    Pattern("Letter J", "letters", "U F U' D R D'",
            "The letter J on the front face"),   # ..#/..#/###
    Pattern("Letter L", "letters", "U' F' U D' L' D",
            "The letter L on the front face"),   # #../#../###
    Pattern("Letter P", "letters", "L' D' L D",
            "The letter P on the front face"),   # ###/###/#..
    Pattern("Letter T", "letters", "D2 B U2 D2 F U2",
            "The letter T on the front face"),   # ###/.#./.#.
    Pattern("Letter U", "letters", "R2 D2 L2",
            "The letter U on the front face"),   # #.#/#.#/###
    Pattern("Letter V", "letters", "R' B' L' B D R",
            "The letter V on the front face"),   # #.#/#.#/.#.
    Pattern("Letter Y", "letters", "R B L B' U' R'",
            "The letter Y on the front face"),   # #.#/.#./.#.
]

# Shapes discovered by search_face_shape() and verified before being added.
# They are stored as explicit states because the sequence the search happened
# to find is irrelevant -- route() recomputes the way there from your cube.
DISCOVERED = []


def all_patterns():
    """Every pattern, grouped by category.

    Grouped rather than in definition order so the listing and the picker grid
    do not show the same category heading twice. Stable within a category, so
    the order inside each block is the order they are written above.
    """
    everything = CATALOGUE + GLYPHS + DISCOVERED
    order = []
    for pat in everything:
        if pat.category not in order:
            order.append(pat.category)
    return sorted(everything, key=lambda pat: order.index(pat.category))


def by_name(name):
    wanted = name.strip().lower()
    for p in all_patterns():
        if p.name.lower() == wanted:
            return p
    for p in all_patterns():                       # allow a unique prefix
        if p.name.lower().startswith(wanted):
            return p
    return None


def categories():
    out = []
    for p in all_patterns():
        if p.category not in out:
            out.append(p.category)
    return out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class RouteError(RuntimeError):
    pass


def _solver():
    if _solve is None:
        raise RouteError("solve.py could not be imported")
    return _solve.solve_state


def moves_from_solved(pattern, solve_fn=None):
    """A sequence taking a solved cube to this pattern."""
    if pattern.moves:
        return list(pattern.moves)
    solve_fn = solve_fn or _solver()
    return cube.invert_sequence(solve_fn(pattern.target()))


def route(current, pattern, solve_fn=None):
    """Moves taking `current` to `pattern`, from any starting position.

    Returns a list of ordinary face turns. See the module docstring for why
    this is about twenty moves rather than the forty you get from solving
    first and then applying the pattern.
    """
    solve_fn = solve_fn or _solver()
    target = pattern.target()
    cube.validate(current)
    if current == target:
        return []

    naive = solve_fn(current) + moves_from_solved(pattern, solve_fn)

    # Any solution of this state has exactly the effect of `naive`.
    relative = cube.apply_sequence(cube.SOLVED, cube.invert_sequence(naive))
    optimised = solve_fn(relative)

    best = min((seq for seq in (optimised, naive)
                if cube.apply_sequence(current, seq) == target),
               key=len, default=None)
    if best is None:
        raise RouteError(
            "internal error: no route verified. Refusing to show moves that "
            f"do not reach the pattern. current={current} pattern={pattern.name}"
        )
    return best


# ---------------------------------------------------------------------------
# Finding new shapes
# ---------------------------------------------------------------------------
# A 3x3 shape drawn on one face. The centre sticker can never move, so it is
# always the face's own colour -- which means a shape whose middle cell is
# part of the drawing must be drawn IN the face colour, and one whose middle
# cell is not must be drawn in a contrasting colour around it. Both read fine;
# the mask below says which cells are the face colour.

SHAPES = {
    "cross":     "010111010",
    "plus":      "010111010",
    "x":         "101010101",
    "o":         "111101111",
    "dot":       "000010000",
    "heart":     "101111010",
    "t":         "111010010",
    "h":         "101111101",
    "i":         "111010111",
    "l":         "100100111",
    "u":         "101101111",
    "c":         "111100111",
    "s":         "111010111",
    "z":         "111010111",
    "checker":   "101010101",
    "corners":   "101000101",
    "stripe":    "010010010",
    "bar":       "000111000",
}


def face_signature(state, face="F"):
    """(mask, background) for one face, or (mask, None) if not uniform.

    mask[i] is True where the sticker is the face's own colour. background is
    the single colour of everything else, when there is one.
    """
    base = cube.FACE_ORDER.index(face) * 9
    stickers = state[base:base + 9]
    own = stickers[4]
    mask = tuple(c == own for c in stickers)
    rest = {c for c in stickers if c != own}
    return mask, (rest.pop() if len(rest) == 1 else None)


def mask_from_string(text):
    """"010111010" -> a 9-tuple of booleans, reading row by row."""
    flat = "".join(text.split())
    if len(flat) != 9 or set(flat) - set("01"):
        raise ValueError("a shape is 9 characters of 0 and 1")
    return tuple(c == "1" for c in flat)


def search_face_shape(mask, face="F", max_depth=7, uniform_background=True,
                      progress=None):
    """Shortest move sequence showing `mask` on `face`. None if not found.

    Iterative deepening over face turns, skipping consecutive turns of the
    same face (two in a row always collapse into one). Depth 7 is about 90
    million sequences, so this is a minutes-not-seconds tool -- which is why
    the results it finds are cached into DISCOVERED rather than searched for
    at runtime.
    """
    moves = [f + suffix for f in "URFDLB" for suffix in ("", "'", "2")]
    perms = {m: cube.ALL_TURNS[m] for m in moves}
    base = cube.FACE_ORDER.index(face) * 9

    def matches(state):
        stickers = state[base:base + 9]
        own = stickers[4]
        if tuple(c == own for c in stickers) != mask:
            return False
        if not uniform_background:
            return True
        rest = {c for c in stickers if c != own}
        return len(rest) <= 1

    if matches(cube.SOLVED):
        return []

    def descend(state, depth, last_face, trail):
        for m in moves:
            if m[0] == last_face:
                continue
            perm = perms[m]
            nxt = "".join(state[perm[i]] for i in range(54))
            trail.append(m)
            if matches(nxt):
                return list(trail)
            if depth > 1:
                found = descend(nxt, depth - 1, m[0], trail)
                if found:
                    return found
            trail.pop()
        return None

    for depth in range(1, max_depth + 1):
        if progress:
            progress(depth)
        found = descend(cube.SOLVED, depth, None, [])
        if found:
            return found
    return None


def register_discovered(name, category, state, note=""):
    """Add a searched shape to the catalogue after checking it is a real cube."""
    cube.validate(state)
    DISCOVERED.append(Pattern(name, category, state=state, note=note))
    return DISCOVERED[-1]
