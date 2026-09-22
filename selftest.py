#!/usr/bin/env python3
"""
Self-tests. Run this after touching cube.py, scan.py's IN_PLANE_ROT, or solve.py.

    python3 selftest.py

The tests that matter most are the round trips: they are the only things that
can catch a wrong facelet index or a wrong in-plane rotation, because such a
bug produces a state that looks entirely legitimate and only fails when you
actually try to solve the physical cube.
"""

import random
import sys

import arrows
import colors
import cube
import guide
import scan
import solve

PASS, FAIL = "  ok  ", " FAIL "


def _synthetic_face(size=480):
    """A plain 3x3 cube face on a flat background, for the detector test."""
    import numpy as np
    import cv2
    frame = np.full((size, size + 160, 3), 205, np.uint8)
    cell = size // 6
    x0 = y0 = size // 4
    for i in range(3):
        for j in range(3):
            letter = "GRWYGBOGR"[3 * i + j]
            cv2.rectangle(frame, (x0 + j * cell + 3, y0 + i * cell + 3),
                          (x0 + j * cell + cell - 3, y0 + i * cell + cell - 3),
                          colors.DISPLAY_BGR[letter], -1)
    cv2.rectangle(frame, (x0, y0), (x0 + 3 * cell, y0 + 3 * cell),
                  (30, 30, 30), 3)
    return frame

results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print((PASS if condition else FAIL) + name + (f"  {detail}" if detail and not condition else ""))


def group(title):
    print()
    print(title)
    print("-" * len(title))


try:
    import kociemba as _k  # noqa: F401
    _HAVE_SOLVER = True
except ImportError:
    try:
        import twophase.solver as _t  # noqa: F401
        _HAVE_SOLVER = True
    except ImportError:
        _HAVE_SOLVER = False

# ---------------------------------------------------------------------------
group("cube geometry and move tables")

check("54 unique facelet keys", len(cube.FACELET_INDEX) == 54)
check("8 corners and 12 edges", len(cube.CORNERS) == 8 and len(cube.EDGES) == 12)

for m in "URFDLB":
    check(f"{m}^4 = identity", cube.apply_sequence(cube.SOLVED, [m] * 4) == cube.SOLVED)
for r in "XYZ":
    check(f"{r}^4 = identity", cube.apply_sequence(cube.SOLVED, [r] * 4) == cube.SOLVED)

t = cube.SOLVED
for _ in range(6):
    t = cube.apply_sequence(t, "R U R' U'")
check("(R U R' U')^6 = identity", t == cube.SOLVED)

# The order of (R U) in the cube group is famously 105, and it is 105 only
# if BOTH move tables and their relative orientation are right -- a much
# sharper test than each move having order 4.
def _order(seq, cap=500):
    t, n = cube.SOLVED, 0
    while n <= cap:
        t = cube.apply_sequence(t, seq)
        n += 1
        if t == cube.SOLVED:
            return n
    return None

check("(R U) has order 105", _order("R U") == 105, f"got {_order('R U')}")
check("(R U R' U') has order 6", _order("R U R' U'") == 6)

check("a whole-cube rotation keeps every face uniform",
      all(len(set(cube.apply_sequence(cube.SOLVED, [r])[i * 9:i * 9 + 9])) == 1
          for r in "XYZ" for i in range(6)))

check("inverse of a sequence undoes it",
      cube.apply_sequence(
          cube.apply_sequence(cube.SOLVED, "R U F' D2 L B'"),
          cube.invert_sequence("R U F' D2 L B'")) == cube.SOLVED)

# ---------------------------------------------------------------------------
group("state validation")

cube.validate(cube.SOLVED)
random.seed(0)
ok = True
for _ in range(300):
    seq = [random.choice("URFDLB") + random.choice(["", "'", "2"]) for _ in range(25)]
    try:
        cube.validate(cube.apply_sequence(cube.SOLVED, seq))
    except cube.InvalidCube:
        ok = False
        break
check("300 random scrambles all validate", ok)

bad = list(cube.SOLVED); bad[0], bad[9] = bad[9], bad[0]
try:
    cube.validate("".join(bad)); rejected = False
except cube.InvalidCube:
    rejected = True
check("a swapped pair of stickers is rejected", rejected)

try:
    cube.validate("U" * 54); rejected = False
except cube.InvalidCube:
    rejected = True
check("an all-one-colour cube is rejected", rejected)

# ---------------------------------------------------------------------------
group("colour classification (measured references)")

MEASURED = {"B": (153, 93, 36), "O": (63, 91, 216), "Y": (77, 178, 195),
            "R": (65, 53, 209), "W": (204, 205, 209), "G": (134, 197, 52)}
for letter, bgr in MEASURED.items():
    check(f"centre {colors.NAMES[letter]:6} classifies as itself",
          colors.classify(bgr) == letter)

check("red and orange are separated by the sign of G-B",
      colors.features(MEASURED["R"])[2] < 0 < colors.features(MEASURED["O"])[2])
check("white is the only low-saturation colour",
      colors.features(MEASURED["W"])[1] < colors.WHITE_SAT_MAX
      and all(colors.features(v)[1] > colors.WHITE_SAT_MAX
              for k, v in MEASURED.items() if k != "W"))

full = [MEASURED[k] for k in "WROYGB" for _ in range(9)]
check("assign_all respects the 9-of-each quota",
      colors.assign_all(full) == [k for k in "WROYGB" for _ in range(9)])

# A sticker nudged most of the way toward orange still gets called red when
# red's quota is open and orange's is full -- the point of the constraint.
nudged = list(full)
nudged[9] = (64, 78, 212)
out = colors.assign_all(nudged)
check("quota assignment still yields 9 of each on a marginal sticker",
      all(out.count(c) == 9 for c in "WROYGB"))

# ---------------------------------------------------------------------------
group("scan -> state string")

for scheme in (
    {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"},   # one chirality
    {"U": "W", "R": "O", "F": "G", "D": "Y", "L": "R", "B": "B"},   # the mirror
    {"U": "G", "R": "W", "F": "R", "D": "B", "L": "Y", "B": "O"},   # arbitrary
):
    faces = {s: [[scheme[s]] * 3 for _ in range(3)] for s in "URFDLB"}
    state, _c2f = scan.build_state(faces)
    check(f"solved cube scans to SOLVED (scheme {''.join(scheme.values())})",
          state == cube.SOLVED)

check("all six in-plane rotations are zero by construction",
      set(scan.IN_PLANE_ROT.values()) == {0})

# A scrambled virtual cube, rendered into per-face grids the way the camera
# would see them, must scan back to the same state. This is the test that
# catches an in-plane rotation error.
random.seed(3)
ok = True
for _ in range(50):
    seq = [random.choice("URFDLB") + random.choice(["", "'", "2"]) for _ in range(20)]
    state = cube.apply_sequence(cube.SOLVED, seq)
    faces = {s: [[state[cube.FACE_ORDER.index(s) * 9 + 3 * i + j]
                  for j in range(3)] for i in range(3)] for s in cube.FACE_ORDER}
    back, _ = scan.build_state(faces)
    if back != state:
        ok = False
        break
check("50 scrambles survive the grid -> state round trip", ok)

# ---------------------------------------------------------------------------
group("face lock")

check("opposites are an involution with no fixed points",
      all(scan.OPPOSITES[scan.OPPOSITES[c]] == c and scan.OPPOSITES[c] != c
          for c in colors.COLOURS))

exp = scan.expected_colours("W", "G")
check("declaring up+front pins four faces immediately",
      exp == {"U": "W", "D": "Y", "F": "G", "B": "B"})
check("the remaining pair is exactly the two side colours",
      sorted(scan.side_candidates("W", "G")) == ["O", "R"])
check("seeing the right face pins all six",
      scan.expected_colours("W", "G", "R")
      == {"U": "W", "D": "Y", "F": "G", "B": "B", "R": "R", "L": "O"})

for slot, allowed in [("F", ["G"]), ("B", ["B"]), ("U", ["W"]), ("D", ["Y"])]:
    check(f"step {slot} accepts exactly one colour",
          scan._allowed_for(slot, "W", "G", None) == allowed)
check("step R accepts exactly two before chirality is known",
      len(scan._allowed_for("R", "W", "G", None)) == 2)
check("step L is pinned once the right face is known",
      scan._allowed_for("L", "W", "G", "R") == ["O"])

check("a wrong face is refused", scan._check("F", "B", ["G"], []) is not None)
check("an already-seen face is refused",
      scan._check("R", "G", ["G"], ["G"]) is not None)
check("the right face is accepted", scan._check("F", "G", ["G"], []) is None)
try:
    scan.expected_colours("W", "Y")
    rejected = False
except scan.ScanError:
    rejected = True
check("opposite colours cannot be up and front together", rejected)

# ---- hold-to-capture ------------------------------------------------------
# These are regression tests for a bug that made auto-capture impossible: the
# run length was measured from the oldest frame of a deque that had just been
# trimmed to the hold duration, so it could never exceed that duration. The
# only reason it ever fired was perfectly spaced synthetic timestamps, so any
# test using exact arithmetic passes even with the bug present. Every test
# here therefore feeds JITTERED frame times.

def _replay(jitter=0.003, fps=30.0, seconds=25.0, seed=1, flaky=0.0,
            gap=None, hold=3.0):
    """Run frames through a HoldTracker; return the time it captured, or None."""
    import random
    rng = random.Random(seed)
    tracker = scan.HoldTracker(hold=hold)
    now = 0.0
    while now < seconds:
        now += 1.0 / fps + rng.uniform(-jitter, jitter)
        labels = ["G"] * 9
        if rng.random() < flaky:
            labels[8] = "O"
        acceptable = not (gap and gap[0] <= now < gap[1])
        result = tracker.update(now, tuple(labels), [1.0] * 9, acceptable)
        if result["captured"] is not None:
            return now
    return None

for jitter in (0.0, 0.0005, 0.002, 0.005, 0.02):
    fired = _replay(jitter=jitter)
    check(f"auto-capture fires with +/-{jitter * 1000:.1f}ms frame jitter",
          fired is not None and 2.9 <= fired <= 3.6,
          f"fired at {fired}")

check("auto-capture works on a slow 8fps camera",
      (lambda f: f is not None and 2.9 <= f <= 4.0)(_replay(fps=8)))
check("auto-capture tolerates a sticker flickering on 15% of frames",
      (lambda f: f is not None and f < 5.0)(_replay(flaky=0.15)))
# Genuinely ambiguous readings must never fire: no majority can form.
check("auto-capture never fires while a sticker is genuinely ambiguous",
      _replay(flaky=0.5, seconds=25.0) is None)
check("...nor at 45% or 60%",
      _replay(flaky=0.45, seconds=25.0) is None
      and _replay(flaky=0.6, seconds=25.0) is None)
# Note the deliberate limit: if a sticker reads wrong on the great majority of
# frames, that reading IS the steady one and it gets captured. No amount of
# temporal filtering can recover a colour the camera consistently disagrees
# with; that is what the 9-of-each quota assignment and the review-the-net
# step are for.
check("a custom hold duration is honoured",
      (lambda f: f is not None and 1.4 <= f <= 2.1)(_replay(hold=1.5)))

# Taking the face away must restart the count, not resume it.
fired = _replay(gap=(2.0, 2.4))
check("an interruption restarts the countdown rather than resuming",
      fired is not None and fired > 5.0, f"fired at {fired}")

# The run length must be free to grow past the hold duration -- this is the
# exact invariant the original bug violated.
tracker = scan.HoldTracker(hold=3.0)
now = 0.0
for k in range(300):
    now += 1.0 / 30.0 + 0.0007
    reading = tracker.update(now, ("G",) * 9, [1.0] * 9, True)
check("the measured run keeps growing past the hold duration",
      reading["elapsed"] > 6.0, f"elapsed stalled at {reading['elapsed']:.4f}")

# The escape hatch relaxes steadiness only, and only after the full duration.
tracker = scan.HoldTracker(hold=3.0)
check("no escape hatch before the hold has elapsed",
      tracker.update(0.1, ("G",) * 9, [1.0] * 9, True)["long_enough"] is False)
check("a rejected face resets the tracker completely",
      tracker.update(0.2, ("B",) * 9, [1.0] * 9, False)["elapsed"] == 0.0)
tracker = scan.HoldTracker(hold=0.5)
now = 0.0
for k in range(40):
    now += 1.0 / 30.0
    tracker.update(now, ("G",) * 9 if k % 2 else ("O",) + ("G",) * 8,
                   [1.0] * 9, True)
check("the escape hatch becomes available on a long unsteady run",
      tracker.update(now, ("G",) * 9, [1.0] * 9, True)["long_enough"]
      and tracker.force() is not None)
check("unstable cells are reported so the user knows where to look",
      0 in tracker.update(now, ("O",) + ("G",) * 8, [1.0] * 9,
                          True)["unstable"])

# The temporal median: an outlier frame must not reach the scan.
import numpy as np
window = [(0.0, ("G",) * 9, [np.array([10.0, 200.0, 20.0])] * 9),
          (0.1, ("G",) * 9, [np.array([12.0, 198.0, 22.0])] * 9),
          (0.2, ("W",) * 9, [np.array([250.0, 250.0, 250.0])] * 9),
          (0.3, ("G",) * 9, [np.array([11.0, 202.0, 21.0])] * 9)]
merged = scan._consolidate(window, ("G",) * 9)
check("hold-window median ignores the disagreeing frame",
      abs(merged[0][1] - 200.0) < 3.0, f"got {merged[0]}")

# ---- the wrong-way turn: detection and recovery ---------------------------
# Turning the cube the wrong way about the vertical axis on step 2 is the only
# scanning mistake that yields a plausible-looking cube, because both colours
# are legitimate candidates there. It records the L and R faces into each
# other's slots, which is exactly recoverable.

check("all 24 orientations of a standard cube are enumerated",
      len(scan.STANDARD_RIGHT) == 24)
check("white up + green front means red on the right",
      scan.standard_right("W", "G") == "R")
check("the table is consistent under a whole-cube turn",
      scan.standard_right("W", "B") == "O"
      and scan.standard_right("Y", "G") == "O")
check("an impossible pairing has no standard answer",
      scan.standard_right("W", "Y") is None)


def _grids(state):
    """Per-face 3x3 grids exactly as a correct scan would record them."""
    return {f: [[state[cube.FACE_ORDER.index(f) * 9 + 3 * i + j]
                 for j in range(3)] for i in range(3)]
            for f in cube.FACE_ORDER}


random.seed(17)
recovered = legal_when_wrong = swap_of_good_is_legal = 0
TRIALS = 200
for _ in range(TRIALS):
    seq = [random.choice("URFDLB") + random.choice(["", "'", "2"])
           for _ in range(25)]
    truth = cube.apply_sequence(cube.SOLVED, seq)
    good = _grids(truth)
    wrong = scan.swap_sides(good)               # what a wrong-way scan records

    bad_state, _ = scan.build_state(wrong)
    try:
        cube.validate(bad_state)
        legal_when_wrong += 1                   # would slip through undetected
    except cube.InvalidCube:
        pass

    # Undoing the swap must give back the true cube, exactly.
    back_state, _ = scan.build_state(scan.swap_sides(wrong))
    if back_state == truth:
        recovered += 1

    # And the correction must not fire on a scan that was already fine.
    try:
        cube.validate(bad_state)
    except cube.InvalidCube:
        pass
    try:
        cube.validate(scan.build_state(scan.swap_sides(good))[0])
        swap_of_good_is_legal += 1
    except cube.InvalidCube:
        pass

check("undoing the swap restores the true state exactly",
      recovered == TRIALS, f"{recovered}/{TRIALS}")
check("a wrong-way scan is always caught as illegal",
      legal_when_wrong == 0, f"{legal_when_wrong}/{TRIALS} slipped through")
check("swapping a GOOD scan makes it illegal, so recovery cannot misfire",
      swap_of_good_is_legal == 0,
      f"{swap_of_good_is_legal}/{TRIALS} would be wrongly 'corrected'")
check("swap_sides is its own inverse",
      scan.swap_sides(scan.swap_sides(good)) == good)
check("swap_sides leaves the other four faces alone",
      all(scan.swap_sides(good)[f] == good[f] for f in "UFDB"))

# Regression fixture: the real scan a user reported, and the cube they had.
_reported = {"U": ["RRR", "WWW", "WWW"], "R": ["WOO", "WOO", "GGG"],
             "F": ["GGG", "GGG", "RRY"], "D": ["YYO", "YYO", "YYO"],
             "L": ["RRY", "RRY", "BBB"], "B": ["BBB", "BBB", "WOO"]}
_actual = {"U": ["RRR", "WWW", "WWW"], "R": ["RRY", "RRY", "BBB"],
           "F": ["GGG", "GGG", "RRY"], "D": ["YYO", "YYO", "YYO"],
           "L": ["WOO", "WOO", "GGG"], "B": ["BBB", "BBB", "WOO"]}
_as_grids = lambda d: {f: [list(r) for r in d[f]] for f in cube.FACE_ORDER}
_bad, _ = scan.build_state(_as_grids(_reported))
_fix, _ = scan.build_state(scan.swap_sides(_as_grids(_reported)))
_true, _ = scan.build_state(_as_grids(_actual))
try:
    cube.validate(_bad)
    _rejected = False
except cube.InvalidCube:
    _rejected = True
check("reported scan is rejected", _rejected)
check("the correction reproduces the cube the user actually had",
      _fix == _true)
cube.validate(_fix)
check("the corrected cube validates", True)

# ---------------------------------------------------------------------------
group("visual guide geometry")

def _layer_of(move):
    """Facelet indices in the turned layer, on the three visible faces.

    That is the facelets whose contents move, PLUS the turned face's own
    centre sticker -- the centre spins in place, so nothing about its content
    changes, but it is unquestionably part of the layer and the drawing
    highlights it.
    """
    perm = cube.ALL_TURNS[move]
    tokens = list(range(54))
    after = [tokens[perm[i]] for i in range(54)]
    visible = range(27)          # U, R, F occupy indices 0..26
    moved = {i for i in visible if after[i] != tokens[i]}
    return moved | {cube.FACE_ORDER.index(move[0]) * 9 + 4}

for face in "URF":
    flagged = set()
    for k, vis in enumerate(("U", "R", "F")):
        for i in range(3):
            for j in range(3):
                if arrows._moving(vis, i, j, face):
                    flagged.add(k * 9 + 3 * i + j)
    check(f"highlighted layer for {face} matches the real move",
          flagged == _layer_of(face),
          f"drawn {sorted(flagged)} vs actual {sorted(_layer_of(face))}")

# The AR arrows claim specific directions. Check them against the move tables
# rather than against the drawing: U must carry the front face's top row to
# the LEFT face, and R must carry its right column UP to the top face.
def _destination(move, src):
    perm = cube.ALL_TURNS[move]
    for dst in range(54):
        if perm[dst] == src:
            return cube.face_of(dst)
    return None

check("U carries the front top row to the left face (arrow points left)",
      all(_destination("U", 18 + j) == "L" for j in range(3)))
check("R carries the front right column up to the top face (arrow points up)",
      all(_destination("R", 20 + 3 * i) == "U" for i in range(3)))

# Sweeping an arc from u toward v must read as clockwise: at the top of the
# face the motion has to be toward the face's own "view right".
for face in "URF":
    u, v = arrows.FACE_BASIS[face]
    start, end = arrows._sweep("")
    pts = arrows._arc(arrows.FACE_CENTRE[face], u, v, 1.0, start, end,
                      (0.0, 0.0), 100.0, steps=72)
    top = min(range(len(pts)), key=lambda k: pts[k][1])
    ahead = pts[min(top + 3, len(pts) - 1)]
    right_on_screen = arrows._project(
        tuple(arrows.FACE_CENTRE[face][d] + u[d] for d in range(3)),
        (0.0, 0.0), 100.0)
    centre_on_screen = arrows._project(arrows.FACE_CENTRE[face], (0.0, 0.0), 100.0)
    same_way = ((ahead[0] - pts[top][0]) *
                (right_on_screen[0] - centre_on_screen[0])) > 0
    check(f"unprimed arc on {face} sweeps clockwise", same_way)

check("a primed turn sweeps the other way",
      arrows._sweep("'")[1] < arrows._sweep("'")[0]
      and arrows._sweep("")[1] > arrows._sweep("")[0])

check("cube detection finds a square face and ignores a blank frame",
      arrows.detect_cube_quad(_synthetic_face()) is not None
      and arrows.detect_cube_quad(
          np.full((480, 640, 3), 205, np.uint8)) is None)

# ---- the preview must actually depict the state ---------------------------
# Read the rendered image back: sample the centre pixel of every drawn sticker
# and check it holds the colour that facelet has. This is the only way to
# catch a transposed row or column in the isometric mapping, which would make
# the preview disagree with the cube in your hands while every other test here
# still passed.
import numpy as _np
import cv2 as _cv2

_scheme = {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}
_state = cube.apply_sequence(cube.SOLVED, "R U2 F' D L B' R2 U F")
_img = _np.full((620, 760, 3), 24, _np.uint8)
_scale, _origin = 70.0, (380.0, 300.0)
# Drawn with no move, so no arrow can cover a sticker we are about to sample.
arrows.draw_cube(_img, _state, _scheme, _origin, _scale)
_wrong = []
for _face in ("U", "R", "F"):
    _base = cube.FACE_ORDER.index(_face) * 9
    for _i in range(3):
        for _j in range(3):
            _pts = _np.array([arrows._project(q, _origin, _scale)
                              for q in arrows._sticker_corners(_face, _i, _j)],
                             float)
            _cx, _cy = _pts.mean(axis=0).astype(int)
            _got = tuple(int(v) for v in _img[_cy, _cx])
            _want = colors.DISPLAY_BGR[_scheme[_state[_base + 3 * _i + _j]]]
            if _got != _want:
                _wrong.append((_face, _i, _j))
check("all 27 visible stickers in the preview match the state",
      not _wrong, f"wrong: {_wrong[:5]}")

# The view is from above and to the right, so FRONT must be drawn to the LEFT
# of RIGHT, and TOP above both. If this ever flips, the preview is mirrored.
def _span(face):
    xs, ys = [], []
    for i in range(3):
        for j in range(3):
            for q in arrows._sticker_corners(face, i, j):
                sx, sy = arrows._project(q, _origin, _scale)
                xs.append(sx)
                ys.append(sy)
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

_fx, _fy = _span("F")
_rx, _ry = _span("R")
_ux, _uy = _span("U")
check("the preview is not mirrored (front drawn left of right)", _fx < _rx)
check("the top face is drawn above the side faces", _uy < _fy and _uy < _ry)

# Each face's (row 0, col 0) has to be the corner a person would call first.
check("front (0,0) is drawn upper-left of front (2,2)",
      (lambda a, b: a[0] < b[0] and a[1] < b[1])(
          arrows._project(arrows._sticker_corners("F", 0, 0)[0], _origin, _scale),
          arrows._project(arrows._sticker_corners("F", 2, 2)[0], _origin, _scale)))

# ---- the starting orientation must be stated ------------------------------
# Regression test for a real omission: scanning ends with the cube tipped away
# from the reference orientation (the last capture brings the bottom face up
# to the camera), but nothing told the user to go back before step 1. Follow
# the moves from the wrong orientation and nothing matches -- the preview
# included, which is how the bug got noticed.
check("the home orientation is named from the scan's own centres",
      arrows.home_words(_scheme) == "WHITE on top and GREEN facing you")
check("face letters instead of colours produce no bogus wording",
      arrows.home_words({f: f for f in cube.FACE_ORDER}) is None)

_card = arrows.render_home(_state, _scheme, size=(620, 520))
check("an orientation card is rendered before the first move",
      _card.shape == (520, 620, 3) and _card.std() > 20)

import io
import contextlib
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    guide.print_sequence(
        [{"kind": "turn", "move": "R", "notation": "R", "text": "x"}],
        colour_to_face={c: f for f, c in _scheme.items()})
_out = _buf.getvalue()
check("the terminal header states the starting orientation",
      "WHITE on top and GREEN facing you" in _out and "FIRST" in _out,
      _out[:120])

# ---------------------------------------------------------------------------
group("camera settings")

import config

_saved = config.PATH.read_text() if config.PATH.exists() else None
try:
    check("the default camera is 1, not 0",
          config.DEFAULT_CAMERA == 1,
          "macOS Continuity Camera takes 0; the built-in webcam is 1")
    config.set_value("camera", 4)
    check("a saved camera index persists", config.get("camera") == 4)

    import main as _main
    def _resolve(argv):
        a = _main.parse_args(argv)
        return a.camera if a.camera is not None else config.get("camera")

    check("a bare run uses the saved camera", _resolve([]) == 4)
    check("--patterns uses the saved camera too", _resolve(["--patterns"]) == 4)
    check("--ar uses the saved camera too", _resolve(["--ar"]) == 4)
    check("--camera overrides for one run without saving",
          _resolve(["--camera", "0"]) == 0 and config.get("camera") == 4)

    config.PATH.write_text("{ not json at all")
    check("a corrupt settings file falls back to the default",
          config.get("camera") == config.DEFAULT_CAMERA)
    config.PATH.write_text('{"camera": 2, "unknown": "ignored"}')
    check("unknown keys in the settings file are ignored",
          config.get("camera") == 2 and "unknown" not in config.load())
finally:
    if _saved is None:
        config.PATH.unlink(missing_ok=True)
    else:
        config.PATH.write_text(_saved)

# ---------------------------------------------------------------------------
group("prompt ordering and the picker")

import patterns

# Regression test for a bug that looked exactly like the program hanging:
# the pattern picker (a window) opened BEFORE the scan-confirmation prompt (a
# terminal input). You picked a pattern, the window closed, and the program
# sat on a blocking input() hidden behind it. Nothing was slow -- routing to a
# pattern measures at about a hundredth of a second -- it was waiting for a
# keypress you could not see.
import io
import contextlib
import main as _main

_order = []


def _fake_confirm(prompt=""):
    _order.append("confirm")
    return True


def _fake_picker(catalogue, face_to_colour, **kw):
    _order.append("picker")
    return patterns.by_name("Heart")


_real_confirm, _real_picker = guide.confirm, arrows.pick_pattern
_real_walk = arrows.walk_through_visual
guide.confirm, arrows.pick_pattern = _fake_confirm, _fake_picker
arrows.walk_through_visual = lambda *a, **k: True
try:
    _scramble = cube.apply_sequence(cube.SOLVED, "R U2 F' D L B'")
    _main.get_state = lambda args: (_scramble, {f: f for f in cube.FACE_ORDER}, None)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            _main.main(["--patterns"])
        except SystemExit:
            # With no solver installed main() exits when it tries to route.
            # That is fine: both prompts have already happened by then, which
            # is the only thing this test is about.
            pass
finally:
    guide.confirm, arrows.pick_pattern = _real_confirm, _real_picker
    arrows.walk_through_visual = _real_walk

check("the scan is confirmed BEFORE the picker window opens",
      _order[:2] == ["confirm", "picker"], f"order was {_order}")

# Picker key handling, testable without a display.
_n, _cols = 31, 5
check("picker: arrows move and clamp at the ends",
      arrows._picker_key(83, 0, _cols, _n) == (1, None)
      and arrows._picker_key(81, 0, _cols, _n) == (0, None)
      and arrows._picker_key(84, 0, _cols, _n) == (_cols, None)
      and arrows._picker_key(83, _n - 1, _cols, _n) == (_n - 1, None)
      and arrows._picker_key(84, _n - 2, _cols, _n) == (_n - 1, None))
check("picker: ENTER chooses, q cancels, other keys do nothing",
      arrows._picker_key(13, 12, _cols, _n) == (12, 12)
      and arrows._picker_key(ord("q"), 12, _cols, _n) == (12, -1)
      and arrows._picker_key(ord("z"), 12, _cols, _n) == (12, None))

# Routing has to be fast enough that a wait is never the program thinking.
if _HAVE_SOLVER:
    import time
    _st = cube.apply_sequence(cube.SOLVED, "R U2 F' D L B' R2 U F2 D'")
    _t = time.time()
    for _p in patterns.all_patterns():
        patterns.route(_st, _p)
    _elapsed = time.time() - _t
    check(f"routing to all {len(patterns.all_patterns())} patterns is quick",
          _elapsed < 20, f"took {_elapsed:.1f}s")

# ---------------------------------------------------------------------------
group("pattern catalogue")

_pats = patterns.all_patterns()
check("the catalogue is not empty", len(_pats) >= 30, f"{len(_pats)}")
check("every pattern target is a legal cube",
      all(cube.validate(p.target()) is None for p in _pats))
check("no two patterns are the same picture",
      len({p.target() for p in _pats}) == len(_pats),
      "duplicate targets present")
check("no pattern is just the solved cube",
      all(p.target() != cube.SOLVED for p in _pats))
check("patterns are grouped, so no category heading repeats",
      len([1 for a, b in zip(_pats, _pats[1:]) if a.category != b.category])
      == len(patterns.categories()) - 1)
check("lookup by name and by prefix both work",
      patterns.by_name("Heart") is not None
      and patterns.by_name("checkerb") is not None
      and patterns.by_name("nonsense") is None)

# A glyph must actually draw its glyph: one contrasting colour on the front
# face, and the centre sticker necessarily the face's own colour.
_base = cube.FACE_ORDER.index("F") * 9
_bad_glyph = []
for p in patterns.GLYPHS:
    face = p.target()[_base:_base + 9]
    if len({c for c in face if c != face[4]}) != 1:
        _bad_glyph.append(p.name)
check("every glyph shows exactly one contrasting colour on the front face",
      not _bad_glyph, str(_bad_glyph))

# The slice moves exist only to define patterns; they must never reach a user.
check("slice moves are available for pattern definitions",
      all(m in cube.ALL_TURNS for m in ("M", "E", "S", "M2", "E'", "S2")))
for _m in "MES":
    check(f"{_m}^4 = identity",
          cube.apply_sequence(cube.SOLVED, [_m] * 4) == cube.SOLVED)
check("a slice move turns exactly 12 stickers",
      all(sum(1 for i in range(54)
              if cube.apply_move(cube.SOLVED, m)[i] != cube.SOLVED[i]) == 12
          for m in "MES"))

check("a state has exactly 24 orientations",
      len(cube.orientations(cube.SOLVED)) == 24
      and len(cube.orientations(patterns.by_name("Heart").target())) == 24)

# The constructor trap that actually bit: a description landing in `state`.
try:
    patterns.Pattern("x", "y", "R", "note", "not-a-state")
    _caught = False
except TypeError:
    _caught = True
check("a pattern's state cannot be passed positionally", _caught)

# ---- routing from anywhere, which is the whole point ----------------------
if _HAVE_SOLVER:
    random.seed(31)
    _fail_route = _fail_human = 0
    _naive, _direct = [], []
    for _p in _pats:
        _seq = [random.choice("URFDLB") + random.choice(["", "'", "2"])
                for _ in range(22)]
        _start = cube.apply_sequence(cube.SOLVED, _seq)
        _moves = patterns.route(_start, _p)
        if cube.apply_sequence(_start, _moves) != _p.target():
            _fail_route += 1
            continue
        _steps = solve.humanize(_start, _moves, target=_p.target())
        if cube.apply_sequence(_start, [x["move"] for x in _steps]) \
                not in cube.orientations(_p.target()):
            _fail_human += 1
        _naive.append(len(solve.solve_state(_start))
                      + len(patterns.moves_from_solved(_p)))
        _direct.append(len(_moves))
    check(f"routing reaches all {len(_pats)} patterns from a random scramble",
          _fail_route == 0, f"{_fail_route} failed")
    check("the rewritten pattern routes also arrive",
          _fail_human == 0, f"{_fail_human} failed")
    check("routing beats solve-then-apply",
          sum(_direct) < sum(_naive),
          f"direct {sum(_direct)} vs naive {sum(_naive)}")
    print(f"         (avg {sum(_direct)/len(_direct):.1f} moves direct vs "
          f"{sum(_naive)/len(_naive):.1f} solving first)")

    # Pattern to pattern, and already-there.
    _a, _b = patterns.by_name("Checkerboard"), patterns.by_name("Superflip")
    _mv = patterns.route(_a.target(), _b)
    check("routing works from one pattern straight to another",
          cube.apply_sequence(_a.target(), _mv) == _b.target())
    check("routing to where you already are is a no-op",
          patterns.route(_b.target(), _b) == [])
else:
    print("  skip  routing checks need a solver installed")

# ---------------------------------------------------------------------------
group("solution rewriting")

random.seed(11)
ok_reach = ok_solve = True
for _ in range(200):
    seq = [random.choice("URFDLB") + random.choice(["", "'", "2"]) for _ in range(20)]
    state = cube.apply_sequence(cube.SOLVED, seq)
    steps = solve.humanize(state, cube.invert_sequence(seq))   # raises if wrong
    if any(s["kind"] == "turn" and s["move"][0] not in solve.REACHABLE for s in steps):
        ok_reach = False
    result = cube.apply_sequence(state, [s["move"] for s in steps])
    if not all(len(set(result[i * 9:i * 9 + 9])) == 1 for i in range(6)):
        ok_solve = False
check("rewritten sequences use only U/R/F turns", ok_reach)
check("rewritten sequences solve the cube", ok_solve)

# ---------------------------------------------------------------------------
group("cross-check against an independent implementation")

# This is the strongest test in the file. RubikTwoPhase ships its own,
# independently written facelet/cubie definitions. If our geometry-derived
# move tables agree with theirs on every move and on long random sequences,
# then both the facelet numbering AND the turn directions are right -- there
# is no room left for a sign error or a transposed column to hide in.
# Importing twophase.cubie is cheap; only twophase.solver builds pruning
# tables, so this check does not pay that cost.
try:
    import twophase.cubie as _tp_cubie
    import twophase.face as _tp_face
except ImportError:
    print("  skip  RubikTwoPhase not installed (pip install RubikTwoPhase)")
else:
    def _theirs(seq):
        cc = _tp_face.FaceCube().to_cubie_cube()
        for mv in seq:
            base = "URFDLB".index(mv[0])
            for _ in range(({"": 1, "2": 2, "'": 3})[mv[1:]]):
                cc.multiply(_tp_cubie.basicMoveCube[base])
        return cc.to_facelet_cube().to_string()

    check("solved-state string agrees", _theirs([]) == cube.SOLVED)
    same = all(_theirs([m + suf]) == cube.apply_move(cube.SOLVED, m + suf)
               for m in "URFDLB" for suf in ("", "'", "2"))
    check("all 18 single moves agree", same)

    random.seed(2)
    same = True
    for _ in range(200):
        seq = [random.choice("URFDLB") + random.choice(["", "'", "2"])
               for _ in range(20)]
        if _theirs(seq) != cube.apply_sequence(cube.SOLVED, seq):
            same = False
            print("      first mismatch on:", " ".join(seq))
            break
    check("200 random 20-move sequences agree", same)

# ---------------------------------------------------------------------------
group("real solver round trip")

try:
    import kociemba  # noqa: F401
    have = "kociemba"
except ImportError:
    try:
        import twophase.solver  # noqa: F401
        have = "RubikTwoPhase"
    except ImportError:
        have = None

if have is None:
    print("  skip  no solver installed (pip install kociemba) -- this is the")
    print("        test that proves the facelet convention matches the solver's,")
    print("        so please install one and re-run before trusting the output.")
else:
    random.seed(5)
    ok_raw = ok_human = True
    for _ in range(5):
        seq = [random.choice("URFDLB") + random.choice(["", "'", "2"]) for _ in range(25)]
        state = cube.apply_sequence(cube.SOLVED, seq)
        solution = solve.solve_state(state)
        if not solve.check_solution(state, solution):
            ok_raw = False
            print("      failing state:", state, "solution:", " ".join(solution))
            break
        steps = solve.humanize(state, solution)
        result = cube.apply_sequence(state, [s["move"] for s in steps])
        if not all(len(set(result[i * 9:i * 9 + 9])) == 1 for i in range(6)):
            ok_human = False
    check(f"{have}: solutions actually solve the scanned state", ok_raw)
    check(f"{have}: rewritten solutions also solve it", ok_human)

# ---------------------------------------------------------------------------
print()
print("=" * 58)
failed = results.count(False)
print(f"{len(results) - failed}/{len(results)} checks passed"
      + (f"  --  {failed} FAILED" if failed else ""))
print("=" * 58)
sys.exit(1 if failed else 0)
