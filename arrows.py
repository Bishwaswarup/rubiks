"""
Visual rotation guide: draw the cube and an arrow showing the turn to make.

Two display modes, both built on the same geometry.

`render_step` draws an isometric cube -- the three faces you can actually see
while holding it, painted in the CUBE'S REAL CURRENT COLOURS (the simulator in
cube.py tells us what they are at every point in the solution), with the
moving layer picked out and a curved arrow over it. Being able to compare the
drawing against the cube in your hands is most of the value: if they do not
match, you know immediately rather than twenty moves later.

`ar_overlay` draws onto the live camera image instead. It finds the cube's
front face and puts the arrow on it. This is honest about what it is: the
front face is located in the image, but no 3D pose is recovered, so the arrow
is drawn in the image plane rather than wrapped onto the cube's surface. For
the three turns this program ever asks for, that is enough to be unambiguous:

  a FRONT turn is a rotation in the image plane, so a curved arrow on the
  face is exactly right;
  a TOP turn slides the front face's top row sideways (U carries the front
  stickers to the left face -- see the derivation in cube.py), so a straight
  arrow along the top edge says it;
  a RIGHT turn carries the front face's right column upward (R takes F to U),
  so a straight arrow up the right edge says it.

WHY THE ARROWS POINT WHERE THEY DO
----------------------------------
Each visible face gets a pair of in-plane basis vectors (u, v) = (the
direction that looks RIGHT when you face it, the direction that looks DOWN).
Sweeping an arc from u toward v is then clockwise as seen from outside that
face, which is what an unprimed move means. The three pairs:

    U   u = +x   v = +z      (facing down at the top: front is toward you,
                              so +z reads as "down" in that view)
    R   u = -z   v = -y      (facing the right side: the front edge is on
                              your left, so -z reads as "right")
    F   u = +x   v = -y

Whole-cube rotations get one big arc around the cube, in the plane
perpendicular to the rotation axis, with the basis ordered so that sweeping
from e1 to e2 is the direction the cube should turn:

    X (top comes toward you)      e1 = +y  ->  e2 = +z
    Y (right comes toward you)    e1 = +x  ->  e2 = +z
    Z (rolls clockwise)           e1 = +y  ->  e2 = +x
"""

import math

import numpy as np
import cv2

import colors
import cube

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Isometric projection constants: 30-degree axonometric.
_A = math.cos(math.radians(30))
_C = math.sin(math.radians(30))

# (view-right, view-down) for each visible face -- see the module docstring.
FACE_BASIS = {
    "U": ((1, 0, 0), (0, 0, 1)),
    "R": ((0, 0, -1), (0, -1, 0)),
    "F": ((1, 0, 0), (0, -1, 0)),
}
FACE_NORMAL = {"U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1)}
FACE_CENTRE = {"U": (1.5, 3, 1.5), "R": (3, 1.5, 1.5), "F": (1.5, 1.5, 3)}

# Plane basis for whole-cube rotations, ordered along the direction of turn.
ROT_BASIS = {
    "X": ((0, 1, 0), (0, 0, 1)),
    "Y": ((1, 0, 0), (0, 0, 1)),
    "Z": ((0, 1, 0), (1, 0, 0)),
}


def _project(p, origin, scale):
    x, y, z = p
    ox, oy = origin
    return (int(round(ox + (x - z) * _A * scale)),
            int(round(oy + (x + z) * _C * scale - y * scale)))


def _sticker_corners(face, i, j):
    """The four 3-D corners of sticker (row i, col j) on a visible face."""
    if face == "F":
        x0, y0 = j, 2 - i
        return [(x0, y0, 3), (x0 + 1, y0, 3), (x0 + 1, y0 + 1, 3), (x0, y0 + 1, 3)]
    if face == "U":
        x0, z0 = j, i
        return [(x0, 3, z0), (x0 + 1, 3, z0), (x0 + 1, 3, z0 + 1), (x0, 3, z0 + 1)]
    y0, z0 = 2 - i, 2 - j
    return [(3, y0, z0), (3, y0, z0 + 1), (3, y0 + 1, z0 + 1), (3, y0 + 1, z0)]


def _moving(face, i, j, turn_face):
    """Is this sticker part of the layer that `turn_face` moves?

    The layer is the outer slice along the turned face's axis, so it is that
    whole face plus one row or column of each of the two visible neighbours.
    """
    if face == turn_face:
        return True
    if turn_face == "U":
        return (face == "F" and i == 0) or (face == "R" and i == 0)
    if turn_face == "R":
        return (face == "F" and j == 2) or (face == "U" and j == 2)
    if turn_face == "F":
        return (face == "U" and i == 2) or (face == "R" and j == 0)
    return False


def _arc(centre, u, v, r, start_deg, end_deg, origin, scale, steps=48):
    pts = []
    for k in range(steps + 1):
        t = math.radians(start_deg + (end_deg - start_deg) * k / steps)
        p = tuple(centre[d] + r * (math.cos(t) * u[d] + math.sin(t) * v[d])
                  for d in range(3))
        pts.append(_project(p, origin, scale))
    return pts


def _draw_arrow(img, pts, colour, thickness=6, head=26):
    """Polyline with an arrowhead at the far end, outlined for contrast."""
    poly = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [poly], False, (15, 15, 15), thickness + 5,
                  lineType=cv2.LINE_AA)
    cv2.polylines(img, [poly], False, colour, thickness, lineType=cv2.LINE_AA)

    if len(pts) < 2:
        return
    # Take the heading from a point a little way back along the line, but
    # never index past the start: the straight arrows are only three points.
    p1 = np.array(pts[-1], float)
    d = np.zeros(2)
    for back in range(2, min(len(pts), 5) + 1):
        d = p1 - np.array(pts[-back], float)
        if np.linalg.norm(d) > 1e-6:
            break
    n = np.linalg.norm(d)
    if n < 1e-6:
        return
    d /= n
    perp = np.array([-d[1], d[0]])
    tip = p1 + d * head * 0.45
    left = p1 - d * head * 0.55 + perp * head * 0.42
    right = p1 - d * head * 0.55 - perp * head * 0.42
    tri = np.array([tip, left, right], dtype=np.int32)
    cv2.drawContours(img, [tri], 0, (15, 15, 15), thickness + 3,
                     lineType=cv2.LINE_AA)
    cv2.fillPoly(img, [tri], colour, lineType=cv2.LINE_AA)


def _sweep(suffix):
    """(start, end) degrees for a turn, so that sweeping is the right way."""
    if suffix == "2":
        return -150, 150
    if suffix == "'":
        return 125, -125
    return -125, 125


def draw_cube(img, state, face_to_colour, origin, scale, turn_face=None,
              labels=False):
    """Paint the three visible faces in the cube's real current colours.

    `labels` names the faces: True for all three, or an iterable of face
    letters. Worth having, because this is a view from above and to the right,
    so the FRONT face is drawn on the lower-LEFT, which is easy to misread as
    the left face when comparing the picture against a cube in your hands.
    On a move card only the faces that are NOT turning get labelled -- the
    turning one is named in the heading, and its label would sit under the
    arrow.
    """
    for face in ("U", "R", "F"):
        base = cube.FACE_ORDER.index(face) * 9
        for i in range(3):
            for j in range(3):
                letter = state[base + 3 * i + j]
                colour = colors.DISPLAY_BGR.get(
                    face_to_colour.get(letter, letter), (170, 170, 170))
                if turn_face is not None and not _moving(face, i, j, turn_face):
                    # Mute the stationary stickers so the moving layer reads.
                    colour = tuple(int(0.42 * c + 0.58 * 70) for c in colour)
                quad = np.array([_project(p, origin, scale)
                                 for p in _sticker_corners(face, i, j)],
                                dtype=np.int32)
                cv2.fillConvexPoly(img, quad, colour, lineType=cv2.LINE_AA)
                cv2.polylines(img, [quad], True, (25, 25, 25), 2,
                              lineType=cv2.LINE_AA)

    if labels:
        wanted = ("U", "F", "R") if labels is True else tuple(labels)
        for face, name in (("U", "TOP"), ("F", "FRONT"), ("R", "RIGHT")):
            if face not in wanted:
                continue
            cx, cy = _project(FACE_CENTRE[face], origin, scale)
            wide = cv2.getTextSize(name, FONT, 0.5, 2)[0][0]
            cv2.putText(img, name, (cx - wide // 2, cy + 5), FONT, 0.5,
                        (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(img, name, (cx - wide // 2, cy + 5), FONT, 0.5,
                        (245, 245, 245), 1, cv2.LINE_AA)


# Unfolded-net layout: which (column, row) each face occupies, in face units.
#          U
#      L   F   R   B
#          D
NET_LAYOUT = {"U": (3, 0), "L": (0, 3), "F": (3, 3),
              "R": (6, 3), "B": (9, 3), "D": (3, 6)}


def draw_net(img, state, face_to_colour, x0, y0, cell, gap=1,
             highlight=None):
    """Draw all six faces unfolded. Shows the whole cube at once, which an
    isometric view cannot -- the point of a pattern is usually how it reads
    across every face."""
    for face, (fx, fy) in NET_LAYOUT.items():
        base = cube.FACE_ORDER.index(face) * 9
        for i in range(3):
            for j in range(3):
                letter = state[base + 3 * i + j]
                colour = colors.DISPLAY_BGR.get(
                    face_to_colour.get(letter, letter), (170, 170, 170))
                x = x0 + (fx + j) * cell
                y = y0 + (fy + i) * cell
                cv2.rectangle(img, (x + gap, y + gap),
                              (x + cell - gap, y + cell - gap), colour, -1)
    if highlight in NET_LAYOUT:
        # Glyph patterns only draw on one face; ring it so the eye goes there
        # instead of hunting across the net for which block is the picture.
        fx, fy = NET_LAYOUT[highlight]
        cv2.rectangle(img, (x0 + fx * cell - 1, y0 + fy * cell - 1),
                      (x0 + (fx + 3) * cell, y0 + (fy + 3) * cell),
                      (235, 235, 235), 1, cv2.LINE_AA)
    return img


def net_size(cell):
    return 12 * cell, 9 * cell


def render_pattern_card(state, name, note, face_to_colour, size=(320, 250),
                        selected=False, highlight=None):
    """One tile of the catalogue: the pattern's net, its name, one line of note."""
    w, h = size
    img = np.full((h, w, 3), 38 if selected else 24, dtype=np.uint8)
    cell = min((w - 24) // 12, (h - 74) // 9)
    nw, nh = net_size(cell)
    draw_net(img, state, face_to_colour, (w - nw) // 2, 44, cell,
             highlight=highlight)
    cv2.putText(img, name[:26], (12, 28), FONT, 0.56,
                (120, 235, 255) if selected else (235, 235, 235), 1,
                cv2.LINE_AA)
    if note:
        for k, line in enumerate(_wrap(note, 42)[:2]):
            cv2.putText(img, line, (12, 58 + nh + 18 * k), FONT, 0.4,
                        (150, 150, 150), 1, cv2.LINE_AA)
    if selected:
        cv2.rectangle(img, (2, 2), (w - 3, h - 3), (120, 235, 255), 2)
    return img


def render_step(state, step, index, total, face_to_colour,
                size=(760, 620), compact=False):
    """One instruction as a picture. Returns a BGR image.

    `compact` drops the prose and gives the cube the whole panel, for use as
    an inset over the camera image where there is no room for sentences.
    """
    w, h = size
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    top = 44 if compact else 190
    scale = min(w / 7.0, (h - top) / 6.6)
    origin = (w / 2, (36 if compact else 150) + 3 * scale)

    is_turn = step["kind"] == "turn"
    turn_face = step["move"][0] if is_turn else None
    suffix = step["move"][1:]

    draw_cube(img, state, face_to_colour, origin, scale, turn_face,
              labels=[f for f in ("U", "F", "R") if f != turn_face])

    if is_turn:
        u, v = FACE_BASIS[turn_face]
        n = FACE_NORMAL[turn_face]
        centre = tuple(FACE_CENTRE[turn_face][d] + 0.45 * n[d] for d in range(3))
        start, end = _sweep(suffix)
        pts = _arc(centre, u, v, 1.05, start, end, origin, scale)
        _draw_arrow(img, pts, (90, 235, 255))
    else:
        axis = step["move"][0]
        e1, e2 = ROT_BASIS[axis]
        if step["move"].endswith("'"):
            e1, e2 = e2, e1
        span = 150 if step["move"].endswith("2") else 105
        pts = _arc((1.5, 1.5, 1.5), e1, e2, 2.95, -span / 2, span / 2,
                   origin, scale)
        _draw_arrow(img, pts, (120, 170, 255), thickness=7, head=30)

    if compact:
        cv2.putText(img, step["move"], (12, 30), FONT, 0.9, accent_of(step), 2)
        cv2.putText(img, f"{index}/{total}", (w - 78, 26), FONT, 0.5,
                    (150, 150, 150), 1)
        return img

    # Header
    cv2.putText(img, f"STEP {index}/{total}", (22, 42), FONT, 0.8,
                (150, 150, 150), 2)
    tag = step["move"]
    cv2.putText(img, tag, (w - 40 - 34 * len(tag), 46), FONT, 1.4,
                (90, 235, 255) if is_turn else (120, 170, 255), 3)

    if is_turn:
        head = f"{['','ANTI'][suffix == chr(39)]}CLOCKWISE" if suffix != "2" \
            else "HALF TURN"
        cv2.putText(img, f"{solve_words(turn_face)} FACE - {head}",
                    (22, 92), FONT, 0.78, (255, 255, 255), 2)
        cv2.putText(img, "only that layer - do not turn the whole cube",
                    (22, 122), FONT, 0.52, (150, 150, 150), 1)
    else:
        cv2.putText(img, "REORIENT THE WHOLE CUBE", (22, 92), FONT, 0.78,
                    (120, 170, 255), 2)
        for k, line in enumerate(_wrap(step["text"], 58)):
            cv2.putText(img, line, (22, 120 + 22 * k), FONT, 0.52,
                        (200, 200, 200), 1)

    cv2.putText(img, "SPACE / -> next    <- back    q quit",
                (22, h - 16), FONT, 0.5, (140, 140, 140), 1)
    return img


def accent_of(step):
    return (90, 235, 255) if step["kind"] == "turn" else (120, 170, 255)


def solve_words(face):
    return {"U": "TOP", "R": "RIGHT", "F": "FRONT",
            "D": "BOTTOM", "L": "LEFT", "B": "BACK"}[face]


def _wrap(text, width):
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Live camera overlay
# ---------------------------------------------------------------------------

def detect_cube_quad(frame, min_frac=0.03):
    """Find the cube's front face in the image. Returns 4 points, or None.

    Deliberately simple: threshold on edges, take convex quadrilateral
    contours that are big and roughly square, and keep the largest. It finds a
    cube held up to the camera against an ordinary background, which is the
    only situation this is used in. It is not a tracker and it is not a pose
    estimate.
    """
    h, w = frame.shape[:2]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(grey, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, min_frac * w * h
    for c in contours:
        area = cv2.contourArea(c)
        if area < best_area:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(float)
        side = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        if min(side) < 0.62 * max(side):        # roughly square
            continue
        best, best_area = pts, area
    return best


def _order_quad(pts):
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    centre = pts.mean(axis=0)
    ordered = sorted(pts, key=lambda p: math.atan2(p[1] - centre[1],
                                                   p[0] - centre[0]))
    ordered = np.array(ordered)
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def ar_overlay(frame, step, index, total, quad=None):
    """Draw the current instruction onto a live camera frame.

    See the module docstring for what each arrow shape means and why it is
    correct for that turn.
    """
    img = frame.copy()
    h, w = img.shape[:2]
    is_turn = step["kind"] == "turn"
    face = step["move"][0] if is_turn else None
    suffix = step["move"][1:]
    accent = (90, 235, 255) if is_turn else (120, 170, 255)

    if quad is not None and is_turn and face in ("F", "U", "R"):
        tl, tr, br, bl = _order_quad(quad)
        centre = (tl + tr + br + bl) / 4.0
        span = float(np.linalg.norm(tr - tl))
        cv2.polylines(img, [np.array([tl, tr, br, bl], np.int32)], True,
                      (60, 200, 60), 2, lineType=cv2.LINE_AA)

        if face == "F":
            # A front turn is a rotation in the image plane: draw it directly.
            r = span * 0.33
            a0, a1 = (-150, 150) if suffix == "2" else \
                ((120, -120) if suffix == "'" else (-120, 120))
            pts = [(int(centre[0] + r * math.cos(math.radians(t))),
                    int(centre[1] + r * math.sin(math.radians(t))))
                   for t in np.linspace(a0, a1, 48)]
            _draw_arrow(img, pts, accent)
        elif face == "U":
            # U carries the front face's top row to the LEFT face.
            mid = (tl + tr) / 2.0
            left = suffix != "'"
            dx = span * (-0.42 if left else 0.42)
            start = mid + np.array([-dx * 0.9, -span * 0.10])
            end = mid + np.array([dx * 1.05, -span * 0.10])
            _draw_arrow(img, [tuple(start.astype(int)),
                              tuple(((start + end) / 2).astype(int)),
                              tuple(end.astype(int))], accent)
        else:
            # R carries the front face's right column UP.
            mid = (tr + br) / 2.0
            up = suffix != "'"
            dy = span * (-0.42 if up else 0.42)
            start = mid + np.array([span * 0.10, -dy * 0.9])
            end = mid + np.array([span * 0.10, dy * 1.05])
            _draw_arrow(img, [tuple(start.astype(int)),
                              tuple(((start + end) / 2).astype(int)),
                              tuple(end.astype(int))], accent)
        if suffix == "2" and face != "F":
            cv2.putText(img, "x2", tuple(np.int32(centre + [span * 0.3, 0])),
                        FONT, 1.0, accent, 3)

    # Instruction band.
    band = 104
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, band), (22, 22, 22), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.putText(img, f"{index}/{total}", (16, 38), FONT, 0.72,
                (150, 150, 150), 2)
    cv2.putText(img, step["move"], (110, 42), FONT, 1.2, accent, 3)
    text = (f"{solve_words(face)} face "
            f"{'half turn' if suffix == '2' else ('anticlockwise' if suffix == chr(39) else 'clockwise')}") \
        if is_turn else step["text"]
    for k, line in enumerate(_wrap(text, 58)):
        cv2.putText(img, line, (16, 74 + 22 * k), FONT, 0.55,
                    (255, 255, 255), 1)
    if quad is None:
        cv2.putText(img, "cube not detected - hold it up to the camera",
                    (16, h - 14), FONT, 0.5, (140, 140, 255), 1)
    else:
        cv2.putText(img, "SPACE / -> next    <- back    q quit",
                    (16, h - 14), FONT, 0.5, (150, 150, 150), 1)
    return img


# ---------------------------------------------------------------------------
# Interactive walkthroughs
# ---------------------------------------------------------------------------

def home_words(face_to_colour):
    """"WHITE on top and GREEN facing you", from the scan's own centres.

    Returns None when the "colours" are really face letters (--demo/--state),
    where naming them would be meaningless.
    """
    if all(k == v for k, v in face_to_colour.items()):
        return None
    return (f"{colors.NAMES[face_to_colour['U']]} on top and "
            f"{colors.NAMES[face_to_colour['F']]} facing you")


def render_home(state, face_to_colour, size=(760, 620)):
    """The "start from here" card.

    This exists because scanning does NOT leave the cube in the reference
    orientation. The last capture step has the user tip the cube so the bottom
    face comes up to the camera, which ends with the front colour on top. The
    solution, and every picture in this module, is expressed in the home frame
    the scan started from. Without being told to go back to it, the user
    follows step 1 while holding the cube a quarter turn out, and from then on
    nothing matches -- including this preview, which is how the omission
    announces itself.
    """
    w, h = size
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    scale = min(w / 7.0, (h - 210) / 6.6)
    origin = (w / 2, 168 + 3 * scale)
    draw_cube(img, state, face_to_colour, origin, scale, labels=True)

    cv2.putText(img, "BEFORE YOU START", (22, 44), FONT, 0.9,
                (120, 235, 130), 2)
    words = home_words(face_to_colour)
    line = f"Hold the cube with {words}." if words else \
        "Hold the cube in the orientation you scanned it in."
    for k, part in enumerate(_wrap(line, 52)):
        cv2.putText(img, part, (22, 84 + 26 * k), FONT, 0.62,
                    (255, 255, 255), 1)
    cv2.putText(img, "Scanning left the cube turned away from this position.",
                (22, 140), FONT, 0.5, (170, 170, 170), 1)
    cv2.putText(img, "Every move assumes you are back here.",
                (22, 162), FONT, 0.5, (170, 170, 170), 1)
    cv2.putText(img, "SPACE to begin", (22, h - 16), FONT, 0.6,
                (120, 235, 130), 2)
    return img


def pick_pattern(patterns_list, face_to_colour, cols=5, rows=3,
                 title="PICK A PATTERN"):
    """Browsable grid of rendered patterns. Returns the chosen one, or None.

    Every tile is the pattern's own target state drawn as an unfolded net, so
    you are choosing from the actual picture rather than from a name. A net
    rather than the isometric view because a pattern's whole point is usually
    how it reads across all six faces at once.
    """
    per_page = cols * rows
    card = (300, 235)
    w, h = cols * card[0], rows * card[1] + 64
    index, window = 0, "Pattern catalogue"
    drawn = None            # the index the current image was composed for

    while True:
        page = index // per_page
        if drawn == index:
            # Nothing has changed, so do not redraw. Recomposing fifteen cards
            # -- 54 rectangles and two lines of text each -- thirty times a
            # second is pure waste and makes the picker feel heavy.
            key = cv2.waitKey(30) & 0xFF
            index, decision = _picker_key(key, index, cols, len(patterns_list))
            if decision is not None:
                cv2.destroyWindow(window)
                return patterns_list[decision] if decision >= 0 else None
            continue

        img = np.full((h, w), 18, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        shown = patterns_list[page * per_page:(page + 1) * per_page]
        for k, pat in enumerate(shown):
            tile = render_pattern_card(
                pat.target(), pat.name, pat.note, face_to_colour, size=card,
                selected=(page * per_page + k) == index,
                highlight="F" if pat.category in ("shapes", "letters") else None)
            r, c = divmod(k, cols)
            img[64 + r * card[1]:64 + (r + 1) * card[1],
                c * card[0]:(c + 1) * card[0]] = tile

        chosen = patterns_list[index]
        cv2.putText(img, title, (16, 30), FONT, 0.8, (120, 235, 255), 2)
        cv2.putText(img, f"{chosen.name}  --  {chosen.category}"
                         f"   ({index + 1} of {len(patterns_list)})",
                    (16, 54), FONT, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "arrows move   ENTER choose   q cancel",
                    (w - 400, 54), FONT, 0.5, (140, 140, 140), 1)
        cv2.imshow(window, img)
        drawn = index

        key = cv2.waitKey(30) & 0xFF
        index, decision = _picker_key(key, index, cols, len(patterns_list))
        if decision is not None:
            cv2.destroyWindow(window)
            return patterns_list[decision] if decision >= 0 else None


def _picker_key(key, index, cols, count):
    """Apply one keypress. Returns (new index, decision).

    decision is None to keep going, -1 to cancel, or the chosen index.
    Separated out so the key handling can be tested without a display.
    """
    if key in (ord("q"), 27):
        return index, -1
    if key in (13, 10, 32):
        return index, index
    if key in (83, ord("d")):
        return min(index + 1, count - 1), None
    if key in (81, ord("a")):
        return max(index - 1, 0), None
    if key in (84, ord("s")):
        return min(index + cols, count - 1), None
    if key in (82, ord("w")):
        return max(index - cols, 0), None
    return index, None


def _states_along(state, steps):
    """The cube state before each step, plus the final one."""
    out = [state]
    for s in steps:
        state = cube.apply_move(state, s["move"])
        out.append(state)
    return out


def walk_through_visual(state, steps, colour_to_face, mirror=True,
                        goal="SOLVED"):
    """Isometric window, one step at a time. Returns True if finished."""
    face_to_colour = {f: c for c, f in colour_to_face.items()}
    states = _states_along(state, steps)
    i = -1                      # -1 is the orientation card, shown first
    window = "Cube solver - press SPACE for the next move"
    while i < len(steps):
        img = (render_home(states[0], face_to_colour) if i < 0 else
               render_step(states[i], steps[i], i + 1, len(steps),
                           face_to_colour))
        cv2.imshow(window, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            return False
        if key in (32, 13, 10, ord("d"), 83):
            i += 1
        elif key in (ord("a"), 81) and i > -1:
            i -= 1

    final = np.full((360, 700, 3), 24, dtype=np.uint8)
    draw_cube(final, states[-1], face_to_colour, (350, 190), 44)
    label = goal.upper()
    wide = cv2.getTextSize(label, FONT, 1.4, 3)[0][0]
    cv2.putText(final, label, (350 - wide // 2, 60), FONT, 1.4,
                (90, 235, 130), 3)
    cv2.putText(final, "press any key to close", (240, 340), FONT, 0.5,
                (150, 150, 150), 1)
    cv2.imshow(window, final)
    cv2.waitKey(0)
    cv2.destroyWindow(window)
    return True


def walk_through_ar(state, steps, colour_to_face, camera=0, mirror=True,
                    goal="SOLVED"):
    """Live-camera walkthrough with the arrow drawn on the detected cube."""
    from scan import _open_camera

    states = _states_along(state, steps)
    face_to_colour = {f: c for c, f in colour_to_face.items()}
    cap = _open_camera(camera)
    window = "Cube solver (AR)"
    i, frames, quad = -1, 0, None
    try:
        while i < len(steps):
            if i < 0:
                # Same orientation card as the plain visual guide: scanning
                # does not leave the cube where the solution starts from.
                cv2.imshow(window, render_home(states[0], face_to_colour))
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    return False
                if key in (32, 13, 10, ord("d"), 83):
                    i = 0
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("lost the camera")
            if mirror:
                frame = cv2.flip(frame, 1)
            # Detection is the expensive part; every third frame is plenty.
            if frames % 3 == 0:
                quad = detect_cube_quad(frame)
            frames += 1

            img = ar_overlay(frame, steps[i], i + 1, len(steps), quad)
            # Small isometric inset, so the true colours are still visible.
            inset = render_step(states[i], steps[i], i + 1, len(steps),
                                face_to_colour, size=(330, 290),
                                compact=True)
            ih, iw = inset.shape[:2]
            img[img.shape[0] - ih - 10:img.shape[0] - 10,
                img.shape[1] - iw - 10:img.shape[1] - 10] = inset

            cv2.imshow(window, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return False
            if key in (32, 13, 10, ord("d"), 83):
                i += 1
            elif key in (ord("a"), 81) and i > -1:
                i -= 1
        return True
    finally:
        cap.release()
        cv2.destroyAllWindows()
