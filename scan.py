"""
Guided six-face scanning: webcam, or typed entry when there is no camera.

WHY THE CAPTURE ORDER IS THE WAY IT IS
--------------------------------------
The face a camera photographs has to be written into the canonical facelet
order (see cube.py). In general that needs an in-plane rotation of 0, 90, 180
or 270 degrees per face, and getting one of those six numbers wrong is the
classic bug in this kind of project: the state validates, the solver returns a
solution, and the solution does not solve the cube.

This particular sequence is chosen so that ALL SIX ROTATIONS ARE ZERO. The
photographed 3x3, read row-major from the raw camera frame, is already in
canonical order for every face. The derivation, with U on top and F toward the
camera as the home position:

  F, R, B, L   Captured with the up-face still up, so image-up is the U edge.
               Canonical order for a side face is also read with U at the top.
               Rotation 0. (The columns work out too: turning the cube so the
               right face comes to the front carries R's F-adjacent edge to
               the image left, and canonical R1 is exactly the F-adjacent
               corner of the top row.)

  U            Captured by tipping the cube so the top face turns toward the
               camera and the front face goes to the bottom. The U edge
               furthest from the camera -- the B-adjacent one -- sweeps up and
               over to the TOP of the image, like opening a book cover toward
               you. Canonical U is read with B at the top. Rotation 0.

  D            Captured by tipping the other way, so the bottom face comes up
               toward the camera and the front face goes to the top. D's
               F-adjacent edge ends up at the TOP of the image. Canonical D is
               read with F at the top. Rotation 0.

If you change the wording of the prompts and a user interprets a tilt the
other way, the fix is IN_PLANE_ROT below -- not scattered index arithmetic.
Run selftest.py after any change to it.

THE FACE LOCK
-------------
Every step knows exactly which colour must be facing the camera, and refuses
to capture anything else. There is no operator override, because an override
is only ever used by mistake: a face captured into the wrong slot produces a
cube that either fails validation with a confusing message or, worse, is a
legal-looking mirror image. Refusing costs the user two seconds; accepting
costs them the whole scan.

The expected colours are derived, not guessed. Declaring the up and front
colours fixes four faces immediately (a face and its opposite are the two
colours that can never be adjacent). The remaining pair -- left and right --
is only ambiguous until the second face is captured, because a cube's
chirality is not knowable in advance; from then on all six are pinned. So step
2 accepts either of exactly two colours, and every other step accepts exactly
one.
"""

import sys
import time
from collections import Counter, deque

import cv2

import colors
from cube import FACE_ORDER

# Quarter-turns to rotate each captured 3x3 clockwise before writing it into
# canonical order. All zero by construction -- see the module docstring.
IN_PLANE_ROT = {"F": 0, "R": 0, "B": 0, "L": 0, "U": 0, "D": 0}

# Opposite face colours. Standard for a six-colour cube; override if yours is
# painted differently. Only used to work out which face to expect next.
OPPOSITES = {"W": "Y", "Y": "W", "R": "O", "O": "R", "G": "B", "B": "G"}

# The ordinary six-colour scheme: white up, green front puts RED on the right.
# Used only for a HINT while scanning and for diagnosing a failed scan -- never
# to decide anything, because mirror-scheme cubes exist and this program reads
# the scheme off the centres like everything else.
STANDARD_SCHEME = {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}


def _standard_orientations():
    """{(up_colour, front_colour): right_colour} for all 24 orientations."""
    import cube as _cube
    start = "".join(STANDARD_SCHEME[f] * 9 for f in _cube.FACE_ORDER)
    seen, frontier, table = {start}, [start], {}
    while frontier:
        st = frontier.pop()
        table[(st[4], st[22])] = st[13]          # U centre, F centre -> R centre
        for rot in ("X", "Y", "Z", "X'", "Y'", "Z'"):
            nxt = _cube.apply_move(st, rot)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return table


STANDARD_RIGHT = _standard_orientations()


def standard_right(up, front):
    """What a standard-scheme cube would have on the right. None if unknown."""
    return STANDARD_RIGHT.get((up, front))


def swap_sides(faces):
    """Exchange the L and R captures: the correction for a wrong-way turn.

    Turning the cube the wrong way about the vertical axis on step 2 shows the
    faces in the order F, L, B, R while the program is storing them into slots
    F, R, B, L. Every grid is still in its own correct in-plane orientation --
    when the physical L face reaches the camera, the B face is on its left,
    and canonical L is read with B on the left, so the rows and columns land
    correctly. Only the SLOT is wrong, and only for L and R (two wrong-way
    turns put B at the front either way, and four bring you back home, so U
    and D are unaffected).

    That makes the mistake exactly recoverable, which is worth doing rather
    than making the user rescan six faces.
    """
    out = dict(faces)
    out["L"], out["R"] = faces["R"], faces["L"]
    return out


HOLD_SECONDS = 3.0        # how long the correct face must be held
STABLE_FRACTION = 0.8     # of frames in the window that must agree
MIN_FRAMES = 12           # do not fire on a handful of frames from a slow camera

# The guided sequence. Each step is (slot, instruction, hint).
CAPTURE_STEPS = [
    ("F", "Hold the cube with {up} on TOP and {front} FACING THE CAMERA.",
     "This is the home position. Everything else is relative to it."),
    ("R", "Turn the cube so the face that was on your RIGHT now faces the "
          "camera - {front} moves off to your LEFT. Keep {up} on top.",
     "Spin the whole cube about the vertical axis. Do not turn a layer."),
    ("B", "Turn the cube the same way again -- whatever is now on your RIGHT "
          "comes to the camera, the front moves LEFT. Keep {up} on top.", ""),
    ("L", "Turn the cube the same way once more. Keep {up} on top.", ""),
    ("U", "Turn the same way ONE more time to get back to {front} facing the "
          "camera, then tip the cube so {up} faces the camera and {front} "
          "ends up on the BOTTOM.",
     "Like opening a book cover toward yourself."),
    ("D", "Return to the home position ({front} facing you, {up} on top), "
          "then tip the cube the OTHER way so the bottom face comes up to "
          "face the camera and {front} ends up on TOP.", ""),
]


class ScanError(RuntimeError):
    pass


class HoldTracker:
    """Decides when a face has been held long enough to capture.

    This lives in its own class rather than inline in the camera loop because
    the first version of it could not fire at all, and the reason was
    invisible while the logic was tangled up with drawing code.

    THE BUG THIS CLASS EXISTS TO PREVENT
    ------------------------------------
    There are two different time windows here and they must not be the same
    one:

      the RUN -- how long the correct face has been shown without a break.
                 This is what "hold it for three seconds" means, and it has
                 to be free to grow past three seconds.

      the VOTE -- the recent frames whose readings get compared to decide
                 whether the picture is steady. This one is deliberately
                 trimmed to the last `hold` seconds, so old flickering frames
                 stop counting against you once they age out.

    The original code measured the run from the oldest frame in the *trimmed*
    window. Trimming guarantees that frame is younger than `hold`, so the
    measured run was always just under the threshold it was being compared
    against, and auto-capture was arithmetically impossible: at 30 fps with
    half a millisecond of frame jitter it would sit at 2.999984s forever. Only
    with perfectly spaced timestamps did it ever fire, which is why it looked
    fine in a synthetic test and never worked on a real camera.

    So: `_run_start` is a plain timestamp that trimming cannot touch, and the
    deque is only ever used for the stability vote.
    """

    def __init__(self, hold=HOLD_SECONDS, stable_fraction=STABLE_FRACTION,
                 min_frames=MIN_FRAMES):
        self.hold = hold
        self.stable_fraction = stable_fraction
        self.min_frames = min_frames
        self.reset()

    def reset(self):
        self._votes = deque()
        self._run_start = None

    def update(self, now, labels, patches, acceptable):
        """Feed one frame. Returns a dict describing where we are.

        Keys: elapsed, progress (0..1), steady, ready, long_enough,
              captured (the 9 medians, or None), unstable (cell indices).
        """
        if not acceptable:
            self.reset()
            return {"elapsed": 0.0, "progress": 0.0, "steady": False,
                    "ready": False, "long_enough": False,
                    "captured": None, "unstable": []}

        if self._run_start is None:
            self._run_start = now
        self._votes.append((now, labels, patches))
        while self._votes and now - self._votes[0][0] > self.hold:
            self._votes.popleft()

        elapsed = now - self._run_start           # NOT from the trimmed deque
        counts = Counter(entry[1] for entry in self._votes)
        modal, agree = counts.most_common(1)[0]
        fraction = agree / len(self._votes)
        steady = fraction >= self.stable_fraction

        unstable = [i for i in range(9)
                    if len({entry[1][i] for entry in self._votes}) > 1]

        long_enough = (elapsed >= self.hold
                       and len(self._votes) >= self.min_frames)
        ready = long_enough and steady

        # While unsteady the bar stalls short of full, so it is obvious that
        # the countdown is not actually running.
        progress = min(elapsed / self.hold, 1.0)
        if not steady:
            progress = min(progress, 0.34)

        return {"elapsed": elapsed, "progress": progress, "steady": steady,
                "ready": ready, "long_enough": long_enough,
                "unstable": unstable,
                "captured": _consolidate(self._votes, modal) if ready else None}

    def force(self):
        """Capture despite unsteady readings, for the manual escape hatch.

        Only relaxes the STABILITY requirement, never the face lock: the
        caller has already established that the right face is showing.
        """
        if not self._votes:
            return None
        modal, _ = Counter(e[1] for e in self._votes).most_common(1)[0]
        return _consolidate(self._votes, modal)


# ---------------------------------------------------------------------------
# Which face should be facing the camera at each step
# ---------------------------------------------------------------------------

def expected_colours(up, front, right=None):
    """slot -> the one colour allowed there, as far as is currently known."""
    if up == front or OPPOSITES.get(up) == front:
        raise ScanError(
            f"{colors.NAMES[up]} and {colors.NAMES[front]} cannot be adjacent "
            "faces, so they cannot be up and front at the same time."
        )
    exp = {"U": up, "D": OPPOSITES[up], "F": front, "B": OPPOSITES[front]}
    if right is not None:
        exp["R"] = right
        exp["L"] = OPPOSITES[right]
    return exp


def side_candidates(up, front):
    """The two colours that could be on the right, before chirality is known."""
    fixed = {up, OPPOSITES[up], front, OPPOSITES[front]}
    return [c for c in colors.COLOURS if c not in fixed]


def _allowed_for(slot, up, front, right):
    exp = expected_colours(up, front, right)
    if slot in exp:
        return [exp[slot]]
    return side_candidates(up, front)          # slot == "R", chirality unknown


def _describe(allowed):
    return " or ".join(colors.NAMES[c] for c in allowed)


def _rotate_grid(grid, quarter_turns):
    """Rotate a 3x3 list-of-lists clockwise by n quarter turns."""
    out = [row[:] for row in grid]
    for _ in range(quarter_turns % 4):
        out = [[out[2 - c][r] for c in range(3)] for r in range(3)]
    return out


def grid_points(width, height, cell, gap):
    """Centres of the 3x3 sample patches, in raw-frame coordinates.

    Row-major: index 3*i+j is row i, column j as the camera sees it, which is
    canonical order for every face in this capture sequence.
    """
    cx, cy = width // 2, height // 2
    step = cell + gap
    return [(cx + (j - 1) * step, cy + (i - 1) * step)
            for i in range(3) for j in range(3)]


# ---------------------------------------------------------------------------
# Camera plumbing
# ---------------------------------------------------------------------------

def _backend():
    """AVFoundation on macOS, default elsewhere.

    Naming the backend explicitly matters on macOS: the default probe order
    can pick up a Continuity Camera (a nearby iPhone) as index 0, ahead of the
    built-in webcam, which is almost never what you want mid-scan.
    """
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _macos_camera_names():
    """Camera names as macOS reports them, best-effort, in system order.

    system_profiler's order usually but not always matches the index
    OpenCV/AVFoundation assigns, so these are shown as a hint, never used to
    choose a camera. An empty list just means no hints.
    """
    if sys.platform != "darwin":
        return []
    try:
        import json
        import subprocess
        out = subprocess.run(
            ["system_profiler", "-json", "SPCameraDataType"],
            capture_output=True, text=True, timeout=10)
        return [c.get("_name", "?")
                for c in json.loads(out.stdout).get("SPCameraDataType", [])]
    except Exception:
        return []


def list_cameras(max_index=6, quiet=False):
    """Probe camera indices and report which ones actually deliver frames."""
    try:
        prior = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        prior = None

    found = []
    try:
        for index in range(max_index):
            cap = cv2.VideoCapture(index, _backend())
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    found.append((index, w, h))
            cap.release()
    finally:
        if prior is not None:
            cv2.utils.logging.setLogLevel(prior)

    if not quiet:
        import config
        saved = config.get("camera")
        names = _macos_camera_names()
        print("Working camera indices:")
        for index, w, h in found:
            mark = "  <- current default" if index == saved else ""
            print(f"  --camera {index}    {w}x{h}{mark}")
        if not found:
            print("  (none) -- check Camera permission for your terminal in")
            print("  System Settings > Privacy & Security > Camera")
        if names:
            print("\nCameras macOS knows about (order is a hint, not the index):")
            for n in names:
                print(f"  - {n}")
            print("\nIf an iPhone is listed, that is Continuity Camera and it")
            print("often takes index 0. Pass --camera for the built-in one,")
            print("press 'c' while scanning to cycle, or turn it off on the")
            print("phone: Settings > General > AirPlay & Continuity.")
    return found


def _open_camera(index):
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        # The saved index can be wrong on a different machine, or when the
        # iPhone that was pushing the webcam to index 1 is no longer around.
        # Rather than failing, take the only working camera if there is
        # exactly one; if there are several, say which so the user can choose
        # instead of being handed the phone by accident.
        working = [i for i, _w, _h in list_cameras(quiet=True)]
        if len(working) == 1:
            print(f"  camera {index} did not open; using camera {working[0]}")
            index = working[0]
            cap = cv2.VideoCapture(index, _backend())
        elif working:
            raise ScanError(
                f"camera {index} did not open. Working indices: "
                f"{', '.join(str(i) for i in working)}. Pick one with "
                f"--camera N (and --set-camera N to remember it)."
            )
    if not cap.isOpened():
        raise ScanError(
            f"could not open camera {index}. Run 'python3 main.py "
            "--list-cameras' to see which indices work. On macOS the app "
            "running python also needs Camera permission (System Settings > "
            "Privacy & Security > Camera). Or use --manual to type the "
            "colours in."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _wrap(text, width=64):
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines


def _draw_overlay(display, pts, labels, mirror, cell, step_no, instruction,
                  status, ok, progress):
    h, w = display.shape[:2]
    half = cell // 2

    for (x, y), letter in zip(pts, labels):
        dx = (w - 1 - x) if mirror else x       # where that pixel is on screen
        colour = (90, 220, 90) if ok else (200, 200, 200)
        cv2.rectangle(display, (dx - half, y - half), (dx + half, y + half),
                      colour, 2)
        if letter:
            cv2.rectangle(display, (dx - half + 3, y - half + 3),
                          (dx - half + 26, y - half + 26),
                          colors.DISPLAY_BGR[letter], -1)
            cv2.putText(display, letter, (dx - half + 8, y - half + 22),
                        FONT, 0.6, (0, 0, 0), 2)

    # Header band with the instruction, wrapped.
    lines = _wrap(instruction, 62)
    band = 40 + 26 * len(lines) + 34
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, band), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.78, display, 0.22, 0, display)

    cv2.putText(display, f"STEP {step_no}/6", (14, 30), FONT, 0.7,
                (120, 200, 255), 2)
    for i, line in enumerate(lines):
        cv2.putText(display, line, (14, 58 + 26 * i), FONT, 0.6,
                    (255, 255, 255), 1)
    cv2.putText(display, status, (14, band - 12), FONT, 0.55,
                (120, 235, 120) if ok else (140, 140, 255), 1)

    # Hold-to-capture progress bar along the bottom.
    if progress > 0:
        bar_w = int((w - 28) * min(progress, 1.0))
        cv2.rectangle(display, (14, h - 46), (w - 14, h - 26), (60, 60, 60), -1)
        cv2.rectangle(display, (14, h - 46), (14 + bar_w, h - 26),
                      (90, 220, 90), -1)
        secs = max(0.0, HOLD_SECONDS * (1.0 - min(progress, 1.0)))
        cv2.putText(display, f"holding... {secs:.1f}s",
                    (w // 2 - 70, h - 31), FONT, 0.55, (0, 0, 0), 2)

    cv2.putText(display, "hold the face steady - captures itself   "
                         "+/- grid   m mirror   c camera   r redo   q quit",
                (14, h - 8), FONT, 0.45, (170, 170, 170), 1)


# ---------------------------------------------------------------------------
# The scan loop
# ---------------------------------------------------------------------------

def scan_with_camera(up="W", front="G", camera=0, mirror=True,
                     hold=HOLD_SECONDS):
    """Walk the user through the six captures. Returns {slot: 3x3 of BGR}.

    Capture is automatic: hold the expected face steady and it fires after
    `hold` seconds. There is no key to capture a face the program did not ask
    for -- see the module docstring on the face lock.

    The nine sampled patches are a TEMPORAL median over the frames that agreed
    during the hold window, not a single frame, so a moment of motion blur or
    a passing highlight cannot land in the scan.
    """
    expected_colours(up, front)                # fail fast on a bad pairing
    cap = _open_camera(camera)
    faces, seen, right = {}, [], None
    cell, gap = 90, 14
    available = None
    step_index = 0

    try:
        while step_index < len(CAPTURE_STEPS):
            slot, template, hint = CAPTURE_STEPS[step_index]
            instruction = template.format(up=colors.NAMES[up],
                                          front=colors.NAMES[front])
            allowed = _allowed_for(slot, up, front, right)
            expect_std = standard_right(up, front) if slot == "R" else None
            if expect_std:
                instruction += (f"  (on an ordinary cube that is "
                                f"{colors.NAMES[expect_std]})")
            tracker = HoldTracker(hold=hold)
            captured = None

            while captured is None:
                ok_frame, frame = cap.read()
                if not ok_frame or frame is None:
                    raise ScanError("lost the camera mid-scan")
                h, w = frame.shape[:2]
                pts = grid_points(w, h, cell, gap)
                patches = [colors.sample_patch(frame, x, y, cell // 3)
                           for x, y in pts]
                labels = tuple(colors.classify(p) for p in patches)

                problem = _check(slot, labels[4], allowed, seen)
                held = tracker.update(time.monotonic(), labels, patches,
                                      problem is None)
                captured = held["captured"]
                progress = held["progress"]

                if problem:
                    status = problem
                elif not held["steady"]:
                    cells = ", ".join(f"r{i // 3 + 1}c{i % 3 + 1}"
                                      for i in held["unstable"][:4])
                    status = ("readings unsteady"
                              + (f" at {cells}" if cells else "")
                              + " - hold still, fill the grid, avoid glare"
                              + ("   [SPACE to accept anyway]"
                                 if held["long_enough"] else ""))
                elif progress < 1.0:
                    status = (f"{colors.NAMES[labels[4]]} centre - holding, "
                              f"{max(0.0, hold - held['elapsed']):.1f}s to go")
                else:
                    status = f"{colors.NAMES[labels[4]]} centre - capturing"

                # A standard cube turned the wrong way shows the mirror colour
                # here. Mirror-scheme cubes are real, so this only warns.
                if (expect_std and problem is None
                        and labels[4] != expect_std):
                    status = (f"{colors.NAMES[labels[4]]}? an ordinary cube "
                              f"has {colors.NAMES[expect_std]} here - check "
                              "you turned the right way. " + status)

                # Flip FIRST, then draw. The sample point (x,y) in the raw
                # frame appears at (w-1-x, y) once flipped, so the box has to
                # be drawn there -- otherwise the label the user sees sits
                # under a different sticker than the one it describes.
                display = cv2.flip(frame, 1) if mirror else frame.copy()
                _draw_overlay(display, pts, labels, mirror, cell,
                              step_index + 1, instruction, status,
                              problem is None, progress)
                cv2.imshow("Cube scanner", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise ScanError("scan cancelled")
                if key in (ord("+"), ord("=")):
                    cell = min(cell + 6, min(h, w) // 3 - gap)
                    tracker.reset()
                if key in (ord("-"), ord("_")):
                    cell = max(cell - 6, 30)
                    tracker.reset()
                if key == 32 and problem is None and held["long_enough"]:
                    # Relaxes only the steadiness requirement. The face lock
                    # still had to pass to get here, so this cannot capture a
                    # face the program did not ask for.
                    captured = tracker.force()
                    print("  accepted unsteady reading on request")
                elif key == 32:
                    print("  not capturing: "
                          + (problem or "hold the face a moment longer"))
                if key == ord("m"):
                    mirror = not mirror
                if key == ord("r") and step_index > 0:
                    prev = CAPTURE_STEPS[step_index - 1][0]
                    faces.pop(prev, None)
                    if seen:
                        dropped = seen.pop()
                        if prev == "R":
                            right = None
                        print(f"  redoing {prev} (was "
                              f"{colors.NAMES.get(dropped, dropped)})")
                    step_index -= 1
                    break
                if key == ord("c"):
                    camera, cap, available = _next_camera(cap, camera, available)
                    tracker.reset()

            if captured is None:
                continue                        # 'r' was pressed

            grid = [[captured[3 * i + j] for j in range(3)] for i in range(3)]
            faces[slot] = _rotate_grid(grid, IN_PLANE_ROT[slot])
            centre = colors.classify(captured[4])
            seen.append(centre)
            if slot == "R":
                right = centre                  # chirality now known
            print(f"  captured {slot}: {colors.NAMES[centre]}")
            step_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return faces


def _consolidate(window, modal):
    """Per-sticker median BGR over the frames whose labels matched the mode."""
    import numpy as np
    agreeing = [patches for _t, labels, patches in window if labels == modal]
    return [np.median(np.array([p[i] for p in agreeing]), axis=0)
            for i in range(9)]


def _check(slot, centre, allowed, seen):
    """Return a problem string, or None if this face may be captured.

    The check is on the centre sticker, which is the one thing that identifies
    a face unambiguously: a physical cube has exactly one centre of each
    colour, and centres never move relative to their face. It does NOT try to
    read the adjacent faces' centres to confirm the whole orientation -- if
    the face fills enough of the frame to sample reliably, the neighbours are
    not in shot anyway.

    The one mistake no centre check can catch is turning the cube the wrong
    way about the vertical axis on step 2: both candidates are legitimate
    there, because chirality is not knowable in advance. That records a mirror
    image of the real cube, which has legal-looking pieces but the wrong
    chirality, so the solver rejects it and solve.py says so in as many words.
    """
    if centre not in allowed:
        return (f"show {_describe(allowed)} - this is "
                f"{colors.NAMES[centre]}, not capturing it")
    if centre in seen:
        return f"{colors.NAMES[centre]} was already captured - turn the cube"
    return None


def _next_camera(cap, camera, available):
    """Cycle to the next working camera index, keeping captures so far."""
    if available is None:
        cap.release()
        available = [i for i, _w, _h in list_cameras(quiet=True)]
    if len(available) > 1:
        pos = (available.index(camera) + 1) % len(available) \
            if camera in available else 0
        camera = available[pos]
        cap.release()
        cap = _open_camera(camera)
        print(f"  switched to camera {camera}")
    else:
        if not cap.isOpened():
            cap = _open_camera(camera)
        print("  only one working camera found")
    return camera, cap, available


# ---------------------------------------------------------------------------
# Typed entry, for testing and for when the camera is unavailable
# ---------------------------------------------------------------------------

def scan_manually(up="W", front="G"):
    """Type 9 letters per face, in the same order the camera would see them."""
    print("Type each face as 9 letters from WROYGB, reading left-to-right and\n"
          "top-to-bottom as you see the face held toward you.\n")
    faces, seen, right = {}, [], None
    for slot, template, _hint in CAPTURE_STEPS:
        print(f"  {slot}: " + template.format(up=colors.NAMES[up],
                                              front=colors.NAMES[front]))
        allowed = _allowed_for(slot, up, front, right)
        while True:
            raw = input(f"  {slot} > ").strip().upper().replace(" ", "")
            if len(raw) != 9 or any(c not in colors.COLOURS for c in raw):
                print("    need exactly 9 letters from WROYGB")
                continue
            problem = _check(slot, raw[4], allowed, seen)
            if problem:
                print("    " + problem)
                continue
            faces[slot] = [[raw[3 * i + j] for j in range(3)] for i in range(3)]
            seen.append(raw[4])
            if slot == "R":
                right = raw[4]
            break
    return faces


# ---------------------------------------------------------------------------
# Turning six captured faces into a state string
# ---------------------------------------------------------------------------

def build_state(faces, recalibrate=True):
    """{slot: 3x3 of BGR or letters} -> (54-char state string, colour_to_face).

    Accepts either raw BGR patches (from the camera, which is what you want,
    because it lets the global 9-of-each assignment do its job) or letters
    already decided (from typed entry).
    """
    slots = list(FACE_ORDER)
    missing = [s for s in slots if s not in faces]
    if missing:
        raise ScanError(f"never captured: {', '.join(missing)}")

    flat = [faces[s][i][j] for s in slots for i in range(3) for j in range(3)]

    if isinstance(flat[0], str):
        letters = flat
    else:
        refs = None
        if recalibrate:
            centres = {}
            for s in slots:
                bgr = faces[s][1][1]
                centres[colors.classify(bgr)] = bgr
            if len(centres) == 6:
                refs = colors.calibrate_from_centres(centres)
        letters = colors.assign_all(flat, refs=refs)

    centre_of = {s: letters[9 * k + 4] for k, s in enumerate(slots)}
    if len(set(centre_of.values())) != 6:
        raise ScanError(
            "the six centre stickers did not come out as six different "
            f"colours: {centre_of}. Rescan."
        )
    colour_to_face = {colour: slot for slot, colour in centre_of.items()}
    state = "".join(colour_to_face[c] for c in letters)
    return state, colour_to_face


def review(state, colour_to_face):
    """Print the scan as a net in the user's own colour names, for eyeballing."""
    from cube import pretty
    face_to_colour = {f: c for c, f in colour_to_face.items()}
    return pretty(state, colours=face_to_colour)
