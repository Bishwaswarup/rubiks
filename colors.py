"""
Sticker colour measurement and classification.

CALIBRATION DATA
----------------
Measured 2026-09-10 from six photographs of this specific cube (indoor
fluorescent tube plus daylight through a window, phone camera), by taking the
median of a 70x70 px patch at each of the 54 sticker positions.

OpenCV conventions throughout: H in [0,179], S in [0,255], V in [0,255].
Hue is UNWRAPPED to [-30,150] by unwrap_hue() so that red sits just below
zero instead of straddling the 179/0 seam -- otherwise red's hue averages to
something meaningless and every comparison needs special-casing.

  face     hue        sat        G-B          G/R
  RED      -3 .. -1   180..209    -15 ..  -8   0.18..0.29
  ORANGE    5 ..  6   167..191    +23 .. +29   0.37..0.45
  YELLOW   25 .. 27   140..157    +83 ..+106   0.89..0.93
  GREEN    76 .. 79   165..231    +43 .. +79   2.84..10.5
  BLUE    105 ..107   160..199    -65 .. -54   1.72..2.73
  WHITE      (n/a)      3.. 12     -3 ..  +5   0.95..1.00

Two conclusions drive the code below.

1. White is a non-problem on this cube. Saturation is <= 12 for white and
   >= 140 for every chromatic face -- an enormous margin. One saturation
   threshold is safe. (The white/yellow horror stories come from faded
   sticker cubes; this is pigmented stickerless plastic.)

2. Red and orange are only ~6 hue units apart (-1 vs +5). The gap is real in
   this lighting but thin, and it moves with colour temperature. The robust
   discriminator is the SIGN OF (G - B): red is -15..-8, orange is +23..+29,
   a gap of 31 units straddling zero -- five times the margin hue gives, and
   because it is a channel difference rather than an absolute it barely moves
   when the light dims. So G-B is the primary red/orange test and hue is only
   a sanity check.

Values are always the MEDIAN of a patch, never the mean: a ceiling light puts
a specular highlight on one or two stickers in most shots, and a mean gets
dragged toward white by it.
"""

import numpy as np
import cv2

COLOURS = "WROYGB"          # white, red, orange, yellow, green, blue

# Reference (hue, sat, g_minus_b) per colour, from the table above.
# calibrate_from_centres() replaces these per scan; these are the fallback.
REFERENCE = {
    "W": (None, 6, 1),
    "R": (-2, 190, -12),
    "O": (5, 181, 28),
    "Y": (26, 154, 95),
    "G": (77, 188, 61),
    "B": (106, 195, -60),
}

WHITE_SAT_MAX = 60      # below this a sticker is achromatic (measured max 12)
RED_ORANGE_HUE_MAX = 15  # above this we are out of the red/orange region
GB_SPLIT = 5            # G-B midpoint between red (-15..-8) and orange (+23..+29)

DISPLAY_BGR = {         # for drawing overlays, roughly the measured values
    "W": (215, 215, 215), "R": (65, 53, 209), "O": (63, 91, 216),
    "Y": (77, 178, 195), "G": (134, 197, 52), "B": (153, 93, 36),
}
NAMES = {"W": "WHITE", "R": "RED", "O": "ORANGE",
         "Y": "YELLOW", "G": "GREEN", "B": "BLUE"}


def unwrap_hue(h):
    """Map OpenCV hue to [-30,150] so red does not straddle the 179/0 seam."""
    h = int(h)
    return h - 180 if h > 150 else h


def sample_patch(bgr_image, cx, cy, half=25):
    """Median BGR over a square patch centred on (cx, cy).

    Median rather than mean, to reject specular highlights -- see module docs.
    """
    h, w = bgr_image.shape[:2]
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    patch = bgr_image[y0:y1, x0:x1].reshape(-1, 3)
    if patch.size == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.median(patch, axis=0)


def features(bgr):
    """(hue_unwrapped, saturation, g_minus_b) for one BGR triple."""
    b, g, r = (float(v) for v in bgr)
    px = np.uint8([[[b, g, r]]])
    h, s, _v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return unwrap_hue(h), int(s), g - b


def classify(bgr, refs=None):
    """Classify a single sticker into one of WROYGB.

    Independent per-sticker classification. Fine for the live overlay while
    the user is aiming the camera, but for the committed scan use
    assign_all() over the full 54, which is far more forgiving.
    """
    refs = refs or REFERENCE
    hue, sat, gb = features(bgr)

    if sat < WHITE_SAT_MAX:
        return "W"
    if hue < RED_ORANGE_HUE_MAX:
        return "O" if gb > GB_SPLIT else "R"

    best, best_d = None, float("inf")
    for name in "YGB":
        ref_hue = refs[name][0]
        if ref_hue is None:
            continue
        d = abs(hue - ref_hue)
        if d < best_d:
            best, best_d = name, d
    return best or "Y"


def calibrate_from_centres(centre_bgrs):
    """Rebuild the reference table from this scan's six centre stickers.

    Do this every single scan. It costs nothing -- you are already reading the
    centres to identify each face -- and it absorbs whatever the room lighting
    happens to be doing today, which is the whole reason fixed thresholds fail
    in someone else's kitchen.

    centre_bgrs: {colour_letter: bgr_triple}
    """
    refs = dict(REFERENCE)
    for letter, bgr in centre_bgrs.items():
        refs[letter] = features(bgr)
    return refs


def _cost(feat, colour, refs):
    """Distance from one sticker's features to one colour's reference."""
    hue, sat, gb = feat
    ref_hue, ref_sat, ref_gb = refs[colour]

    if colour == "W":
        # White is defined by absence of saturation; hue is noise there.
        return abs(sat - ref_sat) * 2.0 + (300.0 if sat >= WHITE_SAT_MAX else 0.0)

    penalty = 300.0 if sat < WHITE_SAT_MAX else 0.0   # grey is not chromatic
    hue_term = abs(hue - (ref_hue if ref_hue is not None else hue)) * 3.0
    gb_term = abs(gb - ref_gb) * 0.5
    sat_term = abs(sat - ref_sat) * 0.2
    return hue_term + gb_term + sat_term + penalty


def assign_all(sticker_bgrs, refs=None, quota=9):
    """Label all 54 stickers at once, forcing exactly `quota` of each colour.

    This is the single biggest robustness win available in the whole project.
    Classifying each sticker independently means six threshold decisions that
    can each go wrong on their own. Classifying them jointly under the
    9-of-each-colour constraint means an ambiguous sticker gets resolved for
    free by the fact that its preferred colour's quota is already full.

    Uses scipy's Hungarian solver when scipy is installed, and otherwise a
    greedy assignment by confidence margin. Given how wide the measured
    separations are on this cube the two agree in practice; the exact solver
    only earns its keep in bad light.

    Returns a list of colour letters, same length and order as the input.
    """
    refs = refs or REFERENCE
    feats = [features(b) for b in sticker_bgrs]
    n = len(feats)
    costs = [[_cost(f, c, refs) for c in COLOURS] for f in feats]

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _assign_greedy(costs, quota, n)

    # Expand each colour into `quota` interchangeable slots, then solve the
    # rectangular assignment problem exactly.
    big = np.zeros((n, len(COLOURS) * quota))
    for i in range(n):
        for j in range(len(COLOURS)):
            big[i, j * quota:(j + 1) * quota] = costs[i][j]
    rows, cols = linear_sum_assignment(big)
    out = [None] * n
    for i, c in zip(rows, cols):
        out[i] = COLOURS[c // quota]
    return out


def _assign_greedy(costs, quota, n):
    """Assign in order of confidence, skipping colours whose quota is full."""
    remaining = {c: quota for c in COLOURS}
    order = []
    for i in range(n):
        ranked = sorted(range(len(COLOURS)), key=lambda j: costs[i][j])
        margin = costs[i][ranked[1]] - costs[i][ranked[0]]
        order.append((-margin, i))          # most confident first
    order.sort()

    out = [None] * n
    for _, i in order:
        for j in sorted(range(len(COLOURS)), key=lambda j: costs[i][j]):
            colour = COLOURS[j]
            if remaining[colour] > 0:
                out[i] = colour
                remaining[colour] -= 1
                break
    return out


def confidence(bgr, refs=None):
    """How clear-cut this sticker is: gap between best and runner-up cost.

    Used to flag stickers worth a human glance before solving.
    """
    refs = refs or REFERENCE
    f = features(bgr)
    cs = sorted(_cost(f, c, refs) for c in COLOURS)
    return cs[1] - cs[0]
