# Rubik's cube scanner and solving guide

Point a webcam at each face of a cube, get told which layer to turn.

## Release and reproducibility

This repository is released under the [MIT License](LICENSE). It targets
Python 3.10–3.12 on macOS, Linux, and Windows; the automated test workflow
tests those Python versions on Ubuntu. The scanner itself needs a webcam and
the operating system's camera permission.

Create an isolated environment and install the pinned dependencies:

```
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 selftest.py
```

`data/evaluation/` contains a small, labelled example image set for the cube
used during development. It is a calibration/evaluation record, not a
machine-learning training set and not a general-accuracy benchmark.

Before making a formal release, replace the placeholder repository URL in
[`CITATION.cff`](CITATION.cff) and add the screenshots and demo video listed
in [`docs/media/README.md`](docs/media/README.md).

```
python3 selftest.py          # run this first -- 60-odd checks, a few seconds
python3 main.py --demo       # scramble a virtual cube and solve it, no camera
python3 main.py              # the real thing: scan with the webcam
python3 main.py --ar         # draw the move onto the live camera image
python3 main.py --manual     # type the colours in instead
python3 main.py --text       # step through in the terminal, no window

python3 main.py --patterns          # browse the catalogue, go to one
python3 main.py --pattern Heart     # go straight to a named pattern
python3 main.py --list-patterns     # print the catalogue and exit
```

Scanning is hands-free: hold the face the program asks for, keep it steady,
and it captures itself after three seconds.

## Solver dependency

The supplied pinned environment installs `kociemba`, a C extension that is
fast to start but needs a compiler. It also pins `setuptools` below version 74
because newer versions can fail while building this dependency with
`AttributeError: install_layout`. `solve.py` can instead use the pure-Python
`RubikTwoPhase` package, but its first solve generates roughly 150 MB of
pruning tables and can take about half an hour.

On macOS, the terminal app you run this from needs Camera permission
(System Settings → Privacy & Security → Camera). If the camera will not open,
`--manual` needs no hardware at all.

### Picking the right camera on a Mac

If you get your **iPhone** instead of the built-in webcam, that is Continuity
Camera: macOS offers a nearby iPhone as a capture device and it frequently
lands on index 0, ahead of the FaceTime HD camera.

```
python3 main.py --list-cameras     # which indices work, and what macOS calls them
python3 main.py --set-camera 1     # remember it; every run uses it from now on
python3 main.py --camera 0         # override for a single run, without saving
```

**The default is camera 1, not 0**, for exactly this reason: when an iPhone is
around, Continuity Camera takes index 0 and the built-in webcam gets pushed to
1, and scanning a cube with the phone on your desk is never what you want. On
a machine with no Continuity Camera that guess is wrong, so if the saved index
will not open and exactly one other camera works, it switches to that one and
says so rather than failing. The choice lives in `settings.json` beside the
code.

You can also press **`c`** while scanning to cycle to the next working camera
without restarting, which is the quickest way to find it. Captured faces are
kept when you switch.

OpenCV cannot ask a device for its name, so `--list-cameras` prints the
resolutions it got plus the names macOS reports, and you match them up: the
built-in camera comes back at 1280x720, while an iPhone reports 1920x1080 or
larger. To stop being offered the phone at all, turn it off on the phone:
Settings → General → AirPlay & Continuity → Continuity Camera.

## How it works

```
   camera  ->  scan.py     guided six-face capture on a fixed 3x3 grid
           ->  colors.py   measured HSV references, 9-of-each assignment
           ->  cube.py     canonical 54-char facelet string + validation
           ->  solve.py    two-phase solver, rewritten into R/U/F turns
           ->  arrows.py   isometric cube + arrow, or live-camera overlay
               guide.py    the same thing in the terminal

   patterns.py  31 patterns, and the route to any of them from any cube
```

### Scanning

The program never tries to recognise a face from its pattern — two faces can
look alike, and pattern matching would be a hard computer-vision problem for
no benefit. Instead it relies on two facts:

**The centre sticker identifies the face.** A physical cube has exactly one
centre of each colour, and centres never move relative to their face. So
"centre is white" *is* "this is the white face", full stop.

**The cube's orientation is fixed by the procedure, not detected.** You are
told what to hold and how to turn it, and the program checks the centre colour
at each step to confirm you did. It does not attempt 3D pose estimation, and it
does not try to read the neighbouring faces' centres to verify orientation —
if the face fills enough of the frame to sample reliably, the neighbours are
not in shot anyway.

The capture order is not arbitrary. It is chosen so that **all six faces need
zero in-plane rotation**: the 3×3 you photograph, read left-to-right and
top-to-bottom straight out of the camera, is already in canonical facelet
order for every face. Getting one of those six rotations wrong is *the*
classic bug in this kind of project, because the result is a state that
validates cleanly, produces a solution, and does not solve the cube. The
derivation is written out at the top of `scan.py`; the numbers live in one
place, `scan.IN_PLANE_ROT`, if you ever change the prompts.

#### The face lock

Every step knows exactly which colour must be facing the camera and **refuses
to capture anything else**. There is no operator override, because an override
only ever gets used by mistake: a face captured into the wrong slot yields a
cube that either fails validation with a confusing message or, worse, is a
legal-looking mirror image. Refusing costs you two seconds; accepting costs
you the whole scan.

The expected colours are derived, not guessed. Declaring up and front pins
four faces at once — a face and its opposite are the two colours that can
never be adjacent to it. Left and right stay ambiguous only until the second
capture, because a cube's chirality is not knowable in advance. So step 2
accepts either of exactly two colours and every other step accepts exactly
one.

That leaves precisely one mistake no centre check can catch: turning the cube
the wrong way about the vertical axis on step 2, where both candidates are
legitimate. **It is recoverable, and the program recovers it.**

Turned the wrong way, the faces pass the camera as F, L, B, R while being
recorded into slots F, R, B, L — so the L and R captures land in each other's
slots. Nothing else is affected: two turns put the back face in front either
way, and four bring you home, so U and D are fine. And each grid is still in
its own correct in-plane orientation, because when the physical L face reaches
the camera the B face is on its left, and canonical L is read with B on the
left.

So the whole mistake is one block swap, and undoing it is exact. When a scan
fails validation the program tries the swap; if that makes it legal, it says
what happened, prints the corrected cube, and offers to use it rather than
making you rescan six faces. The tests establish that this cannot misfire: a
wrong-way scan of a scrambled cube is caught as illegal 200/200 times, undoing
the swap restores the true state 200/200 times, and swapping an *already good*
scan makes it illegal 200/200 times — so a swap that turns an illegal cube
legal is near-conclusive evidence of what happened.

It is also flagged before it happens. Step 2 names the colour an ordinary cube
would have there (derived by walking all 24 orientations of the standard
scheme, not hard-coded), and warns if you show a different one. It only warns,
never refuses — mirror-scheme cubes exist, and this program reads the scheme
off your centres like everything else.

#### Hold to capture

No keypress. Hold the requested face steady and it fires after three seconds,
with a progress bar counting down. The timer resets the moment the readings
change, so a hand drifting out of the grid cannot sneak a bad frame in.

Two details make it reliable. The nine samples are a **temporal median across
the frames that agreed during the hold window**, not a single frame, so motion
blur or a passing highlight cannot land in the scan. And it only fires if at
least 80% of the recent frames agree on all nine stickers — a modal vote
rather than requiring perfection, which would deadlock on a hand-held cube
whose edge stickers flicker.

There are **two separate time windows** here and conflating them is what broke
the first version of this (see `HoldTracker`'s docstring for the post-mortem):

- the **run** — how long the correct face has been shown without a break. This
  is what "hold it for three seconds" means, and it must be free to grow past
  three seconds.
- the **vote** — the recent frames compared to decide whether the picture is
  steady. This one *is* trimmed to the last three seconds, so old flickering
  frames stop counting against you once they age out.

Measuring the run from the oldest frame of the trimmed window makes it
arithmetically impossible for the run to reach the threshold, and auto-capture
never fires. It is a nasty bug to spot because perfectly spaced synthetic
timestamps hide it — so all the tests for it feed jittered frame times.

If a sticker flickers so persistently that no majority ever forms, the
countdown stalls, the offending cells are named on screen, and SPACE becomes
an escape hatch. That relaxes the steadiness requirement only — never the
face lock, which has already passed by then. And note the deliberate limit: a
sticker that reads wrong on the *great majority* of frames will be captured as
misread, because no temporal filter can recover a colour the camera
consistently disagrees with. That is what the quota assignment and the
review-the-net step are for.

### Colour

Measured from six photographs of this specific cube, indoor fluorescent plus
daylight, median of a 70×70 patch per sticker (OpenCV HSV, hue unwrapped to
[-30,150] so red does not straddle the 179/0 seam):

| face | hue | sat | G−B | G/R |
|---|---|---|---|---|
| RED | −3 … −1 | 180–209 | −15 … −8 | 0.18–0.29 |
| ORANGE | 5 … 6 | 167–191 | +23 … +29 | 0.37–0.45 |
| YELLOW | 25 … 27 | 140–157 | +83 … +106 | 0.89–0.93 |
| GREEN | 76 … 79 | 165–231 | +43 … +79 | 2.84–10.5 |
| BLUE | 105 … 107 | 160–199 | −65 … −54 | 1.72–2.73 |
| WHITE | — | **3 … 12** | −3 … +5 | 0.95–1.00 |

Two things fall out of that table and both are in the code.

**White is a non-problem on this cube.** Saturation ≤ 12 for white against
≥ 140 for everything chromatic. One threshold, enormous margin. (The
white-versus-yellow horror stories come from faded *sticker* cubes; this is
pigmented stickerless plastic.)

**Red and orange are only ~6 hue units apart** — real separation in this
light, but thin, and it slides with colour temperature. The reliable
discriminator is the **sign of (G − B)**: red −15…−8, orange +23…+29, a gap of
31 straddling zero. Five times the margin hue gives, and being a channel
*difference* it barely moves when the light dims. So G−B decides red/orange
and hue is only a sanity check.

Three further things earn their keep:

- **Medians, never means.** A ceiling light puts a specular highlight on one
  or two stickers in most shots and a mean gets dragged toward white by it.
- **Recalibrate every scan** from the six centre stickers, which you are
  reading anyway to identify the faces. Free, and it absorbs whatever the room
  lighting is doing today — the reason fixed thresholds work on your desk and
  fail in someone else's kitchen.
- **Assign all 54 stickers jointly, forcing exactly 9 of each colour**
  (`colors.assign_all`). This is the single biggest robustness win available.
  Classified independently, each sticker is its own chance to go wrong;
  classified jointly, an ambiguous sticker is resolved for free because its
  preferred colour's quota is already full.

### Solving

`cube.py` derives every move table from geometry at import time rather than
carrying hand-typed facelet cycles, then `validate()` rejects impossible states
up front — nine of each colour, six distinct centres, and every corner and edge
a real cube piece — so you get "two stickers on one corner were probably
swapped" instead of a cryptic solver error or a search that never ends.

The solver's ~20 moves are then rewritten to use **only R, U and F turns**,
with explicit whole-cube rotations inserted where needed
(`solve.humanize`). B and D turns are where beginners come unstuck: reaching
behind or underneath the cube, they reorient it slightly without noticing, and
from then on every remaining instruction is wrong with no way to tell. Trading
about 35% more steps for four unambiguous physical actions is worth it. Which
rotation to insert is chosen by short lookahead, because picking the first
valid one produces sequences that thrash (`X' … X … X' …`) and run about 55%
long instead.

`humanize` verifies its own output solves the cube before returning, and
refuses to print a sequence that does not.

### Showing the move

`arrows.py` draws an isometric cube — the three faces you can actually see
while holding it — painted in the cube's **real current colours**, which the
simulator knows at every point in the solution. The moving layer is picked out
and the rest muted, with a curved arrow over it. Being able to compare the
drawing against the cube in your hands is most of the value: if they diverge
you find out immediately instead of twenty moves later. Arrow keys step
backward and forward.

**It starts with an orientation card**, and that is not decoration. Scanning
does *not* leave the cube in the reference orientation — the last capture step
has you tip the bottom face up to the camera, which ends with the front colour
on top. But the solution and every picture are expressed in the frame the scan
*started* from. Without being told to go back there, you follow step 1 holding
the cube a quarter turn out and nothing matches from then on. The card names
the position in your own colours ("hold the cube with WHITE on top and GREEN
facing you"), and the terminal header says the same thing.

The faces are labelled TOP / FRONT / RIGHT for the same reason: the view is
from above and to the right, so the FRONT face is drawn on the lower-**left**,
which is easy to misread as the left face when you are holding a cube up
against the picture. On a move card only the two faces that are not turning
get labelled — the turning one is named in the heading and its label would sit
under the arrow.

`--ar` draws onto the live camera image instead, finding the cube's front face
and putting the arrow on it, with the isometric cube as an inset so the true
colours stay visible. This is honest about what it is: the face is *located*
in the image, but no 3D pose is recovered, so the arrow is drawn in the image
plane rather than wrapped onto the cube's surface. For the only three turns
this program ever asks for, that is enough to be unambiguous:

- a **front** turn is a rotation in the image plane, so a curved arrow on the
  face is exactly right;
- a **top** turn slides the front face's top row sideways — `U` carries the
  front stickers to the left face — so a straight arrow along the top edge
  says it;
- a **right** turn carries the front face's right column upward — `R` takes F
  to U — so a straight arrow up the right edge says it.

Those two claims about where stickers go are not asserted from memory; the
tests check them against the move tables.

The arrow directions come from one rule rather than six hand-set cases. Each
visible face gets a basis (u, v) = (the direction that looks *right* when you
face it, the direction that looks *down*), and sweeping an arc from u toward v
is then clockwise as seen from outside — which is what an unprimed move means.
Whole-cube rotations get one big arc in the plane perpendicular to the axis,
with the basis ordered along the direction of turn.

## Patterns

The flow is: scan → **confirm the net matches your cube** → pick a pattern →
follow the moves. The confirmation deliberately comes *before* the picker
opens. It is a terminal prompt and the picker is a window; asking afterwards
means the window closes and the program sits on a blocking `input()` hidden
behind it, which is indistinguishable from a hang.

31 patterns: the classics (checkerboard, cube-in-a-cube, superflip, the four
snakes, gift box), plus shapes and letters drawn on the front face. Browse them
as pictures with `--patterns` — each tile is that pattern's real target state
drawn as an unfolded net, so you choose from the picture, not the name.

### Going to a pattern from a *scrambled* cube

Patterns are published as sequences to apply to a *solved* cube, which is no
use when yours is scrambled. The obvious fix — solve it, then apply the
pattern — works but wastes about twenty moves. You don't need them.

A pattern is really a target *state*, and a solver is a machine for computing
the moves between two states; it is only ever pointed at "this state → solved"
by convention. So point it somewhere else. Writing `s · M` for applying moves
`M` to state `s`, we want `M` with `current · M = target`. Take the naive route
`N = solve(current) + pattern_moves`, then build

```
Y = solved · N⁻¹
```

Solving `Y` means finding a sequence that undoes `N⁻¹` — that is, any sequence
with exactly `N`'s effect. Every solution of `Y` is the same cube-group element
as `N`, so it takes `current` to `target` just as `N` does, but the solver
returns its own near-optimal sequence instead of your concatenation.

Measured over all 31 patterns from random scrambles: **20.5 moves instead of
29.1**. The same machinery routes from one pattern straight to another, and
means a pattern's own sequence length stops mattering at all.

### Letters and shapes

The centre sticker can never move, so it is always its face's own colour. A
glyph whose middle cell is part of the stroke must therefore be drawn *in* the
face colour on a contrasting ground; one whose middle cell is not is drawn in
the contrasting colour instead. Both read fine — and it is why "Ring", "Dot"
and "Letter O" are one and the same picture.

Rather than hunt for published sequences, `search_face_shape` enumerates every
move sequence to six moves deep and records, for each possible front-face
pattern, the sequence that gets there. The result: **all 256 possible masks are
reachable within six moves**, so any 3×3 glyph can be drawn. The limit is
legibility, not the cube — `E` fills eight of nine cells and has nothing left
to contrast against, which is why it isn't in the catalogue.

Since routing recomputes the path anyway, sequence length is free, so the
search spends it elsewhere: among all sequences producing a given shape it
keeps the one leaving the *rest* of the cube tidiest, instead of leaving the
other five faces arbitrary.

Slice moves (`M`, `E`, `S`) exist in `cube.py` only so the classic patterns can
be written as published. They never reach you: a pattern becomes a target
state, and the route back comes out as ordinary face turns.

## Tests

`selftest.py` is worth reading before you trust any of this.

- Each move to the fourth power is the identity; `(R U)` has order **105** and
  `(R U R' U')` has order 6 — the 105 in particular only comes out right if
  both move tables *and* their relative orientation are correct.
- A scrambled virtual cube, rendered into per-face grids the way the camera
  would see them, scans back to the same state. This is the test that catches
  an in-plane rotation error.
- Rewritten solutions use only U/R/F turns and still solve the cube, over 200
  random sequences.
- **Cross-check against RubikTwoPhase's own independently written cubie
  definitions**: all 18 moves and 200 random 20-move sequences agree exactly.
  This is the strongest check here — it leaves nowhere for a sign error or a
  transposed column to hide.
- The **highlighted layer in the drawing is checked against the move tables**:
  the stickers drawn as moving must be exactly the ones the permutation moves,
  plus the turned face's own centre (which spins in place). And the arcs are
  checked to sweep clockwise by projecting them, rather than by eye.
- The face lock: four faces pinned from up+front, two candidates at step 2,
  everything pinned after, wrong faces and repeat faces refused.
- **The preview is read back pixel by pixel**: the centre of every one of the
  27 drawn stickers must hold the colour that facelet actually has, plus
  checks that the view is not mirrored (front drawn left of right, top above
  both). Nothing else would catch a transposed row in the isometric mapping —
  the drawing would just quietly disagree with the cube in your hands.
- The starting orientation is named from the scan's own centres, and the
  terminal header is asserted to say it.
- The hold window's temporal median discards a disagreeing frame.
- **Every pattern is reached and verified**: all 31 routed to from random
  scrambles, then re-checked after the R/U/F rewrite. Arrival is judged
  against all 24 orientations of the target, since the rewrite inserts
  whole-cube rotations and the cube can finish correct but held a different
  way up.
- No two patterns are the same picture — which caught two real duplicates:
  `M2 E2 S2` and `F2 B2 U2 D2 L2 R2` are one position with two traditional
  names, and the shape search rediscovered Plus Minus exactly.
- **The scan is confirmed before the picker opens** — a window must never be
  followed by an invisible terminal prompt. Plus the picker's key handling
  (arrows clamp at the ends, ENTER chooses, q cancels), tested without needing
  a display, and a check that routing to all 31 patterns stays fast, so a wait
  is never the program thinking.
- Every glyph shows exactly one contrasting colour on its face, and a
  pattern's target state cannot be passed positionally (a description landing
  in that slot silently made every `target()` return prose).
- **Auto-capture fires at ~3s across frame jitter from 0 to 20ms**, on a slow
  8fps camera, and with a sticker flickering on 15% of frames; it never fires
  on genuinely ambiguous readings (45–60% flicker); an interruption restarts
  the countdown instead of resuming it; and the measured run is asserted to
  keep growing past the hold duration, which is the exact invariant the
  original bug violated.
- With a solver installed, full round trips: scramble → solve → verify.

Verified state as of the last run: **133/133 with both solvers installed**,
122/122 without, including the RubikTwoPhase cross-check and 30 end-to-end
kociemba round trips.

One thing the tests establish that is worth knowing: a **wrong-way turn during
scanning** — the one mistake the face lock cannot catch — is rejected every
time on a scrambled cube, and is then recovered exactly. It only slips through
on an already-solved cube, where it is harmless. So if your scan was accepted,
the recorded state is your cube.

## Files

| file | what's in it |
|---|---|
| `cube.py` | geometry, facelet indexing, move tables, simulator, validation |
| `colors.py` | measured references, classification, quota assignment |
| `scan.py` | webcam capture, capture sequence, typed-entry mode |
| `solve.py` | solver wrapper, R/U/F rewrite with verification |
| `arrows.py` | isometric renderer, arrow geometry, live-camera overlay |
| `patterns.py` | catalogue, routing from any state, shape search |
| `guide.py` | terminal step-through |
| `main.py` | entry point and CLI |
| `selftest.py` | the checks above |

## Where to take it next

Roughly in order of payoff:

1. Click-to-correct on the review net, so one misread sticker never means
   rescanning six faces.
2. Contour-based sticker detection instead of the fixed grid overlay, so the
   cube does not have to be aligned to it. `arrows.detect_cube_quad` already
   finds the face outline; feeding that back into the scanner would let the
   grid follow the cube.
3. Real pose estimation for `--ar`: `solvePnP` on the four detected corners
   gives a rotation and translation, which would let the arrows be drawn
   wrapped onto the cube's actual surface instead of in the image plane. This
   is the natural next step and everything it needs is now in place.
4. Detecting whether the user actually performed the move, by re-reading the
   front face after each step and comparing against the expected state. The
   simulator already knows what the cube should look like at every point.
