#!/usr/bin/env python3
"""
Rubik's cube scanner and solving guide.

    python3 main.py                 # scan with the webcam, then walk through
    python3 main.py --manual        # type the colours in, no camera needed
    python3 main.py --state UUU...  # solve a state string directly
    python3 main.py --demo          # scramble a virtual cube and solve it

The pipeline, and where each piece lives:

    camera  ->  scan.py    guided six-face capture, fixed 3x3 grid
            ->  colors.py  measured HSV references, 9-of-each assignment
            ->  cube.py    canonical facelet string + validation
            ->  solve.py   two-phase solver + rewrite into R/U/F only
            ->  guide.py   one instruction at a time
"""

import argparse
import random
import sys

import cv2

import arrows
import colors
import config
import cube
import guide
import patterns
import scan
import solve


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--manual", action="store_true",
                     help="type the 54 colours instead of using a camera")
    src.add_argument("--state", metavar="STR",
                     help="54-character URFDLB state string, skip scanning")
    src.add_argument("--demo", action="store_true",
                     help="scramble a virtual cube and solve it (no hardware)")
    p.add_argument("--camera", type=int, default=None,
                   help=f"camera index for this run (default "
                        f"{config.get('camera')}, from settings.json)")
    p.add_argument("--set-camera", type=int, metavar="N",
                   help="remember N as the camera to use from now on, and "
                        "exit. On a Mac with Continuity Camera, the built-in "
                        "webcam is usually 1 and the iPhone takes 0")
    p.add_argument("--list-cameras", action="store_true",
                   help="probe camera indices and exit. Use this when macOS "
                        "Continuity Camera has grabbed index 0 for your iPhone")
    p.add_argument("--up", default="W", choices=list(colors.COLOURS),
                   help="colour to hold on top while scanning (default W)")
    p.add_argument("--front", default="G", choices=list(colors.COLOURS),
                   help="colour to face the camera while scanning (default G)")
    p.add_argument("--no-mirror", action="store_true",
                   help="do not mirror the camera preview")
    p.add_argument("--patterns", action="store_true",
                   help="browse the pattern catalogue and go to one instead "
                        "of solving")
    p.add_argument("--pattern", metavar="NAME",
                   help="go straight to a named pattern, e.g. --pattern Heart")
    p.add_argument("--list-patterns", action="store_true",
                   help="print the catalogue and exit")
    p.add_argument("--text", action="store_true",
                   help="step through in the terminal instead of the visual "
                        "window")
    p.add_argument("--ar", action="store_true",
                   help="draw the move onto the live camera image, with the "
                        "arrow placed on the detected cube face")
    p.add_argument("--raw-moves", action="store_true",
                   help="show the solver's own moves, including B and D turns, "
                        "instead of rewriting them into R/U/F")
    p.add_argument("--seed", type=int, help="scramble seed for --demo")
    return p.parse_args(argv)


def get_state(args):
    """Return (state_string, colour_to_face, captured_faces_or_None)."""
    if args.state:
        state = args.state.strip().upper()
        if len(state) != 54:
            sys.exit(f"--state needs exactly 54 characters, got {len(state)}")
        legal = set(colors.COLOURS) | set(cube.FACE_ORDER)
        stray = sorted(set(state) - legal)
        if stray:
            sys.exit(f"--state contains characters that are neither colours "
                     f"(WROYGB) nor faces (URFDLB): {', '.join(stray)}")
        # Accept either face letters or colour letters.
        if set(state) <= set(colors.COLOURS) and not set(state) <= set(cube.FACE_ORDER):
            centre_of = {s: state[9 * k + 4] for k, s in enumerate(cube.FACE_ORDER)}
            c2f = {c: s for s, c in centre_of.items()}
            state = "".join(c2f[c] for c in state)
        else:
            c2f = {f: f for f in cube.FACE_ORDER}
        return state, c2f, None

    if args.demo:
        rng = random.Random(args.seed)
        scramble = [rng.choice("URFDLB") + rng.choice(["", "'", "2"])
                    for _ in range(25)]
        print("  virtual scramble: " + " ".join(scramble))
        return (cube.apply_sequence(cube.SOLVED, scramble),
                {f: f for f in cube.FACE_ORDER}, None)

    if args.up == args.front:
        sys.exit("--up and --front must be different colours")

    if args.manual:
        faces = scan.scan_manually(up=args.up, front=args.front)
    else:
        faces = scan.scan_with_camera(up=args.up, front=args.front,
                                      camera=args.camera,
                                      mirror=not args.no_mirror)
    state, colour_to_face = scan.build_state(faces)
    return state, colour_to_face, faces


def choose_goal(args, colour_to_face):
    """None = solve the cube; a Pattern = go to it; False = user cancelled."""
    if args.pattern:
        pat = patterns.by_name(args.pattern)
        if pat is None:
            sys.exit(f"no pattern called {args.pattern!r}. "
                     "Try --list-patterns.")
        return pat
    if not args.patterns:
        return None

    catalogue = patterns.all_patterns()
    face_to_colour = {f: c for c, f in colour_to_face.items()}
    try:
        # Say so in the terminal too: on a Mac the OpenCV window does not
        # always come to the front, and a picker you cannot see looks like a
        # hang.
        print(f"\n  Opening the pattern catalogue ({len(catalogue)} patterns)"
              " in a window --")
        print("  arrow keys to move, ENTER to choose, q to cancel.")
        pat = arrows.pick_pattern(catalogue, face_to_colour)
    except cv2.error:
        pat = guide.choose_pattern(catalogue)
    if pat is None:
        print("  No pattern chosen.")
        return False
    return pat


def recover(state, colour_to_face, faces, exc, args):
    """Try to explain and undo a failed scan instead of just giving up.

    There is exactly one scanning mistake that produces a plausible-looking
    but illegal cube: turning it the wrong way about the vertical axis on step
    2, which records the L and R faces into each other's slots. Because that
    is a known transformation it can simply be undone -- and because it is the
    ONLY thing that swap fixes, a swap that turns an illegal cube into a legal
    one is near-conclusive evidence that is what happened.
    """
    fixed = None
    if faces is not None:
        try:
            alt_state, alt_c2f = scan.build_state(scan.swap_sides(faces))
            cube.validate(alt_state)
            fixed = (alt_state, alt_c2f)
        except (cube.InvalidCube, scan.ScanError):
            fixed = None

    if fixed is None:
        sys.exit(f"\nThis is not a valid cube: {exc}\n"
                 "Compare the net above with the real cube and rescan. The "
                 "usual cause is the sampling grid overlapping the edge of a "
                 "face, so one row or column reads the neighbouring face -- "
                 "shrink the grid with '-' and fill it with the cube.")

    alt_state, alt_c2f = fixed
    print(f"\nThis scan is not a valid cube: {exc}")
    print()
    print("  But exchanging the LEFT and RIGHT faces makes it valid, which")
    print("  means the cube was turned the WRONG WAY at step 2: the faces came")
    print("  past the camera as F, L, B, R while they were being recorded as")
    print("  F, R, B, L. Nothing else is affected -- two turns put the back")
    print("  face in front either way, and four bring you home, so up and")
    print("  down are fine.")
    print()
    face_to_colour = {f: c for c, f in alt_c2f.items()}
    std = scan.standard_right(face_to_colour.get("U"), face_to_colour.get("F"))
    print("  Corrected cube:")
    print(cube.pretty(alt_state, colours=face_to_colour))
    print()
    if std:
        print(f"  Centres now read: right = "
              f"{colors.NAMES[face_to_colour['R']]}, left = "
              f"{colors.NAMES[face_to_colour['L']]}"
              f"  (an ordinary cube has {colors.NAMES[std]} on the right).")
        print()
    print("  Next time: turn so the face that was on your RIGHT comes to the")
    print("  camera -- the face you were looking at moves off to your LEFT.")
    print()

    if not guide.confirm("  Use the corrected cube? [y/n] "):
        sys.exit("  Nothing changed. Rescan when ready.")
    print()
    return alt_state, alt_c2f


def main(argv=None):
    args = parse_args(argv)

    if args.set_camera is not None:
        config.set_value("camera", args.set_camera)
        print(f"  camera {args.set_camera} saved as the default "
              f"({config.PATH.name})")
        return 0

    if args.camera is None:
        args.camera = config.get("camera")

    if args.list_cameras:
        scan.list_cameras()
        return 0

    if args.list_patterns:
        last = None
        for pat in patterns.all_patterns():
            if pat.category != last:
                print(f"\n-- {pat.category} --")
                last = pat.category
            print(f"  {pat.name:<28} {pat.note}")
        return 0

    try:
        state, colour_to_face, faces = get_state(args)
    except scan.ScanError as exc:
        sys.exit(f"scan failed: {exc}")

    guide.show_scan(state, colour_to_face)

    try:
        cube.validate(state)
    except cube.InvalidCube as exc:
        state, colour_to_face = recover(state, colour_to_face, faces, exc, args)

    # Confirm the scan BEFORE opening the pattern picker, not after. The
    # picker is a window; this is a terminal prompt. Asking afterwards means
    # the window closes and the program sits on a blocking input() hidden
    # behind it, which reads as "it has frozen" rather than "it is waiting".
    if not args.demo and not args.state and not guide.confirm():
        print("  Rescan then. Nothing was changed.")
        return 1

    goal = choose_goal(args, colour_to_face)
    if goal is False:
        return 1

    if goal is None and state == cube.SOLVED:
        print("\n  Already solved. Nothing to do.")
        return 0

    try:
        if goal is None:
            solution = solve.solve_state(state)
            if not solve.check_solution(state, solution):
                sys.exit("\nThe solver returned a sequence that does not "
                         "solve this state. Refusing to show it. Please "
                         f"report this with the state string:\n  {state}")
            goal_name = "SOLVED"
        else:
            print(f"  Finding a route to {goal.name}...", flush=True)
            # route() verifies that its moves land on the pattern before
            # returning, so there is no separate check here.
            solution = patterns.route(state, goal)
            goal_name = goal.name
            if not solution:
                print(f"\n  The cube is already showing {goal.name}.")
                return 0
    except solve.SolverUnavailable as exc:
        sys.exit(f"\n{exc}")
    except (solve.Unsolvable, patterns.RouteError) as exc:
        sys.exit(f"\n{exc}")

    if args.raw_moves:
        steps = [{"kind": "turn", "move": m, "notation": m,
                  "text": f"turn the {solve.MOVE_WORDS[m[0]]} face "
                          f"{solve.TURN_WORDS[m[1:]]}"}
                 for m in solution]
    else:
        steps = solve.humanize(state, solution,
                               target=None if goal is None else goal.target())

    # The isometric renderer only draws the three faces you can see, which is
    # exactly the three the rewriter uses. --raw-moves keeps the solver's own
    # B/D/L turns, so it has to fall back to the terminal.
    if goal is not None:
        print(f"\n  Going to: {goal.name} -- {goal.note}")

    if args.text or args.raw_moves:
        if args.raw_moves and not args.text:
            print("  (--raw-moves keeps back and bottom turns, which the "
                  "visual guide cannot draw; using the terminal walkthrough)")
        guide.walk_through(steps, colour_to_face=colour_to_face)
        return 0

    guide.print_sequence(steps, colour_to_face=colour_to_face)
    try:
        if args.ar:
            arrows.walk_through_ar(state, steps, colour_to_face,
                                   camera=args.camera,
                                   mirror=not args.no_mirror, goal=goal_name)
        else:
            arrows.walk_through_visual(state, steps, colour_to_face,
                                       goal=goal_name)
    except cv2.error as exc:
        # No display (headless, ssh without X). The terminal guide needs none.
        print(f"  (no graphical display available: {str(exc).splitlines()[0]})")
        print("  falling back to the terminal walkthrough.\n")
        guide.walk_through(steps, colour_to_face=colour_to_face)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
