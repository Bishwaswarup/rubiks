"""
Step-through instruction UI, in the terminal.

Deliberately terminal-based rather than a second OpenCV window: it works over
ssh, it needs no GUI event loop competing with the scanner's, and the thing
being communicated is one short sentence plus an arrow, which text does well.
"""

import shutil

import arrows
import colors
import cube

ARROWS = {"": "↻  clockwise", "'": "↺  anticlockwise",
          "2": "↻↻ 180°"}

# Which cell of the little diagram to highlight for each reachable face.
_DIAGRAM = {
    "U": ["  [=======]  ", "  |  TOP  |  ", "  [=======]  ",
          "  |       |  ", "  |       |  ", "  |_______|  "],
    "F": ["  ,-------,  ", "  |       |  ", "  [=======]  ",
          "  | FRONT |  ", "  |       |  ", "  [=======]  "],
    "R": ["  ,-----[=]  ", "  |     [=]  ", "  |     [R]  ",
          "  |     [I]  ", "  |     [G]  ", "  '-----[=]  "],
    "D": ["  ,-------,  ", "  |       |  ", "  |       |  ",
          "  [=======]  ", "  |BOTTOM |  ", "  [=======]  "],
    "L": ["  [=]-----,  ", "  [=]     |  ", "  [L]     |  ",
          "  [E]     |  ", "  [F]     |  ", "  [=]-----'  "],
    "B": ["  ,-------,  ", "  | BACK  |  ", "  | (away |  ",
          "  |  from |  ", "  |  you) |  ", "  '-------'  "],
}


def _rule(char="-"):
    return char * min(shutil.get_terminal_size((72, 24)).columns, 72)


def show_scan(state, colour_to_face):
    face_to_colour = {f: c for c, f in colour_to_face.items()}
    print(_rule("="))
    print("SCANNED CUBE")
    print(_rule("="))
    print(cube.pretty(state, colours=face_to_colour))
    print()
    # In --demo / --state mode the "colours" ARE face letters, and R and B are
    # letters in both alphabets -- so translating per-letter would print
    # "R=RED, F=F". Detect the identity mapping and leave it alone.
    identity = all(k == v for k, v in colour_to_face.items())
    print("  centres:  " + "   ".join(
        f"{slot}={face_to_colour[slot] if identity else colors.NAMES[face_to_colour[slot]]}"
        for slot in cube.FACE_ORDER))
    print(_rule())


def confirm(prompt="  Does that net match your cube? [y/n] "):
    print()
    print(_rule())
    try:
        return input(prompt).strip().lower().startswith("y")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def print_sequence(steps, notation=True, colour_to_face=None):
    """Header: the starting orientation, the length, and the whole sequence."""
    total = len(steps)
    turns = sum(1 for s in steps if s["kind"] == "turn")

    print()
    print(_rule("="))
    print(f"SOLUTION: {turns} face turns"
          f"{f' and {total - turns} whole-cube rotations' if total != turns else ''}")
    print(_rule("="))

    # Scanning does NOT leave the cube in the reference orientation -- the last
    # capture has you tip the bottom face up to the camera. Every move below is
    # expressed in the frame the scan STARTED from, so say so before step 1.
    words = None
    if colour_to_face:
        words = arrows.home_words({f: c for c, f in colour_to_face.items()})
    print("  FIRST: " + (f"hold the cube with {words}."
                         if words else
                         "hold the cube in the orientation you scanned it in."))
    print("  Scanning left it turned away from that position, and every move")
    print("  below assumes you are back there.")
    print()
    print("Then hold the cube the same way throughout. Only turn the layer you")
    print("are told to turn -- do not reorient it unless a step says so.")
    print()
    if notation:
        print("  full sequence: " + " ".join(s["move"] for s in steps))
        print("  solver's own:  " + " ".join(
            s["notation"] for s in steps if s["kind"] == "turn"))
    print(_rule())


def choose_pattern(patterns_list):
    """Terminal fallback for the catalogue picker, grouped by category."""
    print(_rule("="))
    print("PATTERN CATALOGUE")
    print(_rule("="))
    last = None
    for i, pat in enumerate(patterns_list, 1):
        if pat.category != last:
            print(f"\n  -- {pat.category} --")
            last = pat.category
        print(f"  {i:>2}. {pat.name:<26} {pat.note}")
    print()
    while True:
        raw = input("  number (or name, blank to cancel): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(patterns_list):
            return patterns_list[int(raw) - 1]
        for pat in patterns_list:
            if pat.name.lower().startswith(raw.lower()):
                return pat
        print("  no such pattern")


def walk_through(steps, notation=True, colour_to_face=None):
    """Print one instruction at a time, waiting for ENTER between them."""
    total = len(steps)
    print_sequence(steps, notation, colour_to_face)

    for i, step in enumerate(steps, 1):
        print()
        if step["kind"] == "rotate":
            print(f"  Step {i}/{total}      *** REORIENT ***")
            print(f"    {step['text'].capitalize()}.")
            print(f"    ({step['move']})")
        else:
            face = step["move"][0]
            suffix = step["move"][1:]
            print(f"  Step {i}/{total}      {step['move']}"
                  f"      [solver: {step['notation']}]")
            print()
            for line in _DIAGRAM.get(face, []):
                print("      " + line)
            print()
            print(f"    {step['text'].capitalize()}   {ARROWS[suffix]}")
        try:
            input("\n    ENTER for the next step (Ctrl-C to stop) ")
        except (KeyboardInterrupt, EOFError):
            print("\n  stopped.")
            return False

    print()
    print(_rule("="))
    print("  Done -- the cube should be solved.")
    print(_rule("="))
    return True
