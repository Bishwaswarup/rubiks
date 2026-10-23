"""Regenerate the README images without a camera or a window.

    python docs/media/make_screenshots.py

Everything here is drawn by the same functions the live program uses
(arrows.render_home, arrows.render_step, arrows.render_pattern_card), on a
fixed scramble, so the pictures match what a user sees after scanning.
Needs `kociemba` (or RubikTwoPhase) for the solution.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import arrows      # noqa: E402
import cube        # noqa: E402
import patterns    # noqa: E402
import solve       # noqa: E402

OUT = os.path.join(ROOT, "docs", "media")
SCRAMBLE = "R U2 F' L D2 B R' U' F2 D L2 B2 U R2"
# Standard colour scheme in the default home position: white up, green front.
FACE_TO_COLOUR = {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}


def save(name, img):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print("wrote", os.path.relpath(path, ROOT))


def main():
    state = cube.apply_sequence(cube.SOLVED, SCRAMBLE.split())
    steps = solve.humanize(state, solve.solve_state(state))

    save("home_card.png", arrows.render_home(state, FACE_TO_COLOUR))

    # First turn card and first whole-cube rotation card, each drawn on the
    # state the user is holding at that point in the walk-through.
    current, shown = state, set()
    for i, step in enumerate(steps, 1):
        if step["kind"] not in shown:
            img = arrows.render_step(current, step, i, len(steps),
                                     FACE_TO_COLOUR)
            save(f"step_{step['kind']}.png", img)
            shown.add(step["kind"])
        current = cube.apply_move(current, step["move"])
        if len(shown) == 2:
            break

    cards = [arrows.render_pattern_card(p.target(), p.name, p.note,
                                        FACE_TO_COLOUR, selected=(k == 0))
             for k, p in enumerate(patterns.CATALOGUE[1:5])]
    save("pattern_cards.png", np.hstack(cards))


if __name__ == "__main__":
    main()
