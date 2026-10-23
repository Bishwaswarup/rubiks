# Media

| file | what it shows |
|---|---|
| `home_card.png` | the "before you start" orientation card |
| `step_turn.png` | a layer-turn card, with the moving layer highlighted |
| `step_rotate.png` | a whole-cube rotation card |
| `pattern_cards.png` | four tiles from the `--patterns` catalogue |

All four are drawn by the program's own renderer for a fixed scramble, with no
camera or window needed:

```
python docs/media/make_screenshots.py
```

Still to add by hand, because they need a real webcam:

1. A screenshot of the scanning prompt with the 3x3 grid (`python main.py`).
2. A screenshot of the `--ar` overlay on the live camera image.
3. A short demo video (scan, review, guided solve), hosted externally and
   linked from the root README rather than committed.

The photographs in `data/evaluation/` are labelled calibration examples, not
product screenshots.
