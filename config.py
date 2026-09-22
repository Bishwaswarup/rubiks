"""
Small persistent settings file, so preferences survive between runs.

Lives beside the code as `settings.json`. The only thing in it so far is the
camera index, which is worth remembering because working it out is genuinely
annoying: on a Mac with an iPhone nearby, macOS Continuity Camera claims index
0 and the built-in webcam lands on 1, and OpenCV cannot ask a device for its
name to tell you which is which.
"""

import json
import pathlib

PATH = pathlib.Path(__file__).with_name("settings.json")

# Index 1 rather than 0 on purpose. macOS offers a nearby iPhone as a capture
# device through Continuity Camera and it usually takes index 0, pushing the
# built-in webcam to 1. Scanning a cube with the phone that is sitting on the
# desk is never what anyone wants, so the built-in camera is the better guess.
# On a machine with no Continuity Camera this will be wrong, which is why
# _open_camera falls back to whatever does work instead of just failing.
DEFAULT_CAMERA = 1

_DEFAULTS = {"camera": DEFAULT_CAMERA}


def load():
    try:
        data = json.loads(PATH.read_text())
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        return dict(_DEFAULTS)
    settings = dict(_DEFAULTS)
    settings.update({k: v for k, v in data.items() if k in _DEFAULTS})
    return settings


def get(key):
    return load().get(key, _DEFAULTS.get(key))


def set_value(key, value):
    if key not in _DEFAULTS:
        raise KeyError(key)
    settings = load()
    settings[key] = value
    PATH.write_text(json.dumps(settings, indent=2) + "\n")
    return settings
