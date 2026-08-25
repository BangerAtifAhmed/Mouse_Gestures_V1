"""Find the cameras attached to this machine, and name them usefully.

WHY THIS IS NOT TRIVIAL.  OpenCV addresses cameras by index and offers no
way to ask what a device is called -- `getBackendName()` names the API,
not the hardware.  The operating system knows the friendly names but not
the indices.  Neither half is much use alone, so this module gathers both
and joins them where it honestly can.

WHAT IT WILL NOT DO.  Guess.  When the number of OS device names does not
match the number of indices that actually open, the pairing would be a
coin toss, and a confidently mislabelled camera is worse than an honest
"Camera 1" -- so the generic label is used instead.  The resolution and
backend go in either way, because those come from the device that was
really opened and are the details that tell two similar cameras apart.

COST.  Opening a working camera takes about 1.8 s; a missing index fails
in a few milliseconds.  A full sweep therefore costs roughly two seconds
per camera present, which is why every caller here is expected to run it
off the UI thread.
"""
from __future__ import annotations

import subprocess
import sys

import cv2

# How far to look.  Indices are dense in practice -- a machine with two
# cameras uses 0 and 1 -- so a low ceiling costs nothing and keeps a
# refresh from spending seconds probing indices nobody has.
MAX_INDEX = 6

# The backend to probe with.  DirectShow enumerates far faster than the
# default on Windows and does not print a console banner per failure.
_BACKEND = getattr(cv2, "CAP_DSHOW", 0) if sys.platform.startswith("win") \
    else 0


class Camera:
    """One camera that genuinely opened and produced a frame."""

    __slots__ = ("index", "name", "width", "height", "backend")

    def __init__(self, index, name="", width=0, height=0, backend=""):
        self.index = int(index)
        self.name = name or ""
        self.width = int(width)
        self.height = int(height)
        self.backend = backend or ""

    @property
    def label(self) -> str:
        """What the dropdown shows."""
        if self.name:
            return f"Camera {self.index} — {self.name}"
        if self.width and self.height:
            return (f"Camera {self.index} — {self.width}×{self.height}"
                    + (f" {self.backend}" if self.backend else ""))
        return f"Camera {self.index}"

    def __repr__(self):                      # pragma: no cover - debugging
        return f"<Camera {self.index} {self.name!r}>"


def device_names(timeout: float = 20.0) -> list:
    """Friendly names the OS knows, in its own order.  [] if unavailable.

    Windows only for now.  Anything else returns an empty list, which the
    caller treats as "no names available" and falls back to generic
    labels -- the feature degrades rather than breaking.
    """
    if not sys.platform.startswith("win"):
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -Class Camera,Image -Status OK "
             "| Select-Object -ExpandProperty FriendlyName"],
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def probe(index: int):
    """Open one index briefly.  Returns a Camera, or None if unusable.

    "Opened" is not enough: a device can report success and then hand
    back nothing, so a frame has to arrive before the index counts.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(index, _BACKEND)
        if not cap.isOpened():
            return None
        got, frame = cap.read()
        if not got or frame is None:
            return None
        return Camera(
            index,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            backend=cap.getBackendName() or "",
        )
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def enumerate_cameras(max_index: int = MAX_INDEX, skip=()) -> list:
    """Every index that opens and yields a frame, named where possible.

    `skip` lists indices not to touch -- the one the engine currently
    holds, above all.  Probing a camera that is already open fails, and
    a refresh must not make a working camera vanish from its own list,
    so those are carried through as known-present instead.
    """
    skip = {int(i) for i in skip}
    found = []
    for index in range(int(max_index)):
        if index in skip:
            found.append(Camera(index))
            continue
        cam = probe(index)
        if cam is not None:
            found.append(cam)

    # Join OS names to indices ONLY when the counts line up.  See the
    # module docstring: a mismatched pairing is a guess, and a wrong name
    # is worse than no name.
    names = device_names()
    if names and len(names) == len(found):
        for cam, name in zip(found, names):
            cam.name = name
    return found
