"""Display enumeration and target-screen resolution.

Its own module because both sides need it and neither can reach the other:
app.py defers `import hand_cursor_2` because that pulls in mediapipe and
costs seconds, so the GUI cannot import the engine just to fill a dropdown.
Duplicating the logic instead would let the two disagree about which screen
is live, which is precisely the bug the fallback rule exists to prevent.

Nothing here imports anything expensive — stdlib, plus screeninfo if it
happens to be installed.

COORDINATES ARE ABSOLUTE AND MAY BE NEGATIVE.  A monitor placed left of the
primary in Windows display settings starts at x = -1920, and every function
here is written around that rather than assuming an origin at zero.
"""
from __future__ import annotations

import ctypes
import sys

__all__ = ["ALL_SCREENS", "enumerate_monitors", "monitor_labels",
           "monitor_union", "resolve_monitor_target"]

ALL_SCREENS = "All Screens"


def _from_screeninfo():
    """screeninfo if it is installed, else None.  Never raises."""
    try:
        from screeninfo import get_monitors
    except Exception:
        return None
    try:
        found = list(get_monitors())
    except Exception:
        return None

    out = []
    for i, m in enumerate(found):
        try:
            width, height = int(m.width), int(m.height)
            if width <= 0 or height <= 0:
                continue
            out.append({
                "x": int(m.x), "y": int(m.y),
                "width": width, "height": height,
                "name": getattr(m, "name", "") or f"Display {i + 1}",
                "primary": bool(getattr(m, "is_primary", False)),
            })
        except (AttributeError, TypeError, ValueError):
            continue
    return out or None


def _from_win32():
    """EnumDisplayMonitors through ctypes — no third-party dependency.

    A first-class fallback rather than a token one: this is the same API
    screeninfo calls on Windows, so it returns the same rectangles,
    negative origins included, and it needs nothing installed.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        user32 = ctypes.windll.user32
    except Exception:
        return None

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    found = []
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.POINTER(RECT), ctypes.c_double)

    def _collect(_hmonitor, _hdc, rect, _data):
        r = rect.contents
        found.append({
            "x": int(r.left), "y": int(r.top),
            "width": int(r.right - r.left),
            "height": int(r.bottom - r.top),
            "name": f"Display {len(found) + 1}",
            "primary": r.left == 0 and r.top == 0,
        })
        return 1

    try:
        user32.EnumDisplayMonitors(0, 0, callback_type(_collect), 0)
    except Exception:
        return None
    return [m for m in found if m["width"] > 0 and m["height"] > 0] or None


def _virtual_desktop():
    """The whole desktop as one entry, and honest about not being
    a real enumeration.

    Keeps "All Screens" working where per-monitor data is unavailable; the
    per-screen options simply collapse to a single choice.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        user32 = ctypes.windll.user32
        left = user32.GetSystemMetrics(76)
        top = user32.GetSystemMetrics(77)
        width = user32.GetSystemMetrics(78)
        height = user32.GetSystemMetrics(79)
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return [{"x": int(left), "y": int(top), "width": int(width),
             "height": int(height), "name": "Virtual desktop",
             "primary": True}]


def enumerate_monitors(backend=None):
    """Every attached display, as dicts with x / y / width / height.

    Sources in order: screeninfo, ctypes EnumDisplayMonitors, the virtual
    desktop as one screen, and finally whatever `backend` reports.  Returns
    [] only when every one of those fails.

    DPI is already handled upstream — hand_cursor_2 makes the process
    per-monitor aware before any of this runs — so the rectangles are true
    pixels rather than scaled ones.
    """
    for source in (_from_screeninfo, _from_win32, _virtual_desktop):
        try:
            found = source()
        except Exception:
            found = None
        if found:
            return found

    if backend is not None:
        try:
            left, top, width, height = backend.read_geometry()
            if width > 0 and height > 0:
                return [{"x": int(left), "y": int(top), "width": int(width),
                         "height": int(height), "name": "Virtual desktop",
                         "primary": True}]
        except Exception:
            pass
    return []


def monitor_labels(monitors):
    """["All Screens", "Screen 1", "Screen 2", ...] for any N, including 0."""
    return [ALL_SCREENS] + [f"Screen {i + 1}" for i in range(len(monitors))]


def monitor_union(monitors):
    """Absolute box enclosing every monitor, or None when there are none.

    min/max over the corners rather than a sum of widths, because displays
    are not necessarily in a row: a stacked or offset arrangement leaves
    gaps, and the union has to cover the whole arrangement.
    """
    if not monitors:
        return None
    min_x = min(m["x"] for m in monitors)
    min_y = min(m["y"] for m in monitors)
    max_x = max(m["x"] + m["width"] for m in monitors)
    max_y = max(m["y"] + m["height"] for m in monitors)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def resolve_monitor_target(label, monitors):
    """(rect, resolved_label), where rect is (left, top, width, height).

    Falls back rather than failing, because the config outlives the
    hardware: a layout saved with three screens attached gets opened on a
    laptop with one, and the cursor still has to go somewhere sensible.
    "Screen 3" with two monitors present resolves to "Screen 1".
    """
    union = monitor_union(monitors)
    if union is None:
        return None, ALL_SCREENS

    text = str(label or ALL_SCREENS).strip()
    if text.lower() in ("", ALL_SCREENS.lower(), "all", "extended"):
        return union, ALL_SCREENS

    index = None
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        try:
            index = int(digits) - 1
        except ValueError:
            index = None

    if index is None or not (0 <= index < len(monitors)):
        index = 0          # names a screen that is not plugged in today

    m = monitors[index]
    return ((m["x"], m["y"], m["width"], m["height"]),
            f"Screen {index + 1}")
