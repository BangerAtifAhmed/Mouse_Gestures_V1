"""
Hand-Tracking Cursor Controller (v13 — Runtime Sensitivity)
===========================================================
Moves the mouse cursor by tracking the index finger tip (MediaPipe landmark 8).
Smoothing is handled *exclusively* by a One Euro Filter (Casiez et al. 2012),
which adapts its cutoff frequency to hand speed: heavy smoothing at rest,
light smoothing during fast swipes.  Movement only — no click gestures.

The active box maps onto the *virtual desktop* (all monitors combined) and is
rebuilt on the fly when a display is plugged, unplugged or rearranged.

v13 makes CURSOR_SENSITIVITY adjustable while running: '+' / '=' raise it,
'-' / '_' lower it, and the preview reports the current multiplier.  Both of
the loop's waitKey() call sites share one handler — the duplicate-frame skip
path runs far more often than the bottom of the loop, so handling keys in
only one place would swallow most presses.

v12 added centre-scaled CURSOR_SENSITIVITY inside ScreenGeometry, which owns
the desktop centre and bounds the scaling needs.

v11 attacked input lag on three fronts: filter tuning (MIN_CUTOFF / BETA with
a measured response table), zero frame buffering (CAP_PROP_BUFFERSIZE plus an
atomic newest-frame publication), and skipping redundant inference on frames
already processed.

v9 removed the sub-frame interpolation "glide" from v8.  That code called
time.sleep() inside the main loop, which was counter-productive for two
reasons:

  1. It stole ~27 ms from the frame budget *precisely* when the hand was
     moving fastest, delaying the next camera sample.  The larger gap
     produced a larger jump on the following frame, which re-triggered
     interpolation — a positive feedback loop of lag.

  2. On CPython < 3.11 (this project targets 3.8/3.10), time.sleep() on
     Windows is bounded by the ~15.6 ms system timer tick, so each 6.67 ms
     sleep actually blocked for ~15 ms.  The real cost was closer to 60 ms
     per triggering frame.

Controls:  + / =  raise sensitivity      - / _  lower sensitivity
           q      quit

Dependencies:  pip install opencv-python mediapipe==0.8.11
Platform:      Windows only (uses user32.dll)
Python:        3.8 – 3.10  (mediapipe 0.8.11 ships no cp311 wheels)
"""

# PEP 604 unions (`float | None`) are evaluated at def-time on Python < 3.10
# and would raise TypeError there.  Deferring annotation evaluation keeps the
# 3.8/3.9 half of the supported range importable.
from __future__ import annotations

import ctypes
import cv2
import math
import mediapipe as mp
import threading
import time

# ─── Win32 virtual desktop metrics (direct API, no wrapper overhead) ───────
# SM_CXSCREEN (0) / SM_CYSCREEN (1) describe the *primary* monitor only, so a
# cursor driven from them can never leave display 1.  The SM_*VIRTUALSCREEN
# family describes the bounding rectangle enclosing ALL monitors:
#
#   SM_XVIRTUALSCREEN  (76) → left edge of the virtual desktop
#   SM_YVIRTUALSCREEN  (77) → top  edge of the virtual desktop
#   SM_CXVIRTUALSCREEN (78) → total width  of the virtual desktop
#   SM_CYVIRTUALSCREEN (79) → total height of the virtual desktop
#
# The origin matters.  Windows pins the primary monitor at (0, 0), so any
# display arranged to its left or above it produces NEGATIVE coordinates —
# a second monitor on the left reports SM_XVIRTUALSCREEN = −1920.  Mapping
# into [0, width] would therefore strand the cursor on the right-hand
# display, so every mapping below is offset by the origin.  SetCursorPos
# accepts negative virtual-screen coordinates directly.
#
# ctypes leaves restype at the default c_int (signed), so negative origins
# come back correctly; forcing c_uint here would silently wrap them.

SM_XVIRTUALSCREEN  = 76
SM_YVIRTUALSCREEN  = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
SM_CMONITORS       = 80

_user32 = ctypes.windll.user32

# ═══════════════════════════════════════════════════════════════════════════
#  FEEL — sensitivity and latency, the knobs worth touching
# ═══════════════════════════════════════════════════════════════════════════

# ── Cursor sensitivity (speed) ─────────────────────────────────────────────
# The mapping is absolute: each point in the active box corresponds to a
# fixed point on the desktop.  Sensitivity scales that correspondence about
# the desktop CENTRE, so less physical movement covers the same distance:
#
#       offset = mapped − centre
#       target = centre + offset × CURSOR_SENSITIVITY
#       target = clamp(target, desktop bounds)
#
#   1.0  → plain absolute mapping; the box edge is the desktop edge.
#   1.5  → the middle 67% of the box covers the whole desktop.
#   2.0  → the middle 50% does.
#   3.0  → the middle 33% does  (the maximum allowed).
#
# The cost is a saturated band around the box perimeter.  With S > 1 the
# outer (1 − 1/S) of the box maps past the desktop edge and clamps flat, so
# hand movement out there produces no cursor movement at all:
#
#       usable fraction of the box = 1 / CURSOR_SENSITIVITY
#
# The preview draws that live region as an inner amber rectangle, so the
# trade-off stays visible — and it resizes live as you press + / −.
#
# This is the STARTING value; '+' / '=' and '-' / '_' adjust it at runtime,
# strictly within [SENS_MIN, SENS_MAX].  ScreenGeometry reads the module-level
# name on every call, so rebinding it takes effect on the very next frame with
# no object to keep in sync.

CURSOR_SENSITIVITY = 1.0

SENS_STEP = 0.1    # increment per keypress
SENS_MIN  = 1.0    # floor: plain absolute mapping, the entire box is live.
                   # Below 1.0 the mapping would shrink the reachable area to
                   # a sub-region of the desktop and strand the screen edges
                   # — never useful here, so the range excludes it outright.
SENS_MAX  = 3.0    # ceiling: only 1/3 of the box still reaches an edge.
                   # Beyond this the live region gets too small to aim
                   # inside, and key auto-repeat would otherwise run away.

# ── One Euro Filter: MIN_CUTOFF and BETA ───────────────────────────────────
# These two decide how the cursor feels.  The filter's smoothing factor is
#
#     fc    = MIN_CUTOFF + BETA · |velocity|      ← dynamic cutoff, Hz
#     alpha = 1 / (1 + (1/(2π·fc)) / dt)          ← fraction of the new
#                                                   sample that passes through
#
# alpha near 1.0 = the cursor follows the hand instantly (no smoothing);
# alpha near 0.0 = heavy smoothing, and therefore lag.
#
#   MIN_CUTOFF owns the SLOW end.  At rest the velocity term vanishes and
#   fc collapses to MIN_CUTOFF, so this single number sets how sluggish
#   precise aiming feels.  Measured alpha at 60 FPS:
#
#       min_cutoff   alpha    step response (90%)
#            0.5     0.050        46 frames
#            0.8     0.077        29 frames
#            1.0     0.095        24 frames     ← now
#            1.5     0.136        16 frames
#            2.0     0.173        13 frames
#            3.0     0.239         9 frames
#
#   Raise it if slow, careful pointing feels like dragging the cursor
#   through syrup.  Lower it if the cursor visibly trembles while your
#   hand is still — that tremble is MediaPipe landmark noise, and
#   MIN_CUTOFF is the only thing suppressing it.
#
#   BETA owns the MID and HIGH end.  Velocity is measured in *screen*
#   pixels/second, so it runs into the thousands during a swipe and the
#   BETA term dominates fc.  Measured alpha at 60 FPS:
#
#       speed px/s   b=0.005   b=0.010   b=0.015   b=0.030
#              100      0.14      0.17      0.21      0.30
#              300      0.21      0.30      0.37      0.51
#             1000      0.39      0.54      0.63      0.76
#             3000      0.63      0.76      0.83      0.91
#             8000      0.81      0.89      0.93      0.96
#
#   Note how little is left to win above ~3000 px/s: the filter is already
#   ~85% transparent there.  BETA pays off most in the 100–1000 px/s band,
#   which is where ordinary pointing actually happens.  Raise it if the
#   cursor trails your hand during normal movement; lower it if fast moves
#   overshoot or feel twitchy.
#
# NOTE ON SENSITIVITY:  raising CURSOR_SENSITIVITY multiplies the mapped
# velocity by the same factor, which raises the BETA term for a given
# physical hand speed and so opens the filter earlier.  Expect a sensitivity
# increase to feel slightly snappier — and slightly jitterier at rest, since
# landmark noise is magnified too.  If a high multiplier makes the cursor
# tremble, lower MIN_CUTOFF a little to compensate.
#
# Tune in steps of ~0.1 for MIN_CUTOFF and ~0.005 for BETA, one at a time.

MIN_CUTOFF = 1.0     # Hz — cutoff at rest.  Higher = snappier, more jitter.
BETA       = 0.015   # speed coefficient.  Higher = less trailing when moving.
D_CUTOFF   = 1.0     # Hz — cutoff of the velocity estimator itself.

# ── MediaPipe inference cost ───────────────────────────────────────────────
# Inference is the single largest term in the end-to-end delay, far larger
# than anything the filter contributes.  The "lite" graph roughly halves it
# at some cost in landmark precision.
#   0 = lite  (lowest latency)
#   1 = full  (more accurate, slower)
MODEL_COMPLEXITY = 0

# ─── Configuration ───────────────────────────────────────────────────────────

# Webcam settings
CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 60

# ── Display hot-plug polling ───────────────────────────────────────────────
# How often to re-read the virtual desktop rectangle.  GetSystemMetrics is a
# cheap user-mode read, so this is nearly free, but there is no reason to do
# it every frame: a human cannot plug a monitor in faster than this.
POLL_INTERVAL = 2.5   # seconds between display-geometry checks

# ── Dynamic aspect-ratio bounding box (compact, shifted upward) ────────────
# The box is sized from the *horizontal* constraint first, then the height
# is derived to lock to the monitor's aspect ratio.  This makes the box
# smaller than the "largest possible" approach, which increases cursor
# sensitivity — less physical hand displacement is needed to traverse the
# full screen width/height.
#
# Horizontal:
#   SIDE_DEADZONE reserves pixels on each side of the webcam frame.
#   active_width  = CAM_WIDTH − 2 × SIDE_DEADZONE
#
# Vertical (aspect-ratio locked):
#   active_height = active_width / screen_aspect_ratio
#
# Placement:
#   BOTTOM_DEADZONE pushes the box upward so the user's hand isn't cut off.
#   ACTIVE_BOTTOM = CAM_HEIGHT − BOTTOM_DEADZONE
#   ACTIVE_TOP    = ACTIVE_BOTTOM − active_height  (clamped ≥ MIN_TOP_PAD)
#
# Because the ratio comes from the *virtual* desktop, all of this has to be
# recomputed whenever a display is added or removed — hence ScreenGeometry.
#
# This box controls sensitivity *geometrically* (a smaller box needs less
# hand travel); CURSOR_SENSITIVITY then scales the result numerically.  The
# two compose, so shrink the box for a bigger comfortable range of motion
# and reach for CURSOR_SENSITIVITY only when the box is already as small as
# your camera framing allows.

SIDE_DEADZONE   = 60   # px — reserved on each side (controls sensitivity)
BOTTOM_DEADZONE = 150  # px — reserved dead zone at the bottom of the frame
MIN_TOP_PAD     = 10   # px — minimum clearance at the top of the frame

# ─── Win32 cursor function reference ───────────────────────────────────────
# Cache the function reference to avoid repeated attribute lookups in the loop.
_set_cursor_pos = _user32.SetCursorPos

# ─── Helpers ────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Restrict *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def handle_key(key: int) -> bool:
    """Process one keypress from cv2.waitKey().  True means "quit".

    Rebinding the module-level CURSOR_SENSITIVITY is what makes the change
    take effect: ScreenGeometry looks the name up on every to_screen() call
    rather than caching it, so there is no second copy to synchronise.

    Rounding after each step keeps the value clean — repeated float addition
    would drift to 1.5999999999999999 and read badly on the HUD.

    Both waitKey() sites in the main loop funnel through here.  The
    duplicate-frame skip path runs far more often than the bottom of the
    loop, so keys handled in only one place would mostly be swallowed.
    """
    global CURSOR_SENSITIVITY

    if key in (ord('+'), ord('=')):          # '=' is '+' without shift
        CURSOR_SENSITIVITY = round(
            min(SENS_MAX, CURSOR_SENSITIVITY + SENS_STEP), 2)
    elif key in (ord('-'), ord('_')):        # '_' is '-' with shift
        CURSOR_SENSITIVITY = round(
            max(SENS_MIN, CURSOR_SENSITIVITY - SENS_STEP), 2)
    elif key == ord('q'):
        return True
    return False


# ─── Screen geometry (hot-pluggable) ───────────────────────────────────────

class ScreenGeometry:
    """Virtual desktop metrics plus the active box derived from them.

    Every screen-dependent value lives here so that a display change is a
    single atomic swap.  Updating a dozen module-level globals in sequence
    would leave a window in which the main loop maps the cursor using a
    half-rebuilt rectangle.

    Sensitivity scaling also lives here: it is defined relative to the
    desktop centre and clamped to the desktop bounds, both of which this
    object owns and both of which move when a display is hot-plugged.

    Attributes
    ----------
    left, top       Origin of the virtual desktop (may be negative).
    width, height   Size of the virtual desktop, all monitors combined.
    ratio           width / height — drives the active box aspect lock.
    box_*           The active rectangle inside the webcam frame.
    """

    def __init__(self, poll_interval: float = POLL_INTERVAL):
        self._poll_interval = poll_interval
        self._next_poll = 0.0
        self.monitors = 0
        # Seed with whatever the desktop looks like right now.
        self._apply(self._read_metrics())

    # ── Metric acquisition ─────────────────────────────────────────────
    @staticmethod
    def _read_metrics() -> tuple[int, int, int, int]:
        """Snapshot the virtual desktop rectangle as (left, top, w, h)."""
        return (
            _user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            _user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )

    # ── Derived geometry ───────────────────────────────────────────────
    def _apply(self, metrics: tuple[int, int, int, int]) -> None:
        """Adopt *metrics* and rebuild the active box from them."""
        self.left, self.top, self.width, self.height = metrics
        self.monitors = _user32.GetSystemMetrics(SM_CMONITORS)
        self.ratio = self.width / self.height

        # Step 1: width is driven by the side dead zones.
        box_w = CAM_WIDTH - 2 * SIDE_DEADZONE

        # Step 2: height preserves the desktop's aspect ratio.
        box_h = int(box_w / self.ratio)
        box_w = int(box_w)

        # Step 3: anchor the bottom edge above the dead zone.
        bottom = CAM_HEIGHT - BOTTOM_DEADZONE
        top = bottom - box_h

        # Step 4: if the box is too tall, shrink to fit while keeping the ratio.
        if top < MIN_TOP_PAD:
            top = MIN_TOP_PAD
            box_h = bottom - top
            box_w = int(box_h * self.ratio)

        # An extreme desktop ratio (very wide or very tall) can round a
        # dimension down to zero, which would divide by zero in to_screen().
        box_w = max(1, box_w)
        box_h = max(1, box_h)

        # Horizontally centred.
        self.box_left   = (CAM_WIDTH - box_w) // 2
        self.box_right  = self.box_left + box_w
        self.box_top    = top
        self.box_bottom = bottom
        self.box_w      = box_w
        self.box_h      = box_h

    # ── Polling ────────────────────────────────────────────────────────
    def poll(self, now: float) -> bool:
        """Re-read the desktop rectangle at most every *poll_interval* sec.

        Returns True only when the geometry actually changed and was
        rebuilt, so the caller can react (reset filters, log, …).
        """
        if now < self._next_poll:
            return False
        self._next_poll = now + self._poll_interval

        metrics = self._read_metrics()

        # Mid-hotplug the API can transiently report a degenerate rectangle
        # while the driver reconfigures.  Ignore it and keep the last known
        # good geometry; the next poll will pick up the settled values.
        if metrics[2] <= 0 or metrics[3] <= 0:
            return False

        if metrics == (self.left, self.top, self.width, self.height):
            return False

        self._apply(metrics)
        return True

    # ── Mapping ────────────────────────────────────────────────────────
    def to_screen(self, cam_x: float, cam_y: float) -> tuple[float, float]:
        """Map webcam pixel coords → virtual desktop coords.

        Three stages:

          1. Clamp the hand into the active box, so positions outside it
             pin to a desktop edge rather than overshooting.
          2. Map linearly onto the desktop, offset by the desktop origin —
             this is what makes monitors left of / above the primary
             (negative coords) reachable.
          3. Scale about the desktop centre by CURSOR_SENSITIVITY, then
             clamp to the desktop rectangle.

        CURSOR_SENSITIVITY is read fresh on every call, so runtime '+' / '-'
        adjustments apply from the next frame onward.

        The final clamp uses the real desktop bounds rather than [0, W] /
        [0, H]: on a multi-monitor desktop the origin can be negative, and
        clamping to zero would make every display left of the primary
        unreachable.  On a single monitor at (0, 0) the two are identical.
        """
        cx = clamp(cam_x, self.box_left, self.box_right)
        cy = clamp(cam_y, self.box_top,  self.box_bottom)

        # Absolute mapping (this is the CURSOR_SENSITIVITY = 1.0 result).
        mapped_x = self.left + ((cx - self.box_left) / self.box_w) * self.width
        mapped_y = self.top  + ((cy - self.box_top)  / self.box_h) * self.height

        # Centre-scaled sensitivity: push the offset from the middle out by
        # the multiplier, so the desktop edge is reached from a smaller
        # physical displacement.
        mid_x, mid_y = self.center
        mapped_x = mid_x + (mapped_x - mid_x) * CURSOR_SENSITIVITY
        mapped_y = mid_y + (mapped_y - mid_y) * CURSOR_SENSITIVITY

        # Anything the multiplier pushed past an edge saturates there.
        # Clamping *before* the 1€ filter matters: exponential smoothing of
        # in-range values stays in range, so the filter output needs no
        # second clamp and can never be dragged off-screen by history.
        return (
            clamp(mapped_x, self.left, self.left + self.width),
            clamp(mapped_y, self.top,  self.top + self.height),
        )

    @property
    def center(self) -> tuple[float, float]:
        """Centre of the virtual desktop (origin-aware)."""
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    @property
    def effective_box(self) -> tuple[int, int, int, int] | None:
        """Sub-rectangle of the active box that still reaches a desktop edge.

        With CURSOR_SENSITIVITY > 1 the mapping saturates before the box
        perimeter, so the outer band is dead.  Returns (l, t, r, b) for the
        live region, or None when sensitivity does not magnify and the whole
        box is live.  Recomputed per frame, so it resizes as '+' / '-' are
        pressed.
        """
        if CURSOR_SENSITIVITY <= 1.0:
            return None
        mid_x = (self.box_left + self.box_right) / 2.0
        mid_y = (self.box_top  + self.box_bottom) / 2.0
        half_w = (self.box_w / 2.0) / CURSOR_SENSITIVITY
        half_h = (self.box_h / 2.0) / CURSOR_SENSITIVITY
        return (int(mid_x - half_w), int(mid_y - half_h),
                int(mid_x + half_w), int(mid_y + half_h))

    def describe(self) -> str:
        """One-line summary for the console banner / change notices."""
        return (f"{self.width}×{self.height} px  origin ({self.left}, {self.top})  "
                f"ratio {self.ratio:.3f}  monitors {self.monitors}  "
                f"box {self.box_w}×{self.box_h}")


# ─── MediaPipe setup ────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

_hand_kwargs = dict(
    static_image_mode=False,        # video stream mode (faster, uses tracking)
    max_num_hands=1,                # single hand for pointer control
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# model_complexity arrived partway through the 0.8.x line.  Fall back rather
# than crash if this build predates it — the pin must keep working.
try:
    hands = mp_hands.Hands(model_complexity=MODEL_COMPLEXITY, **_hand_kwargs)
    _complexity_note = f"model_complexity={MODEL_COMPLEXITY}"
except TypeError:
    hands = mp_hands.Hands(**_hand_kwargs)
    _complexity_note = "model_complexity unsupported by this build"

# ─── Threaded webcam capture ─────────────────────────────────────────────
#
# Problem:  cv2.VideoCapture.read() blocks the calling thread while the
#           camera hardware transfers a frame.  On many webcams this takes
#           30–70 ms, capping the main loop at ~15–30 FPS regardless of how
#           fast MediaPipe processes the image.
#
# Solution: Move read() into a background daemon thread that runs an
#           infinite grab loop.  The main thread calls stream.read() which
#           returns the latest pre-captured frame instantly (no blocking).
#
#           Main thread              Background thread
#           ───────────              ─────────────────
#           stream.read()   ←────   self._latest (newest)
#           mediapipe.process()      cap.read()  [loops]
#           SetCursorPos()           cap.read()  [loops]
#           cv2.imshow()             cap.read()  [loops]
#           ...                      ...
#
# Staleness:  the grab loop overwrites _latest unconditionally, so any frame
#           the main thread did not collect is simply dropped.  There is no
#           queue and therefore no backlog to drain — read() hands back the
#           newest physical frame every time.  CAP_PROP_BUFFERSIZE = 1 asks
#           the *driver* to do the same one level down; DirectShow does not
#           always honour it, so the value it returns is reported at startup.
#
# Publication:  frame, grabbed flag and sequence number are published as ONE
#           tuple rebind.  Assigning a single attribute is atomic under the
#           GIL, so the reader can never observe a new frame paired with an
#           old sequence number.  Three separate attributes would need a
#           lock; this needs none, and adds no contention to the hot path.

class WebcamStream:
    """Non-blocking webcam reader that always yields the newest frame."""

    def __init__(self, index: int = 0, width: int = 640,
                 height: int = 480, fps: int = 60):
        # Force DirectShow backend on Windows to unlock higher frame rates.
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)

        # Ask the backend to keep at most one frame queued.  Not all
        # backends implement this; keep the result so startup can say so.
        self.buffersize_accepted = bool(self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))

        # Read one frame synchronously so _latest is never None.
        grabbed, frame = self._cap.read()
        self._latest = (grabbed, frame, 0)

        # The stop flag signals the background thread to exit.
        self._stopped = False

        # Launch the capture thread as a daemon so it dies if the main
        # process is killed unexpectedly.
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        """Grab frames continuously, discarding any the reader missed."""
        seq = 0
        while not self._stopped:
            grabbed, frame = self._cap.read()
            if not grabbed:
                continue
            seq += 1
            # Single atomic rebind — no torn reads, no lock needed.
            self._latest = (grabbed, frame, seq)

    def read(self):
        """Return (grabbed, frame, seq) for the newest frame, instantly.

        `seq` increments once per physical frame, so a caller can tell a
        fresh frame from one it has already processed.
        """
        return self._latest

    def stop(self):
        """Signal the thread to stop, wait for it, and release the camera."""
        self._stopped = True
        self._thread.join(timeout=2.0)
        self._cap.release()


# ─── Webcam setup ───────────────────────────────────────────────────────────

stream = WebcamStream(CAM_INDEX, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)

# ─── One Euro Filter implementation ────────────────────────────────────────

class LowPassFilter:
    """Simple first-order exponential low-pass filter.

    Given a smoothing factor α ∈ (0, 1]:
        output = α · input  +  (1 − α) · previous_output

    α = 1 means no filtering;  α → 0 means maximum smoothing.
    """
    def __init__(self, alpha: float, initial: float = 0.0):
        self._y = initial
        self._alpha = alpha
        self._initialised = False

    @property
    def initialised(self) -> bool:
        """True once at least one sample has been absorbed."""
        return self._initialised

    @property
    def last_value(self) -> float:
        """The most recent filtered output."""
        return self._y

    def reset(self) -> None:
        """Discard history so the next sample is adopted verbatim.

        Used when the hand re-enters the frame: the filter must snap to the
        new position rather than glide from wherever the hand was last seen.
        """
        self._initialised = False

    def __call__(self, value: float, alpha: float | None = None) -> float:
        if alpha is not None:
            self._alpha = alpha
        if not self._initialised:
            self._y = value
            self._initialised = True
        else:
            self._y = self._alpha * value + (1.0 - self._alpha) * self._y
        return self._y


class OneEuroFilter:
    """One Euro Filter for a single scalar signal.

    Reference:  Casiez, Roussel & Vogel, "1€ Filter: A Simple Speed-Based
                Low-Pass Filter for Noisy Input in Interactive Systems",
                CHI 2012.  https://cristal.univ-lille.fr/~casiez/1euro/

    Parameters
    ----------
    freq : float      Initial sampling frequency estimate (Hz).
    min_cutoff : float  Minimum cutoff frequency for the position filter.
    beta : float      Speed coefficient — scales how much velocity opens
                      the cutoff.
    d_cutoff : float  Cutoff frequency for the derivative (speed) filter.
    """
    def __init__(self, freq: float, min_cutoff: float = 1.0,
                 beta: float = 0.0, d_cutoff: float = 1.0):
        self._freq0 = freq          # kept so reset() can restore it
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filt = LowPassFilter(self._alpha(min_cutoff))
        self._dx_filt = LowPassFilter(self._alpha(d_cutoff), initial=0.0)
        self._last_time = None

    @staticmethod
    def _alpha(cutoff: float, dt: float = None, freq: float = None) -> float:
        """Compute the smoothing factor α from a cutoff frequency.

            τ  = 1 / (2π · fc)
            α  = 1 / (1 + τ/dt)  =  1 / (1 + 1/(2π · fc · dt))

        Higher cutoff → higher α → less smoothing.
        """
        if dt is None and freq is not None:
            dt = 1.0 / freq
        elif dt is None:
            dt = 1.0 / 30.0  # fallback
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        """Clear all internal state (position, velocity, timing).

        The next __call__ adopts its input verbatim and restarts velocity
        estimation from zero, so the cursor teleports to the hand's real
        position instead of interpolating from a stale one.
        """
        self._x_filt.reset()
        self._dx_filt.reset()
        self._last_time = None
        self._freq = self._freq0

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Filter one sample and return the smoothed value."""
        # Estimate dt from timestamps if available.
        if timestamp is not None and self._last_time is not None:
            dt = timestamp - self._last_time
            if dt > 0:
                self._freq = 1.0 / dt
        self._last_time = timestamp
        dt = 1.0 / self._freq

        # 1) Estimate the derivative (speed) and smooth it.
        if self._x_filt.initialised:
            dx = (x - self._x_filt.last_value) / dt
        else:
            dx = 0.0
        edx = self._dx_filt(dx, alpha=self._alpha(self._d_cutoff, dt=dt))

        # 2) Dynamic cutoff:  fc = min_cutoff + β · |smoothed_speed|
        cutoff = self._min_cutoff + self._beta * abs(edx)

        # 3) Filter the position with the dynamic cutoff.
        return self._x_filt(x, alpha=self._alpha(cutoff, dt=dt))


# ─── State variables ────────────────────────────────────────────────────────

screen = ScreenGeometry(POLL_INTERVAL)

# Create separate One Euro Filters for X and Y axes.
_oef_x = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF)
_oef_y = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF)

# Tracks whether a hand was visible on the *previous* frame.  A False → True
# transition means the hand just re-entered the frame, which is when the
# filters must be reset.  A display change also lowers this flag, reusing the
# same snap-to-target path (see the main loop).
hand_present = False

# Sequence number of the last frame actually processed, so duplicates can be
# skipped instead of re-running inference on data we already consumed.
last_seq = -1

# Last smoothed position — retained only for the on-screen jump readout.
prev_x, prev_y = screen.center

# perf_counter() is monotonic and sub-microsecond.  time.time() on Windows
# is backed by GetSystemTimeAsFileTime, whose ~15.6 ms granularity would
# badly quantise the dt estimate the 1€ filter depends on at 60 FPS.
prev_time = time.perf_counter()

# ─── Main loop ──────────────────────────────────────────────────────────────

print(f"Virtual desk: {screen.describe()}")
print(f"Webcam      : {CAM_WIDTH} × {CAM_HEIGHT} @ {CAM_FPS} FPS")
print(f"Active box  : x[{screen.box_left}–{screen.box_right}]  "
      f"y[{screen.box_top}–{screen.box_bottom}]")
print(f"Sensitivity : {CURSOR_SENSITIVITY}×  start value, "
      f"adjustable {SENS_MIN}–{SENS_MAX} in steps of {SENS_STEP}")
print(f"1€ Filter   : min_cutoff={MIN_CUTOFF}  β={BETA}  d_cutoff={D_CUTOFF}")
print(f"MediaPipe   : {_complexity_note}")
print(f"Smoothing   : filter-only (no blocking interpolation)")
print(f"Camera      : threaded, buffersize=1 "
      f"{'accepted' if stream.buffersize_accepted else 'REFUSED by backend'}")
print(f"Display poll: every {POLL_INTERVAL}s (hot-plug aware)")
print("Controls    : '+'/'=' faster   '-'/'_' slower   'q' quit")
print("(the preview window must have focus for keys to register)\n")

try:
    while True:
        success, frame, seq = stream.read()
        if not success:
            continue

        # ── Duplicate frame guard ───────────────────────────────────────
        # The main loop can outrun the camera.  Re-running MediaPipe on a
        # frame already processed costs a full inference and returns an
        # identical answer, delaying the next *real* frame by that much.
        # Skip it, but keep pumping the GUI — this branch is where most
        # keypresses actually land, so it must handle them too.
        if seq == last_seq:
            if handle_key(cv2.waitKey(1) & 0xFF):
                break
            continue
        last_seq = seq

        # One timestamp per iteration, shared by the display poll, the 1€
        # filter and the FPS counter so they cannot disagree about "now".
        now_ts = time.perf_counter()

        # ── Display hot-plug polling ────────────────────────────────────
        # Cheap rate-limited check; returns True only on a real change.
        # Lowering hand_present routes the next frame through the existing
        # re-entry reset, so the filters snap into the new coordinate space
        # instead of gliding from a position that may no longer exist.
        if screen.poll(now_ts):
            print(f"[display] geometry changed → {screen.describe()}")
            hand_present = False

        # Mirror the frame so it feels natural (like looking in a mirror).
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB for MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mark the buffer read-only.  MediaPipe defensively copies any array
        # it might mutate; flagging it non-writeable lets it borrow the
        # buffer instead, saving a full-frame allocation + memcpy per frame.
        rgb.flags.writeable = False
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Landmark 8 = Index Finger Tip.
            tip = hand.landmark[8]

            # MediaPipe returns normalised coords [0, 1]; convert to pixels.
            raw_x = tip.x * CAM_WIDTH
            raw_y = tip.y * CAM_HEIGHT

            # Clamp into the active box, map onto the virtual desktop, and
            # apply centre-scaled sensitivity — all inside to_screen().
            target_x, target_y = screen.to_screen(raw_x, raw_y)

            # ── Re-entry / geometry reset ────────────────────────────────
            # Either the hand was absent last frame, or the desktop just
            # changed shape.  Without this the filters would still hold a
            # position from the old situation and the cursor would slide
            # across the screen from that stale point.  Clearing them makes
            # the very next filter call adopt the raw target verbatim.
            if not hand_present:
                _oef_x.reset()
                _oef_y.reset()
                prev_x, prev_y = target_x, target_y
                hand_present = True

            # ── One Euro Filter smoothing ────────────────────────────────
            # The filter is the *only* smoothing stage.  It internally tracks
            # time and speed to compute a dynamic cutoff: still → heavy
            # smoothing, fast → light.  Nothing here blocks the loop, so the
            # next camera sample arrives as soon as the hardware has it.
            smooth_x = _oef_x(target_x, timestamp=now_ts)
            smooth_y = _oef_y(target_y, timestamp=now_ts)

            _set_cursor_pos(int(smooth_x), int(smooth_y))

            # Per-frame travel, kept purely as a tuning readout.
            jump = math.hypot(smooth_x - prev_x, smooth_y - prev_y)
            prev_x = smooth_x
            prev_y = smooth_y

            # ── Visualisation overlays ──────────────────────────────────
            # Draw hand skeleton.
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Draw a filled circle at the index finger tip.
            cx, cy = int(raw_x), int(raw_y)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Show how far the cursor moved this frame.
            cv2.putText(
                frame, f"j={jump:.0f}", (cx + 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )
        else:
            # No hand this frame — arm the reset for whenever it returns.
            hand_present = False

        # Draw the active bounding box on the preview.  Read from `screen`
        # so it follows the box when a display is plugged or unplugged.
        cv2.rectangle(
            frame,
            (screen.box_left,  screen.box_top),
            (screen.box_right, screen.box_bottom),
            (255, 0, 255), 2,
        )

        # With sensitivity > 1 only an inner region still reaches the desktop
        # edges; everything outside it is clamped flat.  Drawn every frame so
        # it shrinks and grows live as '+' / '-' are pressed.
        _eff = screen.effective_box
        if _eff is not None:
            cv2.rectangle(frame, (_eff[0], _eff[1]), (_eff[2], _eff[3]),
                          (0, 165, 255), 1)

        # ── HUD ─────────────────────────────────────────────────────────
        # FPS counter — counts only frames that were really processed.
        fps = 1.0 / (now_ts - prev_time) if now_ts > prev_time else 0.0
        prev_time = now_ts
        cv2.putText(
            frame, f"FPS: {int(fps)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        # Current sensitivity, directly below the FPS readout.  Amber to
        # match the live-region rectangle it controls.
        cv2.putText(
            frame, f"Speed: {CURSOR_SENSITIVITY:.1f}x", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
        )

        # Desktop summary underneath.
        cv2.putText(
            frame, f"{screen.width}x{screen.height} ({screen.monitors} mon)",
            (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        # Key hints along the bottom edge.
        cv2.putText(
            frame, "+/- speed   q quit", (10, CAM_HEIGHT - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("Hand Cursor Control", frame)

        # Sensitivity keys and quit, same handler as the skip path above.
        if handle_key(cv2.waitKey(1) & 0xFF):
            break

finally:
    stream.stop()
    cv2.destroyAllWindows()
    hands.close()
    print("Shutdown complete.")
