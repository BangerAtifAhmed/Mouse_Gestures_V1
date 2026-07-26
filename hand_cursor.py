"""
Hand-Tracking Cursor Controller (v9 — Non-Blocking, Filter-Only)
================================================================
Moves the mouse cursor by tracking the index finger tip (MediaPipe landmark 8).
Smoothing is handled *exclusively* by a One Euro Filter (Casiez et al. 2012),
which adapts its cutoff frequency to hand speed: heavy smoothing at rest,
light smoothing during fast swipes.  Movement only — no click gestures.

v9 removes the sub-frame interpolation "glide" from v8.  That code called
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

The One Euro Filter already produces continuous motion without stalling the
capture loop, so the correct fix is to let it do its job unimpeded.

The active box maps onto the *virtual desktop* (all monitors combined), not
just the primary display.

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
SCREEN_LEFT = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
SCREEN_TOP  = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
SCREEN_W    = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
SCREEN_H    = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
MONITORS    = _user32.GetSystemMetrics(SM_CMONITORS)

# ─── Configuration ───────────────────────────────────────────────────────────

# Webcam settings
CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 60

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

SIDE_DEADZONE   = 60   # px — reserved on each side (controls sensitivity)
BOTTOM_DEADZONE = 150  # px — reserved dead zone at the bottom of the frame
MIN_TOP_PAD     = 10   # px — minimum clearance at the top of the frame

_screen_ratio = SCREEN_W / SCREEN_H

# Step 1: width is driven by the side dead zones.
_box_w = CAM_WIDTH - 2 * SIDE_DEADZONE

# Step 2: height preserves the screen's aspect ratio.
_box_h = int(_box_w / _screen_ratio)
_box_w = int(_box_w)

# Step 3: anchor the bottom edge above the dead zone.
ACTIVE_BOTTOM = CAM_HEIGHT - BOTTOM_DEADZONE
ACTIVE_TOP    = ACTIVE_BOTTOM - _box_h

# Step 4: if the box is too tall, shrink to fit while keeping the ratio.
if ACTIVE_TOP < MIN_TOP_PAD:
    ACTIVE_TOP = MIN_TOP_PAD
    _box_h = ACTIVE_BOTTOM - ACTIVE_TOP
    _box_w = int(_box_h * _screen_ratio)

# Horizontally centred.
ACTIVE_LEFT  = (CAM_WIDTH - _box_w) // 2
ACTIVE_RIGHT = ACTIVE_LEFT + _box_w
ACTIVE_W = _box_w
ACTIVE_H = _box_h

# ── One Euro Filter parameters ─────────────────────────────────────────────
# The One Euro Filter (Casiez, Roussel & Vogel, CHI 2012) is an adaptive
# low-pass filter designed for real-time noisy signal smoothing.  It is
# the industry standard for pointer/hand tracking at low frame rates, and
# is now the *only* smoothing stage in this pipeline.
#
# Core idea:
#   - At rest (low speed):  use heavy smoothing to kill jitter.
#   - During movement (high speed):  reduce smoothing to avoid lag.
#
# It achieves this with TWO cascaded exponential smoothing stages:
#
#   1. A first low-pass filter smooths the raw derivative (speed).
#   2. The smoothed speed is used to compute a dynamic cutoff frequency
#      for a second low-pass filter that smooths the position.
#
# Key formulas:
#
#   α(fc)  = 1 / (1 + 1/(2π · fc · dt))     ← smoothing factor from cutoff
#   fc     = MIN_CUTOFF + BETA · |dx_smooth|  ← dynamic cutoff for position
#
# Parameters:
#   MIN_CUTOFF  — cutoff freq when hand is still (Hz).  Lower = smoother.
#   BETA        — how much speed increases the cutoff.  Higher = snappier.
#   D_CUTOFF    — cutoff for the derivative filter (Hz).  Usually ~1.0.
#
# Tuning guide (BETA is the primary knob now that interpolation is gone):
#   - Jittery at rest?  Lower MIN_CUTOFF.
#   - Too laggy on fast moves?  Raise BETA.
#   - Derivative noisy?  Lower D_CUTOFF.

MIN_CUTOFF = 0.8    # Hz — position cutoff at rest  (lower = smoother)
BETA       = 0.01   # speed coefficient  (higher = snappier on fast moves)
D_CUTOFF   = 1.0    # Hz — cutoff for the speed (derivative) filter

# ─── Win32 cursor function reference ───────────────────────────────────────
# Cache the function reference to avoid repeated attribute lookups in the loop.
_set_cursor_pos = _user32.SetCursorPos

# ─── MediaPipe setup ────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,        # video stream mode (faster, uses tracking)
    max_num_hands=1,                # single hand for pointer control
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

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
#           stream.read()   ←────   self._frame (latest)
#           mediapipe.process()      cap.read()  [loops]
#           SetCursorPos()           cap.read()  [loops]
#           cv2.imshow()             cap.read()  [loops]
#           ...                      ...

class WebcamStream:
    """Non-blocking webcam reader using a background thread."""

    def __init__(self, index: int = 0, width: int = 640,
                 height: int = 480, fps: int = 60):
        # Force DirectShow backend on Windows to unlock higher frame rates.
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)

        # Read one frame synchronously so self._frame is never None.
        self._grabbed, self._frame = self._cap.read()

        # The stop flag signals the background thread to exit.
        self._stopped = False

        # Launch the capture thread as a daemon so it dies if the main
        # process is killed unexpectedly.
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        """Continuously grab frames in the background."""
        while not self._stopped:
            grabbed, frame = self._cap.read()
            if grabbed:
                self._grabbed = grabbed
                self._frame = frame

    def read(self):
        """Return the most recent frame instantly (non-blocking)."""
        return self._grabbed, self._frame

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

# Create separate One Euro Filters for X and Y axes.
_oef_x = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF)
_oef_y = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF)

# Tracks whether a hand was visible on the *previous* frame.  A False → True
# transition means the hand just re-entered the frame, which is when the
# filters must be reset (see the main loop).
hand_present = False

# Last smoothed position — retained only for the on-screen jump readout.
# Seeded at the centre of the virtual desktop (origin-aware).
prev_x = SCREEN_LEFT + SCREEN_W / 2.0
prev_y = SCREEN_TOP  + SCREEN_H / 2.0

# perf_counter() is monotonic and sub-microsecond.  time.time() on Windows
# is backed by GetSystemTimeAsFileTime, whose ~15.6 ms granularity would
# badly quantise the dt estimate the 1€ filter depends on at 60 FPS.
prev_time = time.perf_counter()

# ─── Helpers ────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Restrict *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))

# ─── Main loop ──────────────────────────────────────────────────────────────

print(f"Virtual desk: {SCREEN_W} × {SCREEN_H} px  origin ({SCREEN_LEFT}, {SCREEN_TOP})  "
      f"ratio {_screen_ratio:.3f}")
print(f"Monitors    : {MONITORS}")
print(f"Webcam      : {CAM_WIDTH} × {CAM_HEIGHT} @ {CAM_FPS} FPS")
print(f"Active box  : {ACTIVE_W} × {ACTIVE_H} px  "
      f"x[{ACTIVE_LEFT}–{ACTIVE_RIGHT}]  y[{ACTIVE_TOP}–{ACTIVE_BOTTOM}]")
print(f"1€ Filter   : min_cutoff={MIN_CUTOFF}  β={BETA}  d_cutoff={D_CUTOFF}")
print(f"Smoothing   : filter-only (no blocking interpolation)")
print(f"Camera      : threaded (background capture)")
print("Press 'q' in the preview window to quit.\n")

try:
    while True:
        success, frame = stream.read()
        if not success:
            continue

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

            # Clamp into the active bounding box so we never exceed screen edges.
            clamped_x = clamp(raw_x, ACTIVE_LEFT, ACTIVE_RIGHT)
            clamped_y = clamp(raw_y, ACTIVE_TOP,  ACTIVE_BOTTOM)

            # ── Virtual desktop mapping ─────────────────────────────────
            # Linear interpolation from active-area coords → virtual-screen
            # coords, offset by the desktop origin so monitors positioned
            # left of / above the primary (negative origin) are reachable:
            #
            #   screen_x = SCREEN_LEFT
            #            + (clamped_x − ACTIVE_LEFT) / ACTIVE_W × SCREEN_W
            #
            # The active-area edges therefore map to the outer corners of
            # the whole multi-monitor desktop, not just display 1.
            target_x = SCREEN_LEFT + ((clamped_x - ACTIVE_LEFT) / ACTIVE_W) * SCREEN_W
            target_y = SCREEN_TOP  + ((clamped_y - ACTIVE_TOP)  / ACTIVE_H) * SCREEN_H

            # ── Re-entry reset ───────────────────────────────────────────
            # The hand was absent last frame and is back now.  Without this,
            # the filters would still hold the position from wherever the
            # hand vanished, and the cursor would slide across the screen
            # from that stale point.  Clearing them makes the very next
            # filter call adopt the raw target verbatim.
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
            now_ts = time.perf_counter()
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

        # Draw the active bounding box on the preview.
        cv2.rectangle(
            frame,
            (ACTIVE_LEFT, ACTIVE_TOP),
            (ACTIVE_RIGHT, ACTIVE_BOTTOM),
            (255, 0, 255), 2,
        )

        # FPS counter.
        now = time.perf_counter()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now
        cv2.putText(
            frame, f"FPS: {int(fps)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        cv2.imshow("Hand Cursor Control", frame)

        # Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stream.stop()
    cv2.destroyAllWindows()
    hands.close()
    print("Shutdown complete.")
