"""
Hand-Tracking Cursor Controller (v10 — Hot-Pluggable Displays)
==============================================================
Moves the mouse cursor by tracking the index finger tip (MediaPipe landmark 8).
Smoothing is handled *exclusively* by a One Euro Filter (Casiez et al. 2012),
which adapts its cutoff frequency to hand speed: heavy smoothing at rest,
light smoothing during fast swipes.  Movement only — no click gestures.

The active box maps onto the *virtual desktop* (all monitors combined), not
just the primary display.

v10 makes that mapping survive display changes.  Plugging or unplugging a
monitor, or rearranging the desktop, resizes the virtual screen underneath a
running script; every derived constant (aspect ratio, active box, screen
mapping) silently goes stale.  Rather than hook WM_DISPLAYCHANGE, the main
loop re-reads GetSystemMetrics every POLL_INTERVAL seconds and rebuilds the
geometry only when the rectangle actually moves.  The cost is four cheap
user-mode calls per interval.

All screen-dependent state therefore lives inside ScreenGeometry, so a
display change is one atomic recompute rather than a dozen scattered globals
updated in sequence — a half-updated box would map the cursor into a
rectangle that no longer exists.

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

SIDE_DEADZONE   = 60   # px — reserved on each side (controls sensitivity)
BOTTOM_DEADZONE = 150  # px — reserved dead zone at the bottom of the frame
MIN_TOP_PAD     = 10   # px — minimum clearance at the top of the frame

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

# ─── Helpers ────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Restrict *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


# ─── Screen geometry (hot-pluggable) ───────────────────────────────────────

class ScreenGeometry:
    """Virtual desktop metrics plus the active box derived from them.

    Every screen-dependent value lives here so that a display change is a
    single atomic swap.  Updating a dozen module-level globals in sequence
    would leave a window in which the main loop maps the cursor using a
    half-rebuilt rectangle.

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

        The hand is first clamped into the active box, so positions outside
        it pin the cursor to a desktop edge rather than overshooting.  The
        result is offset by the desktop origin, which is what makes monitors
        left of / above the primary (negative coords) reachable.
        """
        cx = clamp(cam_x, self.box_left, self.box_right)
        cy = clamp(cam_y, self.box_top,  self.box_bottom)
        return (
            self.left + ((cx - self.box_left) / self.box_w) * self.width,
            self.top  + ((cy - self.box_top)  / self.box_h) * self.height,
        )

    @property
    def center(self) -> tuple[float, float]:
        """Centre of the virtual desktop (origin-aware)."""
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def describe(self) -> str:
        """One-line summary for the console banner / change notices."""
        return (f"{self.width}×{self.height} px  origin ({self.left}, {self.top})  "
                f"ratio {self.ratio:.3f}  monitors {self.monitors}  "
                f"box {self.box_w}×{self.box_h}")


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

screen = ScreenGeometry(POLL_INTERVAL)

# Create separate One Euro Filters for X and Y axes.
_oef_x = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF)
_oef_y = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF)

# Tracks whether a hand was visible on the *previous* frame.  A False → True
# transition means the hand just re-entered the frame, which is when the
# filters must be reset.  A display change also lowers this flag, reusing the
# same snap-to-target path (see the main loop).
hand_present = False

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
print(f"1€ Filter   : min_cutoff={MIN_CUTOFF}  β={BETA}  d_cutoff={D_CUTOFF}")
print(f"Smoothing   : filter-only (no blocking interpolation)")
print(f"Camera      : threaded (background capture)")
print(f"Display poll: every {POLL_INTERVAL}s (hot-plug aware)")
print("Press 'q' in the preview window to quit.\n")

try:
    while True:
        success, frame = stream.read()
        if not success:
            continue

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

            # Clamp into the active box and map onto the virtual desktop.
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

        # FPS counter.
        fps = 1.0 / (now_ts - prev_time) if now_ts > prev_time else 0.0
        prev_time = now_ts
        cv2.putText(
            frame, f"FPS: {int(fps)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        # Monitor count, so a hot-plug is visible in the preview too.
        cv2.putText(
            frame, f"{screen.width}x{screen.height} ({screen.monitors} mon)",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
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
