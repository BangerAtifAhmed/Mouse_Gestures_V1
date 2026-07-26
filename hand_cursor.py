"""
Hand-Tracking Cursor Controller (v8 — One Euro Filter + Sub-Frame Glide)
========================================================================
Moves the mouse cursor by tracking the index finger tip (MediaPipe landmark 8).
Uses a One Euro Filter (Casiez et al. 2012) for speed-adaptive smoothing that
kills jitter at rest while staying responsive during fast swipes — even at the
camera's hardware-limited 30 Hz.  Sub-frame interpolation ensures large jumps
look like smooth glides, not teleports.  Movement only — no click gestures.

Dependencies:  pip install opencv-python mediapipe==0.8.11
Platform:      Windows only (uses user32.dll)
"""

import ctypes
import cv2
import math
import mediapipe as mp
import threading
import time

# ─── Win32 screen resolution (direct API, no wrapper overhead) ─────────────
# GetSystemMetrics(0) → screen width in pixels  (SM_CXSCREEN)
# GetSystemMetrics(1) → screen height in pixels (SM_CYSCREEN)
_user32 = ctypes.windll.user32
SCREEN_W = _user32.GetSystemMetrics(0)
SCREEN_H = _user32.GetSystemMetrics(1)

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
# the industry standard for pointer/hand tracking at low frame rates.
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
# Tuning guide:
#   - Jittery at rest?  Lower MIN_CUTOFF.
#   - Too laggy on fast moves?  Raise BETA.
#   - Derivative noisy?  Lower D_CUTOFF.

MIN_CUTOFF = 0.8    # Hz — position cutoff at rest  (lower = smoother)
BETA       = 0.01   # speed coefficient  (higher = snappier on fast moves)
D_CUTOFF   = 1.0    # Hz — cutoff for the speed (derivative) filter

# ── Sub-frame interpolation ────────────────────────────────────────────────
# At 30 Hz, a fast hand swipe can jump 200+ screen pixels between frames.
# Even with perfect filtering, a single SetCursorPos() call per frame would
# look like a teleport.  We split large jumps into INTERP_STEPS smaller
# moves spread across a short time window, creating a visible glide.
#
#   If distance > INTERP_THRESHOLD:
#       for step in 1..INTERP_STEPS:
#           lerp_t = step / INTERP_STEPS
#           pos    = prev + lerp_t * (target - prev)
#           SetCursorPos(pos)
#           sleep(frame_budget / INTERP_STEPS)

INTERP_STEPS     = 5      # number of sub-steps for large jumps
INTERP_THRESHOLD = 80.0   # px — jump distance that triggers interpolation

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
        if self._x_filt._initialised:
            dx = (x - self._x_filt._y) / dt
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

# Previous cursor position for sub-frame interpolation.
prev_x = SCREEN_W / 2.0
prev_y = SCREEN_H / 2.0

prev_time = time.time()

# ─── Helpers ────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Restrict *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))

# ─── Main loop ──────────────────────────────────────────────────────────────

print(f"Screen      : {SCREEN_W} × {SCREEN_H}  (ratio {_screen_ratio:.3f})")
print(f"Webcam      : {CAM_WIDTH} × {CAM_HEIGHT} @ {CAM_FPS} FPS")
print(f"Active box  : {ACTIVE_W} × {ACTIVE_H} px  "
      f"x[{ACTIVE_LEFT}–{ACTIVE_RIGHT}]  y[{ACTIVE_TOP}–{ACTIVE_BOTTOM}]")
print(f"1€ Filter   : min_cutoff={MIN_CUTOFF}  β={BETA}  d_cutoff={D_CUTOFF}")
print(f"Interpolation: {INTERP_STEPS} steps when jump > {INTERP_THRESHOLD} px")
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

            # ── Screen mapping ──────────────────────────────────────────
            # Linear interpolation from active-area coords → screen coords:
            #   screen_x = (clamped_x − ACTIVE_LEFT) / ACTIVE_W  ×  SCREEN_W
            # Maps the active area edges to the full screen boundaries.
            target_x = ((clamped_x - ACTIVE_LEFT) / ACTIVE_W) * SCREEN_W
            target_y = ((clamped_y - ACTIVE_TOP)  / ACTIVE_H) * SCREEN_H

            # ── One Euro Filter smoothing ────────────────────────────────
            # Feed each axis through its own 1€ filter instance.
            # The filter internally tracks time and speed to compute a
            # dynamic cutoff: still → heavy smoothing, fast → light.
            now_ts = time.time()
            smooth_x = _oef_x(target_x, timestamp=now_ts)
            smooth_y = _oef_y(target_y, timestamp=now_ts)

            # ── Sub-frame interpolation (anti-teleport) ─────────────────
            # At 30 Hz, even filtered positions can jump 100+ px between
            # frames.  We split the jump into INTERP_STEPS smaller moves
            # spread over the frame budget so the cursor *glides* visibly.
            jump = math.hypot(smooth_x - prev_x, smooth_y - prev_y)

            if jump > INTERP_THRESHOLD and INTERP_STEPS > 1:
                # Spread the glide over ~1 frame period (≈33 ms at 30 Hz).
                step_delay = (1.0 / 30.0) / INTERP_STEPS
                for s in range(1, INTERP_STEPS + 1):
                    t = s / INTERP_STEPS
                    ix = prev_x + t * (smooth_x - prev_x)
                    iy = prev_y + t * (smooth_y - prev_y)
                    _set_cursor_pos(int(ix), int(iy))
                    if s < INTERP_STEPS:
                        time.sleep(step_delay)
            else:
                _set_cursor_pos(int(smooth_x), int(smooth_y))

            # Store for next frame's interpolation baseline.
            prev_x = smooth_x
            prev_y = smooth_y

            # ── Visualisation overlays ──────────────────────────────────
            # Draw hand skeleton.
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Draw a filled circle at the index finger tip.
            cx, cy = int(raw_x), int(raw_y)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Show the filter's effective cutoff and jump distance.
            cv2.putText(
                frame, f"j={jump:.0f}", (cx + 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )

        # Draw the active bounding box on the preview.
        cv2.rectangle(
            frame,
            (ACTIVE_LEFT, ACTIVE_TOP),
            (ACTIVE_RIGHT, ACTIVE_BOTTOM),
            (255, 0, 255), 2,
        )

        # FPS counter.
        now = time.time()
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
