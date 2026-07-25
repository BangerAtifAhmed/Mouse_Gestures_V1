"""
Hand-Tracking Cursor Controller (v7 — Threaded Camera)
======================================================
Moves the mouse cursor by tracking the index finger tip (MediaPipe landmark 8).
Uses a dedicated background thread for frame capture so cap.read() never blocks
the main processing loop.  Active bounding box is auto-sized to the screen's
aspect ratio.  Direct Win32 API calls for zero-latency cursor positioning.
Movement only — no click gestures.

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

# ── Dynamic aspect-ratio bounding box (shifted upward) ─────────────────────
# We compute the largest rectangle that:
#   1. Fits inside the webcam frame (CAM_WIDTH × CAM_HEIGHT)
#   2. Has the exact same aspect ratio as the user's monitor
#   3. Is centred *horizontally* but pushed **upward** vertically
#
# The bottom of the webcam frame is a dead zone — the user's hand gets
# physically cut off there before the cursor can reach the screen bottom.
# BOTTOM_DEADZONE reserves that many pixels at the bottom of the frame.
#
# Vertical layout:
#   ACTIVE_BOTTOM = CAM_HEIGHT − BOTTOM_DEADZONE
#   ACTIVE_TOP    = ACTIVE_BOTTOM − box_h   (clamped to ≥ MIN_TOP_PAD)
#
# If ACTIVE_TOP would fall below MIN_TOP_PAD, the box height is shrunk to
# fit and the width is recalculated to preserve the aspect ratio.

PADDING        = 10   # px — horizontal breathing room on each side
BOTTOM_DEADZONE = 150  # px — reserved dead zone at the bottom of the frame
MIN_TOP_PAD    = 10   # px — minimum clearance at the top of the frame

_screen_ratio = SCREEN_W / SCREEN_H
_cam_ratio    = CAM_WIDTH / CAM_HEIGHT

# Step 1: compute the ideal box size (same as before).
if _screen_ratio > _cam_ratio:
    _box_w = CAM_WIDTH - 2 * PADDING
    _box_h = _box_w / _screen_ratio
else:
    _box_h = CAM_HEIGHT - 2 * PADDING
    _box_w = _box_h * _screen_ratio

_box_w = int(_box_w)
_box_h = int(_box_h)

# Step 2: anchor the bottom edge above the dead zone.
ACTIVE_BOTTOM = CAM_HEIGHT - BOTTOM_DEADZONE
ACTIVE_TOP    = ACTIVE_BOTTOM - _box_h

# Step 3: if the box is too tall, shrink it to fit while keeping the ratio.
if ACTIVE_TOP < MIN_TOP_PAD:
    ACTIVE_TOP = MIN_TOP_PAD
    _box_h = ACTIVE_BOTTOM - ACTIVE_TOP
    _box_w = int(_box_h * _screen_ratio)

# Horizontally centred (unchanged).
ACTIVE_LEFT  = (CAM_WIDTH - _box_w) // 2
ACTIVE_RIGHT = ACTIVE_LEFT + _box_w
ACTIVE_W = _box_w
ACTIVE_H = _box_h

# ── Adaptive EMA parameters ────────────────────────────────────────────────
# Instead of a fixed α, we compute α per frame based on how far the new
# target is from the previous smoothed position (Euclidean distance in
# screen-space pixels).
#
#   distance = √( (target_x − prev_x)² + (target_y − prev_y)² )
#
# We then linearly interpolate α between two extremes:
#
#   • ALPHA_MIN  (e.g. 0.05)  — used when distance ≤ DIST_SLOW
#       → aggressive jitter suppression when the hand is nearly still
#
#   • ALPHA_MAX  (e.g. 0.55)  — used when distance ≥ DIST_FAST
#       → minimal lag when the user swipes quickly
#
# For distances between DIST_SLOW and DIST_FAST, α is linearly ramped:
#
#   t     = clamp( (distance − DIST_SLOW) / (DIST_FAST − DIST_SLOW) , 0, 1 )
#   alpha = ALPHA_MIN + t × (ALPHA_MAX − ALPHA_MIN)
#
# This gives a smooth, speed-dependent transition: steady hands feel locked
# in place while fast movements track with near-zero latency.

ALPHA_MIN  = 0.02    # α when nearly still  (heavier smoothing at 60 FPS)
ALPHA_MAX  = 0.55    # α during fast swipes  (light smoothing)
DIST_SLOW  = 8.0     # px — below this, hand is "still" (wider dead zone)
DIST_FAST  = 120.0   # px — above this, hand is "fast"

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

# ─── State variables ────────────────────────────────────────────────────────

# Previous smoothed screen coordinates (initialised to screen centre).
prev_x = SCREEN_W / 2.0
prev_y = SCREEN_H / 2.0

prev_time = time.time()

# ─── Helpers ────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Restrict *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def adaptive_alpha(distance: float) -> float:
    """Return a dynamic EMA α based on the displacement *distance* (px).

    Linear ramp between ALPHA_MIN and ALPHA_MAX over [DIST_SLOW, DIST_FAST].

        t     = clamp((distance − DIST_SLOW) / (DIST_FAST − DIST_SLOW), 0, 1)
        alpha = ALPHA_MIN  +  t · (ALPHA_MAX − ALPHA_MIN)

    Small distance  →  low α   →  heavy smoothing  (kills jitter)
    Large distance  →  high α  →  light smoothing   (kills lag)
    """
    t = clamp((distance - DIST_SLOW) / (DIST_FAST - DIST_SLOW), 0.0, 1.0)
    return ALPHA_MIN + t * (ALPHA_MAX - ALPHA_MIN)

# ─── Main loop ──────────────────────────────────────────────────────────────

print(f"Screen      : {SCREEN_W} × {SCREEN_H}  (ratio {_screen_ratio:.3f})")
print(f"Webcam      : {CAM_WIDTH} × {CAM_HEIGHT} @ {CAM_FPS} FPS")
print(f"Active box  : {ACTIVE_W} × {ACTIVE_H} px  "
      f"x[{ACTIVE_LEFT}–{ACTIVE_RIGHT}]  y[{ACTIVE_TOP}–{ACTIVE_BOTTOM}]")
print(f"Adaptive EMA: α ∈ [{ALPHA_MIN}, {ALPHA_MAX}]  "
      f"speed range [{DIST_SLOW}, {DIST_FAST}] px")
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

            # ── Adaptive EMA smoothing ──────────────────────────────────
            # 1) Compute Euclidean distance from the last smoothed position
            #    to the new target.  This measures "how fast" the hand moved.
            distance = math.hypot(target_x - prev_x, target_y - prev_y)

            # 2) Derive α dynamically from the distance.
            alpha = adaptive_alpha(distance)

            # 3) Apply EMA:  S_t = α · X_t  +  (1 − α) · S_{t−1}
            smooth_x = alpha * target_x + (1 - alpha) * prev_x
            smooth_y = alpha * target_y + (1 - alpha) * prev_y

            # Store for next frame.
            prev_x = smooth_x
            prev_y = smooth_y

            # ── Move cursor via Win32 SetCursorPos (lowest possible latency)
            _set_cursor_pos(int(smooth_x), int(smooth_y))

            # ── Visualisation overlays ──────────────────────────────────
            # Draw hand skeleton.
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Draw a filled circle at the index finger tip.
            cx, cy = int(raw_x), int(raw_y)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Show the current adaptive α value near the fingertip.
            cv2.putText(
                frame, f"a={alpha:.2f}", (cx + 15, cy - 10),
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
