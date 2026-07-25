"""
Hand-Tracking Cursor Controller (v4 — Windows Low-Latency + Pinch Click)
========================================================================
Moves the mouse cursor by tracking the index finger tip (MediaPipe landmark 8)
and performs left-click via a thumb-index pinch gesture.  Uses direct Win32 API
calls (ctypes) for the absolute lowest latency — no wrapper libraries.

Dependencies:  pip install opencv-python mediapipe==0.8.11
Platform:      Windows only (uses user32.dll)
"""

import ctypes
import cv2
import math
import mediapipe as mp
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
CAM_FPS = 30

# Asymmetric active bounding box margins (pixels inside the webcam frame).
# Only hand positions within this inner rectangle are mapped to the screen.
# Using a larger BOTTOM margin means the user doesn't have to lower their
# hand as far to reach the bottom of the monitor.
#
#   ┌──────────────────────────────────┐  ← webcam frame (640×480)
#   │         MARGIN_TOP (50)          │
#   │  M_L ┌────────────────────┐ M_R  │
#   │ (100)│   ACTIVE AREA      │(100) │
#   │      │   (mapped to       │      │
#   │      │    full screen)    │      │
#   │      └────────────────────┘      │
#   │       MARGIN_BOTTOM (200)        │
#   └──────────────────────────────────┘
MARGIN_TOP    = 50
MARGIN_BOTTOM = 200
MARGIN_LEFT   = 100
MARGIN_RIGHT  = 100

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

ALPHA_MIN  = 0.05    # α when nearly still  (heavy smoothing)
ALPHA_MAX  = 0.55    # α during fast swipes  (light smoothing)
DIST_SLOW  = 5.0     # px — below this, hand is "still"
DIST_FAST  = 120.0   # px — above this, hand is "fast"

# ── Pinch-to-click parameters ──────────────────────────────────────────────
# A "pinch" is detected when the Euclidean distance (in webcam pixels)
# between the Thumb tip (landmark 4) and the Index Finger tip (landmark 8)
# falls below PINCH_THRESHOLD.  Increase the value to make clicking easier
# (more forgiving); decrease for tighter precision.
PINCH_THRESHOLD = 40.0   # px — distance below which a pinch is registered

# ─── Derived constants ──────────────────────────────────────────────────────

# Active area boundaries inside the webcam frame
ACTIVE_LEFT   = MARGIN_LEFT
ACTIVE_TOP    = MARGIN_TOP
ACTIVE_RIGHT  = CAM_WIDTH  - MARGIN_RIGHT
ACTIVE_BOTTOM = CAM_HEIGHT - MARGIN_BOTTOM
ACTIVE_W = ACTIVE_RIGHT  - ACTIVE_LEFT
ACTIVE_H = ACTIVE_BOTTOM - ACTIVE_TOP

# ─── Win32 API references (cached for hot-loop performance) ───────────────
_set_cursor_pos = _user32.SetCursorPos
_mouse_event    = _user32.mouse_event

# Win32 mouse_event flags for left button simulation.
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004

# ─── MediaPipe setup ────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,        # video stream mode (faster, uses tracking)
    max_num_hands=1,                # single hand for pointer control
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ─── Webcam setup ───────────────────────────────────────────────────────────

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

# ─── State variables ────────────────────────────────────────────────────────

# Previous smoothed screen coordinates (initialised to screen centre).
prev_x = SCREEN_W / 2.0
prev_y = SCREEN_H / 2.0

# Click state flag.  Ensures LEFTDOWN fires exactly once when the pinch
# begins and LEFTUP fires exactly once when the fingers separate.
is_clicking = False

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

print(f"Screen      : {SCREEN_W} × {SCREEN_H}")
print(f"Webcam      : {CAM_WIDTH} × {CAM_HEIGHT} @ {CAM_FPS} FPS")
print(f"Active area : x[{ACTIVE_LEFT}–{ACTIVE_RIGHT}]  y[{ACTIVE_TOP}–{ACTIVE_BOTTOM}]")
print(f"Adaptive EMA: α ∈ [{ALPHA_MIN}, {ALPHA_MAX}]  "
      f"speed range [{DIST_SLOW}, {DIST_FAST}] px")
print(f"Pinch click : threshold = {PINCH_THRESHOLD} px")
print("Press 'q' in the preview window to quit.\n")

try:
    while cap.isOpened():
        success, frame = cap.read()
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

            # ── Pinch-to-click detection ────────────────────────────────
            # Euclidean distance between Thumb tip (4) and Index tip (8)
            # in webcam-pixel space.
            thumb = hand.landmark[4]
            pinch_dist = math.hypot(
                (tip.x - thumb.x) * CAM_WIDTH,
                (tip.y - thumb.y) * CAM_HEIGHT,
            )

            # State machine: fire LEFTDOWN once on pinch start, LEFTUP
            # once on release.  The boolean flag prevents event spamming.
            if pinch_dist < PINCH_THRESHOLD:
                if not is_clicking:
                    # Fingers just came together → press down.
                    _mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    is_clicking = True
            else:
                if is_clicking:
                    # Fingers just separated → release.
                    _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    is_clicking = False

            # ── Visualisation overlays ──────────────────────────────────
            # Colour feedback: GREEN = idle, RED = click held.
            indicator_colour = (0, 0, 255) if is_clicking else (0, 255, 0)

            # Draw hand skeleton.
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Draw a filled circle at the index finger tip.
            cx, cy = int(raw_x), int(raw_y)
            cv2.circle(frame, (cx, cy), 10, indicator_colour, cv2.FILLED)

            # Also draw a small circle on the thumb tip for pinch clarity.
            tx, ty = int(thumb.x * CAM_WIDTH), int(thumb.y * CAM_HEIGHT)
            cv2.circle(frame, (tx, ty), 8, indicator_colour, cv2.FILLED)

            # Line between thumb and index tip — turns red on pinch.
            cv2.line(frame, (tx, ty), (cx, cy), indicator_colour, 2)

            # Show adaptive α and pinch distance near the fingertip.
            cv2.putText(
                frame, f"a={alpha:.2f}  d={pinch_dist:.0f}",
                (cx + 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )

        # Draw the active bounding box — red when clicking, magenta otherwise.
        box_colour = (0, 0, 255) if is_clicking else (255, 0, 255)
        cv2.rectangle(
            frame,
            (ACTIVE_LEFT, ACTIVE_TOP),
            (ACTIVE_RIGHT, ACTIVE_BOTTOM),
            box_colour, 2,
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
    # Ensure the left button is released if we exit mid-click.
    if is_clicking:
        _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Shutdown complete.")
