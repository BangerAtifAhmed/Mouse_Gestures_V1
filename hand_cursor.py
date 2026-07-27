"""
Hand-Tracking Cursor Controller (v29 — Decoupled Inference Resolution)
======================================================================
v29 separates what MediaPipe sees from what the preview shows, and trims
the per-frame Python maths.

Inference is fed a 1/N integer downscale of the preview (320×180 by
default) while bgr_buf stays full size for display.  Measured here, that is
worth about 5 ms: 37.3 ms → 32.2 ms.  Not more, because the palm and
landmark networks run at fixed internal input sizes — an 11× drop in pixels
bought a 1.16× drop in time.  The divisor is an integer so the rescale hits
OpenCV's fast path (123 µs, against 1536 µs for a fractional ratio) and so
the aspect ratio survives exactly, which is what lets the normalised
landmarks be used with no coordinate correction at all.

Also in v29: detect_gesture compares squared distances and reads each
landmark once (no sqrt, no per-call closure), and to_screen replaces
np.interp with the same arithmetic written out — np.interp's array dispatch
cost 12× the maths it performed on two scalars.  Both were verified against
the previous implementations for bit-identical output.


Moves the mouse cursor by tracking the middle-finger knuckle (MediaPipe
landmark 9) and reports a gesture state read from finger geometry.

MOVEMENT AND REPORTING ONLY.  No click, press, release or drag is issued
anywhere in this file.

v28 changes two things and they work together.

ANCHOR.  Tracking moved from landmark 8 (index tip) to landmark 9 (middle
MCP).  A fingertip is the most mobile point on the hand — every gesture
moves it by definition — so anchoring there means the cursor lurches
whenever the pose changes.  The MCP knuckle is carried by the palm rather
than the finger, so it holds still while the fingers do the signalling.
Landmark 8 is still read, but only as gesture input.

STATE MACHINE.  detect_gesture() classifies point / peace / grip / open,
with idle as the fallback for anything in between.  It is arithmetic only:
a finger counts as extended when its tip sits further from the wrist than
its PIP joint does, which is self-normalising and needs no palm divisor,
no threshold tuning per camera, and no classifier.  The thumb gets its own
test because it folds across the palm instead of curling radially.

TRANSITION FREEZE.  Changing pose still shifts the hand slightly, so every
state change pins the target for STATE_FREEZE_MS and then releases it.  The
frozen point is fed through the 1€ filter rather than bypassing it, leaving
the filter settled there so movement resumes by gliding rather than jumping.

Smoothing is handled *exclusively* by a One Euro Filter (Casiez et al. 2012),
which adapts its cutoff frequency to hand speed: heavy smoothing at rest,
light smoothing during fast swipes.  Movement only — no click gestures.

The active box maps onto the whole virtual desktop (all monitors combined)
and is rebuilt on the fly when a display is plugged, unplugged or rearranged.

v26 routes the pointer through pynput, with the ctypes path kept as a
fallback.  pynput gives one API across Windows and Linux and a clean click
interface for gesture work — mouse.click(Button.left, 1) — which in raw
ctypes means mouse_event/SendInput on Windows and XTest on X11.

It does not change multi-monitor reach, and it is worth being precise about
why: on Windows pynput's position setter IS user32.SetCursorPos, the same
call this script has always made.  Measured side by side on a 3840×1080
dual desktop, the two agreed exactly at every corner including (3839, 1079).
The boundary bug worth avoiding is pyautogui's — it clamps to the primary
display and cannot address negative coordinates — and pyautogui has never
been in this file.

v25 splits the vertical margin in two.  A hand raises well above shoulder
height but stops against the desk or the chest going down, so a symmetric
window wasted travel at the top and ran out before the bottom — the taskbar
was simply out of reach.  MARGIN_TOP (0.10) and MARGIN_BOTTOM (0.28) put the
live band at 10%–72% down the frame, so the screen bottom now arrives where
the arm actually gets to.  The window sits in the upper part of the frame,
which also means a hand resting at chest height already sits near the bottom
of the screen instead of the middle.

v24 adds a tkinter screen probe as a portable fallback, and widens the
horizontal margin to 0.15.

The probe reads the VIRTUAL ROOT, not winfo_screenwidth().  Measured on a
dual-monitor Windows desktop, winfo_screenwidth() reported 1536×864 — the
primary display alone, DPI-scaled — where winfo_vrootwidth() reported the
true 3840×1080.  Sizing from the former would silently amputate the second
monitor.  Native ctypes stays the preferred source because it alone reports
a negative origin and a monitor count, and is cheap enough to re-read every
poll (which is what makes hot-plug detection work); tkinter takes over only
if that fails, and is cached because building a Tk root per poll would cost
far more than it returns.

ctypes itself is standard library and stays for one job the standard
library cannot otherwise do: moving the cursor.  tkinter has no mouse API.

v23 stops forcing DirectShow.  DSHOW was pinned for its low latency, which
suits physical webcams and breaks some virtual ones: it would open the
DroidCam port, report success, then return empty frames forever — a black
preview at 0 FPS.  Captures now pass the index alone and let OpenCV pick a
backend that the driver actually registered with.  MSMF is slower to first
frame than DSHOW was, so the scan and the stream's open both retry more
patiently; MSMF also ignores CAP_PROP_BUFFERSIZE, which costs nothing here
because the capture thread already drops unclaimed frames itself.
CAMERA_API can pin a backend again if one is ever needed.

v22 puts mirroring on the 'm' key.  Front-facing webcams are normally shown
mirrored (some drivers pre-mirror the feed themselves); rear-facing phone
cameras are not, and the wrong choice sends the cursor backwards.  Because
MediaPipe reads the very buffer that is displayed, flipping the picture
flips the control direction with it — so the toggle needs to touch only
cv2.flip, and the two can never disagree.  Flipping the frame *and*
separately inverting the maths would cancel out and leave 'm' doing nothing
but restyling the preview.  INVERT_CURSOR_X survives on 'i' as the
independent control-only trim, for a driver that mirrors internally.

v21 drops the aspect lock and builds the active box straight from the
margins in NORMALISED landmark coordinates, mapping it onto the virtual
desktop with np.interp (which saturates outside its input range, so the
clamp is the interpolation's own behaviour).

The v20 box was locked to the desktop aspect ratio, which on a 3.56:1
virtual screen left a 640×360 frame with a box only 146 px tall — trimmed
by the margins to a 111 px strip, 31% of the frame height.  All vertical
steering had to happen inside that band.  The box is now 76%×76% of the
frame, about 2.5× the vertical room.

The cost is that gain is no longer isotropic: mapping a 16:9 frame onto a
3.56:1 desktop gives roughly twice the horizontal gain as vertical, so a
diagonal sweep no longer traces a straight diagonal.  The startup banner
prints the measured ratio and the MARGIN_Y that would rebalance it.

v20 insets the active region.  Reaching a corner used to mean pushing the
fingertip to the very perimeter of the tracking box — exactly where it is
half out of frame and MediaPipe is least confident.  MARGIN_X / MARGIN_Y now
reserve a band on each side, and the window inside them is what spans the
desktop, so the edges arrive early and the outer band pins the cursor to the
boundary.  The margins compose with CURSOR_SENSITIVITY rather than replacing
it: effective gain is SENSITIVITY / (1 − 2·MARGIN).

v19 makes the horizontal control direction a flag.  The frame is mirrored
with cv2.flip before MediaPipe sees it, so a plain webcam already maps the
right way — but a source that mirrors somewhere in its own pipeline cancels
that flip and reverses the cursor.  INVERT_CURSOR_X reflects the hand inside
the active box, on the control path only: the preview stays mirrored and the
fingertip marker stays on the fingertip.

v18 caps the size frames are processed at.  A phone driver ignores the
requested 640×480 and sends 1080p, which costs 6.75× the inference work and
opens a preview window larger than the desktop — Windows then clips it, so
the right and bottom of the overlay simply are not on screen.  The capture
thread now downscales oversized frames before publishing them, so nothing
downstream ever sees the large image.  It is a resize, not a crop: the whole
field of view survives.  The budget is a bounding box rather than a fixed
size, because forcing 16:9 into a literal 640×480 would squash the hand by a
third — a 1080p feed becomes 640×360.

v17 stops assuming the camera delivers 640×480.  The active box was computed
from that constant while the preview buffers were sized from the real frame,
so a 1280×720 phone feed drew the box into the top-left quadrant and scaled
the fingertip to half its true position.  The frame size is now measured from
the first frame WebcamStream receives, owned by ScreenGeometry, and rebuilt
if it ever changes.  The dead zones scale with resolution too, so the box
covers the same relative region on a 4:3 laptop sensor and a 16:9 phone.

v16 stops treating the camera scan as authoritative.  A phone-backed driver
such as DroidCam can need several hundred milliseconds to produce its first
frame, so a fast probe reports it as absent — and v15 would then conclude
"only one camera exists" and start on the built-in webcam without asking.

The scan is now slow enough to give those drivers a chance (repeated read
attempts, a settle pause after each release) and prints its verdict per
index.  More importantly it no longer decides anything: the prompt always
appears, and any index in 0..CAM_MANUAL_MAX can be typed whether or not the
scan confirmed it.  A bare Enter takes the highest detected index.

v14 trimmed resource use and dropped the Windows-only assumption:

  * model_complexity=0 pins MediaPipe to the "Lite" landmark graph.
  * The per-frame flip and BGR→RGB conversion now write into two preallocated
    buffers instead of allocating fresh arrays every frame.  At 640×480×3
    that is ~1.8 MB/frame of allocation churn removed, ~55 MB/s at 30 FPS.
  * The RGB buffer is marked non-writeable around hands.process() so
    MediaPipe borrows it instead of copying, then marked writeable again —
    which is mandatory here, since the next frame reuses that same buffer.
  * All OS-specific calls sit behind a small backend object, so the same
    file runs on Windows (user32) and Linux/X11 (libX11) with nothing but
    ctypes underneath.

v13 made CURSOR_SENSITIVITY adjustable at runtime, clamped to [1.0, 3.0].
v12 added the centre-scaled sensitivity math inside ScreenGeometry.
v11 attacked input lag: filter tuning, zero frame buffering, and skipping
redundant inference on frames already processed.
v9 removed the blocking sub-frame interpolation that made fast motion worse.

Controls:  + / =  raise sensitivity      - / _  lower sensitivity
           m      mirror on/off (picture AND control direction)
           i      invert X, control only (picture unchanged)
           q      quit

Dependencies:  pip install opencv-python mediapipe==0.8.11
Platform:      Windows (user32) and Linux/X11 (libX11)
Python:        3.8 – 3.10  (mediapipe 0.8.11 ships no cp311 wheels)
"""

# PEP 604 unions (`float | None`) are evaluated at def-time on Python < 3.10
# and would raise TypeError there.  Deferring annotation evaluation keeps the
# 3.8/3.9 half of the supported range importable.
from __future__ import annotations

import ctypes
import math
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np          # already a hard dependency of cv2 and mediapipe

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

# ── Active-region inset margins ────────────────────────────────────────────
# Reaching a screen corner should not require pushing the hand to the very
# perimeter of the tracking box, where the fingertip is half out of frame and
# MediaPipe's confidence is at its worst.  These margins reserve a band on
# each side of the box; the region *inside* them is what maps to the full
# desktop, so the edges arrive early and the outer band holds the cursor
# pinned against the boundary:
#
#       usable window = [MARGIN, 1 − MARGIN]  of the box, per axis
#       0.12          → the middle 76% covers the whole screen
#
# Per axis, deliberately: the box is already short and wide on a multi-
# monitor desktop, so vertical and horizontal strain are not the same
# problem and do not want the same number.
#
# RELATIONSHIP TO CURSOR_SENSITIVITY:  both shrink the live sub-region, and
# they COMPOSE rather than override.  The effective gain is
#
#       gain = CURSOR_SENSITIVITY / (1 − 2·MARGIN)
#
# so 0.12 margins at 1.0× already behave like 1.32×, and the 3.0× ceiling
# now reaches ~3.95× of effective travel.  Think of the margins as the
# baseline comfort setting and '+' / '-' as the runtime trim on top.  The
# amber rectangle in the preview draws the combined live region, so the
# total is always visible rather than inferred.
#
# Values are sanitised at use, so a margin of 0.5 or more cannot collapse
# the window to zero width.
# Vertical is deliberately ASYMMETRIC.  A hand can raise far above shoulder
# height but stops dead against the desk or the chest on the way down, so a
# symmetric window wastes travel at the top and runs out before the bottom —
# which is exactly why the taskbar was unreachable.  Ending the window at
# 1 − MARGIN_BOTTOM means the screen bottom arrives at 72% down the frame
# instead of 85%, inside the range the arm actually covers.
#
#       vertical window = [MARGIN_TOP, 1 − MARGIN_BOTTOM]
#       0.10 / 0.28     → the band from 10% to 72% down the frame
#
# The window is 62% of the frame height and sits in its UPPER portion, so
# resting the hand at chest height already puts the cursor near the bottom
# of the screen rather than in the middle.
MARGIN_X      = 0.15
MARGIN_TOP    = 0.10
MARGIN_BOTTOM = 0.28

# ── Mirroring — the 'm' key ────────────────────────────────────────────────
# Cameras disagree about handedness.  A front-facing webcam is usually shown
# mirrored (and some drivers pre-mirror the feed themselves); a rear-facing
# phone camera is not.  Get it wrong and the cursor runs the wrong way.
#
# IS_MIRRORED gates cv2.flip on the ONE buffer that is both displayed and
# handed to MediaPipe.  That single fact is what makes the toggle useful:
# because the landmarks come from the same image you are looking at, tip.x
# is always in *display* coordinates, so flipping the picture flips the
# control direction with it.  Press 'm' until the cursor follows your hand.
#
# WORTH KNOWING:  flipping the frame *and* separately inverting the maths
# would cancel out — two reversals leave the direction unchanged and 'm'
# would appear to do nothing but restyle the preview.  So the flip is the
# only thing 'm' touches, and the two stay coupled by construction.
#
# INVERT_CURSOR_X below is the independent trim, for the case where the
# picture looks right but the cursor still runs backwards (a driver that
# mirrors internally, say).  Between them the four combinations cover every
# camera; 'm' is the one to reach for first.
IS_MIRRORED = True

# ── Horizontal direction (independent trim) ────────────────────────────────
# Reflects the CONTROL path only, leaving the picture alone.  Toggle with
# 'i' at runtime.  The preview stays as-is and the fingertip marker stays on
# the fingertip, because the drawing code keeps using the unreflected
# coordinate.
INVERT_CURSOR_X = True

# ── Gesture state machine ──────────────────────────────────────────────────
# Pure geometry: per-finger extension tests plus a thumb test, classified
# into a handful of named states.  No gesture library, no classifier, no
# training data — every decision below is a comparison of two distances.
#
# EXTENSION TEST.  A finger counts as extended when its TIP is further from
# the wrist than its PIP joint is:
#
#       extended  ⇔  |tip − wrist|  >  |pip − wrist| × EXTEND_MARGIN
#
# Curling a finger folds the tip back toward the palm, so the radial
# distance drops below the knuckle's.  Comparing two distances from the
# same origin makes the test self-normalising: it needs no palm-size
# divisor and holds at any hand distance, any frame resolution, and any
# in-plane rotation.  The margin is slack against landmark jitter for a
# finger held near the boundary.
#
# The thumb needs its own test because it does not curl radially — it folds
# ACROSS the palm, ending up near the index knuckle without its distance
# from the wrist changing much.  So it is measured against the index MCP
# and scaled by palm length, which is what makes that threshold portable.
EXTEND_MARGIN   = 1.05   # slack on the tip-vs-pip comparison
THUMB_OUT_RATIO = 0.60   # |thumb4 − indexMCP5| / palm, above which it is out

# Both tests compare one distance against another, and a square root is
# monotonic — so the comparison can be made on SQUARED distances and every
# sqrt dropped.  Squaring the thresholds once here keeps the comparisons
# algebraically identical:
#       a > b·k        ⇔   a² > b²·k²          (a, b, k ≥ 0)
_EXTEND_MARGIN_SQ   = EXTEND_MARGIN * EXTEND_MARGIN
_THUMB_OUT_RATIO_SQ = THUMB_OUT_RATIO * THUMB_OUT_RATIO
_PALM_MIN_SQ        = 1e-12          # was |palm| <= 1e-6

# ── State-transition freeze (anti-drift) ───────────────────────────────────
# Changing gesture moves the whole hand a little: fingers closing or opening
# drag the knuckles with them, and the cursor lurches at exactly the moment
# the user meant to signal something, not move.  Holding the target still
# for a moment after every transition absorbs that twitch.
#
# Landmark 9 already removes most of it (see the anchor note below); this
# covers the residual.  Long enough to outlast the twitch, short enough that
# an intentional move right after a transition does not feel blocked.
STATE_FREEZE_MS = 150

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
# Inference is the single largest CPU consumer in this pipeline, larger than
# everything else combined.  model_complexity selects the landmark graph:
#   0 = "Lite"  — roughly half the CPU, slightly noisier landmarks
#   1 = "Full"  — more precise, noticeably heavier
# Lite is the right default here because the 1€ filter already suppresses
# landmark noise; paying for precision the filter then smooths away is waste.
MODEL_COMPLEXITY = 0

# ── Confidence thresholds ──────────────────────────────────────────────────
# These decide how readily MediaPipe drops out of cheap tracking and back
# into the expensive palm detector, which is the single biggest swing in
# per-frame cost that is still under our control.
#
#   MIN_TRACKING_CONFIDENCE is the one that matters for speed.  While the
#   tracker holds, only the landmark model runs.  The moment its confidence
#   falls below this, MediaPipe re-runs full-frame palm detection — the
#   expensive path.  Lower keeps the tracker engaged through motion blur and
#   partial occlusion; too low lets it drift on after the hand is gone.
#
#   MIN_DETECTION_CONFIDENCE only gates the initial acquisition, so it costs
#   nothing per frame.  Lowering it makes the hand easier to pick up; too low
#   invites false positives on hand-shaped background clutter.
#
# Measured here at 320x180 with no hand in frame (detection path throughout),
# the thresholds made no difference worth reporting — 31.5 to 34.4 ms across
# 0.3/0.5/0.7/0.9, which is scheduler noise.  That is expected: confidence
# changes WHICH path runs, and with nothing to track every frame takes the
# detector.  The saving shows up only with a real hand held in view, so
# treat the values below as a starting point and watch the FPS readout.
#
# 0.6 / 0.5 lowers tracking from the previous 0.7 to make the tracker
# stickier, while keeping detection high enough to avoid false grabs.
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE  = 0.5

# ── OpenCV internal threading ──────────────────────────────────────────────
# OpenCV farms operations out to a thread pool.  For 640×480 flips and colour
# conversions the pool's synchronisation overhead exceeds the work itself,
# and those idle worker threads still burn scheduler time.  Pinning to one
# thread measurably lowers CPU here.  Set to 0 to restore OpenCV's automatic
# choice if you move to much larger frames.
OPENCV_THREADS = 1

# ─── Configuration ───────────────────────────────────────────────────────────

# Webcam settings
#
# CAM_INDEX = None  → probe indices 0..CAM_SCAN_MAX at startup, then ALWAYS
#                     prompt, whatever the scan found.
# CAM_INDEX = <int> → use that index directly, skipping both the scan and the
#                     prompt.  Set this once you know which camera you want,
#                     or when running the script unattended.
CAM_INDEX = None

CAM_SCAN_MAX = 4       # highest index the scanner probes

# ── Capture backend ────────────────────────────────────────────────────────
# None → hand cv2.VideoCapture the index alone and let OpenCV negotiate.
#
# This used to force cv2.CAP_DSHOW on Windows for its lower latency, which
# is fine for physical webcams and actively broken for some virtual ones:
# DirectShow would open the DroidCam port, report success, and then hand
# back empty frames forever — a black preview at 0 FPS.  Auto-negotiation
# picks whichever backend the driver actually registered with (usually MSMF
# for virtual cameras on Windows, V4L2 on Linux).
#
# Two things get worse in exchange, both handled below rather than hidden:
#   * MSMF takes appreciably longer to produce its first frame than DSHOW,
#     so the scan needs more patience — see CAM_SCAN_READ_TRIES.
#   * MSMF generally ignores CAP_PROP_BUFFERSIZE.  The capture thread already
#     drops unclaimed frames on its own, so this costs nothing but a line in
#     the startup banner saying the request was refused.
#
# Set to cv2.CAP_DSHOW / cv2.CAP_MSMF / cv2.CAP_V4L2 to force one.
CAMERA_API = None

# Virtual-camera drivers (DroidCam, OBS, Iriun, ManyCam) route frames through
# a phone or another application, so they need noticeably longer to answer
# than a physical USB webcam.  A scan that opens and releases each index in
# a few milliseconds will often miss them entirely, and can leave the driver
# wedged for the next attempt.  Three settings slow the loop down enough for
# them to respond:
# Raised from 3 when DirectShow was dropped: MSMF is slower to hand over a
# first frame than DSHOW was, so the same retry count bought fewer real
# chances and a slow driver could go back to looking absent.
CAM_SCAN_READ_TRIES = 5     # read attempts before writing an index off
CAM_SCAN_READ_WAIT  = 0.12  # seconds between those attempts (warm-up)
CAM_SCAN_SETTLE     = 0.1   # seconds after release, before the next index

# The scan is a hint, not the truth — a driver that was still waking up can
# be typed in by hand even though it never answered.  This bounds what the
# prompt will accept, nothing more.
CAM_MANUAL_MAX = 9

# The resolution *requested* from the driver.  It is a request, not a fact:
# cameras are free to ignore it and many do, so nothing downstream may assume
# frames arrive at this size.  The authoritative numbers come from the first
# frame WebcamStream actually receives (stream.width / stream.height), and
# every piece of geometry is built from those.
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 60

# ── Processing resolution budget ───────────────────────────────────────────
# A phone-backed driver ignores the request above and hands over 1080p or
# 720p regardless.  Working at that size costs three ways:
#
#   1. MediaPipe inference scales with pixel count — a 1080p frame is 6.75×
#      the work of 640×480, which is what drags the frame rate down.
#   2. cv2.imshow opens a window the size of the frame.  A 1920×1080 preview
#      on a 1920×1080 monitor is larger than the usable desktop, so Windows
#      clips it and the right/bottom of the overlay is simply not on screen.
#   3. The flip and colour-convert buffers grow with it.
#
# So oversized frames are downscaled in the capture thread, before anything
# else touches them.  This is a RESIZE, not a crop: the whole field of view
# survives, just with fewer pixels.
#
# The budget is a bounding box, not a fixed size.  Forcing a 16:9 feed into
# a literal 640×480 would squash it horizontally by a third, distorting the
# hand MediaPipe is trying to measure and making the preview look wrong; a
# 1920×1080 source therefore becomes 640×360, not 640×480.  Set either value
# to 0 to disable downscaling entirely.
PROC_MAX_WIDTH  = 640
PROC_MAX_HEIGHT = 480

# ── Inference resolution (decoupled from the preview) ──────────────────────
# The preview stays at the processing size so it looks right; MediaPipe gets
# a smaller copy.  Measured on this machine with model_complexity=0:
#
#       input      pixels   median process()   vs 640x360
#       640x360   230,400        37.29 ms         100%
#       480x270   129,600        34.35 ms          92%
#       320x180    57,600        32.20 ms          86%
#       192x108    20,736        32.34 ms          87%
#
# Note the shape of that: pixel count falls 11x, inference time falls 1.16x.
# The palm and landmark networks run at fixed internal input sizes, so
# shrinking the frame only saves MediaPipe's own letterbox resize — real,
# but nothing like proportional.  Expect ~5 ms, not ~30 ms.
#
# WHY 320 AND NOT 256.  The extra downscale is not free, and its cost is
# wildly non-linear in the ratio:
#
#       640x360 -> 320x180   123 us   (exact 2:1, OpenCV fast path)
#       640x360 -> 256x144  1536 us   (fractional, generic path — 12x worse)
#
# 256x144 saves less inference AND costs ten times more to produce, so it
# comes out behind.  The divisor below is therefore an INTEGER, which keeps
# the fast path and — more importantly — keeps the aspect ratio exact.
#
# ASPECT MATTERS MORE THAN SPEED HERE.  MediaPipe returns landmarks
# normalised to [0, 1] of whatever it was given, so a correctly-scaled copy
# needs no coordinate correction at all (see the note in the main loop).
# That equivalence only holds while both images frame the same scene at the
# same aspect; a stretched copy would skew every landmark.
#
# Set to 0 to hand MediaPipe the full-size frame.
INFER_MAX_WIDTH = 320

# ── Display hot-plug polling ───────────────────────────────────────────────
# How often to re-read the virtual desktop rectangle.  The query is a cheap
# user-mode call (Windows) or one X round-trip (Linux), so this is nearly
# free, but there is no reason to do it every frame: a human cannot plug a
# monitor in faster than this.
POLL_INTERVAL = 2.5   # seconds between display-geometry checks

# ── The active box IS the margin rectangle ─────────────────────────────────
# Up to v20 the box was aspect-locked to the desktop: its height was derived
# as width / desktop_ratio, so both axes carried identical gain and a
# diagonal hand movement produced a diagonal cursor movement.
#
# That geometry collapses on a wide desktop.  Locked to a 3.56:1 virtual
# screen, a 640×360 camera frame yielded a box only 146 px tall — and the
# margins then trimmed it to a 111 px strip, 31% of the frame height.  Every
# vertical move had to happen inside that band, which is exactly the
# "reaching the edges needs physical extremes" complaint.
#
# The box is now simply the frame inset by the margins, computed
# from NORMALISED landmark coordinates.  On the same setup that is 76% of
# the frame in both directions — 274 px of vertical travel instead of 111,
# about 2.5× the room.
#
# THE TRADE-OFF, stated plainly: gain is no longer isotropic.  Mapping 76%
# of a 16:9 frame onto a 3.56:1 desktop gives ~7.9 screen px per camera px
# horizontally against ~3.9 vertically — a 2:1 ratio, where it used to be
# 1:1.  Vertical movement is therefore half as sensitive as horizontal.  On
# a very wide desktop that arguably matches intuition, since there is far
# more screen to cross sideways, but a diagonal sweep will not trace a
# straight diagonal.  Narrow the vertical band relative to MARGIN_X to
# rebalance: equal gain needs
#     (1 − MARGIN_TOP − MARGIN_BOTTOM) / (1 − 2·MARGIN_X)
#         = frame_ratio / desktop_ratio
# The startup banner prints the pair that would satisfy it.
#
# Because the box is built from normalised coordinates it is automatically
# correct on any sensor — 640×480, 1280×720 or a downscaled 640×360 all
# produce the same relative rectangle, with no reference resolution to
# calibrate against.

# ─── Helpers ────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Restrict *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


# ── MediaPipe hand landmark indices ────────────────────────────────────────
# Named rather than inlined: hand.landmark[14] is unreadable, and the tip /
# pip / mcp distinction is the entire basis of the extension test below.
LM_WRIST      = 0
LM_THUMB_TIP  = 4
LM_INDEX_MCP  = 5
LM_INDEX_PIP  = 6
LM_INDEX_TIP  = 8
LM_MIDDLE_MCP = 9        # the cursor anchor — see ANCHOR note in the header
LM_MIDDLE_PIP = 10
LM_MIDDLE_TIP = 12
LM_RING_PIP   = 14
LM_RING_TIP   = 16
LM_PINKY_PIP  = 18
LM_PINKY_TIP  = 20

# (name, pip, tip) for the four fingers that curl radially.  Kept for
# reference and for the overlay's per-finger labels; detect_gesture unrolls
# these rather than looping, to avoid rebuilding anything per frame.
_FINGERS = (
    ("index",  LM_INDEX_PIP,  LM_INDEX_TIP),
    ("middle", LM_MIDDLE_PIP, LM_MIDDLE_TIP),
    ("ring",   LM_RING_PIP,   LM_RING_TIP),
    ("pinky",  LM_PINKY_PIP,  LM_PINKY_TIP),
)

# Shared immutables, so the common paths allocate nothing at all.
_NO_FINGERS = (False, False, False, False)
_HINT_TEXT = "+/- speed   m mirror   i invert-x   q quit"
# Upper/lower initials for the overlay flags, indexed by the boolean.
_FINGER_CHARS = (("i", "I"), ("m", "M"), ("r", "R"), ("p", "P"))


def detect_gesture(hand_landmarks, frame_w=1.0, frame_h=1.0):
    """Classify a hand pose from landmark geometry alone.

    Returns (state, extended, thumb_out) where *state* is one of
    "point", "peace", "grip", "open" or "idle", *extended* is a 4-tuple of
    booleans for index/middle/ring/pinky, and *thumb_out* is a bool.  The
    extra detail is returned rather than recomputed so the overlay can show
    what the classifier actually saw.

    frame_w / frame_h scale the normalised landmarks back into pixels.
    They default to 1.0 so the function is callable with landmarks alone,
    but passing the real frame size matters: normalised x and y are divided
    by different dimensions, so on a 16:9 feed an unscaled "distance" is
    stretched horizontally and the finger tests skew with it.

    "idle" is the fallback for any combination that is not one of the named
    poses — a half-curled hand mid-transition lands here rather than being
    forced into the nearest match.
    """
    # Written flat on purpose.  A dist() closure would be rebuilt on every
    # call and every landmark lookup would repeat; here each point is read
    # once into a local, and the comparisons run on SQUARED distances so no
    # square root is taken anywhere in this function.
    lm = hand_landmarks.landmark
    fw = frame_w
    fh = frame_h

    p = lm[LM_WRIST]
    wx = p.x * fw
    wy = p.y * fh

    # Palm length: wrist to middle MCP.  Fixed by skeleton, unaffected by
    # finger articulation, so it is the natural scale reference.
    p = lm[LM_MIDDLE_MCP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    palm_sq = dx * dx + dy * dy
    if palm_sq <= _PALM_MIN_SQ:
        # Hand edge-on or a degenerate frame; refuse to guess.
        return "idle", _NO_FINGERS, False

    # Index.
    p = lm[LM_INDEX_PIP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    pip_sq = dx * dx + dy * dy
    p = lm[LM_INDEX_TIP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    index = dx * dx + dy * dy > pip_sq * _EXTEND_MARGIN_SQ

    # Middle.
    p = lm[LM_MIDDLE_PIP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    pip_sq = dx * dx + dy * dy
    p = lm[LM_MIDDLE_TIP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    middle = dx * dx + dy * dy > pip_sq * _EXTEND_MARGIN_SQ

    # Ring.
    p = lm[LM_RING_PIP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    pip_sq = dx * dx + dy * dy
    p = lm[LM_RING_TIP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    ring = dx * dx + dy * dy > pip_sq * _EXTEND_MARGIN_SQ

    # Pinky.
    p = lm[LM_PINKY_PIP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    pip_sq = dx * dx + dy * dy
    p = lm[LM_PINKY_TIP]
    dx = p.x * fw - wx
    dy = p.y * fh - wy
    pinky = dx * dx + dy * dy > pip_sq * _EXTEND_MARGIN_SQ

    # Thumb, measured against the index knuckle rather than the wrist.
    q = lm[LM_INDEX_MCP]
    p = lm[LM_THUMB_TIP]
    dx = (p.x - q.x) * fw
    dy = (p.y - q.y) * fh
    thumb_out = dx * dx + dy * dy > palm_sq * _THUMB_OUT_RATIO_SQ

    extended = (index, middle, ring, pinky)

    if not any(extended) and not thumb_out:
        state = "grip"                       # closed fist
    elif index and not middle and not ring and not pinky:
        state = "point"                      # index only
    elif index and middle and not ring and not pinky:
        state = "peace"                      # index + middle
    elif all(extended):
        state = "open"                       # flat palm
    else:
        state = "idle"                       # anything in between

    return state, extended, thumb_out


def pick_infer_size(width: int, height: int):
    """Choose the size MediaPipe is fed, given the display frame size.

    Returns (infer_w, infer_h, divisor, interpolation).  A divisor of 1
    means "hand over the display frame unchanged".

    An INTEGER divisor is used rather than a free target resolution, for
    two reasons that both matter more than hitting an exact pixel count:

      * cv2.resize has a fast path for exact integer ratios.  Measured,
        640x360 -> 320x180 costs 123 us while 640x360 -> 256x144 costs
        1536 us — the fractional case is twelve times dearer and wipes out
        the inference it was meant to save.
      * An integer divisor divides both axes identically, so the aspect
        ratio is preserved to the pixel.  MediaPipe normalises landmarks to
        [0, 1] of its input, so an exactly-scaled copy needs no coordinate
        correction; a stretched one would skew every landmark and there is
        no clean way to undo that afterwards.

    INTER_AREA is the right kernel for an integer downscale and costs the
    same as INTER_LINEAR at that ratio; if a frame size ever forces a
    fractional divisor, INTER_LINEAR is chosen instead to dodge the slow
    generic INTER_AREA path.
    """
    if INFER_MAX_WIDTH <= 0 or width <= INFER_MAX_WIDTH:
        return width, height, 1, cv2.INTER_AREA

    # Smallest integer divisor that brings the width within budget.
    divisor = -(-width // INFER_MAX_WIDTH)          # ceil, int arithmetic
    infer_w = max(1, width // divisor)
    infer_h = max(1, height // divisor)

    # Exact only when the divisor divides both axes without remainder.
    exact = (width % divisor == 0) and (height % divisor == 0)
    interp = cv2.INTER_AREA if exact else cv2.INTER_LINEAR
    return infer_w, infer_h, divisor, interp


def sanitise_margins() -> tuple[float, float, float]:
    """Return usable (margin_x, margin_top, margin_bottom).

    Sanitising at the point of use rather than at import means editing the
    constants takes effect with no derived value left to fall out of sync,
    and a nonsense setting cannot collapse the mapping window to zero width
    and divide by zero.

    The vertical pair is scaled together when it overruns, because the two
    are not independent: MARGIN_TOP + MARGIN_BOTTOM must leave a band to map
    from.  Scaling preserves the top/bottom RATIO, which is the part that
    encodes the ergonomics — a hand reaches higher than it reaches low — so
    an over-large pair degrades to the same shape rather than to a
    symmetric one.
    """
    mx = min(max(MARGIN_X, 0.0), 0.49)
    mt = max(MARGIN_TOP, 0.0)
    mb = max(MARGIN_BOTTOM, 0.0)

    # Keep at least 5% of the frame height as the live band.
    if mt + mb > 0.95:
        scale = 0.95 / (mt + mb)
        mt, mb = mt * scale, mb * scale

    return mx, mt, mb


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
    global CURSOR_SENSITIVITY, IS_MIRRORED, INVERT_CURSOR_X

    if key in (ord('+'), ord('=')):          # '=' is '+' without shift
        CURSOR_SENSITIVITY = round(
            min(SENS_MAX, CURSOR_SENSITIVITY + SENS_STEP), 2)
    elif key in (ord('-'), ord('_')):        # '_' is '-' with shift
        CURSOR_SENSITIVITY = round(
            max(SENS_MIN, CURSOR_SENSITIVITY - SENS_STEP), 2)
    elif key in (ord('m'), ord('M')):
        # Flips the picture and, with it, the control direction — the
        # landmarks come from the displayed buffer, so the two cannot
        # disagree.  The main loop notices the change and resets the
        # filters, because the hand's mapped position mirrors instantly.
        IS_MIRRORED = not IS_MIRRORED
        print(f"[mirror] preview {'MIRRORED' if IS_MIRRORED else 'RAW'} "
              f"— control direction follows the picture")
    elif key in (ord('i'), ord('I')):
        INVERT_CURSOR_X = not INVERT_CURSOR_X
        print(f"[invert] control-only X inversion "
              f"{'ON' if INVERT_CURSOR_X else 'OFF'} (picture unchanged)")
    elif key == ord('q'):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  CAMERA DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def open_capture(index: int, api=None):
    """Open a VideoCapture on *index*, negotiating the backend by default.

    With api=None the index is passed on its own, which is what tells
    OpenCV to try its registered backends in preference order and keep the
    first that opens.  Naming a backend explicitly is still possible, but
    it is opt-in: forcing DirectShow is precisely what left virtual cameras
    open-but-silent.

    Every capture in this file goes through here so there is one place to
    change if a future driver needs pinning again.
    """
    return cv2.VideoCapture(index) if api is None else cv2.VideoCapture(index, api)


def describe_api(api) -> str:
    """Human-readable name for a backend constant, for the banner."""
    if api is None:
        return "auto-negotiated"
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_V4L2", "CAP_GSTREAMER",
                 "CAP_AVFOUNDATION", "CAP_ANY"):
        if getattr(cv2, name, None) == api:
            return name.replace("CAP_", "")
    return f"backend {api}"


def scan_cameras(max_index: int = CAM_SCAN_MAX, api=None) -> dict:
    """Probe indices 0..max_index and return those that actually deliver.

    isOpened() alone is not a sufficient test: virtual-camera drivers and
    stale device nodes frequently report an open handle and then never
    produce a frame.  Each candidate therefore has to survive a real
    read() before it counts.

    The converse failure matters just as much, and is the reason this loop
    is deliberately unhurried.  A phone-backed driver such as DroidCam can
    take several hundred milliseconds to hand over its first frame; probed
    at full speed it looks identical to an empty slot, and hammering it can
    leave the driver wedged for the following attempt.  So each opened
    device gets CAM_SCAN_READ_TRIES chances with a short wait between them,
    and every index is followed by CAM_SCAN_SETTLE seconds of quiet after
    release.

    Progress is printed per index because this scan is fallible by nature —
    when a camera you know is connected does not appear, the line that
    names it tells you whether it failed to open at all or opened and then
    stayed silent.  Those two failures have different fixes.

    Every handle is released before moving on, so nothing is left holding a
    device when WebcamStream opens the chosen one.

    Returns
    -------
    dict  {index: (width, height)} for every index that delivered a frame.
          The resolution is read back from the device so the prompt can show
          what each camera actually produces — a 4:3 laptop sensor and a 16:9
          phone feed need very different bounding boxes.
    """
    found = {}

    for index in range(max_index + 1):
        print(f"[SCAN] Checking index {index}...")

        cap = open_capture(index, api)
        verdict = "not available (no device or backend refused)"
        try:
            if cap.isOpened():
                for attempt in range(1, CAM_SCAN_READ_TRIES + 1):
                    grabbed, frame = cap.read()
                    if grabbed and frame is not None:
                        # What the driver claims…
                        claim_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        claim_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                        # …and what it actually handed over.  These disagree
                        # more often than one would like: some backends echo
                        # back the requested size, others return 0.  The array
                        # cannot lie, so it wins.
                        real_h, real_w = frame.shape[:2]
                        found[index] = (real_w, real_h)

                        verdict = (f"OK — {real_w}×{real_h} "
                                   f"on attempt {attempt}/{CAM_SCAN_READ_TRIES}")
                        if (claim_w, claim_h) != (real_w, real_h):
                            verdict += (f"  [driver reported "
                                        f"{claim_w}×{claim_h}]")
                        break
                    time.sleep(CAM_SCAN_READ_WAIT)
                else:
                    verdict = (f"opened but produced no frame in "
                               f"{CAM_SCAN_READ_TRIES} tries (slow driver?)")
        finally:
            cap.release()    # never leave a device held

        print(f"[SCAN]   index {index}: {verdict}")

        # Give the driver a moment to fully let go before the next open.
        time.sleep(CAM_SCAN_SETTLE)

    return found


def pick_best_camera(available: list) -> int:
    """Default choice: the highest index the scan found.

    Index 0 is almost always the laptop's built-in webcam.  External USB
    and virtual cameras are enumerated after it, so the largest index is
    the one most likely to be the camera that was deliberately added.
    """
    return max(available)


def choose_camera(available: list) -> int:
    """Prompt for a camera index.  Always asks — the scan only suggests.

    There is no auto-select shortcut, not even when exactly one source
    answers.  A virtual driver that was still waking up is indistinguishable
    from an absent one, so "only one camera exists" is a conclusion the scan
    is not entitled to draw.  Typing an index the scan missed is therefore
    allowed: anything in 0..CAM_MANUAL_MAX is accepted, with a warning when
    it is not one the scan confirmed.

    A bare Enter takes the highest detected index.  Missing stdin (piped
    input, no console) falls back to the same default rather than hanging.
    """
    if available:
        best = pick_best_camera(available)
        print(f"\n[camera] scan answered on {len(available)} index/indices:")
        for idx in sorted(available):
            w, h = available[idx]
            print(f"           [{idx}]  {w}×{h}  ({w / h:.2f}:1)")
        print(f"         index 0 is normally the built-in webcam; "
              f"higher indices are USB or virtual cameras")
    else:
        best = None
        print("\n[camera] no index answered the scan.")
        print("         a virtual driver (DroidCam, OBS, Iriun…) may still be "
              "running but too slow to reply —")
        print("         if you know its index, type it anyway.")

    print(f"         any index 0–{CAM_MANUAL_MAX} is accepted, detected or not.")

    while True:
        prompt = (f"         camera index [Enter = {best}]: " if best is not None
                  else f"         camera index (0–{CAM_MANUAL_MAX}, no default): ")
        try:
            raw = input(prompt).strip()
        except EOFError:
            # No interactive console (piped stdin, service, IDE runner).
            if best is None:
                raise RuntimeError(
                    "no camera answered the scan and stdin is unavailable, so "
                    "no index could be chosen — set CAM_INDEX explicitly."
                )
            print(f"\n[camera] stdin unavailable — defaulting to {best}\n")
            return best

        if not raw:
            if best is None:
                print("         nothing was detected, so there is no default "
                      "— please type an index.")
                continue
            print(f"[camera] using index {best} (default)\n")
            return best

        if raw.isdigit() and 0 <= int(raw) <= CAM_MANUAL_MAX:
            index = int(raw)
            if index not in available:
                print(f"         note: index {index} did not answer the scan "
                      f"— trying it anyway.")
            print(f"[camera] using index {index}\n")
            return index

        print(f"         '{raw}' is not a number in 0–{CAM_MANUAL_MAX} "
              f"— try again.")


# ═══════════════════════════════════════════════════════════════════════════
#  CURSOR OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
#
# Which library actually moves the pointer:
#   "auto"    pynput if importable, else the ctypes path   (default)
#   "pynput"  require pynput; fail loudly if it is missing
#   "native"  ctypes only, no third-party import at all
#
# pynput gives one API across Windows and Linux and, more usefully, a clean
# click interface for gesture work later — mouse.click(Button.left, 1) —
# which in raw ctypes means SendInput on Windows and XTest on X11.
#
# What it does NOT change is multi-monitor reach.  On Windows pynput's
# position setter *is* user32.SetCursorPos, the same call the native path
# has always used; measured side by side on a 3840×1080 dual desktop the two
# returned identical coordinates at every corner, including (3839, 1079).
# The boundary problem worth avoiding belongs to pyautogui, which clamps to
# the primary display and cannot address negative coordinates at all — this
# script has never used it.
#
# "auto" keeps a ctypes fallback because pynput is a third-party package and
# needs python-xlib on Linux; a missing optional dependency should not stop
# the cursor from working.
CURSOR_BACKEND = "auto"


class PynputCursor:
    """Pointer via pynput.  Understands the virtual desktop on both OSes."""

    name = "pynput"

    def __init__(self):
        from pynput.mouse import Button, Controller
        self._button = Button
        self._mouse = Controller()
        # Touch the property once so an unusable backend (no DISPLAY, no
        # python-xlib) raises here, while there is still a fallback, rather
        # than on the first frame.
        _ = self._mouse.position

    def move(self, x: int, y: int) -> None:
        self._mouse.position = (int(x), int(y))

    def click(self, button: str = "left", count: int = 1) -> None:
        """Send a click.  No gesture calls this yet; it is here for the
        pinch/click work rather than left to be bolted on later."""
        self._mouse.click(getattr(self._button, button), count)

    def close(self) -> None:
        pass


class NativeCursor:
    """Pointer via the platform backend's own ctypes call."""

    name = "ctypes"

    def __init__(self, backend):
        self._backend = backend

    def move(self, x: int, y: int) -> None:
        self._backend.move_cursor(int(x), int(y))

    def click(self, button: str = "left", count: int = 1) -> None:
        self._backend.click(button, count)

    def close(self) -> None:
        pass


def make_cursor(backend):
    """Pick the cursor output, honouring CURSOR_BACKEND."""
    if CURSOR_BACKEND in ("auto", "pynput"):
        try:
            return PynputCursor()
        except Exception as exc:
            if CURSOR_BACKEND == "pynput":
                raise RuntimeError(
                    f"CURSOR_BACKEND='pynput' but pynput is unusable: {exc}. "
                    "Install it with 'pip install pynput' (Linux also needs "
                    "python-xlib), or set CURSOR_BACKEND='native'."
                ) from exc
            print(f"[cursor] pynput unavailable ({exc.__class__.__name__}) — "
                  f"falling back to ctypes")
    return NativeCursor(backend)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN DETECTION — portable probe
# ═══════════════════════════════════════════════════════════════════════════
#
# Which source supplies the desktop rectangle:
#   "auto"     native first, tkinter if that fails  (default)
#   "tkinter"  force the portable path
#   "native"   force ctypes, fail loudly if unavailable
SCREEN_SOURCE = "auto"


def probe_screen_tkinter():
    """Desktop rectangle via tkinter.  Returns (left, top, w, h) or None.

    Uses the VIRTUAL ROOT metrics, not winfo_screenwidth().  That
    distinction is the whole reason this function is worth having: on a
    dual-monitor Windows desktop measured here, winfo_screenwidth() returned
    1536×864 — the primary monitor alone, and DPI-scaled at that — while
    winfo_vrootwidth() returned the true 3840×1080.  Sizing the mapping from
    winfo_screenwidth() would silently amputate the second display.

    tkinter is standard library, so this adds no dependency, but it is not
    free: it builds and tears down a Tk root, needs python3-tk present on
    Linux (not installed by default on minimal images), and reports the
    virtual origin as (0, 0) on Windows even when a monitor sits left of the
    primary.  Hence "fallback" rather than "primary".
    """
    try:
        import tkinter
    except Exception:
        return None

    root = None
    try:
        root = tkinter.Tk()
        root.withdraw()                     # never show the helper window
        width = int(root.winfo_vrootwidth())
        height = int(root.winfo_vrootheight())
        left = int(root.winfo_vrootx())
        top = int(root.winfo_vrooty())

        # Some window managers leave the vroot unset; fall back to the
        # single-screen numbers rather than returning a zero rectangle.
        if width <= 0 or height <= 0:
            width = int(root.winfo_screenwidth())
            height = int(root.winfo_screenheight())
            left = top = 0

        return (left, top, width, height) if width > 0 and height > 0 else None
    except Exception:
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  PLATFORM BACKENDS
# ═══════════════════════════════════════════════════════════════════════════
#
# NOTE ON ctypes:  ctypes is part of the standard library — there is no
# win32api, pywin32 or any other wrapper anywhere in this file, and never
# was.  It stays for one reason: MOVING THE CURSOR.  Python's standard
# library has no mouse API at all, tkinter included; the only alternatives
# are third-party packages, which the "no frameworks" rule rules out.  So
# the cursor goes through user32.SetCursorPos / XWarpPointer, and screen
# *measurement* is what gained the portable tkinter path above.
#
# Everything OS-specific is confined to these two classes so the rest of the
# file is portable.  Both talk to the native library through plain ctypes —
# no wrapper packages, matching how the Win32 path has always worked.
#
# Each backend exposes:
#     read_geometry()  → (left, top, width, height) of the whole desktop
#     monitor_count()  → number of attached displays
#     move_cursor(x,y) → absolute cursor placement
#     camera_api       → the cv2.CAP_* backend to open the webcam with
#     close()          → release any native handles

class Win32Backend:
    """Desktop control through user32.dll."""

    # SM_CXSCREEN (0) / SM_CYSCREEN (1) describe the *primary* monitor only,
    # so a cursor driven from them could never leave display 1.  The
    # SM_*VIRTUALSCREEN family describes the rectangle enclosing ALL monitors.
    SM_XVIRTUALSCREEN  = 76   # left edge of the virtual desktop
    SM_YVIRTUALSCREEN  = 77   # top  edge of the virtual desktop
    SM_CXVIRTUALSCREEN = 78   # total width
    SM_CYVIRTUALSCREEN = 79   # total height
    SM_CMONITORS       = 80   # number of display monitors

    name = "Windows / user32"
    camera_api = CAMERA_API         # None → OpenCV negotiates (see CAMERA_API)

    def __init__(self):
        self._u32 = ctypes.windll.user32
        # Cache the bound methods: these run in the hot loop.
        self._metric = self._u32.GetSystemMetrics
        self._set_pos = self._u32.SetCursorPos
        # ctypes leaves restype at the default c_int (signed), which is what
        # we want — a monitor left of the primary reports a NEGATIVE origin
        # (e.g. −1920), and c_uint would silently wrap it.

    def read_geometry(self) -> tuple[int, int, int, int]:
        m = self._metric
        return (m(self.SM_XVIRTUALSCREEN), m(self.SM_YVIRTUALSCREEN),
                m(self.SM_CXVIRTUALSCREEN), m(self.SM_CYVIRTUALSCREEN))

    def monitor_count(self) -> int:
        return self._metric(self.SM_CMONITORS)

    def move_cursor(self, x: int, y: int) -> None:
        # SetCursorPos accepts negative virtual-screen coordinates directly.
        # This is the identical call pynput makes on Windows.
        self._set_pos(x, y)

    # mouse_event flags; SendInput is the modern call but mouse_event is
    # still honoured and needs no struct definitions.
    _BTN = {"left":   (0x0002, 0x0004),      # LEFTDOWN,   LEFTUP
            "right":  (0x0008, 0x0010),      # RIGHTDOWN,  RIGHTUP
            "middle": (0x0020, 0x0040)}      # MIDDLEDOWN, MIDDLEUP

    def click(self, button: str = "left", count: int = 1) -> None:
        down, up = self._BTN[button]
        for _ in range(count):
            self._u32.mouse_event(down, 0, 0, 0, 0)
            self._u32.mouse_event(up, 0, 0, 0, 0)

    def close(self) -> None:
        pass


class X11Backend:
    """Desktop control through libX11, via ctypes only.

    Notes
    -----
    * The X root window spans the entire virtual screen and its origin is
      always (0, 0), unlike Windows where a display placed left of the
      primary yields a negative origin.  read_geometry() therefore returns
      zeros for left/top, and the rest of the pipeline needs no special case.
    * XGetGeometry is a server round-trip rather than a cached value, so the
      hot-plug poll sees RandR resolution changes without reconnecting.
    * Under a native Wayland session XWarpPointer is ignored by the
      compositor.  An XWayland-backed session works; a pure Wayland one does
      not, and no amount of ctypes will change that.
    """

    name = "Linux / X11"
    camera_api = CAMERA_API         # None → OpenCV negotiates (usually V4L2)

    def __init__(self):
        try:
            self._xlib = ctypes.CDLL("libX11.so.6")
        except OSError as exc:      # pragma: no cover - platform specific
            raise RuntimeError(
                "libX11.so.6 not found — install libx11 (e.g. "
                "'sudo apt install libx11-6') or run under X11/XWayland."
            ) from exc

        x = self._xlib
        # Declaring restype/argtypes is not optional on 64-bit: a Display*
        # returned as the default c_int would be truncated and segfault.
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XDefaultRootWindow.restype = ctypes.c_ulong
        x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x.XGetGeometry.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
        ]
        x.XWarpPointer.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_int, ctypes.c_int,
        ]
        x.XFlush.argtypes = [ctypes.c_void_p]
        x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x.XFree.argtypes = [ctypes.c_void_p]

        self._dpy = x.XOpenDisplay(None)
        if not self._dpy:
            raise RuntimeError(
                "cannot open an X display — is DISPLAY set?"
            )
        self._root = x.XDefaultRootWindow(self._dpy)

        # XTest is optional and only needed for clicking; loaded here so a
        # missing extension is discovered at startup rather than mid-gesture.
        self._xtest = None
        try:                        # pragma: no cover - platform specific
            xt = ctypes.CDLL("libXtst.so.6")
            xt.XTestFakeButtonEvent.argtypes = [
                ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
            self._xtest = xt
        except OSError:
            pass

        # Xinerama is optional; without it we simply report one monitor.
        self._xinerama = None
        try:                        # pragma: no cover - platform specific
            xin = ctypes.CDLL("libXinerama.so.1")
            xin.XineramaQueryScreens.restype = ctypes.c_void_p
            xin.XineramaQueryScreens.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            self._xinerama = xin
        except OSError:
            pass

        # Scratch out-params, allocated once instead of per poll.
        self._g_root = ctypes.c_ulong()
        self._g_x, self._g_y = ctypes.c_int(), ctypes.c_int()
        self._g_w, self._g_h = ctypes.c_uint(), ctypes.c_uint()
        self._g_bw, self._g_depth = ctypes.c_uint(), ctypes.c_uint()

    def read_geometry(self) -> tuple[int, int, int, int]:
        self._xlib.XGetGeometry(
            self._dpy, self._root,
            ctypes.byref(self._g_root),
            ctypes.byref(self._g_x), ctypes.byref(self._g_y),
            ctypes.byref(self._g_w), ctypes.byref(self._g_h),
            ctypes.byref(self._g_bw), ctypes.byref(self._g_depth),
        )
        # The X root window always starts at (0, 0).
        return (0, 0, int(self._g_w.value), int(self._g_h.value))

    def monitor_count(self) -> int:
        if self._xinerama is None:
            return 1
        n = ctypes.c_int(0)
        ptr = self._xinerama.XineramaQueryScreens(self._dpy, ctypes.byref(n))
        if not ptr:
            return 1
        self._xlib.XFree(ctypes.c_void_p(ptr))
        return max(1, int(n.value))

    def move_cursor(self, x: int, y: int) -> None:
        # src_window 0 (None) means "move regardless of current position".
        self._xlib.XWarpPointer(self._dpy, 0, self._root, 0, 0, 0, 0, x, y)
        # Without a flush the request sits in the output buffer.
        self._xlib.XFlush(self._dpy)

    def click(self, button: str = "left", count: int = 1) -> None:
        """Click through XTest, if the extension is present.

        XWarpPointer can move the pointer but cannot press anything, so this
        needs libXtst.  Rather than half-implement it, an absent extension
        says so and points at pynput, which carries its own X backend.
        """
        if self._xtest is None:
            raise RuntimeError(
                "clicking on X11 needs libXtst (install libxtst6) or "
                "pynput — set CURSOR_BACKEND='pynput'."
            )
        code = {"left": 1, "middle": 2, "right": 3}[button]
        for _ in range(count):
            self._xtest.XTestFakeButtonEvent(self._dpy, code, True, 0)
            self._xtest.XTestFakeButtonEvent(self._dpy, code, False, 0)
        self._xlib.XFlush(self._dpy)

    def close(self) -> None:
        if getattr(self, "_dpy", None):
            self._xlib.XCloseDisplay(self._dpy)
            self._dpy = None


def make_backend():
    """Pick the desktop backend for the running platform."""
    if sys.platform.startswith("win"):
        return Win32Backend()
    if sys.platform.startswith(("linux", "freebsd")):
        return X11Backend()
    raise RuntimeError(
        f"unsupported platform {sys.platform!r} — this script drives the "
        "cursor through user32 (Windows) or libX11 (Linux)."
    )


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
    left, top       Origin of the virtual desktop (negative on Windows when
                    a display sits left of / above the primary; always 0 on
                    X11, where the root window starts at the origin).
    width, height   Size of the virtual desktop, all monitors combined.
    ratio           width / height — drives the active box aspect lock.
    box_*           The active rectangle inside the webcam frame.
    """

    def __init__(self, backend, cam_width: int, cam_height: int,
                 poll_interval: float = POLL_INTERVAL):
        self._backend = backend
        self._poll_interval = poll_interval
        self._next_poll = 0.0
        self.monitors = 0
        # The camera's real frame size.  Held here rather than read from a
        # module constant so that swapping a 4:3 laptop sensor for a 16:9
        # phone feed rebuilds the box instead of silently drawing one sized
        # for the wrong image.
        self.cam_w = int(cam_width)
        self.cam_h = int(cam_height)

        # Where the desktop rectangle comes from, decided once.  Native is
        # preferred: it is the only source that reports a negative origin
        # (a monitor left of the primary) and a monitor count, and it is
        # cheap enough to re-read on every poll, which is what makes
        # hot-plug detection possible.
        self._native_ok = SCREEN_SOURCE != "tkinter"
        self._tk_geometry = None
        self.source = "native"

        metrics = self._read_geometry()
        if metrics is None:
            raise RuntimeError(
                "could not determine the desktop size from either the native "
                "API or tkinter — on Linux install python3-tk, or set "
                "SCREEN_SOURCE and check your display connection."
            )
        self._apply(metrics)

    # ── Geometry source ────────────────────────────────────────────────
    def _read_geometry(self):
        """Desktop rectangle, native first and tkinter as the fallback.

        The tkinter probe is cached after its first success.  Building a Tk
        root costs tens of milliseconds and can flash a window; doing that
        every POLL_INTERVAL would be wasteful, so the portable path trades
        away hot-plug detection rather than pay it repeatedly.
        """
        if self._native_ok:
            try:
                m = self._backend.read_geometry()
                if m and m[2] > 0 and m[3] > 0:
                    self.source = "native"
                    return m
            except Exception:
                pass
            if SCREEN_SOURCE == "native":
                return None
            # One failure is enough; stop paying for it every poll.
            self._native_ok = False
            print("[screen] native geometry unavailable — falling back to "
                  "tkinter (hot-plug detection disabled)")

        if self._tk_geometry is None:
            self._tk_geometry = probe_screen_tkinter()
        if self._tk_geometry is not None:
            self.source = "tkinter"
        return self._tk_geometry

    # ── Camera size ────────────────────────────────────────────────────
    def set_camera_size(self, width: int, height: int) -> bool:
        """Adopt a new camera frame size and rebuild the box.

        Returns True only when the size actually changed.  Called whenever
        the incoming frames stop matching what the box was built for.
        """
        if (int(width), int(height)) == (self.cam_w, self.cam_h):
            return False
        self.cam_w, self.cam_h = int(width), int(height)
        # Rebuild against the desktop metrics already held.
        self._apply((self.left, self.top, self.width, self.height))
        return True

    # ── Derived geometry ───────────────────────────────────────────────
    def _apply(self, metrics: tuple[int, int, int, int]) -> None:
        """Adopt *metrics* and rebuild the active box from them.

        Two independent shapes feed this: the desktop rectangle (which sets
        the aspect ratio the box must lock to) and the camera frame (which
        sets the canvas the box lives on).  Either can change at runtime —
        a monitor hot-plug or a camera swap — so both are read from state
        rather than from constants.
        """
        self.left, self.top, self.width, self.height = metrics
        # The monitor count is a native-only nicety; tkinter cannot report
        # it, so the banner just says 1 rather than guessing.
        try:
            self.monitors = self._backend.monitor_count() if self._native_ok else 1
        except Exception:
            self.monitors = 1
        self.ratio = self.width / self.height

        # The active box is the frame inset by the margins, in normalised
        # coordinates.  No reference resolution and no aspect lock: the same
        # fractions land on the same relative rectangle whatever the sensor
        # delivers, so a 640×480 laptop cam and a downscaled 640×360 phone
        # feed both give a 76%×76% window at the default 0.12 margins.
        mx, mt, mb = sanitise_margins()

        self.box_left   = int(round(mx * self.cam_w))
        self.box_right  = int(round((1.0 - mx) * self.cam_w))
        self.box_top    = int(round(mt * self.cam_h))
        self.box_bottom = int(round((1.0 - mb) * self.cam_h))

        # A one-pixel frame, or margins rounding both edges together, must
        # still leave something to divide by in to_screen().
        if self.box_right <= self.box_left:
            self.box_left, self.box_right = 0, max(1, self.cam_w)
        if self.box_bottom <= self.box_top:
            self.box_top, self.box_bottom = 0, max(1, self.cam_h)

        self.box_w = self.box_right - self.box_left
        self.box_h = self.box_bottom - self.box_top

        # ── Per-frame constants, computed once per geometry change ──────
        # Endpoints and slopes for the box→desktop map, plus the desktop
        # centre as plain floats.  All of this used to be recomputed inside
        # to_screen() every frame; none of it changes between rebuilds.
        self._map_x0 = float(self.left)
        self._map_x1 = float(self.left + self.width)
        self._map_y0 = float(self.top)
        self._map_y1 = float(self.top + self.height)
        self._slope_x = (self._map_x1 - self._map_x0) / self.box_w
        self._slope_y = (self._map_y1 - self._map_y0) / self.box_h
        self._centre_x = self.left + self.width / 2.0
        self._centre_y = self.top + self.height / 2.0

    # ── Polling ────────────────────────────────────────────────────────
    def poll(self, now: float) -> bool:
        """Re-read the desktop rectangle at most every *poll_interval* sec.

        Returns True only when the geometry actually changed and was
        rebuilt, so the caller can react (reset filters, log, …).
        """
        if now < self._next_poll:
            return False
        self._next_poll = now + self._poll_interval

        # Only the native source is re-read; the tkinter fallback is cached
        # and would cost a Tk root per poll for no benefit.
        if not self._native_ok:
            return False

        metrics = self._read_geometry()
        if metrics is None:
            return False

        # Mid-hotplug the OS can transiently report a degenerate rectangle
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

        Stages:

          1. Reflect horizontally when INVERT_CURSOR_X is set, so physical
             left/right matches on-screen left/right.
          2. Interpolate the inset window — [MARGIN, 1 − MARGIN] of the
             frame, held here as box_left..box_right — onto the virtual
             desktop span [left, left + width].  np.interp clamps outside
             its input range by design, so a hand past the box holds the
             cursor flat against the screen edge instead of running off it.
          3. Scale about the desktop centre by CURSOR_SENSITIVITY, then
             clamp again, since the multiplier can push a mid-box position
             beyond an edge.

        CURSOR_SENSITIVITY is read fresh on every call, so runtime '+' / '-'
        adjustments apply from the next frame onward.  It composes with the
        margins: effective gain is SENSITIVITY / (1 − 2·MARGIN).

        The destination is [left, left + width], not [0, width]: on a
        multi-monitor Windows desktop the origin can be negative, and
        interpolating onto a zero-based span would make every display left
        of the primary unreachable.  Written this way the rightmost pixel of
        the furthest secondary monitor is the endpoint of the range.
        """
        cx, cy = cam_x, cam_y

        # Horizontal direction correction, applied about the FRAME centre.
        # The margin box is symmetric about that centre, so this maps
        # box_left onto box_right exactly.  Reflecting here rather than on
        # the raw landmark keeps the fingertip marker on the fingertip: the
        # overlay is drawn from the unreflected coordinate.
        if INVERT_CURSOR_X:
            cx = self.cam_w - cx

        # ── Inset-window interpolation ──────────────────────────────────
        # Algebraically this is np.interp(v, (lo, hi), (a, b)) — the same
        # piecewise-linear map, saturating outside the input range.  It is
        # written out because np.interp is a general array routine: on two
        # scalars per frame its dispatch and array-boxing overhead measured
        # ~12.9 us, roughly 12x the arithmetic itself.  The slope form below
        # matches numpy's own (slope·(x − xp0) + fp0), and the two clamps
        # reproduce its saturation exactly.
        #
        # Slopes are precomputed in _apply(), so a frame does two multiplies
        # and two adds per axis.
        if cx <= self.box_left:
            mapped_x = self._map_x0
        elif cx >= self.box_right:
            mapped_x = self._map_x1
        else:
            mapped_x = self._slope_x * (cx - self.box_left) + self._map_x0

        if cy <= self.box_top:
            mapped_y = self._map_y0
        elif cy >= self.box_bottom:
            mapped_y = self._map_y1
        else:
            mapped_y = self._slope_y * (cy - self.box_top) + self._map_y0

        # Centre-scaled sensitivity: push the offset from the middle out by
        # the multiplier, so the desktop edge is reached from a smaller
        # physical displacement.  Reading the cached floats avoids building
        # a tuple through the .center property on every frame.
        mid_x = self._centre_x
        mid_y = self._centre_y
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
        """Centre of the virtual desktop (origin-aware).

        Convenience accessor for setup code.  to_screen() reads the cached
        floats directly instead, so the hot path never builds this tuple.
        """
        return (self._centre_x, self._centre_y)

    @property
    def effective_box(self) -> tuple[int, int, int, int] | None:
        """Sub-rectangle of the active box that still reaches a desktop edge.

        The margins now define the box itself — the magenta rectangle IS
        the inset window, and its edges are where the screen edges are
        reached.  So the only thing left to shrink the live area is
        CURSOR_SENSITIVITY, which narrows it to 1/S about the centre.

        Everything between this rectangle and the box maps past a desktop
        edge and clamps flat, so hand movement there does nothing.  Returns
        (l, t, r, b), or None at 1.0× where the whole box is live and the
        two rectangles would sit on top of each other.
        """
        if CURSOR_SENSITIVITY <= 1.0:
            return None

        live = 1.0 / CURSOR_SENSITIVITY
        mid_x = (self.box_left + self.box_right) / 2.0
        mid_y = (self.box_top  + self.box_bottom) / 2.0
        half_w = (self.box_w * live) / 2.0
        half_h = (self.box_h * live) / 2.0
        return (int(mid_x - half_w), int(mid_y - half_h),
                int(mid_x + half_w), int(mid_y + half_h))

    def describe(self) -> str:
        """One-line summary for the console banner / change notices."""
        return (f"{self.width}×{self.height} px  origin ({self.left}, {self.top})  "
                f"ratio {self.ratio:.3f}  monitors {self.monitors}  "
                f"box {self.box_w}×{self.box_h}")


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
#           move_cursor()            cap.read()  [loops]
#           cv2.imshow()             cap.read()  [loops]
#           ...                      ...
#
# Staleness:  the grab loop overwrites _latest unconditionally, so any frame
#           the main thread did not collect is simply dropped.  There is no
#           queue and therefore no backlog to drain — read() hands back the
#           newest physical frame every time.  CAP_PROP_BUFFERSIZE = 1 asks
#           the *driver* to do the same one level down; not every backend
#           honours it, so the value it returns is reported at startup.
#
# Publication:  frame, grabbed flag and sequence number are published as ONE
#           tuple rebind.  Assigning a single attribute is atomic under the
#           GIL, so the reader can never observe a new frame paired with an
#           old sequence number.  Three separate attributes would need a
#           lock; this needs none, and adds no contention to the hot path.

class WebcamStream:
    """Non-blocking webcam reader that always yields the newest frame."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: int = 60, api=None,
                 proc_max: tuple[int, int] = (PROC_MAX_WIDTH, PROC_MAX_HEIGHT)):
        self._cap = open_capture(index, api)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)

        # Ask the backend to keep at most one frame queued.  Not all
        # backends implement this; keep the result so startup can say so.
        self.buffersize_accepted = bool(self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))

        # Read one frame synchronously so _latest is never None.
        #
        # Retried rather than one-shot: an auto-negotiated backend (MSMF on
        # Windows) can take appreciably longer than DirectShow did to hand
        # over its first frame, and a phone-backed driver longer still.  A
        # single failed read here would reject a camera the scan had just
        # confirmed working.  The failsafe itself is unchanged — a device
        # that never produces pixels is still refused, with the reason.
        grabbed, frame = False, None
        for _ in range(CAM_SCAN_READ_TRIES):
            grabbed, frame = self._cap.read()
            if grabbed and frame is not None:
                break
            time.sleep(CAM_SCAN_READ_WAIT)

        if not grabbed or frame is None:
            self._cap.release()
            raise RuntimeError(
                f"camera {index} opened but produced no frame in "
                f"{CAM_SCAN_READ_TRIES} attempts — it may be in use by "
                f"another application, or the driver may need a different "
                f"backend (set CAMERA_API)."
            )
        # ── Native frame size ───────────────────────────────────────────
        # This has to be measured here, not carried over from the scan: the
        # cap.set() calls above may have moved the camera to a different mode
        # than the one the scanner saw.
        #
        # frame.shape is the ground truth.  CAP_PROP_FRAME_* is queried too,
        # because a mismatch is worth surfacing: it usually means the driver
        # silently refused the requested mode.
        self.native_height, self.native_width = frame.shape[:2]
        self.reported_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.reported_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.size_mismatch = ((self.reported_width, self.reported_height)
                              != (self.native_width, self.native_height))

        # ── Processing frame size ───────────────────────────────────────
        # width/height describe what read() actually hands back, which is
        # what every consumer must build geometry from.  When the camera
        # overshoots the budget they are the downscaled size, not the
        # native one.
        self.width, self.height = self._fit_within(
            self.native_width, self.native_height, proc_max)
        self.downscaled = ((self.width, self.height)
                           != (self.native_width, self.native_height))
        self.scale = self.width / self.native_width if self.downscaled else 1.0

        self._latest = (grabbed, self._shrink(frame), 0)

        # The stop flag signals the background thread to exit.
        self._stopped = False

        # Launch the capture thread as a daemon so it dies if the main
        # process is killed unexpectedly.
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    @staticmethod
    def _fit_within(w: int, h: int, budget: tuple[int, int]) -> tuple[int, int]:
        """Largest size inside *budget* that keeps the source aspect ratio.

        Returns (w, h) unchanged when the frame already fits, or when the
        budget is disabled with a zero.  A single scale factor is applied to
        both axes, which is what stops a 16:9 feed being squashed into 4:3.
        """
        max_w, max_h = budget
        if max_w <= 0 or max_h <= 0:
            return w, h
        if w <= max_w and h <= max_h:
            return w, h
        scale = min(max_w / w, max_h / h)
        return max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    def _shrink(self, frame):
        """Downscale *frame* to the processing size, or pass it through.

        INTER_AREA is the right kernel for shrinking: it averages every
        source pixel that falls inside a destination pixel, so detail is
        attenuated instead of aliasing into moiré the way INTER_LINEAR
        would.  It is also the fastest of the quality options for downscale.

        The result is a NEW array on purpose.  The publication below is
        lock-free precisely because each frame handed to the main thread is
        a distinct object; resizing into one reused destination would let
        the capture thread overwrite pixels the main thread is still
        reading, halfway through an inference.
        """
        if not self.downscaled:
            return frame
        return cv2.resize(frame, (self.width, self.height),
                          interpolation=cv2.INTER_AREA)

    def _update(self):
        """Grab frames continuously, discarding any the reader missed.

        The resize happens here rather than in the main loop so the large
        frame never leaves this thread: MediaPipe, the flip/convert buffers,
        the overlay and the preview window all see only the reduced size.
        This thread is otherwise blocked in the driver waiting on hardware,
        so the scaling is close to free in wall-clock terms.
        """
        seq = 0
        while not self._stopped:
            grabbed, frame = self._cap.read()
            if not grabbed:
                continue
            seq += 1
            # Single atomic rebind — no torn reads, no lock needed.
            self._latest = (grabbed, self._shrink(frame), seq)

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


# ═══════════════════════════════════════════════════════════════════════════
#  RUNTIME SETUP
# ═══════════════════════════════════════════════════════════════════════════

# Pin OpenCV's internal thread pool before anything uses it.
if OPENCV_THREADS:
    cv2.setNumThreads(OPENCV_THREADS)

backend = make_backend()
cursor = make_cursor(backend)

# ─── Camera selection and open ──────────────────────────────────────────────
# Both happen before MediaPipe loads: the model takes a moment to initialise
# and would delay the prompt, and its loader writes to stderr, which would
# scroll the question the user is meant to answer.
#
# Opening is retried rather than fatal.  Since an index the scan never
# confirmed can be typed on purpose, "that one does not work" is an ordinary
# outcome here, and the right response is another prompt — not a stack trace
# and a re-run of the whole scan.
if CAM_INDEX is None:
    print(f"[camera] scanning indices 0–{CAM_SCAN_MAX}, backend "
          f"{describe_api(backend.camera_api)} "
          f"({CAM_SCAN_READ_TRIES} read attempts each)…\n")
    _available = scan_cameras(CAM_SCAN_MAX, backend.camera_api)

    while True:
        camera_index = choose_camera(_available)
        try:
            stream = WebcamStream(camera_index, CAM_WIDTH, CAM_HEIGHT,
                                  CAM_FPS, api=backend.camera_api)
            break
        except RuntimeError as exc:
            print(f"[camera] index {camera_index} could not be opened: {exc}")
            print("         pick a different one.")
else:
    camera_index = CAM_INDEX
    print(f"[camera] CAM_INDEX pinned to {camera_index}, skipping scan\n")
    stream = WebcamStream(camera_index, CAM_WIDTH, CAM_HEIGHT, CAM_FPS,
                          api=backend.camera_api)

# Rebind the camera constants to the size frames will REALLY have after any
# downscale.  Nothing left in the file reads them for geometry — the box and
# the landmark scaling both go through ScreenGeometry — but leaving them at
# the requested 640×480 would be a loaded gun for the next person who does.
CAM_WIDTH, CAM_HEIGHT = stream.width, stream.height

if stream.downscaled:
    print(f"[camera] native {stream.native_width}×{stream.native_height} "
          f"→ processing at {stream.width}×{stream.height} "
          f"({stream.scale:.2f}× scale, full field of view kept)")
    print(f"         {(1 - (stream.width * stream.height) / (stream.native_width * stream.native_height)) * 100:.0f}% "
          f"fewer pixels per frame for MediaPipe to chew through\n")
else:
    print(f"[camera] {stream.width}×{stream.height} native, no rescale needed\n")

# ─── MediaPipe setup ────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

_hand_kwargs = dict(
    static_image_mode=False,        # video stream mode (faster, uses tracking)
    max_num_hands=1,                # single hand for pointer control
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# model_complexity arrived partway through the 0.8.x line.  Falling back
# rather than crashing keeps every 0.8.x build usable; on 0.8.11 the Lite
# graph is selected normally.
try:
    hands = mp_hands.Hands(model_complexity=MODEL_COMPLEXITY, **_hand_kwargs)
    _complexity_note = f"model_complexity={MODEL_COMPLEXITY} (Lite)"
except TypeError:
    hands = mp_hands.Hands(**_hand_kwargs)
    _complexity_note = "model_complexity unsupported by this build"

# ─── State variables ────────────────────────────────────────────────────────

screen = ScreenGeometry(backend, stream.width, stream.height, POLL_INTERVAL)

# Create separate One Euro Filters for X and Y axes.
_oef_x = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF)
_oef_y = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF)

# Tracks whether a hand was visible on the *previous* frame.  A False → True
# transition means the hand just re-entered the frame, which is when the
# filters must be reset.  A display change also lowers this flag, reusing the
# same snap-to-target path (see the main loop).
hand_present = False

# Mirror state as of the previous frame, so a mid-run 'm' can be spotted and
# the filters reset before the cursor tries to glide to the mirrored position.
was_mirrored = IS_MIRRORED

# ── Gesture state ──────────────────────────────────────────────────────────
# gesture_state is the pose as of the last frame; a mismatch is a transition
# and starts the freeze.  None means "no hand yet", which is distinct from
# "idle" (a hand present in no recognised pose).
gesture_state = None
gesture = "idle"
fingers_ext = (False, False, False, False)
thumb_out = False
frozen_target = None
freeze_until = 0.0
gesture_changes = 0

# Sequence number of the last frame actually processed, so duplicates can be
# skipped instead of re-running inference on data we already consumed.
last_seq = -1

# Reusable frame buffers.  cv2.flip() and cv2.cvtColor() each allocate a
# fresh array when given no destination; writing into preallocated buffers
# instead removes two full-frame allocations per frame (~1.8 MB at 640×480×3,
# roughly 55 MB/s of churn at 30 FPS) along with the matching GC pressure.
# Allocated lazily because the camera may not honour the requested size.
bgr_buf = None      # mirrored frame: drawn on and displayed, full size
infer_bgr = None    # downscaled BGR staging buffer (None when fed 1:1)
rgb_buf = None      # colour-converted copy handed to MediaPipe
infer_w = infer_h = 0
infer_div = 1
infer_interp = cv2.INTER_AREA

# Last smoothed position — retained only for the on-screen jump readout.
prev_x, prev_y = screen.center

# perf_counter() is monotonic and sub-microsecond.  time.time() on Windows
# is backed by GetSystemTimeAsFileTime, whose ~15.6 ms granularity would
# badly quantise the dt estimate the 1€ filter depends on at 60 FPS.
prev_time = time.perf_counter()

# ─── Main loop ──────────────────────────────────────────────────────────────

print(f"Platform    : {backend.name}")
print(f"Cursor out  : {cursor.name}"
      + ("  (one API for Windows+Linux; click ready for gestures)"
         if cursor.name == "pynput"
         else "  (ctypes fallback — pynput not importable)"))
print(f"Screen src  : {screen.source}"
      + ("  (ctypes virtual desktop — origin, monitor count, hot-plug)"
         if screen.source == "native"
         else "  (tkinter vroot — no negative origin, no hot-plug)"))
print(f"Virtual desk: {screen.describe()}")
print(f"Webcam      : index {camera_index}, native {stream.native_width}×"
      f"{stream.native_height} @ {CAM_FPS} FPS requested")
print(f"Processing  : {stream.width} × {stream.height} "
      f"({stream.width / stream.height:.2f}:1)"
      + ("  [downscaled, aspect preserved]" if stream.downscaled
         else "  [native, no rescale]"))
if stream.size_mismatch:
    print(f"              driver reports {stream.reported_width}×"
          f"{stream.reported_height}; using the delivered frame size")
print(f"Active box  : {screen.box_w} × {screen.box_h} px  "
      f"x[{screen.box_left}–{screen.box_right}]  "
      f"y[{screen.box_top}–{screen.box_bottom}]")
print(f"Sensitivity : {CURSOR_SENSITIVITY}× start value, "
      f"adjustable {SENS_MIN}–{SENS_MAX} in steps of {SENS_STEP}")
_mx, _mt, _mb = sanitise_margins()
print(f"Edge margins: x {_mx:.0%} each side; y {_mt:.0%} top / {_mb:.0%} bottom")
print(f"              → live band is x[{_mx:.0%}–{1 - _mx:.0%}] "
      f"y[{_mt:.0%}–{1 - _mb:.0%}] of the frame "
      f"({1 - 2 * _mx:.0%}×{1 - _mt - _mb:.0%})")
print(f"              screen bottom is reached at {1 - _mb:.0%} down the "
      f"frame, not {100:.0f}% — no need to drop the hand past the desk")
_gx = screen.width / max(1, screen.box_w)
_gy = screen.height / max(1, screen.box_h)
print(f"Gain        : {_gx:.1f} screen px per camera px horizontally, "
      f"{_gy:.1f} vertically  (ratio {_gx / _gy:.2f})")
if abs(_gx / _gy - 1.0) > 0.15:
    print(f"              anisotropic — a diagonal sweep will not trace a "
          f"straight diagonal.")
    # Equal gain needs the vertical band to be this fraction of the frame.
    _want_band = ((1 - 2 * _mx) * (screen.height / screen.width)
                  * (stream.width / stream.height))
    if 0.05 < _want_band < 1.0:
        # Shrink the current top/bottom split to that band, keeping its ratio.
        _shrink = (1 - _want_band) / max(1e-6, _mt + _mb)
        print(f"              MARGIN_TOP ≈ {_mt * _shrink:.2f} / "
              f"MARGIN_BOTTOM ≈ {_mb * _shrink:.2f} would equalise it "
              f"(at the cost of vertical room)")
print(f"Mirroring   : {'ON' if IS_MIRRORED else 'OFF'} — press 'm' to flip "
      f"the picture AND the control direction together")
print(f"Invert X    : {'ON' if INVERT_CURSOR_X else 'OFF'} — press 'i' for "
      f"control-only inversion (picture unchanged)")
print(f"Anchor      : landmark {LM_MIDDLE_MCP} (middle-finger MCP) — the "
      f"knuckle, not the index tip")
print(f"              it barely moves when fingers bend, so gesturing does "
      f"not drag the cursor")
print(f"Gestures    : point / peace / grip / open / idle, from finger "
      f"geometry only")
print(f"              transitions freeze the target for {STATE_FREEZE_MS} ms "
      f"to absorb the hand twitch")
print(f"              MOVEMENT ONLY — no click, press or drag is issued")
print(f"1€ Filter   : min_cutoff={MIN_CUTOFF}  β={BETA}  d_cutoff={D_CUTOFF}")
print(f"MediaPipe   : {_complexity_note}, "
      f"det={MIN_DETECTION_CONFIDENCE} track={MIN_TRACKING_CONFIDENCE}")
_iw, _ih, _idiv, _iint = pick_infer_size(stream.width, stream.height)
if _idiv > 1:
    print(f"Inference at: {_iw}×{_ih} (1/{_idiv} of the preview) — preview "
          f"stays {stream.width}×{stream.height}")
    print(f"              landmarks are normalised, so no coordinate "
          f"correction is applied or needed")
else:
    print(f"Inference at: {_iw}×{_ih} — same as the preview")
print(f"Backend     : {describe_api(backend.camera_api)}"
      f"  (set CAMERA_API to pin one)")
_buf_note = ("accepted" if stream.buffersize_accepted else
             "refused — harmless, the capture thread already drops stale frames")
print(f"OpenCV      : {cv2.getNumThreads()} thread(s), buffersize=1 {_buf_note}")
print(f"Frame bufs  : preallocated (no per-frame flip/convert allocation)")
print(f"Display poll: every {POLL_INTERVAL}s (hot-plug aware)")
print("Controls    : '+'/'=' faster   '-'/'_' slower   'q' quit")
print("(the preview window must have focus for keys to register)\n")

try:
    while True:
        success, raw_frame, seq = stream.read()
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

        # ── Buffer preparation (allocation-free steady state) ───────────
        # The capture thread owns raw_frame, so it is never written to in
        # place; both transforms write into buffers this loop owns.
        if bgr_buf is None or bgr_buf.shape != raw_frame.shape:
            bgr_buf = np.empty_like(raw_frame)
            frame_h, frame_w = raw_frame.shape[:2]

            # ── Decoupled inference resolution ──────────────────────────
            # bgr_buf stays at the full processing size: it is what gets
            # drawn on and shown.  MediaPipe gets its own smaller pair of
            # buffers, so the preview quality is independent of how hard
            # the model is being pushed.
            infer_w, infer_h, infer_div, infer_interp = pick_infer_size(
                frame_w, frame_h)
            if infer_div > 1:
                # Two buffers: resize lands in BGR, colour conversion in RGB.
                # Both preallocated once, so the steady state still allocates
                # nothing per frame.
                infer_bgr = np.empty((infer_h, infer_w, 3), raw_frame.dtype)
                rgb_buf = np.empty((infer_h, infer_w, 3), raw_frame.dtype)
                print(f"[infer] MediaPipe fed {infer_w}×{infer_h} "
                      f"(1/{infer_div} of the {frame_w}×{frame_h} preview, "
                      f"{'exact' if infer_interp == cv2.INTER_AREA else 'fractional'})")
            else:
                infer_bgr = None
                rgb_buf = np.empty_like(raw_frame)
                print(f"[infer] MediaPipe fed the full {frame_w}×{frame_h} "
                      f"frame (INFER_MAX_WIDTH disabled or already small)")
            # The frame size is the canvas the box is drawn on, so a change
            # here invalidates the box.  Rebuilding from the same numbers the
            # buffers were sized with is what keeps the two in step.
            if screen.set_camera_size(frame_w, frame_h):
                print(f"[camera] frame size now {frame_w}×{frame_h} → "
                      f"box rebuilt to {screen.box_w}×{screen.box_h} "
                      f"at x[{screen.box_left}–{screen.box_right}] "
                      f"y[{screen.box_top}–{screen.box_bottom}]")
                hand_present = False    # snap rather than glide into the new box

        # ── Mirror, or don't ────────────────────────────────────────────
        # This one buffer is both what gets displayed and what MediaPipe
        # reads, so the choice here sets the control direction as well as
        # the picture.  np.copyto keeps the un-mirrored path allocation-free
        # too, rather than dropping the preallocated buffer on the floor.
        if IS_MIRRORED:
            cv2.flip(raw_frame, 1, dst=bgr_buf)
        else:
            np.copyto(bgr_buf, raw_frame)

        # A toggle mirrors the hand's mapped position instantly.  Route the
        # next frame through the re-entry reset so the cursor snaps to the
        # new spot rather than sweeping across the desktop to reach it.
        if IS_MIRRORED != was_mirrored:
            was_mirrored = IS_MIRRORED
            hand_present = False

        # ── Feed MediaPipe ──────────────────────────────────────────────
        # Downscale first, then convert colour: cvtColor on the small image
        # is a quarter of the work it would be on the large one, so this
        # ordering is cheaper than converting and then shrinking.
        #
        # NO COORDINATE CORRECTION IS NEEDED after this, and adding one
        # would break the mapping.  MediaPipe returns landmarks normalised
        # to [0, 1] of the image it was handed; because the small copy is an
        # exact rescale of the preview, "40% across" means the same place in
        # both.  Multiplying by screen.cam_w below therefore lands in the
        # PREVIEW's pixel space regardless of what size the model saw.
        if infer_bgr is not None:
            cv2.resize(bgr_buf, (infer_w, infer_h), dst=infer_bgr,
                       interpolation=infer_interp)
            cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB, dst=rgb_buf)
        else:
            cv2.cvtColor(bgr_buf, cv2.COLOR_BGR2RGB, dst=rgb_buf)

        # Mark the buffer read-only so MediaPipe borrows it rather than
        # defensively copying a full frame.  process() is synchronous and
        # does not retain the array, so reusing it next iteration is safe —
        # but the flag MUST be cleared afterwards, otherwise the cvtColor
        # above would fail on a read-only destination on the next frame.
        rgb_buf.flags.writeable = False
        results = hands.process(rgb_buf)
        rgb_buf.flags.writeable = True

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # ── Cursor anchor: landmark 9, the middle-finger knuckle ────
            # Not the index tip.  A fingertip is the most mobile point on
            # the hand: every gesture moves it by design, so pointing with
            # it means the cursor lurches whenever the pose changes.  The
            # MCP knuckle barely moves when fingers bend — it is carried by
            # the palm, not the finger — so the tracking signal stays put
            # while the fingers do the signalling.  Landmark 8 is still
            # read below, but only as gesture input.
            anchor = hand.landmark[LM_MIDDLE_MCP]

            # MediaPipe returns normalised coords [0, 1]; convert to pixels
            # of the REAL frame.  Scaling by a hardcoded 640×480 here would
            # place the anchor at a fraction of its true position on any
            # other sensor, and the overlay would drift away from the hand.
            raw_x = anchor.x * screen.cam_w
            raw_y = anchor.y * screen.cam_h

            # ── Gesture classification (geometry only) ──────────────────
            gesture, fingers_ext, thumb_out = detect_gesture(
                hand, screen.cam_w, screen.cam_h)

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
                # A hand that just arrived carries no gesture history, and
                # a stale freeze would pin the cursor to coordinates from
                # the previous appearance.
                gesture_state = None
                frozen_target = None
                freeze_until = 0.0

            # ── State-transition freeze ─────────────────────────────────
            # Changing pose moves the whole hand slightly.  Capturing the
            # target on the transition frame and holding it for
            # STATE_FREEZE_MS absorbs that twitch, then movement resumes so
            # the pose can still be dragged around once settled.
            if gesture != gesture_state:
                previous = gesture_state
                gesture_state = gesture
                frozen_target = (target_x, target_y)
                freeze_until = now_ts + STATE_FREEZE_MS / 1000.0
                gesture_changes += 1
                print(f"[gesture] {previous or '-'} -> {gesture}"
                      f"   (hold {STATE_FREEZE_MS} ms)")

            if frozen_target is not None:
                if now_ts < freeze_until:
                    # The frozen point is fed THROUGH the filter rather than
                    # bypassing it, so the filter stays settled there; on
                    # release it glides back to the hand instead of jumping.
                    target_x, target_y = frozen_target
                else:
                    frozen_target = None

            # ── One Euro Filter smoothing ────────────────────────────────
            # The filter is the *only* smoothing stage.  It internally tracks
            # time and speed to compute a dynamic cutoff: still → heavy
            # smoothing, fast → light.  Nothing here blocks the loop, so the
            # next camera sample arrives as soon as the hardware has it.
            smooth_x = _oef_x(target_x, timestamp=now_ts)
            smooth_y = _oef_y(target_y, timestamp=now_ts)

            cursor.move(int(smooth_x), int(smooth_y))

            # Per-frame travel, kept purely as a tuning readout.
            jump = math.hypot(smooth_x - prev_x, smooth_y - prev_y)
            prev_x = smooth_x
            prev_y = smooth_y

            # ── Visualisation overlays (drawn on the writeable BGR buffer)
            mp_draw.draw_landmarks(bgr_buf, hand, mp_hands.HAND_CONNECTIONS)

            # Marker on the ANCHOR — the middle-finger knuckle the cursor
            # actually follows, not the index tip.  Uses the UNREFLECTED
            # coordinate so it sits on the hand as the preview shows it,
            # whatever the control path did.  Amber while frozen.
            cx, cy = int(raw_x), int(raw_y)
            _frozen = frozen_target is not None and now_ts < freeze_until
            _anchor_col = (0, 165, 255) if _frozen else (0, 255, 0)
            cv2.circle(bgr_buf, (cx, cy), 11, _anchor_col, cv2.FILLED)
            cv2.circle(bgr_buf, (cx, cy), 15, _anchor_col, 2)

            # The index tip is still drawn, small and hollow, to make the
            # anchor change legible: it moves when gesturing, the anchor
            # does not.
            _it = hand.landmark[LM_INDEX_TIP]
            cv2.circle(bgr_buf,
                       (int(_it.x * screen.cam_w), int(_it.y * screen.cam_h)),
                       5, (200, 200, 200), 1)

            # Per-finger verdicts, in the order the classifier sees them.
            # Indexing a precomputed char pair beats building a generator
            # and calling .upper() four times a frame; the resulting string
            # is identical.
            _flags = (_FINGER_CHARS[0][fingers_ext[0]]
                      + _FINGER_CHARS[1][fingers_ext[1]]
                      + _FINGER_CHARS[2][fingers_ext[2]]
                      + _FINGER_CHARS[3][fingers_ext[3]]
                      + ("T" if thumb_out else "t"))
            cv2.putText(
                bgr_buf, _flags, (cx + 20, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            # Show how far the cursor moved this frame.
            cv2.putText(
                bgr_buf, f"j={jump:.0f}", (cx + 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )
        else:
            # No hand this frame — arm the reset for whenever it returns.
            hand_present = False
            # Drop the gesture too.  A hand that leaves mid-freeze would
            # otherwise keep the cursor pinned, and the next appearance
            # would not register as a transition if the pose happened to
            # match the stale one.
            if gesture_state is not None:
                print(f"[gesture] {gesture_state} -> (hand lost)")
            gesture_state = None
            gesture = "idle"
            fingers_ext = (False, False, False, False)
            thumb_out = False
            frozen_target = None
            freeze_until = 0.0

        # Draw the active bounding box on the preview.  Read from `screen`
        # so it follows the box when a display is plugged or unplugged.
        cv2.rectangle(
            bgr_buf,
            (screen.box_left,  screen.box_top),
            (screen.box_right, screen.box_bottom),
            (255, 0, 255), 2,
        )

        # With sensitivity > 1 only an inner region still reaches the desktop
        # edges; everything outside it is clamped flat.  Drawn every frame so
        # it shrinks and grows live as '+' / '-' are pressed.
        _eff = screen.effective_box
        if _eff is not None:
            cv2.rectangle(bgr_buf, (_eff[0], _eff[1]), (_eff[2], _eff[3]),
                          (0, 165, 255), 1)

        # ── HUD ─────────────────────────────────────────────────────────
        # FPS counter — counts only frames that were really processed.
        fps = 1.0 / (now_ts - prev_time) if now_ts > prev_time else 0.0
        prev_time = now_ts
        cv2.putText(
            bgr_buf, f"FPS: {int(fps)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        # Current sensitivity, directly below the FPS readout, rounded to one
        # decimal place.  Amber to match the live-region rectangle it controls.
        cv2.putText(
            bgr_buf, f"Speed: {CURSOR_SENSITIVITY:.1f}x", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
        )

        # Mirror / inversion state — the two things 'm' and 'i' change.
        # Green when mirrored (the usual webcam case), red when raw, so a
        # glance is enough to tell which mode is live.
        cv2.putText(
            bgr_buf,
            f"Mirror: {'ON' if IS_MIRRORED else 'OFF'}"
            f"{'  invX' if INVERT_CURSOR_X else ''}",
            (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 220, 0) if IS_MIRRORED else (0, 80, 255), 2,
        )

        # Gesture state, and whether the transition freeze is holding the
        # cursor right now.  Amber while frozen so the lock is unmistakable.
        _held = frozen_target is not None and now_ts < freeze_until
        if _held:
            _left_ms = (freeze_until - now_ts) * 1000.0
            cv2.putText(
                bgr_buf, f"{gesture.upper()}  LOCK {_left_ms:3.0f}ms",
                (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
            )
        else:
            cv2.putText(
                bgr_buf, f"{gesture.upper()}  ({gesture_changes})",
                (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        # Desktop summary underneath.
        cv2.putText(
            bgr_buf, f"{screen.width}x{screen.height} ({screen.monitors} mon)",
            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        # Key hints along the bottom edge of the real frame.  The text is a
        # module-level constant rather than a literal rebuilt each frame.
        cv2.putText(
            bgr_buf, _HINT_TEXT, (10, frame_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("Hand Cursor Control", bgr_buf)

        # Sensitivity keys and quit, same handler as the skip path above.
        if handle_key(cv2.waitKey(1) & 0xFF):
            break

finally:
    stream.stop()
    cv2.destroyAllWindows()
    hands.close()
    cursor.close()
    backend.close()
    print("Shutdown complete.")