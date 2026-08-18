"""
Hand-Tracking Cursor Controller
===============================
Moves the mouse by tracking the middle-finger knuckle (MediaPipe landmark 9)
and drives clicks, drags and OS macros from hand pose.

Two perception paths, split by what each is good for:

    geometry  (30 Hz)  -->  cursor movement, click, drag
    YOLOv10n  (18 Hz)  -->  semantic labels, keyboard macros

    main loop  --frame.copy()-->  queue(maxsize=1)  -->  YoloWorker
    main loop  <--get_nowait()--  queue(maxsize=8)  <--  results

Both hand-offs are non-blocking.  submit() replaces the pending frame rather
than waiting for room, so the worker always classifies the freshest hand and
stale frames are dropped.  Neither the 1 Euro filter nor cursor.move ever
waits on inference.

ANCHOR.  Landmark 9 (middle MCP), not landmark 8 (index tip).  A fingertip
is the most mobile point on the hand, so anchoring there makes the cursor
lurch whenever the pose changes.  The knuckle is carried by the palm, so it
holds still while the fingers do the signalling.

GESTURE STATES.  detect_gesture() classifies point / peace / grip / open
with idle as the fallback.  Arithmetic only: a finger counts as extended
when its tip sits further from the wrist than its PIP joint does, which is
self-normalising and needs no per-camera threshold and no classifier.  The
thumb gets its own test because it folds across the palm rather than curling
radially.

IDLE IS AN ABSTENTION, NOT A POSE, and this is the whole reason clicking
works.  Curling from point to grip physically passes through a half-curled
hand, which reads as idle.  The FSM fires only when the state immediately
before the target was the source, so a stabilised idle in between silently
becomes that predecessor and the click never comes.  Measured at 30 Hz with
idle fed in, the click survives a 33 ms curl and dies at 66 ms — it never
works for a real hand.  So idle never reaches the FSM.

TRANSITION FREEZE.  Changing pose shifts the whole hand slightly, so every
state change pins the target for STATE_FREEZE_MS.  The frozen point is fed
THROUGH the 1 Euro filter rather than bypassing it, leaving the filter
settled there so movement resumes by gliding rather than jumping.

SMOOTHING is handled exclusively by a One Euro Filter (Casiez et al. 2012),
which adapts its cutoff to hand speed: heavy smoothing at rest, light
smoothing during fast swipes.

DECOUPLED INFERENCE.  MediaPipe is fed a 1/N integer downscale of the
preview while the display buffer stays full size.  The divisor is an integer
so the rescale hits OpenCV's fast path and the aspect ratio survives
exactly, which is what lets the normalised landmarks be used with no
coordinate correction at all.

MAPPING.  The active box is the frame inset by MARGIN_X / MARGIN_TOP /
MARGIN_BOTTOM, mapped onto the whole virtual desktop and rebuilt on the fly
when a display is plugged, unplugged or rearranged.  The vertical margins
are asymmetric because a hand raises well above shoulder height but stops
against the desk going down.

OUTPUT goes through pynput, with a ctypes path as fallback.  On Windows
pynput's position setter IS user32.SetCursorPos; measured side by side on a
3840x1080 dual desktop the two agreed exactly at every corner.

FAIL SOFT.  A missing model, an absent ultralytics, a missing gesture_fsm.py
or an unavailable keyboard leaves that feature off and the cursor path
completely unchanged.

Controls:  + / =  raise sensitivity      - / _  lower sensitivity
           m      mirror on/off (picture AND control direction)
           i      invert X, control only (picture unchanged)
           q      quit

Dependencies:  pip install opencv-python mediapipe numpy pynput ultralytics
Platform:      Windows (user32) and Linux/X11 (libX11)
"""

# PEP 604 unions (`float | None`) are evaluated at def-time on Python < 3.10
# and would raise TypeError there.  Deferring annotation evaluation keeps the
# 3.8/3.9 half of the supported range importable.
from __future__ import annotations

import contextlib
import ctypes
import math
import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np          # already a hard dependency of cv2 and mediapipe

from monitors import (ALL_SCREENS, enumerate_monitors, monitor_labels,
                      monitor_union, resolve_monitor_target)

try:
    from gesture_fsm import (DOUBLE_CLICK, DRAG_START, DRAG_STOP, LEFT_CLICK,
                             ActionExecutor, GestureFSM, MajorityStabilizer,
                             load_config)
    _FSM_AVAILABLE = True
except Exception:                       # pragma: no cover - optional branch
    DOUBLE_CLICK = "DOUBLE_CLICK"
    LEFT_CLICK = "LEFT_CLICK"
    DRAG_START = "DRAG_START"
    DRAG_STOP = "DRAG_STOP"
    ActionExecutor = None
    GestureFSM = None
    MajorityStabilizer = None
    load_config = None
    _FSM_AVAILABLE = False

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

CURSOR_SENSITIVITY = 1.4

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
MARGIN_X = 0.20
MARGIN_TOP = 0.10
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
#
# MUST DEFAULT OFF while IS_MIRRORED is on.  These are the two reversals the
# note above warns about: mirroring the buffer already puts the landmarks in
# display space, so reflecting the maths as well cancels it out and the
# cursor runs backwards under a preview that looks correct.  Only one of the
# two may be on at a time —
#
#   IS_MIRRORED=True,  INVERT_CURSOR_X=False  → mirror, correct  ← default
#   IS_MIRRORED=False, INVERT_CURSOR_X=True   → raw preview,    correct
#   both on / both off                        → cursor runs backwards
INVERT_CURSOR_X = False

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
EXTEND_MARGIN = 1.05   # slack on the tip-vs-pip comparison
THUMB_OUT_RATIO = 0.60   # |thumb4 − indexMCP5| / palm, above which it is out

# Both tests compare one distance against another, and a square root is
# monotonic — so the comparison can be made on SQUARED distances and every
# sqrt dropped.  Squaring the thresholds once here keeps the comparisons
# algebraically identical:
#       a > b·k        ⇔   a² > b²·k²          (a, b, k ≥ 0)
_EXTEND_MARGIN_SQ = EXTEND_MARGIN * EXTEND_MARGIN
_THUMB_OUT_RATIO_SQ = THUMB_OUT_RATIO * THUMB_OUT_RATIO
_PALM_MIN_SQ = 1e-12          # was |palm| <= 1e-6

# ── State-transition freeze (anti-drift) ───────────────────────────────────
# Changing gesture moves the whole hand a little: fingers closing or opening
# drag the knuckles with them, and the cursor lurches at exactly the moment
# the user meant to signal something, not move.  Holding the target still
# for a moment after every transition absorbs that twitch.
#
# Landmark 9 already removes most of it (see the anchor note below); this
# covers the residual.  Long enough to outlast the twitch, short enough that
# an intentional move right after a transition does not feel blocked.
STATE_FREEZE_MS = 75

# MediaPipe prints a TFLite banner and two absl warnings while its graph
# builds.  They are harmless and unactionable, and they are the only thing
# this program writes to stderr that it did not choose to write.  Set False
# to see them again — worth doing if MediaPipe ever fails to initialise,
# since a real error would arrive by the same route.
SUPPRESS_NATIVE_WARNINGS = True

# Launch system_monitor.py alongside the preview.  Set False to run without
# it — the main script neither reads from it nor waits on it.
SYSTEM_MONITOR_ENABLED = True

# Size of the throwaway frame used to build the graph at startup.  Any size
# works: landmarks come back normalised and the result is discarded.
INFER_WARMUP_W = 320
INFER_WARMUP_H = 180

SENS_STEP = 0.1

# Floor: plain absolute mapping, the entire box live.  Below 1.0 the mapping
# would shrink the reachable area to a sub-region of the desktop and strand
# the screen edges — never useful, so the range excludes it outright.
SENS_MIN = 1.0

# Ceiling: only 1/3 of the box still reaches an edge.  Beyond this the live
# region gets too small to aim inside, and key auto-repeat would run away.
SENS_MAX = 3.0

# ── AI confidence ─────────────────────────────────────────────────────────
# One number driving both models: MediaPipe's detection/tracking gates and
# the YOLO score floor.  Low finds a hand in poor light and invents poses
# in clutter; high is certain and drops out when you move.  A custom
# gesture set is exactly when this needs tuning, which is why it is on a
# slider rather than baked in.
AI_CONFIDENCE_MIN = 0.1
AI_CONFIDENCE_MAX = 0.9
AI_CONFIDENCE_DEFAULT = 0.6

# ── Gesture prediction caption ─────────────────────────────────────────────
# The one overlay left on the frame.  BGR, not RGB: bright green reads as
# "recognised" against skin, wood and painted walls alike, and grey keeps
# "none" from competing with it for attention.  The black pass underneath is
# what makes either legible on a pale background.
_PREDICT_ORIGIN = (14, 40)
_PREDICT_SCALE = 0.85
_PREDICT_COLOUR = (0, 255, 120)        # bright green
_PREDICT_IDLE_COLOUR = (170, 170, 170)  # grey, for "none" / loading
_PREDICT_SHADOW = (0, 0, 0)

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
BETA = 0.015   # speed coefficient.  Higher = less trailing when moving.
D_CUTOFF = 1.0     # Hz — cutoff of the velocity estimator itself.

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
MIN_TRACKING_CONFIDENCE = 0.5

# ═══════════════════════════════════════════════════════════════════════════
#  YOLO GESTURE BRANCH  (background thread — never blocks the cursor)
# ═══════════════════════════════════════════════════════════════════════════
#
# The whole preview frame goes to the network, exactly as model/ptmodel.py
# feeds it; ultralytics letterboxes it to the model's input size.  Measured
# warm, a forward pass costs 55 ms — about 18 Hz, against a MediaPipe frame
# of roughly 32 ms.  Running it inline would still halve the cursor rate, so
# it lives on its own thread fed by a depth-1 queue that always holds the
# NEWEST frame: when the worker is busy the main loop overwrites the pending
# item and moves on, so nothing ever waits.
YOLO_ENABLED = True
YOLO_MODEL_PATH = os.path.join("model", "YOLOv10n_gestures.pt")
YOLO_INPUT_SIZE = 640

# "auto" lets ultralytics choose, which picks CUDA when torch can see a GPU.
# Worth leaving alone: on this machine CUDA is 5.3× faster than CPU, close
# enough to camera rate that the semantic path stops feeling like a separate,
# laggy system.  Force with "cpu" or 0 if the GPU is needed elsewhere.
YOLO_DEVICE = "auto"
YOLO_SCORE_THRESHOLD = 0.35

# Diagnostic: dump what the network is actually being shown.  Written on the
# worker thread, never the cursor path.  Off by default — it is a JPEG
# encode per inference.
YOLO_DEBUG_CROP = False
YOLO_DEBUG_CROP_PATH = "debug_crop.jpg"

# Print every change in the YOLO label, so the "[yolo] a -> b" lines sit
# alongside the "[gesture] a -> b" lines from the geometry classifier.  The
# two are separate subsystems and only the geometric one reaches the FSM;
# without this line there is no way to tell them apart in the console.
YOLO_LOG_CHANGES = True

# ── Semantic macros: the YOLO branch's payload ─────────────────────────────
# This is what the slow path is FOR.  A quarter-second of lag is invisible
# on "show me the desktop" and disqualifying on a click, which is why these
# hang off the network and the mouse hangs off the geometry.
#
# Keys are raw HaGRIDv2 class names, as reported by the model.
# EMPTY BY DEFAULT, and it must stay that way.  This dict binds gestures to
# keyboard macros entirely outside gesture_config.json, so anything listed
# here fires on a blank-slate install with no rule to explain it — which is
# exactly the surprise the zero-defaults policy exists to prevent.  Bind
# SHOW_DESKTOP to a hold in the GUI instead; that path is visible, editable
# and deletable.  Populate this only to hardcode a macro on purpose.
YOLO_MACROS = {}

# Two independent guards, because a cooldown alone does not do what it
# sounds like.  At 18 Hz a 2 s cooldown still fires five times if the hand is
# held up for ten seconds — it throttles the spam rather than stopping it.
# So the macro is EDGE-triggered: it fires on entering the gesture and will
# not fire again until the label has been something else.  The cooldown then
# remains as a backstop against a label flickering in and out.
YOLO_MACRO_COOLDOWN = 2.0

# Macros are disruptive and irreversible in a way a click is not, so they
# demand better evidence than the 0.35 used to merely display a label.
YOLO_MACRO_MIN_SCORE = 0.60

YOLO_MACRO_ENABLED = True

# Shown in the banner: the geometric classifier runs once per captured
# frame, so its rate is the camera's, against the network's measured 4.7 Hz.
FPS_NOTE = "30 Hz"

# ── Rule source: gesture_config.json, and nothing else ────────────────────
# There are deliberately NO binding constants here.  Every transition, hold,
# action and timing value is read from gesture_config.json at start-up by
# HandTrackerEngine._apply_config(); an empty config means an empty rule set
# and no gesture does anything.
#
# This block used to hardcode point->grip = click and open->grip = drag as
# fallbacks.  They were already dead — the engine stopped reading them when
# it became config-driven — but leaving them here advertised defaults the
# program no longer has, which is worse than not having them.
#
# The stability window and double-click timing that used to live here are
# now "window_size" / "stability_threshold" / "double_click_sec" under
# "settings" in the config, so the GUI can reach them too.

# detect_gesture returns "idle" for anything half-curled, and curling from
# point to grip physically passes through one.  "idle" is the classifier
# abstaining, not a pose, so it must not reach the FSM: the FSM fires only
# when the state immediately before the target was the source, and a
# stabilised "idle" in between silently becomes that predecessor.
#
# Measured at 30 Hz with idle fed in, the click survives a 33 ms curl and
# dies at 66 ms — i.e. it never works for a real hand.  Widening the window
# to 500 ms does fix it, at the cost of a 500 ms click.  Filtering is the
# version that keeps both.
FSM_IGNORED_STATES = ("idle",)

# The training order.  Index 5 is "three3" whatever it gets called; the
# worker prefers the model's own names when the export carries them and
# falls back to this tuple otherwise.
YOLO_CLASS_NAMES = (
    "grabbing", "grip", "holy", "point", "call", "three3", "timeout",
    "xsign", "hand_heart", "hand_heart2", "little_finger", "middle_finger",
    "take_picture", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted", "three_gun",
    "thumb_index", "thumb_index2", "no_gesture",
)

# Poses whose orientation is decided by MediaPipe landmarks instead of by
# the network, and which axis decides each one.  Deliberately tiny — every
# entry is a place we second-guess a trained model.
#
#   three_gun : the model has no three_gun_inverted at all, so the choice
#               is between our estimate and nothing.  Read off the BARREL
#               (wrist -> index tip), because a gun is held in profile and
#               its knuckles overlap on x.
#   stop      : read off the KNUCKLES (index MCP -> pinky MCP).  A stop is
#               a wide flat hand held square to the camera, which is the
#               pose where that spread is widest and steadiest.
#
# The trained "stop_inverted" class used to collide with this: the same
# pose could arrive under two names depending on which subsystem decided
# it.  LABEL_ALIASES below now folds that name into "stop_inverse" before
# this gate is consulted, so both routes end at one spelling and it is
# safe for "stop" to be listed here.
ORIENTATION_GESTURES = ("three_gun", "stop")

# Class names rewritten the instant they leave the network, before any of
# our own modifiers are considered.  One spelling reaches the FSM, the
# caption and the gestures/ folder: "_inverse", never "_inverted".
#
# Two separate problems, one fix.
#
# THE COLLISION, which only "stop" has.  The network ships a trained
# "stop_inverted" AND our knuckle override produces "stop_inverse" for the
# same physical pose, so a rule bound to one would silently ignore the
# other.  Aliasing collapses them:
#
#   network says "stop_inverted" -> "stop_inverse" immediately, which is
#   NOT in ORIENTATION_GESTURES, so the knuckle math is skipped (the
#   network already decided) and only the handedness suffix is added.
#
#   network says "stop"          -> stays "stop", the knuckle math runs and
#   appends "_inverse" itself when the back of the hand is showing.
#
# THE MISMATCH, which peace and two_up have.  Nothing competes for these —
# the network is the only thing that decides them — but it spells them
# "_inverted" while the artwork in gestures/ is named "_inverse".  Since
# the alias table was removed from app.py, a filename IS a label, so the
# PNGs were unreachable.  Renaming here makes the model agree with the
# folder instead of the other way round.
LABEL_ALIASES = {
    "stop_inverted": "stop_inverse",
    "peace_inverted": "peace_inverse",
    "two_up_inverted": "two_up_inverse",
}


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
CAM_SCAN_READ_WAIT = 0.12  # seconds between those attempts (warm-up)
CAM_SCAN_SETTLE = 0.1   # seconds after release, before the next index

# The scan is a hint, not the truth — a driver that was still waking up can
# be typed in by hand even though it never answered.  This bounds what the
# prompt will accept, nothing more.
CAM_MANUAL_MAX = 9

# The resolution *requested* from the driver.  It is a request, not a fact:
# cameras are free to ignore it and many do, so nothing downstream may assume
# frames arrive at this size.  The authoritative numbers come from the first
# frame WebcamStream actually receives (stream.width / stream.height), and
# every piece of geometry is built from those.
CAM_WIDTH = 1280
CAM_HEIGHT = 720
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
# Raised to 720p for the gesture network.  Point 1 above no longer applies:
# MediaPipe is fed a decoupled 320×180 either way (see pick_infer_size), so
# its cost does not move at all.  What this buys is real sensor detail —
# measured, the hand carries 1.50× the genuine pixels at every distance.
#
# That matters here because the whole frame is letterboxed into the model's
# input and the hand keeps only its own fraction of it, so sensor resolution
# is the only thing standing between the model and a hand a few dozen pixels
# tall.
#
# What it costs is the full-size work: flip, downscale and the frame copy go
# from 258 µs to 2984 µs per frame — 8% of a 30 FPS budget, against the
# ~32 ms MediaPipe still spends.  The 1280→320 downscale is 2496 µs of that
# and is now the second-largest cost in the loop.
#
# Point 2 above still stands and is the reason not to go further: at 1080p
# the preview window would exceed most desktops.
PROC_MAX_WIDTH = 1280
PROC_MAX_HEIGHT = 720

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
# Deliberately UNCHANGED by the 720p upgrade.  1280 / 320 = 4 exactly, and
# 720 / 4 = 180 exactly, so a 720p feed reaches MediaPipe as 320×180 on the
# same integer fast path a 640×480 feed used at divisor 2.  The model's cost
# is identical at both sensor sizes; only the preview buffer grew.
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


@contextlib.contextmanager
def quiet_stderr(enabled: bool = SUPPRESS_NATIVE_WARNINGS):
    """Silence writes to file descriptor 2 for the duration of the block.

    MediaPipe's TFLite banner and absl warnings are emitted by C++ before
    absl's logger is initialised — the "All log messages before
    absl::InitializeLog()" line says so itself.  They therefore go straight
    to fd 2, where nothing in Python can reach them: GLOG_minloglevel,
    TF_CPP_MIN_LOG_LEVEL, absl.logging.set_verbosity and
    warnings.filterwarnings were all measured here, and all four left the
    output completely untouched.

    Redirecting the descriptor is the only thing that works.  It is
    deliberately scoped to one statement rather than the whole program: the
    saved descriptor is restored in a finally, so a genuine error raised a
    moment later still reaches the terminal.
    """
    if not enabled:
        yield
        return

    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def clamp(value: float, lo: float, hi: float) -> float:
    """Restrict *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


# ── How many hands MediaPipe tracks ────────────────────────────────────────
# 2, for the two-handed half of the HaGRIDv2 vocabulary (timeout, hand_heart,
# xsign, take_picture …).  Only the YOLO branch uses the second hand: the
# cursor stays locked to one hand throughout.
#
# The cost is conditional, not constant.  Palm detection runs once either
# way; the landmark model runs per hand found, so a one-handed frame is
# priced exactly as it was before and only a genuinely two-handed frame pays
# for the second pass.
MAX_NUM_HANDS = 2

# MediaPipe does not promise that multi_hand_landmarks keeps a stable order
# between frames — the list is ordered by detection, and a hand that is lost
# and re-acquired, or a second hand entering, can reorder it.  Taking [0]
# every frame would therefore let the cursor jump between hands, which is
# the precise thing two-hand support must not do.  With this on, the primary
# hand is instead the one nearest to where the primary hand was last frame.
# Set False to take multi_hand_landmarks[0] verbatim.
PRIMARY_HAND_TRACKING = True

# ── MediaPipe hand landmark indices ────────────────────────────────────────
# Named rather than inlined: hand.landmark[14] is unreadable, and the tip /
# pip / mcp distinction is the entire basis of the extension test below.
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_MCP = 5
LM_INDEX_PIP = 6
LM_INDEX_TIP = 8
LM_MIDDLE_MCP = 9        # the cursor anchor — see ANCHOR note in the header
LM_MIDDLE_PIP = 10
LM_MIDDLE_TIP = 12
LM_RING_PIP = 14
LM_RING_TIP = 16
LM_PINKY_MCP = 17        # with LM_INDEX_MCP, the palm's left/right axis
LM_PINKY_PIP = 18
LM_PINKY_TIP = 20

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


def _as_hand_list(hand_landmarks):
    """Accept one hand or a sequence of them, return a tuple of hands.

    A single NormalizedLandmarkList carries `.landmark`; MediaPipe's
    multi_hand_landmarks is a plain list of them.  Sniffing the attribute
    rather than the type keeps this working against the fixtures too.
    """
    if hand_landmarks is None:
        return ()
    if hasattr(hand_landmarks, "landmark"):
        return (hand_landmarks,)
    return tuple(hand_landmarks)


def handedness_of(results, hand, mirrored: bool) -> str:
    """"left" / "right" for *hand*, or "" when it cannot be determined.

    Two things make this more than a lookup.

    THE RIGHT HAND, LITERALLY.  multi_handedness runs parallel to
    multi_hand_landmarks, so the label has to be taken at the index of the
    hand the cursor actually follows — pick_primary_hand() may well have
    chosen the second entry, and index 0 would then name the other hand.
    Matched by identity, because comparing protobuf messages with == is
    both slow and not what is meant here.

    THE MIRROR.  MediaPipe's own docs: "it determines handedness assuming
    the input image is mirrored".  We hand it the flipped buffer when
    IS_MIRRORED is on, which is exactly that assumption, so the label is
    already correct.  With mirroring off it sees the raw camera image and
    every label comes back inverted, so it is swapped here.  Getting this
    backwards would name every gesture after the wrong hand.

    Returns "" rather than guessing when the skeleton is missing, which is
    the caller's cue to fall back to the bare gesture name.
    """
    try:
        landmarks = results.multi_hand_landmarks
        labels = results.multi_handedness
        if not landmarks or not labels:
            return ""
        index = next((i for i, h in enumerate(landmarks) if h is hand), None)
        if index is None or index >= len(labels):
            return ""
        label = labels[index].classification[0].label.strip().lower()
    except (AttributeError, IndexError, TypeError, ValueError):
        return ""

    if label not in ("left", "right"):
        return ""
    if not mirrored:
        label = "right" if label == "left" else "left"
    return label


# How far apart the two knuckles must be, as a fraction of frame width,
# before the palm/back verdict is trusted.  Edge-on they project onto
# nearly the same x and the comparison is reading noise, so below this the
# answer is "don't know" rather than a coin flip that flickers the gesture
# name every frame.
PALM_DEADBAND = 0.02


def palm_is_facing(hand, side: str, mirrored: bool):
    """True palm-to-camera, False back-to-camera, None when undecidable.

    The knuckle order along x tells you which way round the hand is.  The
    base case is calibrated against the camera, not derived from anatomy:

        RIGHT hand, RAW image, palm to camera  ->  x[5] > x[17]

    Turn the hand over and they swap.  A left hand reverses it, and
    mirroring the buffer reverses it once more, which is why both flags
    are applied on top of that base case.

    THE SIGN IS EMPIRICAL AND HAS BEEN WRONG TWICE.  Both of the relative
    corrections below are self-evidently right — mirroring flips it, the
    other hand flips it — so a test that only checks "mirroring changes the
    answer" or "the two hands disagree" passes with the base case inverted
    and every verdict backwards.  Only holding a real palm to a real camera
    settles it.  If it ever reads backwards again, flip THIS comparison and
    nothing else: an error here is global, identical for both hands and
    both mirror states, so one sign fixes all eight permutations.

    `side` must be the TRUE physical hand, which is what handedness_of()
    returns; without it there is no way to read the order, so an unknown
    side gives an unknown orientation.

    LIMITATION, stated plainly: this compares x only, so it assumes a
    roughly upright hand.  Rotate the wrist toward horizontal and the two
    knuckles line up vertically instead, the deadband trips, and the
    verdict becomes None rather than wrong.
    """
    if side not in ("left", "right"):
        return None
    try:
        landmarks = hand.landmark
        spread = landmarks[LM_INDEX_MCP].x - landmarks[LM_PINKY_MCP].x
    except (AttributeError, IndexError, TypeError):
        return None
    if not isinstance(spread, float) or spread != spread:
        return None
    if abs(spread) < PALM_DEADBAND:
        return None

    # x[5] > x[17]  ->  palm, for a right hand in a raw image.
    facing = spread > 0.0
    if mirrored:
        facing = not facing
    if side == "left":
        facing = not facing
    return facing


# The gun barrel is a far longer baseline than the palm is wide, so it can
# afford a far wider deadband and still decide.  In a gun pose the wrist and
# index tip are most of a hand apart (~0.15–0.25 of frame width) while the
# two knuckles sit within ~0.02 of each other — which is why the knuckle
# reading flickers on this pose and this one does not.
GUN_DEADBAND = 0.05


def gun_is_facing(hand, side: str, mirrored: bool):
    """Palm/back for a POINTING hand, read off the barrel instead of the palm.

    palm_is_facing() compares the index and pinky knuckles, which works for
    an upright hand and fails badly for a gun: the pose is held in profile,
    the two knuckles project onto nearly the same x, and the verdict becomes
    a coin flip that changes the gesture name every frame.

    A gun has a much better axis available — wrist (0) to index tip (8), the
    barrel — and in this pose that axis is nearly horizontal, which is
    exactly where the knuckle axis is useless.  The trade is one assumption
    for another, and the new one suits the pose:

        knuckles : needs a roughly UPRIGHT hand
        barrel   : needs a roughly THUMB-UP gun

    Base case, calibrated the same empirical way as palm_is_facing():

        RIGHT hand, RAW image, palm to camera, thumb up  ->  x[8] > x[0]

    Turn the hand over and the barrel swings across; a left hand reverses
    it, and mirroring the buffer reverses it once more.

    LIMITATION, stated plainly: roll the wrist until the thumb points down
    and the verdict inverts, because the barrel swings with it while the
    palm does not.  A gun held thumb-down is not a pose this reads
    correctly, and there is no way to tell from x alone that it happened.
    """
    if side not in ("left", "right"):
        return None
    try:
        landmarks = hand.landmark
        reach = landmarks[LM_INDEX_TIP].x - landmarks[LM_WRIST].x
    except (AttributeError, IndexError, TypeError):
        return None
    if not isinstance(reach, float) or reach != reach:
        return None
    if abs(reach) < GUN_DEADBAND:
        return None

    # x[8] > x[0]  ->  palm, for a right hand in a raw image.
    facing = reach > 0.0
    if mirrored:
        facing = not facing
    if side == "left":
        facing = not facing
    return facing


# ── Downward open hand: a pose the network cannot see ─────────────────────
# The model has no class for a flat hand held fingers-down — it returns
# "no_gesture" — so there is nothing for LABEL_ALIASES or the orientation
# override to work on.  This is the one pose recognised from landmarks
# alone, which is why the test is written to be hard to trigger by accident
# rather than easy to trigger at all.
#
# Both thresholds are RELATIVE to the palm, not to the frame.  A fixed
# fraction of frame height would stop working the moment the hand moved
# closer to or further from the camera.
DOWN_MIN_PALM = 0.03     # wrist -> middle MCP drop, as a fraction of height
DOWN_TIP_MARGIN = 0.25   # how far past its PIP a tip must sit, in palm spans


# ── Poses the network confuses because their thumbs agree ─────────────────
# A thumbs-down and a downward open hand share the one feature the model
# leans on hardest: an extended thumb pointing away from a downward hand.
# They differ completely everywhere else, and the difference is not subtle
# — it is four fingers.
#
#       dislike        thumb out, four fingers FOLDED   (a fist)
#       stop_inverse   thumb out, four fingers EXTENDED (an open hand)
#
# So the thumb is exactly the wrong thing to discriminate on, and the
# finger states are exactly the right thing.  detect_gesture() already
# computes them every frame with the radial test — |tip-wrist| against
# |pip-wrist| — which is self-normalising and holds at any hand distance
# and any in-plane rotation, so it stays right where a y-comparison would
# fall over on a tilted hand.
#
# Listed as a tuple because the same argument applies to any class the
# model only ever assigns to a closed hand ("like", "fist"); adding one
# here is a one-word change.
FIST_SHAPED_CLASSES = ("dislike",)

# How many of the four fingers must read extended before a fist-shaped
# prediction is treated as contradicted.  Three, not four, so a single
# mis-tracked fingertip cannot veto the correction — while a real fist,
# which extends none of them, is nowhere near the line.
OPEN_HAND_MIN_FINGERS = 3


def extended_finger_count(fingers_ext) -> int:
    """How many of index/middle/ring/pinky read as extended."""
    try:
        return sum(1 for flag in fingers_ext if flag)
    except TypeError:
        return 0


def contradicts_closed_hand(label, fingers_ext) -> bool:
    """True when *label* claims a fist but the landmarks show an open hand.

    Deliberately one-directional.  It can only ever reject a fist-shaped
    class on a demonstrably open hand; it never invents one, never touches
    a prediction that is not in FIST_SHAPED_CLASSES, and never fires on a
    genuine fist, which extends no fingers at all.
    """
    if label not in FIST_SHAPED_CLASSES:
        return False
    return extended_finger_count(fingers_ext) >= OPEN_HAND_MIN_FINGERS


def _orientation_word(facing):
    """palm / back / ? — for logs, where None must not read as False."""
    if facing is True:
        return "palm"
    if facing is False:
        return "back"
    return "?"


def is_hand_pointing_down(hand) -> bool:
    """True only for an OPEN hand held fingers-down.

    In MediaPipe's normalised space y grows DOWNWARD, so "below" is a
    larger y.  Three conditions, each ruling out a different false
    positive:

      1. the palm itself points down — the middle knuckle sits below the
         wrist by a real margin, which rejects an upright or sideways hand
         before any finger is examined;
      2. every fingertip sits below its own PIP joint by a fraction of that
         palm span, which rejects a fist or a half-curled hand (a curled
         finger folds its tip back up toward the palm);
      3. every fingertip sits below the wrist, which rejects a hand angled
         so far over that the fingers trail behind it.

    All four fingers must agree.  The thumb is deliberately ignored: it
    folds across the palm rather than along it, so it says nothing useful
    about which way the hand is pointing.
    """
    try:
        lm = hand.landmark
        wrist_y = lm[LM_WRIST].y
        palm = lm[LM_MIDDLE_MCP].y - wrist_y
    except (AttributeError, IndexError, TypeError):
        return False

    if not isinstance(palm, float) or palm != palm:
        return False
    if palm < DOWN_MIN_PALM:
        return False

    margin = DOWN_TIP_MARGIN * palm
    for tip, pip in ((LM_INDEX_TIP, LM_INDEX_PIP),
                     (LM_MIDDLE_TIP, LM_MIDDLE_PIP),
                     (LM_RING_TIP, LM_RING_PIP),
                     (LM_PINKY_TIP, LM_PINKY_PIP)):
        try:
            tip_y = lm[tip].y
            if tip_y < lm[pip].y + margin:
                return False
            if tip_y <= wrist_y:
                return False
        except (AttributeError, IndexError, TypeError):
            return False
    return True


def pick_primary_hand(hands, previous_anchor=None):
    """The hand the cursor follows, chosen for continuity across frames.

    Nearest-anchor to last frame's anchor.  With one hand, or on the first
    frame after the hand was lost, this is just the first entry — so the
    single-hand path is bit-identical to what it was.
    """
    hands = _as_hand_list(hands)
    if not hands:
        return None
    if len(hands) == 1 or previous_anchor is None or not PRIMARY_HAND_TRACKING:
        return hands[0]

    prev_x, prev_y = previous_anchor
    best, best_dist = hands[0], None
    for hand in hands:
        anchor = hand.landmark[LM_MIDDLE_MCP]
        dx = anchor.x - prev_x
        dy = anchor.y - prev_y
        dist = dx * dx + dy * dy
        if best_dist is None or dist < best_dist:
            best, best_dist = hand, dist
    return best


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
    if api is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, api)


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
                                   f"on attempt {attempt}/"
                                   f"{CAM_SCAN_READ_TRIES}")
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

    print(f"         any index 0–{CAM_MANUAL_MAX} is accepted, "
          f"detected or not.")

    while True:
        prompt = (
            f"         camera index [Enter = {best}]: "
            if best is not None
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
        self._mouse.click(getattr(self._button, button), count)

    def press(self, button: str = "left") -> None:
        self._mouse.press(getattr(self._button, button))

    def release(self, button: str = "left") -> None:
        self._mouse.release(getattr(self._button, button))

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

    def press(self, button: str = "left") -> None:
        self._backend.press(button)

    def release(self, button: str = "left") -> None:
        self._backend.release(button)

    def close(self) -> None:
        pass


class ActionDispatcher:
    """Turns FSM action strings into OS mouse events, and owns the button.

    The drag state lives here rather than in the FSM because this object is
    the only thing that knows whether the press actually reached the OS.  A
    press that succeeded and a release that failed must leave `dragging`
    True so the shutdown path tries again — a button left held down is the
    one failure in this file that outlives the process and makes the
    desktop unusable.

    Every call is wrapped: cursor output can raise (no libXtst, a revoked
    session) and a gesture must never take the program down with it.  The
    first failure is reported and the rest are silent, because the failure
    mode is per-frame and would otherwise flood the console.
    """

    def __init__(self, cursor, button: str = "left"):
        self._cursor = cursor
        self._button = button
        self.dragging = False
        self.click_count = 0
        self.drag_count = 0
        self.last_action = None
        self.last_action_time = 0.0
        self._error_shown = False
        # Built on first use, for the actions a cursor backend cannot do:
        # right/middle click, scroll wheel, and keyboard macros.  Left alone
        # unless a config actually binds one.
        self._executor = None

    def dispatch(self, action: str, now: float) -> None:
        if action in (LEFT_CLICK, DOUBLE_CLICK):
            if not self._safely(self._cursor.click,
                                button=self._button, count=1):
                return
            self.click_count += 1
        elif action == DRAG_START:
            if self.dragging:
                return
            if not self._safely(self._cursor.press, self._button):
                return
            self.dragging = True
            self.drag_count += 1
        elif action == DRAG_STOP:
            if not self.release():
                return
        elif not self._extended(action, now):
            return

        self.last_action = action
        self.last_action_time = now
        print(f"[action] {action}")

    def _extended(self, action, now: float) -> bool:
        """Everything the cursor backend cannot express, via ActionExecutor.

        The left button deliberately stays on the cursor path above, so the
        drag failsafe still has exactly one owner to ask.  These actions are
        all instantaneous — a click, a scroll tick, a chord pressed and
        released inside a finally — so none of them can leave state behind
        for the shutdown path to clean up.
        """
        if ActionExecutor is None:
            return False
        if self._executor is None:
            try:
                self._executor = ActionExecutor()
            except Exception as exc:
                if not self._error_shown:
                    self._error_shown = True
                    print(f"[action] extended actions unavailable: {exc}")
                return False
        return self._executor.dispatch(action, now)

    def release(self) -> bool:
        """Drop the button if we are holding it.  Safe to call any time.

        `dragging` is cleared only on a release the OS accepted, so a failed
        attempt stays pending for the next caller instead of being forgotten.
        """
        if not self.dragging:
            return False
        if not self._safely(self._cursor.release, self._button):
            return False
        self.dragging = False
        return True

    def _safely(self, fn, *args, **kwargs) -> bool:
        try:
            fn(*args, **kwargs)
            return True
        except Exception as exc:
            if not self._error_shown:
                self._error_shown = True
                print(f"[action] cursor output unavailable: {exc}")
            return False


class MacroDispatcher:
    """Fires OS keyboard macros from the YOLO branch's semantic labels.

    EDGE-TRIGGERED, not level-triggered.  A held gesture fires once: the
    label must become something else before the same macro can fire again.
    The cooldown is a second, independent guard for the case the network
    flickers out of the gesture and straight back into it.

    pynput is imported here rather than at module scope so this file still
    runs without it — the same arrangement PynputCursor uses.  A missing
    package costs the macros and nothing else.
    """

    def __init__(self, macros, cooldown=YOLO_MACRO_COOLDOWN,
                 min_score=YOLO_MACRO_MIN_SCORE):
        from pynput.keyboard import Controller, Key
        self._keyboard = Controller()

        # Modifiers first: the sequence is pressed in order and released in
        # reverse, which is what makes Win+D a chord rather than two taps.
        self._table = {
            "show_desktop": (Key.cmd, "d"),
            "minimise_all": (Key.cmd, "m"),
            "task_view":    (Key.cmd, Key.tab),
            "lock_screen":  (Key.cmd, "l"),
        }
        self._macros = dict(macros)
        self._cooldown = float(cooldown)
        self._min_score = float(min_score)

        self._armed_gesture = None
        self._last_fired_at = float("-inf")
        self._error_shown = False

        self.fired_count = 0
        self.last_macro = None
        self.last_macro_time = 0.0

    @property
    def macro_gestures(self):
        return tuple(self._macros)

    def update(self, gesture, score, now: float) -> bool:
        """One non-blocking evaluation.  True only when a macro fired."""
        macro = self._macros.get(gesture) if gesture else None

        if macro is None:
            self._armed_gesture = None
            return False
        if gesture == self._armed_gesture:
            return False
        if score < self._min_score:
            return False
        if now - self._last_fired_at < self._cooldown:
            return False
        if not self._fire(macro):
            return False

        self._armed_gesture = gesture
        self._last_fired_at = now
        self.fired_count += 1
        self.last_macro = macro
        self.last_macro_time = now
        print(f"[macro] {gesture} ({score:.2f}) -> {macro}")
        return True

    def _fire(self, macro: str) -> bool:
        """Press the chord and release it, whatever happens in between.

        The release runs from a finally over the keys actually pressed, so a
        failure halfway through a chord cannot leave the Windows key held
        down — which would be the keyboard's version of the stuck mouse
        button, and considerably harder to escape.
        """
        sequence = self._table.get(macro)
        if sequence is None:
            if not self._error_shown:
                self._error_shown = True
                print(f"[macro] no such macro: {macro}")
            return False

        pressed = []
        try:
            for key in sequence:
                self._keyboard.press(key)
                pressed.append(key)
            return True
        except Exception as exc:
            if not self._error_shown:
                self._error_shown = True
                print(f"[macro] keyboard unavailable: {exc}")
            return False
        finally:
            for key in reversed(pressed):
                try:
                    self._keyboard.release(key)
                except Exception:
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


# ═══════════════════════════════════════════════════════════════════════════
#  WINDOWS DPI AWARENESS
# ═══════════════════════════════════════════════════════════════════════════
#
# A DPI-unaware process is lied to by Windows: metrics come back divided by
# the primary monitor's scale factor, and windows are bitmap-stretched back
# up by the compositor.
#
# MEASURED HERE, on a 3840×1080 desktop whose primary runs at 125%:
#
#     metric                       unaware      aware
#     SM_CXSCREEN   (primary)      1536×864     1920×1080   <- virtualised
#     SM_CXVIRTUALSCREEN (76–79)   3840×1080    3840×1080   <- NOT virtualised
#     tkinter winfo_vroot*         3840×1080    3840×1080   <- NOT virtualised
#     tkinter winfo_screenwidth    1536×864     1920×1080   <- virtualised
#
# So the mapping path was already immune, and that is not luck: it reads the
# VIRTUAL-screen metrics and winfo_vroot*, never the primary-only ones.
# SetCursorPos was likewise exact to the pixel at every corner in both modes.
#
# Awareness is still worth setting.  It removes a configuration-dependent
# class of bug rather than a bug observed here, and it stops the compositor
# bitmap-scaling the preview window — an unaware 1280×720 preview is blown up
# to 1600×900 on a 125% display and resampled on the way.
#
# Three levels, best first.  Per-Monitor V2 is a CONTEXT, not an awareness
# value: shcore's enum stops at 2 = PER_MONITOR_AWARE, which is V1.  V2 needs
# SetProcessDpiAwarenessContext(-4), and it is the one that keeps reporting
# true pixels when a window crosses between monitors of different scale.

def _dpi_context_v2() -> bool:
    return bool(ctypes.windll.user32.SetProcessDpiAwarenessContext(
        ctypes.c_void_p(-4)))


def _dpi_shcore_v1() -> bool:
    return ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0


def _dpi_legacy_system() -> bool:
    return bool(ctypes.windll.user32.SetProcessDPIAware())


_DPI_STRATEGIES = (
    ("per-monitor-v2", _dpi_context_v2),
    ("per-monitor-v1", _dpi_shcore_v1),
    ("system",         _dpi_legacy_system),
)

_dpi_awareness = None


def ensure_windows_dpi_aware() -> str:
    """Make this process DPI aware, once, and report which level took.

    Idempotent and safe to call from anywhere: Windows only honours the
    first successful call and rejects later ones with E_ACCESSDENIED, so
    the result is cached and every strategy is wrapped.  A no-op on Linux,
    where there is no such virtualisation and shcore does not exist.
    """
    global _dpi_awareness
    if _dpi_awareness is not None:
        return _dpi_awareness

    if not sys.platform.startswith("win"):
        _dpi_awareness = "not-windows"
        return _dpi_awareness

    for label, strategy in _DPI_STRATEGIES:
        try:
            if strategy():
                _dpi_awareness = label
                return _dpi_awareness
        except Exception:
            continue

    # Already set by the host (a launcher, or an embedding app), or simply
    # unavailable on this build.  Either way there is nothing to do and
    # nothing to fail over.
    _dpi_awareness = "unchanged"
    return _dpi_awareness


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
    # Before Tk exists: the toolkit latches the process DPI context when it
    # creates its first window, so setting awareness afterwards would leave
    # this probe reading virtualised numbers.
    ensure_windows_dpi_aware()

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
    SM_XVIRTUALSCREEN = 76   # left edge of the virtual desktop
    SM_YVIRTUALSCREEN = 77   # top  edge of the virtual desktop
    SM_CXVIRTUALSCREEN = 78   # total width
    SM_CYVIRTUALSCREEN = 79   # total height
    SM_CMONITORS = 80   # number of display monitors

    name = "Windows / user32"
    camera_api = CAMERA_API         # None → OpenCV negotiates (see CAMERA_API)

    def __init__(self):
        # FIRST, before a single metric is read.  Awareness is latched per
        # process on first use, so a query made ahead of this call would be
        # answered in virtualised coordinates and cached that way.  This is
        # also why it belongs here rather than in a cursor class: the backend
        # is constructed whatever CURSOR_BACKEND is set to, so the pynput
        # path gets it too.
        self.dpi_awareness = ensure_windows_dpi_aware()

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

    def press(self, button: str = "left") -> None:
        self._u32.mouse_event(self._BTN[button][0], 0, 0, 0, 0)

    def release(self, button: str = "left") -> None:
        self._u32.mouse_event(self._BTN[button][1], 0, 0, 0, 0)

    def click(self, button: str = "left", count: int = 1) -> None:
        # Expressed through press/release so a click and a drag can never
        # disagree about which flag pair means what.
        for _ in range(count):
            self.press(button)
            self.release(button)

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

    _X_BTN = {"left": 1, "middle": 2, "right": 3}

    def _button_event(self, button: str, pressed: bool) -> None:
        """One button event through XTest, if the extension is present.

        XWarpPointer can move the pointer but cannot press anything, so this
        needs libXtst.  Rather than half-implement it, an absent extension
        says so and points at pynput, which carries its own X backend.
        """
        if self._xtest is None:
            raise RuntimeError(
                "clicking on X11 needs libXtst (install libxtst6) or "
                "pynput — set CURSOR_BACKEND='pynput'."
            )
        self._xtest.XTestFakeButtonEvent(
            self._dpy, self._X_BTN[button], pressed, 0)
        self._xlib.XFlush(self._dpy)

    def press(self, button: str = "left") -> None:
        self._button_event(button, True)

    def release(self, button: str = "left") -> None:
        self._button_event(button, False)

    def click(self, button: str = "left", count: int = 1) -> None:
        for _ in range(count):
            self.press(button)
            self.release(button)

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
                 poll_interval: float = POLL_INTERVAL,
                 target: str = ALL_SCREENS):
        self._backend = backend
        self._poll_interval = poll_interval
        self._next_poll = 0.0
        self.monitors = 0

        # Which display the cursor is confined to.  ALL_SCREENS maps onto
        # the union of every monitor; "Screen N" onto that one rectangle.
        # Everything downstream — to_screen, the clamp, the active box — is
        # already written against an arbitrary (left, top, w, h) that may
        # start at a negative origin, so restricting the target needs no
        # changes there at all.
        self.target = ALL_SCREENS
        self.monitor_list = []
        self._requested_target = str(target or ALL_SCREENS)
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
        # A specific screen needs per-monitor rectangles, which the backend
        # does not expose — it only knows the virtual desktop.  Enumeration
        # is 0.24 ms measured, and this runs at the poll interval rather
        # than per frame, so it is affordable here and nowhere else.
        if self._requested_target != ALL_SCREENS:
            monitors = enumerate_monitors(self._backend)
            if monitors:
                self.monitor_list = monitors
                rect, resolved = resolve_monitor_target(
                    self._requested_target, monitors)
                if rect is not None:
                    if resolved != self.target:
                        print(f"[screen] cursor confined to {resolved} "
                              f"({rect[2]}×{rect[3]} at {rect[0]},{rect[1]})")
                        if resolved != self._requested_target:
                            print(f"[screen] "
                                  f"{self._requested_target!r} is "
                                  f"not attached — using "
                                  f"{resolved}")
                    self.target = resolved
                    self.source = "native"
                    return rect

        if self._native_ok:
            try:
                m = self._backend.read_geometry()
            except Exception:
                m = None
            else:
                if m and m[2] > 0 and m[3] > 0:
                    self.source = "native"
                    return m
                # The backend answered, it just answered 0×0 — which is what
                # a driver mid-reconfigure reports.  Once we hold a good
                # rectangle that is a transient to sit out, not a reason to
                # abandon the native source: report "nothing new" and leave
                # _native_ok alone.  Demoting here would silently kill
                # hot-plug detection for the rest of the session and pin the
                # geometry to a cached probe.
                #
                # At construction there is no last-known-good to fall back
                # on, so a degenerate first read does drop through to
                # tkinter rather than leaving the caller with nothing.
                if m is not None and getattr(self, "width", 0) > 0:
                    return None
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

    def set_target(self, target: str) -> bool:
        """Confine the cursor to a display.  True when the rectangle moved.

        Takes effect on the next frame: the rebuild is immediate rather
        than waiting for the poll, because a user picking a screen in the
        GUI expects the cursor to move there now.
        """
        requested = str(target or ALL_SCREENS)
        if requested == self._requested_target:
            return False
        self._requested_target = requested
        self.target = ALL_SCREENS if requested == ALL_SCREENS else self.target

        metrics = self._read_geometry()
        if metrics is None:
            return False
        if metrics == (self.left, self.top, self.width, self.height):
            return False
        self._apply(metrics)
        self._next_poll = 0.0
        return True

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
            self.monitors = (self._backend.monitor_count()
                             if self._native_ok else 1)
        except Exception:
            self.monitors = 1
        self.ratio = self.width / self.height

        # The active box is the frame inset by the margins, in normalised
        # coordinates.  No reference resolution and no aspect lock: the same
        # fractions land on the same relative rectangle whatever the sensor
        # delivers, so a 640×480 laptop cam and a downscaled 640×360 phone
        # feed both give a 76%×76% window at the default 0.12 margins.
        mx, mt, mb = sanitise_margins()

        self.box_left = int(round(mx * self.cam_w))
        self.box_right = int(round((1.0 - mx) * self.cam_w))
        self.box_top = int(round(mt * self.cam_h))
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
        mid_y = (self.box_top + self.box_bottom) / 2.0
        half_w = (self.box_w * live) / 2.0
        half_h = (self.box_h * live) / 2.0
        return (int(mid_x - half_w), int(mid_y - half_h),
                int(mid_x + half_w), int(mid_y + half_h))

    def describe(self) -> str:
        """One-line summary for the console banner / change notices."""
        return (f"{self.width}×{self.height} px  "
                f"origin ({self.left}, {self.top})  "
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
                 proc_max: tuple[int, int] = (PROC_MAX_WIDTH,
                                              PROC_MAX_HEIGHT)):
        self._cap = open_capture(index, api)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)

        # Ask the backend to keep at most one frame queued.  Not all
        # backends implement this; keep the result so startup can say so.
        self.buffersize_accepted = bool(
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))

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
        self.reported_height = int(
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
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
    def _fit_within(w: int, h: int,
                    budget: tuple[int, int]) -> tuple[int, int]:
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


# ─── Threaded YOLO gesture worker ──────────────────────────────────────────
#
# Same shape as WebcamStream: a daemon thread doing the slow thing, talking
# to the main loop only through queues.  submit() and poll() are both
# non-blocking, so the cursor path is unaffected by how long inference takes.
#
# The input queue has depth 1 and REPLACES its contents rather than queueing:
# at 30 FPS in and 4.7 Hz out a backlog would only serve stale hand poses, so
# an unconsumed crop is dropped in favour of the newer one.

class YoloWorker(threading.Thread):

    def __init__(self, model_path, class_names, input_size=YOLO_INPUT_SIZE,
                 score_threshold=YOLO_SCORE_THRESHOLD,
                 fsm=None, name="YoloWorker",
                 debug_crop=YOLO_DEBUG_CROP,
                 debug_crop_path=YOLO_DEBUG_CROP_PATH,
                 log_changes=YOLO_LOG_CHANGES,
                 device=YOLO_DEVICE):
        super().__init__(name=name, daemon=True)
        self._log_changes = bool(log_changes)
        self._model_path = model_path
        self._class_names = tuple(class_names)
        self._input_size = int(input_size)
        self._score_threshold = float(score_threshold)
        self._debug_crop = bool(debug_crop)
        self._debug_crop_path = debug_crop_path
        self._device = device

        # The model is loaded on the WORKER THREAD, not here.  `import
        # ultralytics` costs 10.8 s measured on this machine (it drags in
        # torch and torchvision) and YOLO() a further 0.4 s — eleven seconds
        # of frozen startup before the first camera frame if it happened on
        # the main thread.  Deferring it means the preview opens instantly
        # and the semantic branch simply reports nothing until it is warm,
        # which the HUD and the FSM already handle.
        self._net = None
        self._ready = False
        self._load_error = None
        self._predict_kwargs = {} if device in (None, "auto") else {
            "device": device}
        self._fsm = fsm

        self._inputs = queue.Queue(maxsize=1)
        self._outputs = queue.Queue(maxsize=8)
        self._stopped = False
        self._reset_pending = False

        # One tuple, rebound as a unit, is the whole synchronisation story.
        # Three separate attributes would let the main loop read a gesture
        # from one inference and the score from the next — rare, but it
        # would put a wrong confidence next to a label on the HUD.  A single
        # rebind is atomic under the GIL, so a reader gets either the old
        # triple or the new one and never a mix, with no lock for the
        # cursor path to contend on.
        self._state = (None, 0.0, 0.0)
        self._inference_count = 0
        self._dropped_count = 0

    @property
    def score_threshold(self) -> float:
        """Minimum confidence for a prediction to count as a gesture."""
        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, value) -> None:
        """Safe to set from any thread: _infer reads it per inference,
        and a float rebind is atomic under the GIL."""
        try:
            self._score_threshold = float(value)
        except (TypeError, ValueError):
            pass

    @property
    def current_state(self):
        """(gesture, score, latency_ms) — one consistent snapshot."""
        return self._state

    @property
    def current_gesture(self):
        return self._state[0]

    @property
    def current_score(self):
        return self._state[1]

    @property
    def current_latency_ms(self):
        return self._state[2]

    last_gesture = current_gesture
    last_score = current_score
    last_latency_ms = current_latency_ms

    @property
    def ready(self):
        """True once the model is loaded and inference is actually running."""
        return self._ready

    @property
    def load_error(self):
        return self._load_error

    @property
    def inference_count(self):
        return self._inference_count

    @property
    def dropped_count(self):
        return self._dropped_count

    def submit(self, crop):
        if self._stopped or crop is None or crop.size == 0:
            return False
        try:
            self._inputs.put_nowait(crop)
            return True
        except queue.Full:
            try:
                self._inputs.get_nowait()
                self._dropped_count += 1
            except queue.Empty:
                pass
            try:
                self._inputs.put_nowait(crop)
                return True
            except queue.Full:
                return False

    def poll(self):
        try:
            return self._outputs.get_nowait()
        except queue.Empty:
            return None

    def request_reset(self):
        self._reset_pending = True

    def _load_model(self) -> bool:
        """Import ultralytics and build the model, on this thread.

        Both steps are slow and both can fail; neither may take the cursor
        down with it.  A failure here parks the thread and leaves the rest
        of the program exactly as it would be with the branch disabled.
        """
        # Already supplied — an injected model, or a restarted thread.
        if self._net is not None:
            self._ready = True
            return True

        started = time.perf_counter()
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self._load_error = (f"ultralytics not importable "
                                f"({exc.__class__.__name__}: {exc})")
            print(f"[yolo] {self._load_error} — semantic branch off, "
                  f"cursor unaffected")
            return False

        try:
            net = YOLO(self._model_path)
        except Exception as exc:
            self._load_error = f"{exc.__class__.__name__}: {exc}"
            print(f"[yolo] could not load {self._model_path} "
                  f"({self._load_error}) — semantic branch off")
            return False

        # Prefer the model's own class table over the hardcoded one: it is
        # the export's ground truth, so a re-trained model with a different
        # ordering cannot silently mislabel every gesture.  Verified equal
        # to YOLO_CLASS_NAMES for this export.
        names = getattr(net, "names", None)
        if isinstance(names, dict) and names:
            self._class_names = tuple(names[k] for k in sorted(names))
        elif names:
            self._class_names = tuple(names)

        self._net = net
        self._ready = True
        print(f"[yolo] {os.path.basename(self._model_path)} ready in "
              f"{time.perf_counter() - started:.1f}s "
              f"({len(self._class_names)} classes, device={self._device})")
        return True

    def run(self):
        if not self._load_model():
            return

        while not self._stopped:
            try:
                crop = self._inputs.get(timeout=0.05)
            except queue.Empty:
                if self._reset_pending:
                    self._apply_reset()
                continue

            if self._reset_pending:
                self._apply_reset()

            try:
                gesture, score, latency_ms = self._infer(crop)
            except Exception:
                continue

            previous = self._state[0]
            self._state = (gesture, score, latency_ms)
            self._inference_count += 1

            if self._log_changes and gesture != previous:
                print(f"[yolo] {previous or '-'} -> "
                      f"{gesture or '-'}   ({score:.2f}, {latency_ms:.0f} ms)")

            if self._fsm is None:
                continue

            _stable, trigger = self._fsm.update_pair(gesture,
                                                     time.perf_counter())
            if trigger is not None:
                try:
                    self._outputs.put_nowait(trigger)
                except queue.Full:
                    pass

    def stop(self):
        self._stopped = True
        if self.is_alive():
            self.join(timeout=2.0)

    def _apply_reset(self):
        self._reset_pending = False
        if self._fsm is not None:
            self._fsm.reset()
        self._state = (None, 0.0, 0.0)
        while True:
            try:
                self._inputs.get_nowait()
            except queue.Empty:
                break

    def _infer(self, crop):
        if self._debug_crop:
            try:
                cv2.imwrite(self._debug_crop_path, crop)
            except Exception:
                pass

        # Ultralytics owns the whole preprocess: letterbox, BGR→RGB, the
        # /255 scale and the NHWC→NCHW transpose all happen inside predict().
        # The frame goes in as raw BGR.
        started = time.perf_counter()
        results = self._net.predict(crop, imgsz=self._input_size,
                                    verbose=False, **self._predict_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0, latency_ms

        # argmax rather than boxes[0]: ultralytics usually returns the most
        # confident detection first, but "usually" is not a contract, and
        # taking the wrong row would mislabel the gesture rather than fail.
        confidences = boxes.conf
        best = int(confidences.argmax())
        score = float(confidences[best])
        if score < self._score_threshold:
            return None, score, latency_ms

        class_id = int(boxes.cls[best])
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id], score, latency_ms
        return None, score, latency_ms


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
#  TRACKING ENGINE
# ═══════════════════════════════════════════════════════════════════════════
#
# Everything above this line is definitions and costs nothing to import.
# Everything below is the runtime, and it used to run at module scope: the
# camera opened, MediaPipe loaded and a blocking `while True` with
# cv2.imshow took the thread, so `import hand_cursor_2` from a GUI would
# never return.
#
# It is now a class.  The loop body is unchanged — same anchor, same
# gesture classifier, same FSM arbitration, same 1€ filter, same overlays —
# but it lives on a background thread and publishes its frame instead of
# showing it.  cv2.imshow and cv2.waitKey are gone entirely; a host that
# wants a preview asks for get_latest_frame() and draws it however it likes.
#
# The standalone script still exists, at the bottom, and behaves as it
# always did: it drives the same engine and does the imshow/waitKey itself.

# Attribute lookups on an already-imported package: no graph is built and
# no model is loaded here, so these stay at module scope where both the
# engine and any external caller can reach them.  The expensive part —
# mp_hands.Hands(...) — happens in _ensure_mediapipe(), on start().
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class HandTrackerEngine:
    """The tracking pipeline, on its own thread, with no window of its own.

    Lifecycle:

        engine = HandTrackerEngine()
        engine.start(camera_index=0)     # opens hardware, starts the thread
        frame = engine.get_latest_frame()  # newest BGR frame, or None
        engine.stop()                    # releases the camera, keeps the rest
        engine.close()                   # releases everything

    start() is safe to call again with a different index — it stops the
    current capture first — which is what makes camera switching from a GUI
    work without leaking a device.

    The three direction/speed settings stay MODULE-level globals because
    ScreenGeometry.to_screen() reads them by name on every call; the setter
    methods below rebind them, which is what makes a change take effect on
    the very next frame with no object to keep in sync.
    """

    def __init__(self, *, enable_yolo=YOLO_ENABLED,
                 enable_macros=YOLO_MACRO_ENABLED,
                 enable_monitor=False, verbose=True):
        self.verbose = bool(verbose)
        self._enable_yolo = bool(enable_yolo)
        self._enable_macros = bool(enable_macros)
        self._enable_monitor = bool(enable_monitor)

        # Pin OpenCV's internal thread pool before anything uses it.
        if OPENCV_THREADS:
            cv2.setNumThreads(OPENCV_THREADS)

        # Cheap, hardware-free setup.  Constructing the engine must not
        # open a camera, so a GUI can build one before the user has chosen
        # which device to connect to.
        self.backend = make_backend()
        self.cursor = make_cursor(self.backend)
        self.actions = ActionDispatcher(self.cursor)

        self.stream = None
        self.screen = None
        self.hands = None
        self.fsm = None
        self.semantic_fsm = None
        self.macros = None
        self.yolo_worker = None
        self.monitor_process = None
        self.camera_index = None

        self._thread = None
        self._stop_event = threading.Event()

        # Set by the ai_confidence setter, consumed by the loop thread.
        self._ai_confidence = AI_CONFIDENCE_DEFAULT
        self._mp_rebuild = threading.Event()

        # Raised when the mapping rectangle moves under the cursor — a
        # screen switch, today.  Consumed by the loop thread, which owns
        # the 1€ filters; nothing else may touch them while it is running.
        self._filter_reset = threading.Event()

        # Keys already reported by _log_once, so a per-frame failure
        # prints once instead of thousands of times.
        self._reported = set()

        # Published by the loop, read by whoever wants a preview.  A single
        # attribute rebind is atomic under the GIL, so a reader gets either
        # the previous frame or the new one and never a half-written array
        # — the same lock-free arrangement WebcamStream uses.
        self._latest_frame = None

        self.status = {
            "running": False, "camera_index": None, "fps": 0.0,
            "gesture": None, "stable_gesture": None, "dragging": False,
            "clicks": 0, "drags": 0, "yolo": None, "yolo_score": 0.0,
            "yolo_ms": 0.0, "error": None,
        }

        self._apply_config()

    # ── configuration ───────────────────────────────────────────────────

    def _apply_config(self) -> None:
        """Build the FSM and adopt the cursor settings from the config."""
        global CURSOR_SENSITIVITY, IS_MIRRORED, INVERT_CURSOR_X

        if not _FSM_AVAILABLE:
            self._log("[action] gesture_fsm.py not importable — movement only")
            return

        cfg = load_config()
        self.fsm = GestureFSM(cfg)

        # Same rules, separate state.  The geometric stream runs at the
        # camera rate and the semantic one at the network rate, so they
        # need one stabiliser each; sharing would let a 5 Hz label
        # outvote a 30 Hz one inside the same window.
        self.semantic_fsm = GestureFSM(cfg)

        settings = cfg.get("settings") or {}
        try:
            CURSOR_SENSITIVITY = clamp(
                float(settings.get("cursor_sensitivity", CURSOR_SENSITIVITY)),
                SENS_MIN, SENS_MAX)
        except (TypeError, ValueError):
            self._log("[config] cursor_sensitivity is not a number — keeping "
                      f"{CURSOR_SENSITIVITY}")

        try:
            self._ai_confidence = round(clamp(
                float(settings.get("ai_confidence", self._ai_confidence)),
                AI_CONFIDENCE_MIN, AI_CONFIDENCE_MAX), 2)
        except (TypeError, ValueError):
            self._log(f"[config] ai_confidence is not a number — keeping "
                      f"{self._ai_confidence}")

        IS_MIRRORED = bool(settings.get("is_mirrored", IS_MIRRORED))
        INVERT_CURSOR_X = bool(settings.get("invert_cursor_x",
                                            INVERT_CURSOR_X))

        if IS_MIRRORED == INVERT_CURSOR_X:
            self._log(f"[config] is_mirrored and invert_cursor_x are both "
                      f"{str(IS_MIRRORED).upper()} — these cancel out and the "
                      f"cursor will run backwards. Turn exactly one on.")

        self._log("[action] config-driven FSM running — "
                  f"{self.fsm.describe()}")
        for rule in self.fsm.rules:
            self._log(f"          {rule.name:<28} -> {rule.action}")

    def reload_config(self) -> None:
        """Re-read gesture_config.json without restarting the camera."""
        self._apply_config()

    # ── lifecycle ───────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, camera_index: int = 0) -> bool:
        """Open the camera and begin tracking.  True if the thread started.

        Stops any capture already running first, so switching cameras from
        a GUI cannot leave the previous device held open.
        """
        self.stop()
        self._stop_event.clear()
        self.status["error"] = None

        try:
            self.stream = WebcamStream(int(camera_index), CAM_WIDTH,
                                       CAM_HEIGHT, CAM_FPS,
                                       api=self.backend.camera_api)
        except Exception as exc:
            self.stream = None
            message = f"{exc.__class__.__name__}: {exc}"
            self.status["error"] = message
            self._log(f"[camera] index {camera_index} could not be opened: "
                      f"{message}")
            return False

        self.camera_index = int(camera_index)
        self.status["camera_index"] = self.camera_index

        if self.stream.downscaled:
            self._log(f"[camera] native {self.stream.native_width}×"
                      f"{self.stream.native_height} → processing at "
                      f"{self.stream.width}×{self.stream.height}")
        else:
            self._log(f"[camera] {self.stream.width}×{self.stream.height} "
                      f"native, no rescale needed")

        self.screen = ScreenGeometry(
            self.backend, self.stream.width, self.stream.height,
            POLL_INTERVAL,
            target=getattr(self, "_pending_target", ALL_SCREENS))
        self._ensure_mediapipe()
        self._ensure_macros()
        self._ensure_yolo()
        self._ensure_monitor()

        self._thread = threading.Thread(target=self._run, name="HandTracker",
                                        daemon=True)
        self._thread.start()
        self.status["running"] = True
        return True

    def stop(self) -> None:
        """Release the camera and stop the loop.  Safe to call any time.

        The mouse button is dropped FIRST.  A drag interrupted by a camera
        switch has no gesture left to end it, and a button still held after
        the loop exits leaves the desktop selecting text.
        """
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self.actions.release():
            self._log("[action] DRAG_STOP (engine stopped)")
        for machine in (self.fsm, self.semantic_fsm):
            if machine is not None:
                machine.reset()

        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            self.stream = None

        self._latest_frame = None
        self.status["running"] = False
        self.status["camera_index"] = None
        self.status["fps"] = 0.0

    def close(self) -> None:
        """Full teardown: camera, model, worker, monitor, cursor, backend."""
        self.stop()

        if self.yolo_worker is not None:
            try:
                self.yolo_worker.stop()
            except Exception:
                pass
            self.yolo_worker = None

        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
            self.hands = None

        if self.monitor_process is not None and \
                self.monitor_process.poll() is None:
            try:
                self.monitor_process.terminate()
                self.monitor_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
            except Exception:
                pass
            self.monitor_process = None

        for closer in (self.cursor, self.backend):
            try:
                closer.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    # ── frame hand-off ──────────────────────────────────────────────────

    def get_latest_frame(self):
        """Newest BGR frame with overlays drawn, or None before the first.

        Thread-safe without a lock: the loop publishes by rebinding one
        attribute, which is atomic under the GIL, and the array it points
        at is never written to again.
        """
        return self._latest_frame

    # ── live settings ───────────────────────────────────────────────────
    # cursor_speed / is_mirrored / invert_x are properties over the
    # module-level names rather than plain instance state, and that is the
    # point: ScreenGeometry.to_screen() looks CURSOR_SENSITIVITY and
    # INVERT_CURSOR_X up by name on EVERY call, the capture loop reads
    # IS_MIRRORED on every frame, and the standalone preview's
    # '+'/'-'/'m'/'i' keys rebind the same three names.
    #
    # Proxying keeps one source of truth, so a GUI slider and a keypress
    # can never disagree about the current value.  Assigning any of them
    # takes effect on the very next frame — there is nothing to restart
    # and no copy to invalidate.

    @property
    def ai_confidence(self) -> float:
        """Detection/tracking threshold shared by MediaPipe and YOLO."""
        return self._ai_confidence

    @ai_confidence.setter
    def ai_confidence(self, value) -> None:
        """Apply a new threshold to both models, live.

        YOLO takes it immediately — the worker reads its threshold on
        every inference.  MediaPipe cannot: min_detection_confidence and
        min_tracking_confidence are constructor arguments, so the graph
        has to be rebuilt.  That is flagged here and done at the top of
        the next loop iteration, on the loop's own thread, because
        MediaPipe objects are not safe to swap underneath a call in
        flight.
        """
        try:
            new = float(value)
        except (TypeError, ValueError):
            return
        new = round(clamp(new, AI_CONFIDENCE_MIN, AI_CONFIDENCE_MAX), 2)
        if abs(new - self._ai_confidence) < 1e-9:
            return

        self._ai_confidence = new
        if self.yolo_worker is not None:
            self.yolo_worker.score_threshold = new
        self._mp_rebuild.set()
        self._log(f"[ai] confidence -> {new:.2f}")

    @property
    def cursor_speed(self) -> float:
        """Cursor movement multiplier, clamped to [SENS_MIN, SENS_MAX]."""
        return CURSOR_SENSITIVITY

    @cursor_speed.setter
    def cursor_speed(self, value) -> None:
        self.set_sensitivity(value)

    @property
    def is_mirrored(self) -> bool:
        """Flips the captured frame, and with it the control direction."""
        return IS_MIRRORED

    @is_mirrored.setter
    def is_mirrored(self, value) -> None:
        global IS_MIRRORED
        IS_MIRRORED = bool(value)

    @property
    def invert_x(self) -> bool:
        """Reflects the control path only; the preview is left alone."""
        return INVERT_CURSOR_X

    @invert_x.setter
    def invert_x(self, value) -> None:
        global INVERT_CURSOR_X
        INVERT_CURSOR_X = bool(value)

    def apply_settings(self, cursor_speed=None, is_mirrored=None,
                       invert_x=None, ai_confidence=None,
                       target_screen=None) -> None:
        """Set any combination of these in one call.

        Each is optional so a caller can push just the one that changed;
        None means "leave this one alone".
        """
        if cursor_speed is not None:
            self.cursor_speed = cursor_speed
        if is_mirrored is not None:
            self.is_mirrored = is_mirrored
        if invert_x is not None:
            self.invert_x = invert_x
        if ai_confidence is not None:
            self.ai_confidence = ai_confidence
        if target_screen is not None:
            self.target_screen = target_screen

    def _reset_filters(self) -> None:
        """Make the cursor SNAP to its next target instead of gliding there.

        Safe to call from any thread, which is the whole reason it is an
        event rather than a direct assignment: the smoothing state lives in
        two OneEuroFilter objects owned by the loop thread, and clearing
        them from a GUI callback mid-frame would corrupt a filter that is
        part-way through an update.  The flag is raised here and acted on
        at the top of the next iteration.

        Needed whenever the mapping rectangle moves beneath a stationary
        hand — confining the cursor to one screen rewrites the whole
        box→desktop map, so the filters hold a position that no longer
        means anything.  Without this the cursor sweeps across the desktop
        to reach its new home instead of simply appearing there.
        """
        self._filter_reset.set()

    @property
    def target_screen(self) -> str:
        """Which display the cursor is confined to, as the resolved label.

        Reads back what actually took effect, not what was asked for — so
        a request for a screen that is not attached reports the fallback.
        """
        screen = getattr(self, "screen", None)
        if screen is not None:
            return screen.target
        return getattr(self, "_pending_target", ALL_SCREENS)

    @target_screen.setter
    def target_screen(self, value) -> None:
        label = str(value or ALL_SCREENS)
        self._pending_target = label
        screen = getattr(self, "screen", None)
        if screen is None:
            return          # picked up when the geometry is built
        try:
            if screen.set_target(label):
                # The mapping rectangle moved, so the filters hold a
                # position from the old screen.  Clearing them makes the
                # next frame adopt the new target verbatim instead of
                # gliding across the desktop to reach it.
                self._reset_filters()
        except Exception as exc:
            self._log(f"[screen] could not switch to {label}: "
                      f"{exc.__class__.__name__}: {exc}")

    # ── runtime controls (the keys the standalone preview binds) ────────

    def adjust_sensitivity(self, delta: float) -> float:
        global CURSOR_SENSITIVITY
        CURSOR_SENSITIVITY = round(
            clamp(CURSOR_SENSITIVITY + delta, SENS_MIN, SENS_MAX), 2)
        return CURSOR_SENSITIVITY

    def set_sensitivity(self, value: float) -> float:
        global CURSOR_SENSITIVITY
        CURSOR_SENSITIVITY = round(clamp(float(value), SENS_MIN, SENS_MAX), 2)
        return CURSOR_SENSITIVITY

    def toggle_mirror(self) -> bool:
        global IS_MIRRORED
        IS_MIRRORED = not IS_MIRRORED
        self._log(f"[mirror] preview {'MIRRORED' if IS_MIRRORED else 'RAW'}")
        return IS_MIRRORED

    def toggle_invert(self) -> bool:
        global INVERT_CURSOR_X
        INVERT_CURSOR_X = not INVERT_CURSOR_X
        self._log(f"[invert] control-only X inversion "
                  f"{'ON' if INVERT_CURSOR_X else 'OFF'}")
        return INVERT_CURSOR_X

    def handle_key(self, key: int) -> bool:
        """Route one cv2.waitKey code.  True means 'quit' was pressed."""
        return handle_key(key)

    # ── subsystem construction ──────────────────────────────────────────

    def _ensure_mediapipe(self) -> None:
        if self.hands is not None:
            return
        confidence = clamp(float(self._ai_confidence),
                           AI_CONFIDENCE_MIN, AI_CONFIDENCE_MAX)
        kwargs = dict(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        # The noisy moment: four TFLite/absl lines are written to fd 2 by
        # C++ during the FIRST process() call, not the constructor, so the
        # warm-up below is what holds the suppression window open long
        # enough to catch them.  It earns its place anyway by building the
        # graph here instead of stalling the first real frame.
        with quiet_stderr():
            try:
                self.hands = mp_hands.Hands(
                    model_complexity=MODEL_COMPLEXITY, **kwargs)
            except TypeError:
                self.hands = mp_hands.Hands(**kwargs)
            self.hands.process(
                np.zeros((INFER_WARMUP_H, INFER_WARMUP_W, 3), np.uint8))

    def _rebuild_mediapipe(self) -> bool:
        """Swap in a graph built at the current confidence.  True on success.

        Costs the same second the first build does, so it is only reached
        when the value actually changed.  The old graph is closed after
        the new one is up: a failed build leaves the working one in place
        rather than dropping tracking on the floor.
        """
        previous = self.hands
        self.hands = None
        try:
            self._ensure_mediapipe()
        except Exception as exc:
            self.hands = previous
            self._log(f"[ai] could not rebuild MediaPipe at "
                      f"{self._ai_confidence:.2f} ({exc}) — keeping the "
                      f"previous graph")
            return False

        if previous is not None:
            try:
                previous.close()
            except Exception:
                pass
        self._log(f"[ai] MediaPipe rebuilt at {self._ai_confidence:.2f}")
        return True

    def _ensure_macros(self) -> None:
        if self.macros is not None or not self._enable_macros:
            return
        if not YOLO_MACROS:
            return
        try:
            self.macros = MacroDispatcher(YOLO_MACROS)
        except Exception as exc:
            self.macros = None
            self._log(f"[macro] keyboard macros unavailable "
                      f"({exc.__class__.__name__}: {exc})")

    def _ensure_yolo(self) -> None:
        if self.yolo_worker is not None or not self._enable_yolo:
            return
        if not os.path.exists(YOLO_MODEL_PATH):
            self._log(f"[yolo] model not found at {YOLO_MODEL_PATH} — "
                      f"cursor unaffected")
            return
        try:
            self.yolo_worker = YoloWorker(
                YOLO_MODEL_PATH, YOLO_CLASS_NAMES, fsm=None,
                score_threshold=self._ai_confidence)
            self.yolo_worker.start()
            self._log("[yolo] semantic branch starting in the background")
        except Exception as exc:
            self.yolo_worker = None
            self._log(f"[yolo] unavailable ({exc.__class__.__name__}: {exc})")

    def _ensure_monitor(self) -> None:
        if self.monitor_process is not None or not self._enable_monitor:
            return
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "system_monitor.py")
        try:
            self.monitor_process = subprocess.Popen(
                [sys.executable, script, str(os.getpid())])
            self._log(f"[monitor] resource window started "
                      f"(pid {self.monitor_process.pid})")
        except Exception as exc:
            self.monitor_process = None
            self._log(f"[monitor] could not start: "
                      f"{exc.__class__.__name__}: {exc}")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _log_once(self, key: str, message: str) -> None:
        """Report a recurring per-frame failure exactly once.

        The failures this guards are all per-frame by nature — a macro that
        cannot be pressed fails on every frame the pose is held — so
        printing each time would bury the first occurrence under thousands
        of identical lines and slow the loop doing it.
        """
        if key in self._reported:
            return
        self._reported.add(key)
        print(message)

    # ── the loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        """The tracking loop.  Identical to the original, minus the window.

        Everything here is what the module-level `while True` used to do.
        The two differences are at the ends: there is no cv2.waitKey to
        pump a GUI or yield the CPU, so a duplicate frame sleeps briefly
        instead; and the finished frame is published rather than shown.
        """
        stream = self.stream
        screen = self.screen
        hands = self.hands
        fsm = self.fsm
        semantic_fsm = self.semantic_fsm
        actions = self.actions
        yolo_worker = self.yolo_worker
        macros = self.macros

        oef_x = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA,
                              d_cutoff=D_CUTOFF)
        oef_y = OneEuroFilter(freq=30.0, min_cutoff=MIN_CUTOFF, beta=BETA,
                              d_cutoff=D_CUTOFF)

        hand_present = False
        was_mirrored = IS_MIRRORED
        was_inverted = INVERT_CURSOR_X

        gesture_state = None
        gesture = "idle"
        fingers_ext = (False, False, False, False)
        thumb_out = False
        frozen_target = None
        freeze_until = 0.0
        gesture_changes = 0
        stable_gesture = None
        primary_anchor = None
        hand_side = ""            # "left" / "right" / "" when unknown
        hand_facing = None        # True palm, False back, None unknown
        gun_facing = None         # same, read off the barrel (three_gun)
        hand_down = False         # open hand held fingers-down
        last_semantic = ""        # for the change-only label log

        yolo_action_count = 0
        last_seq = -1

        bgr_buf = None
        infer_bgr = None
        rgb_buf = None
        infer_w = infer_h = 0
        infer_div = 1
        infer_interp = cv2.INTER_AREA
        frame_h = frame_w = 0

        prev_x, prev_y = screen.center
        prev_time = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                success, raw_frame, seq = stream.read()
                if not success:
                    time.sleep(0.001)
                    continue

                # ── Duplicate frame guard ───────────────────────────────
                # The loop can outrun the camera.  Re-running MediaPipe on
                # a frame already processed costs a full inference for an
                # identical answer.  The original yielded here through
                # cv2.waitKey(1); with no window there is nothing to pump,
                # so the sleep is what keeps this from spinning a core.
                if seq == last_seq:
                    time.sleep(0.001)
                    continue
                last_seq = seq

                now_ts = time.perf_counter()

                # ── Live AI-confidence change ───────────────────────────
                # Done here, on this thread, because MediaPipe must not
                # be swapped out from under a process() call in flight.
                if self._mp_rebuild.is_set():
                    self._mp_rebuild.clear()
                    if self._rebuild_mediapipe():
                        hands = self.hands
                        hand_present = False

                # Lowering hand_present routes this frame through the same
                # re-entry path a returning hand takes, which clears both
                # filters and adopts the raw target verbatim.
                if self._filter_reset.is_set():
                    self._filter_reset.clear()
                    hand_present = False

                # ── Display hot-plug polling ────────────────────────────
                if screen.poll(now_ts):
                    self._log(f"[display] geometry changed → "
                              f"{screen.describe()}")
                    hand_present = False

                # ── Buffer preparation (allocation-free steady state) ───
                if bgr_buf is None or bgr_buf.shape != raw_frame.shape:
                    bgr_buf = np.empty_like(raw_frame)
                    frame_h, frame_w = raw_frame.shape[:2]

                    infer_w, infer_h, infer_div, infer_interp = \
                        pick_infer_size(frame_w, frame_h)
                    if infer_div > 1:
                        infer_bgr = np.empty((infer_h, infer_w, 3),
                                             raw_frame.dtype)
                        rgb_buf = np.empty((infer_h, infer_w, 3),
                                           raw_frame.dtype)
                        self._log(f"[infer] MediaPipe fed {infer_w}×{infer_h} "
                                  f"(1/{infer_div} of {frame_w}×{frame_h})")
                    else:
                        infer_bgr = None
                        rgb_buf = np.empty_like(raw_frame)
                        self._log(f"[infer] MediaPipe fed the full "
                                  f"{frame_w}×{frame_h} frame")

                    if screen.set_camera_size(frame_w, frame_h):
                        self._log(f"[camera] frame size now {frame_w}×"
                                  f"{frame_h} → box rebuilt to "
                                  f"{screen.box_w}×{screen.box_h}")
                        hand_present = False

                # ── Mirror, or don't ────────────────────────────────────
                if IS_MIRRORED:
                    cv2.flip(raw_frame, 1, dst=bgr_buf)
                else:
                    np.copyto(bgr_buf, raw_frame)

                # Either toggle reflects the mapped position instantly, so
                # both route the next frame through the re-entry reset and
                # the cursor snaps rather than gliding across the desktop.
                if IS_MIRRORED != was_mirrored or \
                        INVERT_CURSOR_X != was_inverted:
                    was_mirrored = IS_MIRRORED
                    was_inverted = INVERT_CURSOR_X
                    hand_present = False

                # ── Feed MediaPipe ──────────────────────────────────────
                if infer_bgr is not None:
                    cv2.resize(bgr_buf, (infer_w, infer_h), dst=infer_bgr,
                               interpolation=infer_interp)
                    cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB, dst=rgb_buf)
                else:
                    cv2.cvtColor(bgr_buf, cv2.COLOR_BGR2RGB, dst=rgb_buf)

                rgb_buf.flags.writeable = False
                results = hands.process(rgb_buf)
                rgb_buf.flags.writeable = True

                if results.multi_hand_landmarks:
                    detected_hands = results.multi_hand_landmarks
                    hand = pick_primary_hand(detected_hands, primary_anchor)
                    hand_side = handedness_of(results, hand, IS_MIRRORED)
                    hand_facing = palm_is_facing(hand, hand_side,
                                                 IS_MIRRORED)
                    # Read here, not in the label block: that block sits
                    # outside this branch and `hand` is stale there on a
                    # frame with no skeleton.  Two float reads, and only
                    # three_gun ever looks at the answer.
                    gun_facing = gun_is_facing(hand, hand_side, IS_MIRRORED)
                    hand_down = is_hand_pointing_down(hand)

                    anchor = hand.landmark[LM_MIDDLE_MCP]
                    primary_anchor = (anchor.x, anchor.y)

                    raw_x = anchor.x * screen.cam_w
                    raw_y = anchor.y * screen.cam_h

                    gesture, fingers_ext, thumb_out = detect_gesture(
                        hand, screen.cam_w, screen.cam_h)

                    if yolo_worker is not None:
                        yolo_worker.submit(bgr_buf.copy())

                    target_x, target_y = screen.to_screen(raw_x, raw_y)

                    if not hand_present:
                        oef_x.reset()
                        oef_y.reset()
                        prev_x, prev_y = target_x, target_y
                        hand_present = True
                        gesture_state = None
                        frozen_target = None
                        freeze_until = 0.0

                    if gesture != gesture_state:
                        previous = gesture_state
                        gesture_state = gesture
                        frozen_target = (target_x, target_y)
                        freeze_until = now_ts + STATE_FREEZE_MS / 1000.0
                        gesture_changes += 1
                        self._log(f"[gesture] {previous or '-'} -> {gesture}"
                                  f"   (hold {STATE_FREEZE_MS} ms)")

                    # ── Gesture actions: FENCED OFF from the cursor ─────
                    # A custom binding can name a macro pynput cannot
                    # press, or an action a backend refuses.  That must
                    # cost the action and nothing else, so the whole
                    # arbitration block is caught here rather than at the
                    # loop's outer try — which would end the run.  The
                    # cursor maths below is deliberately outside it, so a
                    # broken mapping cannot skip a single frame of
                    # movement.
                    if fsm is not None and gesture not in FSM_IGNORED_STATES:
                        try:
                            stable_gesture, action = fsm.update_pair(
                                gesture, now_ts)
                            if action is not None:
                                actions.dispatch(action, now_ts)
                        except Exception as exc:
                            self._log_once(
                                "action",
                                f"[action] binding failed, tracking "
                                f"continues ({exc.__class__.__name__}: "
                                f"{exc})")

                    if frozen_target is not None:
                        if now_ts < freeze_until:
                            target_x, target_y = frozen_target
                        else:
                            frozen_target = None

                    smooth_x = oef_x(target_x, timestamp=now_ts)
                    smooth_y = oef_y(target_y, timestamp=now_ts)

                    # Cursor output can fail too (a revoked session, no
                    # libXtst).  Losing a frame of movement is survivable;
                    # losing the thread is not.
                    try:
                        self.cursor.move(int(smooth_x), int(smooth_y))
                    except Exception as exc:
                        self._log_once(
                            "cursor",
                            f"[cursor] move failed ({exc.__class__.__name__}"
                            f": {exc})")

                    # Kept as the reference point the re-entry reset snaps
                    # to; the per-frame travel readout it used to feed was
                    # part of the removed text block.
                    prev_x = smooth_x
                    prev_y = smooth_y

                    # ── Overlays ────────────────────────────────────────
                    mp_draw.draw_landmarks(bgr_buf, hand,
                                           mp_hands.HAND_CONNECTIONS)

                    cx, cy = int(raw_x), int(raw_y)
                    _frozen = (frozen_target is not None
                               and now_ts < freeze_until)
                    _anchor_col = (0, 165, 255) if _frozen else (0, 255, 0)
                    cv2.circle(bgr_buf, (cx, cy), 11, _anchor_col, cv2.FILLED)
                    cv2.circle(bgr_buf, (cx, cy), 15, _anchor_col, 2)

                    _it = hand.landmark[LM_INDEX_TIP]
                    cv2.circle(bgr_buf,
                               (int(_it.x * screen.cam_w),
                                int(_it.y * screen.cam_h)),
                               5, (200, 200, 200), 1)

                else:
                    hand_present = False
                    # FAILSAFE, and the order matters: drop the button
                    # BEFORE clearing the FSM.  The hand is gone, so no
                    # further transition is coming to end the drag.
                    if actions.release():
                        self._log("[action] DRAG_STOP (hand lost)")
                    for _machine in (fsm, semantic_fsm):
                        if _machine is not None:
                            _machine.reset()
                    stable_gesture = None
                    primary_anchor = None
                    hand_side = ""
                    hand_facing = None
                    gun_facing = None
                    hand_down = False
                    if yolo_worker is not None:
                        yolo_worker.request_reset()
                    if gesture_state is not None:
                        self._log(f"[gesture] {gesture_state} -> (hand lost)")
                    gesture_state = None
                    gesture = "idle"
                    fingers_ext = (False, False, False, False)
                    thumb_out = False
                    frozen_target = None
                    freeze_until = 0.0

                # ── YOLO output drain (non-blocking) ────────────────────
                if yolo_worker is not None:
                    while True:
                        queued = yolo_worker.poll()
                        if queued is None:
                            break
                        yolo_action_count += 1

                # ── Semantic macros (slow path, edge-triggered) ─────────
                # Same fence as the gesture actions: a macro that cannot be
                # pressed costs the macro, not the run.
                if macros is not None and yolo_worker is not None:
                    try:
                        _mg, _ms, _ = yolo_worker.current_state
                        macros.update(_mg, _ms, now_ts)
                    except Exception as exc:
                        self._log_once(
                            "macro",
                            f"[macro] failed, tracking continues "
                            f"({exc.__class__.__name__}: {exc})")

                # ── Boxes ───────────────────────────────────────────────
                cv2.rectangle(bgr_buf,
                              (screen.box_left, screen.box_top),
                              (screen.box_right, screen.box_bottom),
                              (255, 0, 255), 2)

                _eff = screen.effective_box
                if _eff is not None:
                    cv2.rectangle(bgr_buf, (_eff[0], _eff[1]),
                                  (_eff[2], _eff[3]), (0, 165, 255), 1)

                # ── Per-frame state ─────────────────────────────────────
                # The old debug block (FPS, speed, mirror state, stability,
                # desktop size, key hints) stays gone: the GUI shows all of
                # it in real widgets.  The one line that came back is the
                # gesture prediction, because that is the only value here
                # you cannot read off a label — it has to sit next to the
                # hand that produced it to be worth anything.
                fps = 1.0 / (now_ts - prev_time) if now_ts > prev_time else 0.0
                prev_time = now_ts

                if yolo_worker is not None:
                    _yg, _ys, _yms = yolo_worker.current_state

                    # Alias FIRST, before orientation or handedness is even
                    # considered, so everything downstream — the override
                    # gate, the caption, the FSM — sees one spelling.
                    _yg = LABEL_ALIASES.get(_yg, _yg)

                    # ── MediaPipe-only fallback ────────────────────
                    # A flat hand held fingers-down has no class in the
                    # model, so the network reports nothing and every
                    # rule downstream — including the "stop" orientation
                    # override — has nothing to fire on.  Synthesised
                    # here from landmarks alone.
                    #
                    # Only ever fills a GAP: it is gated on the network
                    # having said nothing, so it can never overrule a
                    # real prediction.  "no_gesture" is included because
                    # that is the class name the model uses for nothing,
                    # alongside the None the worker returns below the
                    # confidence floor.
                    _raw = _yg          # kept for the diagnostic below
                    _source = "yolo"

                    # ── Fist/open-hand disagreement ────────────────────
                    # The network calls this a closed-hand pose while the
                    # landmarks show four extended fingers.  Both cannot be
                    # true, and the finger states are measured geometry
                    # rather than an inference, so they win.
                    #
                    # Corrected, not discarded: dropping the label would
                    # lose the pose entirely.  It is cleared here so the
                    # downward-hand rule below can name it properly, and
                    # if that rule does not apply the frame simply reports
                    # nothing rather than the wrong thing.
                    if contradicts_closed_hand(_yg, fingers_ext):
                        self._log_once(
                            "fist-open",
                            f"[label] '{_yg}' claims a closed hand but "
                            f"{extended_finger_count(fingers_ext)} fingers "
                            f"are extended — deferring to the landmarks")
                        _yg = None
                        _ys = None
                        _source = "landmarks"
                    if not _yg or _yg in ("none", "no_gesture"):
                        if hand_down:
                            _yg = "stop_inverse"
                            _source = "landmarks"
                            # _ys still holds the confidence of whatever
                            # the network JUST REJECTED — _infer returns the
                            # score even when it returns no gesture — and
                            # printing that next to a label MediaPipe
                            # invented reads as "this pose scored 0.44".
                            # It scored nothing; there was no detection.
                            _ys = None

                    # ── Label assembly:  name -> _inverse -> _side ──────
                    # YOLO cannot tell the hands apart and we are not
                    # retraining it to; MediaPipe already knows, and is
                    # already running on the same frame.  Gluing the two
                    # gives "three_gun_left" from a model that only ever
                    # learned "three_gun".
                    #
                    # ORIENTATION IS APPLIED TO ALMOST NOTHING, on purpose.
                    # The model already ships its own inverted classes —
                    # peace_inverted, stop_inverted, two_up_inverted — and
                    # those are trained verdicts on the real image.  Second-
                    # guessing them from two landmarks would put our maths
                    # in competition with the network on poses it already
                    # handles, and the network wins.  ORIENTATION_GESTURES
                    # therefore lists only the poses the model has NO
                    # inverted class for, where the choice is between our
                    # estimate and nothing at all.
                    #
                    # The verdict comes from gun_is_facing() (wrist to index
                    # tip), NOT palm_is_facing() (knuckle to knuckle).  A
                    # gun is held in profile, which is precisely where the
                    # knuckles overlap on x and their reading flickers
                    # frame to frame; the barrel is the one axis this pose
                    # keeps wide.
                    #
                    # _inverse goes on BEFORE the side, so the pose name and
                    # its orientation stay one token — "three_gun_inverse"
                    # is the gesture, "_right" is which hand made it.  A
                    # rule bound to "three_gun_inverse" then matches both
                    # hands by prefix.
                    #
                    # Each modifier is independent: an unknown side still
                    # yields "three_gun_inverse", and an undecidable
                    # orientation still yields "three_gun_right".  Neither
                    # can produce a double underscore.
                    _labelled = _yg
                    if _yg:
                        # Which axis decides THIS pose's orientation.  The
                        # two poses need different ones and neither reads
                        # the other well: a gun is held in profile, where
                        # the knuckles project onto nearly the same x and
                        # flicker; a stop is a flat hand square to the
                        # camera, where the knuckle spread is at its widest
                        # while the wrist-to-tip axis is nearly vertical
                        # and says nothing about x.
                        #
                        # Both estimators already fold in the true hand and
                        # IS_MIRRORED, so nothing extra is applied here.
                        if _yg not in ORIENTATION_GESTURES:
                            _facing = None          # trust the network
                        elif _yg == "three_gun":
                            _facing = gun_facing    # wrist -> index tip
                        else:                       # "stop"
                            _facing = hand_facing   # knuckle 5 -> 17

                        if _facing is False:
                            _labelled = f"{_labelled}_inverse"
                        if hand_side:
                            _labelled = f"{_labelled}_{hand_side}"

                    if _labelled != last_semantic:
                        last_semantic = _labelled
                        self._log(
                            f"[label] {_raw or '-'} -> {_labelled or '-'}"
                            f"   (src={_source}"
                            + (f" {_ys:.2f}" if _ys is not None else "")
                            + f", side={hand_side or '-'}"
                            f", knuckles={_orientation_word(hand_facing)}"
                            f", down={hand_down})")

                    self.status["yolo"] = _labelled
                    self.status["label_source"] = _source
                    self.status["yolo_score"] = _ys
                    self.status["yolo_ms"] = _yms
                    self.status["handedness"] = hand_side
                    self.status["palm_facing"] = hand_facing

                    # ── Semantic stream into its own FSM ────────────────
                    # A SEPARATE instance from the geometric one: the two
                    # streams speak different vocabularies at different
                    # rates, and interleaving them in one sliding window
                    # would leave neither able to stabilise.  Same rule
                    # set, same dispatcher — a rule matches whichever
                    # stream actually emits its pose.
                    if semantic_fsm is not None:
                        try:
                            _sem = semantic_fsm.update(_labelled, now_ts)
                            if _sem is not None:
                                actions.dispatch(_sem, now_ts)
                        except Exception as exc:
                            self._log_once(
                                "semantic",
                                f"[action] semantic binding failed, "
                                f"tracking continues "
                                f"({exc.__class__.__name__}: {exc})")

                    # ── Gesture prediction overlay ──────────────────────
                    # Fixed top-left rather than pinned above a box: the
                    # worker returns a label and a score, not the box
                    # geometry, so there is nothing on this frame to anchor
                    # to.  A fixed corner also stops the caption jumping
                    # around while the hand moves.
                    if not yolo_worker.ready:
                        _text = ("Gesture: loading model..."
                                 if yolo_worker.load_error is None
                                 else "Gesture: model unavailable")
                        _colour = _PREDICT_IDLE_COLOUR
                    elif not _labelled:
                        _text = "Gesture: none"
                        _colour = _PREDICT_IDLE_COLOUR
                    elif _ys is None:
                        # No network confidence exists for this one.  Say
                        # where it came from rather than borrow a number.
                        _text = f"Gesture: {_labelled} (landmarks)"
                        _colour = _PREDICT_COLOUR
                    else:
                        # The combined name, so what is on screen is
                        # exactly what the PNG in gestures/ must be
                        # called and exactly what a rule must bind.
                        _text = f"Gesture: {_labelled} ({_ys:.2f})"
                        _colour = _PREDICT_COLOUR

                    # Drawn twice — a thick black pass, then the colour on
                    # top — so the label stays readable over a live camera
                    # image.  A single pass disappears against a pale wall,
                    # which is most of the frame most of the time.
                    for _c, _w in ((_PREDICT_SHADOW, 5), (_colour, 2)):
                        cv2.putText(bgr_buf, _text, _PREDICT_ORIGIN,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    _PREDICT_SCALE, _c, _w, cv2.LINE_AA)

                # ── Publish, instead of cv2.imshow ──────────────────────
                # A copy, because bgr_buf is reused in place next frame and
                # a consumer holding a view would watch it change mid-draw.
                # The rebind itself is atomic, so no lock is needed.
                self._latest_frame = bgr_buf.copy()

                self.status.update({
                    "fps": fps, "gesture": gesture,
                    "stable_gesture": stable_gesture,
                    "dragging": actions.dragging,
                    "clicks": actions.click_count,
                    "drags": actions.drag_count,
                    "gesture_changes": gesture_changes,
                    "frozen": frozen_target is not None
                    and now_ts < freeze_until,
                    "last_action": actions.last_action,
                })

        except Exception as exc:                # pragma: no cover - runtime
            self.status["error"] = f"{exc.__class__.__name__}: {exc}"
            self._log(f"[engine] loop stopped: {self.status['error']}")
        finally:
            # The button must not outlive the loop, whatever ended it.
            if self.actions.release():
                self._log("[action] DRAG_STOP (loop exit)")
            self.status["running"] = False


# ═══════════════════════════════════════════════════════════════════════════
#  STANDALONE SCRIPT
# ═══════════════════════════════════════════════════════════════════════════
#
# Unchanged behaviour: scan, prompt, open a preview window, drive the same
# engine, and honour the same keys.  The imshow/waitKey pair lives here now
# rather than inside the loop, which is exactly what lets a GUI host the
# engine without inheriting a window it did not ask for.

def _choose_camera_interactively(backend):
    if CAM_INDEX is not None:
        print(f"[camera] CAM_INDEX pinned to {CAM_INDEX}, skipping scan\n")
        return CAM_INDEX
    print(f"[camera] scanning indices 0–{CAM_SCAN_MAX}, backend "
          f"{describe_api(backend.camera_api)} "
          f"({CAM_SCAN_READ_TRIES} read attempts each)…\n")
    return choose_camera(scan_cameras(CAM_SCAN_MAX, backend.camera_api))


def main() -> None:
    engine = HandTrackerEngine(enable_monitor=SYSTEM_MONITOR_ENABLED)

    while True:
        index = _choose_camera_interactively(engine.backend)
        if engine.start(index):
            break
        print("         pick a different one.")
        if CAM_INDEX is not None:
            engine.close()
            return

    screen = engine.screen
    stream = engine.stream

    print(f"Platform    : {engine.backend.name}")
    print(f"Cursor out  : {engine.cursor.name}")
    print(f"Screen src  : {screen.source}")
    print(f"Virtual desk: {screen.describe()}")
    print(f"Webcam      : index {engine.camera_index}, native "
          f"{stream.native_width}×{stream.native_height}")
    print(f"Processing  : {stream.width} × {stream.height}")
    print(f"Active box  : {screen.box_w} × {screen.box_h} px")
    print(f"Sensitivity : {CURSOR_SENSITIVITY}× start value, "
          f"adjustable {SENS_MIN}–{SENS_MAX} in steps of {SENS_STEP}")
    print(f"Mirroring   : {'ON' if IS_MIRRORED else 'OFF'}  |  "
          f"Invert X: {'ON' if INVERT_CURSOR_X else 'OFF'}")
    print(f"1€ Filter   : min_cutoff={MIN_CUTOFF}  β={BETA}  "
          f"d_cutoff={D_CUTOFF}")
    print("Controls    : '+'/'=' faster   '-'/'_' slower   "
          "'m' mirror   'i' invert-x   'q' quit")
    print("(the preview window must have focus for keys to register)\n")

    try:
        while engine.running:
            frame = engine.get_latest_frame()
            if frame is not None:
                cv2.imshow("Hand Cursor Control", frame)
            if handle_key(cv2.waitKey(1) & 0xFF):
                break
    finally:
        engine.close()
        if engine.actions.dragging:
            print("[action] WARNING: could not release the mouse button")
        if engine.fsm is not None:
            print(f"[action] {engine.fsm.click_count} single, "
                  f"{engine.fsm.double_click_count} double, "
                  f"{engine.fsm.drag_start_count} drags — "
                  f"{engine.actions.click_count} clicks and "
                  f"{engine.actions.drag_count} presses reached the OS")
        if engine.yolo_worker is not None:
            print(f"[yolo] {engine.yolo_worker.inference_count} inferences, "
                  f"{engine.yolo_worker.dropped_count} crops dropped")
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
