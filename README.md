# Mousegesture

**Vision-based hand gesture mouse controller.** Turns a commodity webcam into a
pointing device: MediaPipe hand landmarks drive the OS cursor in real time,
while a fine-tuned YOLOv10n model recognises hand poses that you bind to mouse
and keyboard actions through a desktop GUI.

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00A67E?logo=google&logoColor=white)
![YOLOv10n](https://img.shields.io/badge/model-YOLOv10n-FFCC00)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%2FX11-0078D6)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Cursor Tracking](#cursor-tracking)
- [Gesture Recognition](#gesture-recognition)
- [Calibration](#calibration)
- [Performance](#performance)
- [Camera Configuration](#camera-configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Problems Solved](#problems-solved)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Resume Summary](#resume-summary)
- [Technical Highlights](#technical-highlights)

---

## Overview

**What it does.** Tracks one hand through a webcam and maps its position onto
the operating system cursor across a multi-monitor desktop. Hand poses are
classified two independent ways — fast geometry and a neural network — and a
configurable state machine turns pose *transitions* and *timed holds* into
mouse clicks, drags, scrolls and keyboard shortcuts.

**Problem it solves.** Pointer control without touching hardware. Useful for
accessibility, presentation, and touchless-interface contexts. Requires no
depth sensor, no wearable and no controller — a standard webcam is the only
sensor.

**How you interact with it.** Launch the GUI, pick a camera, browse a catalogue
of gesture thumbnails, and bind poses to actions. Move your hand to move the
cursor; perform a bound gesture to fire its action. Sensitivity, mirroring,
axis inversion, model confidence and target display are live controls that
persist to a single JSON file.

**Two branches, on purpose.** Cursor motion must answer within a frame;
gesture classification does not. The two run concurrently at different rates
and never block each other — this is the central design decision of the
project.

> **Note on defaults.** The shipped `gesture_config.json` contains **no
> gesture bindings**. On a fresh clone the cursor moves and no gesture fires
> anything until you create bindings in the GUI. This is deliberate: the
> engine contains no hardcoded gesture names at all.

---

## Features

### Implemented and verified

| Feature | Detail |
|---|---|
| **Cursor tracking** | Anchored on MediaPipe landmark 9 (middle-finger MCP), mapped onto the virtual desktop |
| **Hand / landmark tracking** | MediaPipe Hands, 21 landmarks, up to 2 hands, Lite model |
| **Primary-hand lock** | Nearest-anchor continuity so the cursor never jumps between two hands |
| **Landmark validation** | Rejects implausible skeletons; all tests scaled in palm lengths, not pixels |
| **Geometric gesture classifier** | 5 states (`point`, `peace`, `grip`, `open`, `idle`), deterministic, squared-distance math |
| **Neural gesture classifier** | YOLOv10n fine-tuned on HaGRIDv2, 34 classes, CUDA when available |
| **Handedness detection** | True physical hand, corrected for mirroring; appended as `_left` / `_right` |
| **Orientation detection** | Palm vs back of hand for `stop` and `three_gun`; appended as `_inverse` |
| **Gesture state machine** | Config-driven; transitions, timed holds, cooldowns, double-click promotion |
| **Mouse actions** | Left / right / middle / double click, drag start-stop, scroll up/down |
| **Keyboard macros** | 13 named chords plus arbitrary user-defined chords |
| **Smoothing** | One-Euro filter, one instance per axis, speed-adaptive cutoff |
| **Edge handling** | Two independent saturation points; cursor holds flat against a screen edge |
| **Multi-monitor** | Virtual-desktop mapping with negative origins; confine to "All Screens" or any "Screen N" |
| **Display hot-plug** | Desktop rectangle re-read every 2.5 s; mapping rebuilt live |
| **DPI awareness** | Per-monitor v2 on Windows, with two fallbacks |
| **Desktop GUI** | 3-tab Tkinter app: Mapping Studio, Live Camera & Diagnostics, Recycle Bin |
| **Binding management** | Create, edit, duplicate, disable, test-fire, soft-delete with undo and restore |
| **Diagnostics** | Live CPU / RAM / GPU / VRAM plus engine FPS, in-app and as a standalone window |
| **Fail-soft design** | Every optional dependency degrades to a printed line, never an exception |

### Not implemented

These are named explicitly because they are commonly assumed to be present:

| Not implemented | Notes |
|---|---|
| Calibration wizard | System is calibration-free by design — see [Calibration](#calibration) |
| Motion prediction / LERP | Filtering only; there is no predictive stage |
| Edge-push, ease-edge, max-step limiting | Cursor clamps flat at boundaries |
| Adaptive workspace | The active region is a fixed proportion of the frame |
| Dynamic gestures (swipe, zoom) | Every classifier here is per-frame static pose |
| Landmark centroid | The anchor is a single landmark, not an average |
| Automated test suite | See [Testing](#testing) |

---

## Architecture

```
┌─ CAPTURE THREAD ─────────────────────────────────────────────┐
│  cv2.VideoCapture(index)                                     │
│    ↓ read()                                                  │
│    ↓ _shrink()          downscale to 1280×720, INTER_AREA    │
│    ↓ atomic tuple publish (grabbed, frame, seq)  — lock-free │
└──────────────────────────────┬───────────────────────────────┘
                               │ newest frame only
┌─ MAIN LOOP · HandTrackerEngine._run() ───────────────────────┐
│  cv2.flip  → bgr_buf        mirror, in-place, preallocated   │
│  cv2.resize→ 640×360        integer divisor ÷2               │
│  cv2.cvtColor → rgb_buf     writeable=False (zero-copy)      │
│                                                              │
│  MediaPipe Hands.process()  21 landmarks × ≤2 hands          │
│    ↓                                                         │
│  LandmarkValidator.check()  reject implausible skeletons     │
│  pick_primary_hand()        nearest-anchor continuity        │
│    │                                                         │
│    ├── BRANCH A · CURSOR ─────────────────────┐              │
│    │   anchor = landmark[9]                   │              │
│    │     ↓ × cam_w, cam_h                     │              │
│    │   detect_gesture()   → 5 states          │              │
│    │     ↓                                    │              │
│    │   GestureFSM.update_pair()               │              │
│    │     ↓ majority vote → rule match         │              │
│    │   ActionDispatcher.dispatch()            │              │
│    │     ↓                                    │              │
│    │   STATE_FREEZE_MS  pin target 75 ms      │              │
│    │     ↓                                    │              │
│    │   ScreenGeometry.to_screen()             │              │
│    │     inset box → desktop rect             │              │
│    │     centre-scaled sensitivity → clamp    │              │
│    │     ↓                                    │              │
│    │   OneEuroFilter(x) , OneEuroFilter(y)    │              │
│    │     ↓                                    │              │
│    │   cursor.move(int, int)                  │              │
│    └───────────────────────────────────────────┘             │
│                                                              │
│    └── BRANCH B · SEMANTIC ───────────────────┐              │
│        yolo_worker.submit(bgr_buf.copy())     │              │
│          ↓ queue(maxsize=1) newest-wins       │              │
│        [YoloWorker thread]                    │              │
│          YOLO.predict(imgsz=640)              │              │
│          ↓ argmax(boxes.conf) → score gate    │              │
│        LABEL_ALIASES → _inverse → _side       │              │
│          ↓                                    │              │
│        semantic_fsm.update()   separate FSM   │              │
│          ↓                                    │              │
│        MacroDispatcher  edge-trigger+cooldown │              │
│          ↓                                    │              │
│        keyboard chord                         │              │
│    └───────────────────────────────────────────┘             │
│                                                              │
│  overlays → status dict → GUI / cv2.imshow                   │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
      pynput  │  ctypes: user32 (Win)  /  libX11 + XTest (Linux)
                               ▼
                        OS pointer + keyboard
```

**Stage-to-file map**

| Stage | File | Unit |
|---|---|---|
| Frame capture | `hand_cursor_2.py` | `WebcamStream` (daemon thread) |
| Camera discovery | `hand_cursor_2.py` | `scan_cameras`, `choose_camera` |
| Landmark detection | `hand_cursor_2.py` | `mp_hands.Hands` |
| Landmark validation | `hand_cursor_2.py` | `LandmarkValidator` |
| Geometry classifier | `hand_cursor_2.py` | `detect_gesture` |
| Handedness / orientation | `hand_cursor_2.py` | `handedness_of`, `palm_is_facing`, `gun_is_facing` |
| Smoothing | `hand_cursor_2.py` | `OneEuroFilter`, `LowPassFilter` |
| Screen mapping | `hand_cursor_2.py` | `ScreenGeometry` |
| Display enumeration | `monitors.py` | `enumerate_monitors`, `resolve_monitor_target` |
| OS backends | `hand_cursor_2.py` | `Win32Backend`, `X11Backend` |
| Cursor output | `hand_cursor_2.py` | `PynputCursor`, `NativeCursor` |
| Neural inference | `hand_cursor_2.py` | `YoloWorker` (daemon thread) |
| Gesture arbitration | `gesture_fsm.py` | `GestureFSM`, `TransitionRule`, `HoldRule` |
| Action execution | `gesture_fsm.py` | `ActionExecutor` |
| GUI | `app.py` | `GestureStudio(tk.Tk)` |
| Telemetry | `system_monitor.py` | `SystemMonitor` (separate process) |

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.11 |
| **Computer Vision** | OpenCV 4.11, MediaPipe Hands 0.10 (Solutions API, `model_complexity=0`) |
| **ML / DL** | Ultralytics 8.3, PyTorch 2.5.1+cu121, CUDA 12.1, YOLOv10n fine-tuned on HaGRIDv2 (34 classes) |
| **GUI** | Tkinter + ttk, Pillow (gesture thumbnails) |
| **Mouse / System APIs** | `ctypes` → `user32.dll` (`SetCursorPos`, `mouse_event`, `GetSystemMetrics`, `EnumDisplayMonitors`), `shcore` (DPI awareness); `libX11` / `libXtst` / `libXinerama` on Linux; `screeninfo` |
| **Input control** | pynput (mouse + keyboard), with a native ctypes fallback |
| **Concurrency** | `threading`, `queue` (depth-1 newest-wins), `subprocess`, lock-free atomic publication |
| **Telemetry** | psutil, nvidia-ml-py (`pynvml`) |
| **Numerics** | NumPy 2.x, preallocated frame buffers |
| **Testing** | *No automated test suite in the repository* — see [Testing](#testing) |
| **Development tools** | Git, Git LFS (`*.pt`, `*.onnx`), pycodestyle, conda |

---

## Cursor Tracking

### Order of operations

```
1.  anchor       = hand.landmark[9]                    # middle MCP
2.  raw_x, raw_y = anchor.x * cam_w, anchor.y * cam_h
3.  gesture      = detect_gesture(hand, cam_w, cam_h)
4.  yolo_worker.submit(bgr_buf.copy())                 # non-blocking
5.  target       = screen.to_screen(raw_x, raw_y)      # MAP + CLAMP
6.  if re-entry: oef_x.reset(); oef_y.reset()          # snap, don't glide
7.  if pose changed: frozen_target = target
                     freeze_until  = now + 75 ms
8.  fsm.update_pair(gesture, now) → dispatch(action)   # fenced try/except
9.  if now < freeze_until: target = frozen_target      # FREEZE
10. smooth       = oef_x(target_x, now), oef_y(target_y, now)   # FILTER
11. cursor.move(int(smooth_x), int(smooth_y))          # OUTPUT
```

The ordering is **map → freeze → filter → output**. There is no prediction
stage.

### Anchor

A **single landmark: index 9, the middle-finger MCP knuckle.** A fingertip is
the most mobile point on the hand — every gesture displaces it by definition —
so anchoring there makes the cursor lurch at exactly the moment the user meant
to signal rather than move. The knuckle is carried by the palm.

*Measured:* across four pose transitions the worst-moving fingertip travelled
**108 px** while landmark 9 moved **0.0 px**.

### Landmarks used

| Index | Name | Used for |
|---|---|---|
| 0 | Wrist | Scale reference, gun-barrel origin, pointing-down test |
| 4 | Thumb tip | Thumb-out test (measured against index MCP, not wrist) |
| 5 | Index MCP | Palm/back orientation, thumb reference |
| 6, 10, 14, 18 | PIP joints | Finger-extension test |
| 8, 12, 16, 20 | Fingertips | Extension test; landmark 8 also gives the gun-barrel axis |
| **9** | **Middle MCP** | **Cursor anchor** |
| 17 | Pinky MCP | Palm/back orientation |

### Filtering

One-Euro filter (Casiez, Roussel & Vogel), two independent instances — one per
axis:

```
fc    = MIN_CUTOFF + BETA · |velocity|
alpha = 1 / (1 + (1/(2π·fc)) / dt)
```

| Parameter | Value | Role |
|---|---|---|
| `MIN_CUTOFF` | 1.0 Hz | Cutoff at rest — sets how sluggish precise aiming feels |
| `BETA` | 0.015 | Speed coefficient — how much velocity opens the cutoff |
| `D_CUTOFF` | 1.0 Hz | Cutoff of the velocity estimator itself |

Heavy smoothing when the hand is still, nearly transparent during fast motion.
`dt` is measured per call from the frame timestamp rather than assumed.

### Mapping

Inside `ScreenGeometry.to_screen()`, per axis:

1. Optional horizontal reflection about the frame centre if `INVERT_CURSOR_X`.
2. Piecewise-linear interpolation of the **inset box** `[box_left, box_right]`
   onto the desktop span `[left, left + width]`, saturating outside the input
   range.
3. Centre-scaled sensitivity: `mid + (mapped − mid) × CURSOR_SENSITIVITY`.
4. `clamp` to the desktop rectangle.

The destination is `[left, left + width]`, **not** `[0, width]` — on a
multi-monitor Windows desktop the origin can be negative, and a zero-based span
makes every display left of the primary unreachable.

The interpolation is written out longhand rather than using `np.interp`,
because on two scalars per frame numpy's array dispatch measured **~12.9 µs**,
roughly 12× the arithmetic it performs. Slopes are precomputed, so a frame
costs two multiplies and two adds per axis.

### Active region

| Constant | Value | Meaning |
|---|---|---|
| `MARGIN_X` | 0.20 | Inset each side |
| `MARGIN_TOP` | 0.10 | Inset at top |
| `MARGIN_BOTTOM` | 0.28 | Inset at bottom |
| `CURSOR_SENSITIVITY` | 1.4 | Gain, adjustable 1.0–3.0 in steps of 0.1 |

The vertical margins are **asymmetric on purpose**: an arm raises well above
shoulder height but stops against the desk going down, so a symmetric window
wasted travel at the top and ran out before the bottom.

This region is **static** — it does not adapt to the user or the session.

### Boundary handling

Two independent saturation points: the interpolation clamps at the box edge,
and the final `clamp()` catches anything sensitivity pushed past a desktop
edge. A hand beyond the box holds the cursor **flat** against the screen edge
rather than running off it.

Clamping happens **before** the filter. Exponential smoothing of in-range
values stays in range, so the filter output needs no second clamp and can never
be dragged off-screen by its own history.

There is no edge-push, easing, or maximum-step limiter.

### Multi-monitor

`ScreenGeometry` re-reads the desktop rectangle every `POLL_INTERVAL = 2.5 s`,
so hot-plugging a display rebuilds the mapping live. A degenerate `0×0` read
during a driver reconfigure is ignored rather than adopted. The cursor can be
confined to `"All Screens"` (the union of every monitor) or `"Screen N"` (that
rectangle only); a configured screen that is no longer attached falls back to
Screen 1.

---

## Gesture Recognition

> This project has no fixed gesture→action table. It is a **gesture binding
> engine**: poses come from two classifiers, and you map any pose or pose
> transition to any action through the GUI. What follows is the complete
> vocabulary of poses and actions available for binding.

### Geometric poses — detected every frame

| Pose | Detection rule | Type |
|---|---|---|
| `point` | Index extended; middle, ring, pinky curled | Static |
| `peace` | Index + middle extended; ring, pinky curled | Static |
| `grip` | No finger extended **and** thumb not out (closed fist) | Static |
| `open` | All four fingers extended (flat palm) | Static |
| `idle` | Any other combination — **an abstention, never a state** | — |

A finger counts as extended when its tip sits further from the wrist than its
own PIP joint, times `EXTEND_MARGIN = 1.05`. The thumb has its own test:
distance from thumb tip to index MCP exceeding `THUMB_OUT_RATIO = 0.60` × palm
length. All comparisons use squared distances — no `sqrt` anywhere in the
classifier.

### Neural poses — 34 HaGRIDv2 classes

```
grabbing   grip       holy        point       call       three3
timeout    xsign      hand_heart  hand_heart2 little_finger
middle_finger  take_picture  dislike  fist    four       like
mute       ok         one         palm        peace      peace_inverted
rock       stop       stop_inverted  three    three2     two_up
two_up_inverted  three_gun  thumb_index  thumb_index2   no_gesture
```

Each label is decorated before reaching the state machine:

```
raw class → LABEL_ALIASES → [_inverse if orientation says so] → [_side]

  "three_gun"      + back of hand + right hand  →  three_gun_inverse_right
  "peace_inverted" + left hand                  →  peace_inverse_left
  "stop"           + palm         + left hand   →  stop_left
```

The `gestures/` folder ships **27 PNG thumbnails** for the GUI catalogue.

### Available actions

| Action | OS event |
|---|---|
| `LEFT_CLICK` | `mouse.click(Button.left, 1)` |
| `RIGHT_CLICK` | `mouse.click(Button.right, 1)` |
| `MIDDLE_CLICK` | `mouse.click(Button.middle, 1)` |
| `DOUBLE_CLICK` | `mouse.click(Button.left, 2)` |
| `DRAG_START` | `mouse.press(Button.left)` |
| `DRAG_STOP` | `mouse.release(Button.left)` |
| `SCROLL_UP` | `mouse.scroll(0, 2)` |
| `SCROLL_DOWN` | `mouse.scroll(0, -2)` |

**Named keyboard macros**

| Action | Chord | Action | Chord |
|---|---|---|---|
| `SHOW_DESKTOP` | `win+d` | `COPY` | `ctrl+c` |
| `TASK_VIEW` | `win+tab` | `PASTE` | `ctrl+v` |
| `MINIMISE_ALL` | `win+m` | `SCREENSHOT` | `win+shift+s` |
| `LOCK_SCREEN` | `win+l` | `VOLUME_UP` | `volume_up` |
| `SWITCH_WINDOW` | `alt+tab` | `VOLUME_DOWN` | `volume_down` |
| `CLOSE_WINDOW` | `alt+f4` | `MUTE` | `volume_mute` |
| `MEDIA_PLAY_PAUSE` | `media_play_pause` | `KEYBOARD_MACRO` | user-defined |

### Trigger types

| Kind | Fires when | Guards |
|---|---|---|
| **Transition** | Stable pose becomes `to_state` having been `from_state` | `max_time_sec` (0.8 s), `transition_memory` (2), `cooldown_sec` (0.35 s), optional `promote_double` |
| **Hold** | Pose held continuously for `hold_sec` | Edge-triggered; `cooldown_sec`; optional `repeat` every `repeat_sec` (min 0.05 s) |

### False-trigger prevention

Five independent mechanisms, all verified in code:

1. **Sliding-window majority vote** — 2 of the last 3 frames must agree
   (100 ms at 30 Hz). Absorbs single-frame misclassification.
2. **Abstention filtering** — `idle` never enters the vote and never becomes a
   state. Curling from `point` to `grip` physically passes through a
   half-curled hand; without this the intermediate becomes the predecessor and
   the transition never fires.
3. **Bounded transitions** — `max_time_sec` separates "the user curled point
   into grip" from "the user was pointing a minute ago and has now closed
   their hand for an unrelated reason".
4. **Edge-triggered holds** — armed on entry, disarmed only by leaving the
   pose. A cooldown alone does not stop repeat fires; it only throttles them.
5. **Per-rule cooldown** — a second, independent guard behind edge-triggering,
   covering a label that flickers out and straight back in.

Mutual exclusion in the geometric classifier is structural: the five states are
an `if`/`elif` chain over the same 4-tuple of booleans plus `thumb_out`, so
exactly one branch can match.

### Drag safety

A held mouse button outlives the process that pressed it, so `DRAG_STOP` is
emitted from **four independent paths** rather than assumed: hand lost, an
unmapped pose, `stuck_release_sec` (0.5 s) with no hand, and
`drag_timeout_sec` (30 s) regardless. The dispatcher clears its `dragging`
flag only on a release the OS actually accepted, so a failed release stays
pending for the shutdown path to retry.

---

## Calibration

**There is no calibration wizard in this project.** There is no TOP / BOTTOM /
LEFT / RIGHT capture procedure, no consecutive-frame requirement, no median
capture, no validation step, and no calibration data written to or read from
disk.

The system is **calibration-free by design**:

- **Self-normalising geometry.** Finger extension compares tip-to-wrist against
  PIP-to-wrist on the *same hand*, so it needs no per-user threshold. Thumb-out
  is a ratio of palm length. The landmark validator measures everything in palm
  lengths, never pixels.
- **Proportional margins.** The active region is a fraction of the frame, so it
  lands on the same relative rectangle whatever the sensor delivers.
- **Runtime tuning replaces calibration.** Sensitivity, mirroring, axis
  inversion and model confidence are live controls persisted to JSON.
- **Automatic rebuilds.** Camera resolution changes and display hot-plug both
  rebuild the mapping — the class of drift a calibration step would otherwise
  need re-running for.

Settings that *are* persisted live in `gesture_config.json` under `settings`.
No per-user calibration profile exists because none is captured.

---

## Performance

All figures below are **measured** — either recorded in-source beside the
constant they justify, or produced by executing the code. Nothing here is a
target or an estimate.

### MediaPipe inference by input size

`model_complexity=0`, median `process()` time:

| Input | Pixels | Median | vs 640×360 |
|---|---:|---:|---:|
| 640×360 | 230,400 | 37.29 ms | 100% |
| 480×270 | 129,600 | 34.35 ms | 92% |
| 320×180 | 57,600 | 32.20 ms | 86% |
| 192×108 | 20,736 | 32.34 ms | 87% |

**Key finding:** pixel count falls 11×, inference time falls 1.16×. The palm
and landmark networks run at fixed internal input sizes, so shrinking the frame
only saves MediaPipe's own letterbox resize.

### Model complexity

| Input | Model | Time |
|---|---|---:|
| 320×180 | Lite | 35.6 ms |
| 480×270 | Lite | 32.0 ms |
| 640×360 | Lite | 32.8 ms |
| 480×270 | Full | 67.9 ms |

### Inference and loop costs

| Quantity | Value |
|---|---:|
| YOLO warm forward pass (Ultralytics, CUDA) | **55 ms** (≈18 Hz) |
| YOLO forward, `cv2.dnn` + ONNX (superseded) | 554.7 ms (≈1.8 Hz) |
| Ultralytics CPU, same input | 202.0 ms (≈5.0 Hz) |
| Full-size frame work at 1280×720 (flip + resize + copy) | 2,984 µs |
| Same at 640×480 | 258 µs |
| 1280→320 downscale alone | 2,496 µs |
| 640→320 downscale (exact 2:1, fast path) | 123 µs |
| 640→256 downscale (fractional) | 1,536 µs |
| `np.interp` on two scalars | 12.9 µs |
| YOLO queue `submit()` worst case | 6.2 ms |
| YOLO `poll()` on empty queue | 2.4 µs |
| Frames dropped by depth-1 queue | 27% (83% on the ONNX path) |
| Ultralytics import cost | 10.8 s |
| YOLO model load | 0.4 s |
| Display enumeration | 0.24 ms |
| Click latency after pose settles | 33 ms |

### Reference hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop, driver 610.88 (WDDM) |
| CUDA / PyTorch | 12.1 / 2.5.1+cu121 |
| OS | Windows 10.0.26200 |
| Python | 3.11.15 |

### Not measured

Sustained end-to-end FPS under load, true photon-to-cursor latency,
steady-state CPU %, steady-state RAM and per-process VRAM are **not recorded
anywhere in this project**. The application displays CPU / RAM / GPU / VRAM
live, but no figure is logged or asserted.

Per-process VRAM specifically **cannot** be read on the reference hardware: the
GPU runs in WDDM mode, where NVML returns `usedGpuMemory = None` for every
process. Verified against a child process that had provably allocated 256 MB of
CUDA memory.

---

## Camera Configuration

| Setting | Value |
|---|---|
| Requested resolution | 1280 × 720 |
| Requested frame rate | 60 FPS |
| Processing budget | 1280 × 720 (bounding box, aspect preserved) |
| MediaPipe input | 640 × 360 (integer divisor 2, `INTER_AREA`) |
| YOLO input | Whole frame, letterboxed to 640 × 640 by Ultralytics |
| Backend | `CAMERA_API = None` → OpenCV auto-negotiates (MSMF on Windows) |
| Buffer | `CAP_PROP_BUFFERSIZE = 1`, acceptance recorded |
| Scan range | Indices 0–4, 5 read attempts each, 0.12 s apart |

### Property negotiation

The requested size is treated as **a request, not a fact.** `WebcamStream` sets
the three properties, then reads one frame synchronously and takes
`frame.shape` as ground truth. `CAP_PROP_FRAME_*` is queried too and any
mismatch is recorded, because a disagreement usually means the driver silently
refused the mode. Every downstream geometry object is built from the measured
size.

Oversized frames are downscaled **in the capture thread** using `INTER_AREA`,
before anything else touches them — a resize, not a crop, so the full field of
view survives. The budget is a bounding box, so a 1920×1080 source becomes
640×360 rather than being squashed into 640×480.

### Scanner behaviour

`scan_cameras()` returns `{index: (width, height)}` for every index that
actually **delivered a frame**, not merely opened. A device that opens but
stays silent is excluded and the reason is printed, because "failed to open"
and "opened but produced nothing" have different fixes. Retries exist because a
phone-backed virtual camera can take several hundred milliseconds to hand over
its first frame.

### FOURCC / MJPG

There is **no** `CAP_PROP_FOURCC` call and no MJPG or YUY2 handling in this
codebase. See [Future Improvements](#future-improvements).

---

## Project Structure

```
Mousegesture/
├── app.py                    2,854 lines  Tkinter GUI — GestureStudio(tk.Tk)
│                                          3 tabs: Mapping Studio, Live Camera
│                                          & Diagnostics, Recycle Bin
├── hand_cursor_2.py          4,635 lines  ENGINE — HandTrackerEngine and
│                                          everything it needs: WebcamStream,
│                                          YoloWorker, ScreenGeometry,
│                                          OneEuroFilter, Win32/X11 backends,
│                                          cursor adapters, ActionDispatcher,
│                                          MacroDispatcher, detect_gesture,
│                                          LandmarkValidator, DPI awareness
├── gesture_fsm.py            1,366 lines  Config-driven state machine with no
│                                          gesture names inside it: GestureFSM,
│                                          TransitionRule, HoldRule,
│                                          MajorityStabilizer, ActionExecutor,
│                                          action vocabulary, config load/save
├── monitors.py                 201 lines  Display enumeration shared by the GUI
│                                          and the engine
├── system_monitor.py           412 lines  Standalone resource monitor window
├── gesture_config.json                    THE configuration file
├── README.md                              This file
├── .gitattributes                         Git LFS rules for *.pt and *.onnx
├── model/
│   ├── YOLOv10n_gestures.pt               Active model (LFS)
│   ├── YOLOv10n_gestures.onnx             Superseded ONNX export
│   ├── YOLOv10x_gestures.pt               Not loaded by the application
│   ├── YOLOv10x_gestures.onnx             Not loaded by the application
│   ├── convert.py                         .pt → .onnx export script
│   ├── ptmodel.py                         Standalone Ultralytics webcam demo
│   └── onxxmodel.py                       Standalone cv2.dnn webcam demo
├── gestures/                        27×    PNG thumbnails for the GUI catalogue
├── others/
│   ├── requirements.txt                   Incomplete — see Installation
│   ├── Mouse_Gesture_Architecture.png
│   ├── Mouse_Gesture_Pipeline.drawio
│   ├── Future Updates.txt                 Git save notes
│   └── gestures.txt                       Notes on an intended binding
├── runs/detect/                           Ultralytics output directory
├── test.py                                Stale copy of hand_cursor_2.py —
│                                          NOT a test file
└── testmoniter.py                         Stale copy of system_monitor.py —
                                           NOT a test file
```

> `test.py` and `testmoniter.py` are named like tests and are not. They are
> older snapshots of the engine and the monitor, containing zero assertions.

---

## Testing

**There is no automated test suite in this repository.**

| File | Lines | `assert` | `def test_` |
|---|---:|---:|---:|
| `test.py` | 3,701 | 0 | 0 |
| `testmoniter.py` | 413 | 0 | 0 |
| `app.py` | 2,854 | 0 | 0 |
| `hand_cursor_2.py` | 4,635 | 0 | 0 |
| `gesture_fsm.py` | 1,366 | 0 | 0 |

Assertion-based verification harnesses were written **during development** and
run against the source by AST extraction — importing the module opens a camera
and a window, so the harnesses lift out individual classes and execute them
against fakes. They covered screen geometry and hot-plug, sensitivity clamping,
hotkey handling, the One-Euro filter, camera scanning, buffer reuse, the
threaded stream, the state machine, drag safety, macro chords, DPI awareness,
two-hand handling, orientation truth tables and display enumeration. **Those
harnesses are not part of this repository** and are not claimed as a
deliverable.

### Runtime diagnostics that are in the repository

- `system_monitor.py` — always-on-top window: system CPU / RAM / GPU / VRAM
  plus per-process CPU / RAM, 1 s refresh, launched with the engine's PID and
  terminated with it.
- GUI diagnostics tab — the same metrics inside the app, plus live engine FPS.
- Structured console logging: `[gesture]`, `[label]`, `[action]`, `[yolo]`,
  `[screen]`, `[camera]`, `[config]`, with de-duplication so a per-frame
  failure prints once rather than 30 times a second.
- On-frame overlays: landmark skeleton, anchor marker (amber while frozen,
  green otherwise), active-box and effective-box rectangles, live prediction
  caption.
- A **test-fire button** in the GUI with a countdown, which triggers a rule's
  action without needing to perform the gesture.
- `python gesture_fsm.py` runs a scripted self-check: a synthetic hand
  sequence including a stray frame and an unmapped intermediate pose, printing
  which actions fired.

---

## Design Decisions

### Anchor on the knuckle, not a fingertip
A fingertip is the most mobile point on the hand — every gesture displaces it
by definition, so the cursor lurches at exactly the moment the user meant to
signal rather than move. Landmark 9 is carried by the palm. *Measured: 108 px
versus 0.0 px across four pose transitions.*

### One-Euro filter as the only smoothing stage
Fixed-alpha smoothing forces one trade-off for both regimes: enough smoothing
to kill tremor at rest is enough to add visible lag during motion. The One-Euro
cutoff rises with velocity, so it is heavy at rest and nearly transparent
during a swipe. An earlier sub-frame interpolation stage was removed because it
made fast motion *worse*.

### Map → freeze → filter → output
Clamping before the filter matters: exponential smoothing of in-range values
stays in range, so the output needs no second clamp and can never be dragged
off-screen by history. The freeze also precedes the filter, so the pinned point
is fed *through* it and leaves the filter settled there — on release the cursor
glides back to the hand instead of jumping.

### Two branches at two rates, sharing nothing
Cursor motion must answer within a frame; classification does not. YOLO at
18 Hz inline would halve the cursor rate. It runs on its own thread behind a
**depth-1, newest-wins queue**: when the worker is busy the main loop
overwrites the pending frame and moves on. Each branch also gets its **own FSM
instance** — interleaving two vocabularies at two rates in one sliding window
would leave neither able to stabilise.

### `idle` is an abstention, not a state
The most important temporal decision. Curling from `point` to `grip` passes
through a half-curled hand that reads as `idle`. Once `idle` stabilises it
becomes the predecessor and the transition never fires. *Measured at 30 Hz with
`idle` fed in, the click survives a 33 ms curl and dies at 66 ms — it never
works for a real hand.* Widening the stability window also fixes it, at the
cost of a 500 ms click.

### Config-driven engine with zero gesture names in the code
`gesture_fsm.py` does not contain a single gesture name. Every pose,
transition, action and timing value arrives from JSON, so adding a gesture is a
configuration change and never a code change. Hardcoded fallbacks were
deliberately deleted rather than left in place — advertising defaults the
program no longer honoured was worse than having none.

### Empirical justification over intuition
Constants are chosen from measurement and the measurement is recorded beside
the constant. Several findings contradict the obvious optimisation: shrinking
MediaPipe's input 11× buys 1.16×, and a fractional downscale costs 12× an
integer one.

### Fail soft at every optional boundary
pynput, Ultralytics, screeninfo, psutil, NVML, the keyboard, the GPU, libXtst —
every one is optional and every one degrades to a printed line. The cursor path
is the only thing that must not fail, and gesture arbitration is explicitly
fenced in its own `try`/`except` so a broken user binding costs the action and
not a frame of movement.

### ctypes backends behind one interface
`Win32Backend` (user32) and `X11Backend` (libX11 / libXtst / libXinerama)
implement the same five methods, so one file runs on both platforms with
nothing but ctypes underneath.

---

## Problems Solved

| Problem | Cause | Solution |
|---|---|---|
| Cursor lurched on every gesture | Anchored on a fingertip | Moved to landmark 9 — 108 px → 0.0 px |
| Fast motion felt worse after smoothing | Blocking sub-frame interpolation | Removed it; One-Euro became the only smoothing stage |
| Cursor slid across the screen on re-entry | Filters held a position from before the hand left | `reset()` on re-entry — next call adopts the raw target verbatim |
| Click never fired for a real hand | `idle` stabilised between poses and became the predecessor | Abstention filtering; click now lands 33 ms after the pose settles |
| Screen edges and taskbar unreachable | Symmetric vertical margin | Split into `MARGIN_TOP 0.10` / `MARGIN_BOTTOM 0.28` |
| Secondary monitors unreachable | Interpolating onto `[0, width]` with a negative desktop origin | Map onto `[left, left+width]` throughout |
| Scaled displays reported wrong coordinates | Process not DPI-aware; Windows virtualises the metrics | `SetProcessDpiAwareness(2)` before any metric read, two fallbacks |
| Display hot-plug froze the mapping | Geometry read once at start-up | Re-read every 2.5 s; degenerate `0×0` ignored rather than adopted |
| Preview zoomed / cropped on high-res cameras | Driver ignored the requested size | Downscale in the capture thread; all geometry rebuilt from the measured frame |
| Cameras reported working delivered nothing | Scanner accepted "opened" as success | Require a real frame, 5 retries, distinct message per failure mode |
| Fractional downscale 12× slower than integer | 640→256 misses OpenCV's fast path | Integer divisors only — also preserves aspect exactly |
| YOLO dropped gestures and ran at 1.8 Hz | ONNX export through `cv2.dnn` | Switched to `.pt` through Ultralytics: 554.7 ms → 55 ms |
| Eleven-second frozen start-up | `import ultralytics` costs 10.8 s on the main thread | Deferred to the worker thread; construction is 0.1 ms |
| Worker read pixels being overwritten | Handed the worker a view of the preallocated buffer | `bgr_buf.copy()` at the hand-off |
| Macro fired repeatedly while a pose was held | Level-triggered with a cooldown — 5 fires per 10 s hold | Edge-triggered plus cooldown: 1 fire per hold |
| Windows key could be left held down | Chord raised between press and release | Release from a `finally` over the keys actually pressed |
| Mouse button could outlive the process | Drag assumed a matching stop would arrive | Four independent `DRAG_STOP` paths |
| Orientation verdict was exactly backwards | Base-case sign error; both relative corrections were correct, so symmetry tests passed | Flipped the one comparison; anchored verification to a physical ground truth |
| `three_gun` orientation flickered | Knuckles overlap on x in a profile pose — 96% of frames undecidable | Read the wrist→index-tip axis: 25× wider baseline, 252 → 0 flips per 3,000 frames |
| `stop` and `stop_inverted` arrived under two names | Trained class competed with the landmark override | `LABEL_ALIASES` folds both onto one spelling |
| Four TFLite/absl warnings on every start | Written to fd 2 by C++ before absl initialises | fd-level redirect around the graph build, restored in a `finally` |
| Cursor jumped between hands | `multi_hand_landmarks[0]` is not order-stable | `pick_primary_hand()` — nearest anchor: 499 px → 13 px |

---

## Installation

### Requirements

| Requirement | Detail |
|---|---|
| **Python** | 3.11 (the reference environment is 3.11.15) |
| **OS** | Windows 10/11, or Linux with an X11 session |
| **Webcam** | Any UVC device; 720p recommended |
| **GPU** | Optional — CUDA accelerates YOLO; CPU works but is ~4× slower |
| **Git LFS** | Required — the model weights are LFS objects |

> **Linux:** `libx11-6` must be present (it is on any desktop install); add
> `libxtst6` for clicking and `libxinerama1` for a monitor count. A native
> Wayland session will not accept cursor warps — run under X11 or XWayland.

### 1. Clone

```bash
git lfs install
git clone https://github.com/BangerAtifAhmed/Mouse_Gestures_V1.git
cd Mouse_Gestures_V1
```

Git LFS matters: without it `model/YOLOv10n_gestures.pt` arrives as a 133-byte
pointer file and the semantic branch will not load.

### 2. Create an environment

```bash
conda create -n capstone python=3.11
conda activate capstone
```

or

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux
```

### 3. Install dependencies

`others/requirements.txt` is **out of date** — it pins `mediapipe==0.8.11` and
lists neither `torch`, `ultralytics`, `screeninfo`, `psutil` nor
`nvidia-ml-py`. Install the actual dependency set:

```bash
pip install opencv-python mediapipe numpy pynput pillow
pip install ultralytics
pip install screeninfo psutil nvidia-ml-py
```

For CUDA acceleration, install the matching PyTorch build before Ultralytics:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

| Package | Required for |
|---|---|
| `opencv-python`, `mediapipe`, `numpy` | Capture, landmarks, buffers — **mandatory** |
| `pynput` | Mouse and keyboard output (ctypes fallback exists) |
| `pillow` | GUI gesture thumbnails |
| `ultralytics` (+ `torch`) | The neural gesture branch — optional, fails soft |
| `screeninfo` | Per-monitor enumeration (ctypes fallback exists) |
| `psutil`, `nvidia-ml-py` | Diagnostics — optional, fails soft |

### 4. Verify

```bash
python -c "import cv2, mediapipe, numpy; print('core OK')"
python -c "from ultralytics import YOLO; print('model branch OK')"
```

---

## Configuration

Everything lives in **one file**: `gesture_config.json`, next to the source.

```json
{
  "settings": {
    "cursor_sensitivity": 1.4,
    "is_mirrored": true,
    "invert_cursor_x": false,
    "ai_confidence": 0.6,
    "target_screen": "Screen 1"
  },
  "geometry_bindings": [],
  "yolo_bindings": [],
  "deleted_bindings": []
}
```

### Settings

| Key | Default | Range | Meaning |
|---|---|---|---|
| `cursor_sensitivity` | 1.4 | 1.0 – 3.0 | Gain. Higher reaches screen edges from smaller hand movement |
| `is_mirrored` | `true` | bool | Mirrors the preview *and* the control direction |
| `invert_cursor_x` | `false` | bool | Control-only X inversion |
| `ai_confidence` | 0.6 | 0.1 – 0.9 | Shared detection floor for MediaPipe and YOLO |
| `target_screen` | `"All Screens"` | `"All Screens"` or `"Screen N"` | Which display the cursor is confined to |

> `is_mirrored` and `invert_cursor_x` are a **mirror pair — enable exactly
> one.** Both reflect the same axis, so turning on both (or neither) cancels
> out and the cursor runs backwards under a preview that looks correct.

### Binding lists

| List | Purpose |
|---|---|
| `geometry_bindings` | Rules fed by the geometric classifier |
| `yolo_bindings` | Rules fed by the neural classifier |
| `deleted_bindings` | The GUI recycle bin — **never compiled**, so a binned rule cannot fire |

### Rule schema

```jsonc
// Transition rule
{
  "id": "rule-0",
  "name": "point → grip",
  "enabled": true,
  "trigger": "transition",
  "from_state": "point",
  "to_state": "grip",
  "action": "LEFT_CLICK",
  "max_time_sec": 0.8,
  "cooldown_sec": 0.35,
  "promote_double": false
}

// Hold rule
{
  "id": "rule-1",
  "name": "hold timeout",
  "enabled": true,
  "trigger": "hold",
  "pose": "timeout",
  "action": "SHOW_DESKTOP",
  "hold_sec": 0.4,
  "cooldown_sec": 2.0,
  "repeat": false
}
```

A malformed rule is skipped with a printed reason; the rest of the file still
loads.

### FSM tuning defaults

These have working values in `DEFAULT_SETTINGS` and are merged in at
construction, so a fresh config file stays readable:

| Key | Default | Meaning |
|---|---|---|
| `window_size` / `stability_threshold` | 3 / 2 | Sliding-window majority vote |
| `ignored_states` | `["idle", ""]` | Labels treated as abstentions |
| `double_click_sec` | 0.8 | Window for `promote_double` |
| `transition_memory` | 2 | How far back a transition looks for its origin |
| `stuck_release_sec` | 0.5 | Drag with no hand is released |
| `drag_timeout_sec` | 30.0 | No drag survives this |
| `default_cooldown_sec` | 0.35 | Applied to rules without their own |
| `default_hold_sec` | 0.4 | Applied to holds without their own |

---

## Usage

### GUI (recommended)

```bash
python app.py
```

Three tabs:

| Tab | Purpose |
|---|---|
| **Mapping Studio** | Gesture catalogue, rule builder, rule table, cursor settings |
| **Live Camera & Diagnostics** | Camera selection, live preview, CPU/RAM/GPU/VRAM, engine FPS |
| **Recycle Bin** | Restore or permanently purge deleted bindings |

Workflow: connect a camera → pick a pose from the catalogue → choose a trigger
and an action → **Commit**. Bindings save to `gesture_config.json` and the
engine compiles them at start-up.

### Headless preview

```bash
python hand_cursor_2.py
```

Scans cameras, prompts for an index, opens a preview window and starts
tracking.

### Standalone tools

```bash
python system_monitor.py [pid]      # resource monitor window
python gesture_fsm.py               # FSM self-check on a scripted hand
python model/ptmodel.py             # raw YOLO webcam demo
```

### Runtime controls

Keys are read by the **preview window** — it must have focus.

| Key | Action |
|---|---|
| `+` / `=` | Increase sensitivity (step 0.1) |
| `-` / `_` | Decrease sensitivity |
| `m` | Toggle mirroring (picture **and** control direction) |
| `i` | Toggle X inversion (control only) |
| `q` | Quit |

---

## Limitations

- **One hand controls the cursor.** Two hands are tracked and both feed the
  neural branch, but only the primary hand drives the pointer.
- **No default bindings.** A fresh clone moves the cursor and nothing else
  until bindings are created.
- **Static poses only.** No trajectory, swipe or zoom gestures.
- **Orientation detection has pose assumptions.** The knuckle method assumes a
  roughly upright hand; the barrel method assumes a thumb-up gun. Outside those
  the verdict is `None` or wrong.
- **Neural branch runs at ~18 Hz**, so it is unsuitable for latency-critical
  actions. This is why clicking is driven by the geometric branch.
- **Windows is the primary target.** The Linux/X11 path exists and is
  structured but is less exercised; Wayland is not supported.
- **Per-process VRAM is unreadable under WDDM** — the diagnostics panel reports
  this rather than guessing.
- **Lighting and background sensitivity** are inherited from MediaPipe; poor
  light degrades landmark quality and everything downstream.
- **The engine is a single 4,635-line module.** It is structured internally but
  is not packaged.
- **`test.py` and `testmoniter.py` are stale duplicates**, not tests.

---

## Future Improvements

Genuinely planned or clearly implied by the current code. None of these are
implemented.

| Improvement | Rationale |
|---|---|
| **Request MJPG via FOURCC** | Uncompressed YUY2 at 1280×720×30 needs ~442 Mbit/s and will not fit USB 2.0, so cameras silently downgrade. One `CAP_PROP_FOURCC` call before the size set is the standard fix |
| **Move verification harnesses into `tests/`** | The assertion suites exist but live outside the repository; wiring them to pytest would make coverage claimable |
| **Package the engine** | Capture, tracking, mapping and output are already separable — they are simply not separated |
| **Update `others/requirements.txt`** | It pins `mediapipe==0.8.11` and omits torch, ultralytics, screeninfo, psutil and nvidia-ml-py |
| **Add `.gitignore`** | `__pycache__/*.pyc` is currently committed |
| **Remove unused YOLOv10x weights** | 373 MB of LFS storage the application never loads |
| **Calibration wizard** | The mapping already reads its bounds from four `MARGIN_*` constants in one place, so a capture step would only need to replace those numbers |
| **Dynamic gesture support** | Would need a trajectory buffer; the per-frame classifiers cannot express it |

---

## Resume Summary

### Resume Description

Built a real-time hand-gesture mouse controller in Python that maps MediaPipe
hand landmarks to OS cursor events across a multi-monitor virtual desktop,
combined with a fine-tuned YOLOv10n classifier (34 gesture classes) running
concurrently on a separate thread. Designed a dual-branch architecture with a
lock-free frame handoff so latency-critical cursor motion never blocks on
inference, and a JSON-configured finite state machine that turns pose
transitions and timed holds into mouse and keyboard actions.

### Resume Technologies

`Python · OpenCV · MediaPipe · YOLOv10 · PyTorch · CUDA · Tkinter · ctypes/Win32 · Multithreading`

### Resume Metrics

| Achievement | Value |
|---|---|
| YOLO inference latency reduction (ONNX/OpenCV-DNN → Ultralytics/CUDA) | **554.7 ms → 55 ms (10×)** |
| Semantic branch throughput | **1.8 Hz → 18 Hz** |
| Frames dropped by the inference queue | **83% → 27%** |
| Worst-case producer-side handoff latency | **6.2 ms** against a 55 ms inference pass |
| Gesture-induced cursor displacement eliminated | **108 px → 0.0 px** |
| Click latency after pose settles | **33 ms** (one frame at 30 Hz) |
| Orientation stability under landmark noise | **252 → 0 flips per 3,000 frames** |
| Start-up latency removed by deferred import | **10.8 s** |
| Gesture vocabulary | **34 neural + 5 geometric classes** |

---

## Technical Highlights

**A negative result that changed the design.** Shrinking MediaPipe's input
resolution 11× in pixel count bought only a 1.16× reduction in inference time
(37.29 ms → 32.34 ms). The palm and landmark networks run at fixed internal
input sizes, so downscaling only saves MediaPipe's own letterbox resize. The
obvious optimisation was mostly not there, and knowing that redirected effort
to the parts of the loop that could actually be improved.

**A benchmark that reversed an assumption.** Moving from an ONNX export through
`cv2.dnn` to the native `.pt` through Ultralytics was expected to cost
performance for accuracy. Head-to-head on identical inputs it was **10× faster**
(554.7 ms → 55 ms) *and* more confident. The cost was accepting PyTorch as a
runtime dependency.

**Backpressure instead of buffering.** The producer/consumer handoff uses a
depth-1 queue with newest-wins semantics: when the worker is busy the main loop
discards the pending frame and inserts the current one. 27% of frames are
dropped by design, worst-case producer latency is 6.2 ms against a 55 ms
consumer, and the cursor never waits. A conventional buffer would have traded
latency for frames nobody wanted.

**A bug that every symmetry test passed.** The palm-orientation logic shipped
inverted twice. The properties under test — "mirroring flips the verdict", "the
two hands disagree" — remain true when the base sign is globally wrong, so no
amount of symmetry checking could catch it. The fix was to anchor verification
to a single physical ground truth and derive the other seven permutations from
it.

**Choosing the right axis for the pose.** Palm-versus-back detection compares
the index and pinky knuckles, which works for an upright hand and fails for a
"gun" pose held in profile — the knuckles project onto nearly the same x and
96% of frames were undecidable. Switching that one pose to the wrist→fingertip
axis gave a 25× wider baseline and took verdict flips from 252 to 0 per 3,000
frames under realistic landmark noise.

**Output that outlives the process.** A held mouse button survives the program
that pressed it, so `DRAG_STOP` is emitted from four independent paths rather
than assumed, and the dispatcher clears its state only on a release the OS
accepted. The same reasoning applies to keyboard chords, which release from a
`finally` over the keys actually pressed — a chord that fails halfway cannot
leave the Windows key held down.

**Suppressing warnings that Python cannot see.** MediaPipe emits four
TFLite/absl lines written straight to file descriptor 2 by C++ before absl's
logger initialises. `GLOG_minloglevel`, `TF_CPP_MIN_LOG_LEVEL`,
`absl.logging.set_verbosity` and `warnings.filterwarnings` were all measured
and all left the output untouched. Redirecting the descriptor was the only
approach that worked — and it still needed a warm-up inference inside the
window, because the graph builds lazily on worker threads that flush after the
constructor returns.

---

<sub>Documentation reflects the repository state at the time of writing.
Performance figures are measured, not estimated; features not present in the
code are marked as such rather than described.</sub>
