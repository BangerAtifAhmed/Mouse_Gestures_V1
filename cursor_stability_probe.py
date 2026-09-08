"""Measure how far the cursor drifts while a gesture articulates.

READ-ONLY.  No camera, no model, no mouse, no config file: a scripted
hand trajectory is replayed through the REAL components — the project's
CursorStabiliser, its OneEuroFilter, and a real GestureFSM answering
armed_action_origin() — so what is measured is the shipped decision and
not a description of it.

THE TRAJECTORY is calibrated to the numbers recorded in hand_cursor_2's
own constant notes: a `one -> fist` fold travels ~48.6 px at the anchor,
its first frame moves ~4.2 px, and the geometric classifier does not
relabel until the fold is ~56% done.  Camera 30 Hz, YOLO 4.7 Hz.

    python cursor_stability_probe.py          # summary
    python cursor_stability_probe.py -v       # every frame
"""
from __future__ import annotations

import math
import sys

import hand_cursor_2
from hand_cursor_2 import CursorStabiliser, OneEuroFilter
from gesture_fsm import GestureFSM

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

# Windows assembles two ordinary clicks into a double only when they land
# inside its double-click box AND inside its double-click time.  Defaults.
DOUBLE_CLICK_BOX_PX = 4.0
DOUBLE_CLICK_TIME_MS = 500.0

CAMERA_HZ = 30.0
FRAME = 1.0 / CAMERA_HZ
YOLO_PERIOD = 0.213                 # ~4.7 Hz
FOLD_PX = 48.6                      # measured anchor travel of one -> fist
FOLD_FRAMES = 12                    # ~400 ms
JITTER_PX = 0.45                    # measured resting wander, under 1 px

SEMANTIC = {"stability_threshold": 1, "window_size": 1,
            "transition_memory": 6}


def config(*rules):
    return {"settings": {}, "geometry_bindings": list(rules),
            "yolo_bindings": [], "deleted_bindings": []}


def rule(a, b, action, **extra):
    out = {"id": "%s->%s" % (a, b), "trigger": "transition",
           "from_state": a, "to_state": b, "action": action,
           "enabled": True, "cooldown_sec": 0.0, "source": "any",
           "hand": "any"}
    out.update(extra)
    return out


def jitter(index):
    """Deterministic sub-pixel wander, so runs are comparable."""
    return (JITTER_PX * math.sin(index * 2.1),
            JITTER_PX * math.cos(index * 1.7))


class Script:
    """A hand trajectory, as (screen point, yolo label) per frame."""

    def __init__(self):
        self.frames = []
        self.x = 900.0
        self.y = 500.0
        self.marks = {}

    def mark(self, name):
        self.marks[name] = len(self.frames)
        return self

    def still(self, frames, label):
        for _ in range(frames):
            dx, dy = jitter(len(self.frames))
            self.frames.append((self.x + dx, self.y + dy, label))
        return self

    def travel(self, frames, distance, label, angle=0.0):
        """Deliberate pointing: the hand really goes somewhere."""
        step = distance / frames
        for _ in range(frames):
            self.x += step * math.cos(angle)
            self.y += step * math.sin(angle)
            dx, dy = jitter(len(self.frames))
            self.frames.append((self.x + dx, self.y + dy, label))
        return self

    def fold(self, label_from, label_to, frames=FOLD_FRAMES,
             distance=FOLD_PX):
        """A gesture articulating: the knuckle drags, then stops.

        The label flips at 56% of the fold, which is where the geometric
        classifier was measured to relabel.
        """
        start_x, start_y = self.x, self.y
        for index in range(frames):
            phase = (index + 1) / frames
            # Ease-out: the knuckle moves fastest as the fingers start.
            travelled = distance * (1.0 - (1.0 - phase) ** 2)
            self.x = start_x + travelled * 0.80
            self.y = start_y + travelled * 0.60
            dx, dy = jitter(len(self.frames))
            label = label_from if phase < 0.56 else label_to
            self.frames.append((self.x + dx, self.y + dy, label))
        return self

    def unfold(self, label_from, label_to, frames=FOLD_FRAMES,
               distance=FOLD_PX):
        """The hand opening again, between the two clicks of a double."""
        start_x, start_y = self.x, self.y
        for index in range(frames):
            phase = (index + 1) / frames
            travelled = distance * (1.0 - (1.0 - phase) ** 2)
            self.x = start_x - travelled * 0.80
            self.y = start_y - travelled * 0.60
            dx, dy = jitter(len(self.frames))
            label = label_from if phase < 0.56 else label_to
            self.frames.append((self.x + dx, self.y + dy, label))
        return self


class Replay:
    """One run of a trajectory through the real cursor pipeline."""

    def __init__(self, script, rules, label="", settle=True):
        self.script = script
        self.label = label
        # settle=False reproduces the behaviour before the fix, through
        # the very same code path, so the comparison cannot drift.
        self.settle = settle
        self.dragging = False
        self.fsm = GestureFSM(config(*rules), **SEMANTIC)
        self.stab = CursorStabiliser()
        self.oef_x = OneEuroFilter(freq=CAMERA_HZ,
                                   min_cutoff=hand_cursor_2.MIN_CUTOFF,
                                   beta=hand_cursor_2.BETA,
                                   d_cutoff=hand_cursor_2.D_CUTOFF)
        self.oef_y = OneEuroFilter(freq=CAMERA_HZ,
                                   min_cutoff=hand_cursor_2.MIN_CUTOFF,
                                   beta=hand_cursor_2.BETA,
                                   d_cutoff=hand_cursor_2.D_CUTOFF)
        self.cursor = []            # (frame, x, y, frozen, reason)
        self.clicks = []            # (frame, action, x, y)

    def run(self):
        now = 100.0
        next_yolo = now
        geometry_state = None

        for index, (raw_x, raw_y, label) in enumerate(self.script.frames):
            now += FRAME

            # The geometric classifier relabels at the camera rate and
            # arms the pose-change hold, exactly as the loop does.
            if label != geometry_state:
                geometry_state = label
                self.stab.arm_pose_change((raw_x, raw_y), now)

            # ORDER MATTERS, and this is the loop's order: the cursor
            # block runs FIRST and the semantic branch second.  So the
            # stabiliser is asked while the FSM still holds the ORIGIN
            # pose, and the click that the same frame goes on to fire is
            # delivered wherever the cursor was just put.
            armed = self.fsm.armed_action_origin(now, source="semantic")
            held_x, held_y = self.stab.update(raw_x, raw_y, now, armed)

            x = self.oef_x(held_x, timestamp=now)
            y = self.oef_y(held_y, timestamp=now)
            self.cursor.append((index, x, y, self.stab.frozen,
                                self.stab.reason))

            # The semantic branch only speaks at the network rate.
            action = None
            if now >= next_yolo:
                next_yolo = now + YOLO_PERIOD
                action = self.fsm.update(label, now, source="semantic")
            if action is not None:
                self.clicks.append((index, str(action), x, y, now))
                # The engine guards this with `not actions.dragging`,
                # because a drag exists to move the pointer.  Modelled
                # here so the harness cannot flatter the real loop.
                if str(action) == "DRAG_START":
                    self.dragging = True
                elif str(action) == "DRAG_STOP":
                    self.dragging = False
                if self.settle and not self.dragging:
                    self.stab.note_action(now)

            if VERBOSE:
                print("   %3d %-7s raw=(%7.1f,%7.1f) cur=(%7.1f,%7.1f) "
                      "frozen=%-5s %-15s armed=%s%s"
                      % (index, label, raw_x, raw_y, x, y,
                         self.stab.frozen, self.stab.reason,
                         "yes" if armed is not None else "no",
                         "   <<< %s" % action if action else ""))
        return self

    def spread_between_clicks(self):
        if len(self.clicks) < 2:
            return None
        (_i, _a, x1, y1, t1) = self.clicks[0]
        (_j, _b, x2, y2, t2) = self.clicks[1]
        return math.hypot(x2 - x1, y2 - y1), (t2 - t1) * 1000.0

    def drift_over(self, start, end):
        """Max distance from the cursor's position at `start`."""
        base = None
        worst = 0.0
        for index, x, y, _f, _r in self.cursor:
            if index < start:
                continue
            if index > end:
                break
            if base is None:
                base = (x, y)
            worst = max(worst, math.hypot(x - base[0], y - base[1]))
        return worst

    def frozen_frames(self, start, end):
        return sum(1 for index, _x, _y, frozen, _r in self.cursor
                   if start <= index <= end and frozen)


LEFT = rule("one", "fist", "LEFT_CLICK", promote_double=True,
            max_time_sec=2.0)


def double_click_script():
    """Aim, fold, unfold, fold again — the promoted double-click."""
    script = Script()
    script.still(20, "one").mark("fold-1")
    script.fold("one", "fist").mark("held-1")
    script.still(6, "fist").mark("unfold")
    script.unfold("fist", "one").mark("fold-2")
    script.fold("one", "fist").mark("held-2")
    script.still(10, "fist")
    return script


def report(title, replay, notes=()):
    print("\n%s" % title)
    marks = replay.script.marks
    print("   clicks: %s" % ([(i, a) for i, a, _x, _y, _t in replay.clicks]
                             or "none"))
    measured = replay.spread_between_clicks()
    if measured is not None:
        spread, gap_ms = measured
        print("   between the two clicks: %5.1f px  %6.0f ms   "
              "box %s   time %s"
              % (spread, gap_ms,
                 "OK " if spread <= DOUBLE_CLICK_BOX_PX else "FAIL",
                 "OK " if gap_ms <= DOUBLE_CLICK_TIME_MS else "FAIL"))
    for name, note in notes:
        start = marks.get(name)
        if start is None:
            continue
        end = start + FOLD_FRAMES
        print("   %-28s drift %6.1f px   frozen %d/%d frames"
              % (note, replay.drift_over(start, end),
                 replay.frozen_frames(start, end), FOLD_FRAMES + 1))


def main():
    print(__doc__.strip().splitlines()[0])
    print("MOTION_TRIGGER_PX=%.1f  MOTION_CONFIRM_FRAMES=%d  "
          "STATE_FREEZE_MS=%d  SEMANTIC_FREEZE_MAX_MS=%d"
          % (hand_cursor_2.MOTION_TRIGGER_PX,
             hand_cursor_2.MOTION_CONFIRM_FRAMES,
             hand_cursor_2.STATE_FREEZE_MS,
             hand_cursor_2.SEMANTIC_FREEZE_MAX_MS))

    notes = [("fold-1", "during the first fold"),
             ("unfold", "while the hand re-opens"),
             ("fold-2", "during the second fold")]

    report("DOUBLE_CLICK, BEFORE the fix (settle disabled)",
           Replay(double_click_script(), [LEFT], settle=False).run(), notes)
    report("DOUBLE_CLICK, AFTER the fix",
           Replay(double_click_script(), [LEFT]).run(), notes)

    fast = Script().still(20, "one").mark("fold-1")
    fast.fold("one", "fist", frames=6).mark("unfold")
    fast.unfold("fist", "one", frames=5).mark("fold-2")
    fast.fold("one", "fist", frames=6).still(8, "fist")
    report("DOUBLE_CLICK, a BRISK double (fits the OS 500 ms window)",
           Replay(fast, [LEFT]).run(), notes)

    # A single click, for comparison.
    single = Script().still(20, "one").mark("fold").fold("one", "fist")
    single.still(10, "fist")
    report("LEFT_CLICK, BEFORE the fix",
           Replay(single, [LEFT], settle=False).run(),
           [("fold", "during the fold")])
    report("LEFT_CLICK, AFTER the fix", Replay(single, [LEFT]).run(),
           [("fold", "during the fold")])

    # Deliberate pointing while holding an origin pose: must NOT freeze.
    moving = Script().still(10, "one").mark("travel")
    moving.travel(30, 600.0, "one")
    replay = Replay(moving, [LEFT], "pointing").run()
    print("\nPOINTING while holding the origin pose (must stay live)")
    print("   frozen frames during travel: %d/30"
          % replay.frozen_frames(replay.script.marks["travel"],
                                 replay.script.marks["travel"] + 30))

    # A stationary hand with landmark jitter: must NOT freeze either.
    stationary = Script().mark("rest").still(40, "one")
    replay = Replay(stationary, [LEFT], "resting").run()
    print("\nSTATIONARY hand with landmark jitter (must stay live)")
    print("   frozen frames: %d/40  cursor wander: %.2f px"
          % (replay.frozen_frames(0, 40), replay.drift_over(10, 40)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
