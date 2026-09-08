"""Regression tests for what reaches the mouse backend.

WHAT THIS COVERS.  ActionDispatcher — the object that turns an FSM action
string into button presses — driven with a recording cursor, so what is
asserted is the exact sequence of backend calls rather than a description
of them.  The full path is exercised end to end where it matters: a real
GestureFSM produces the ActionEvent, and that same event is handed to a
real ActionDispatcher.

THE DEFECT THESE PIN DOWN.  DOUBLE_CLICK was dispatched as `count=1`,
identical to LEFT_CLICK, so the action depended on the desktop's own
double-click timer fusing two separately dispatched events.  At the
semantic branch's 4.7 Hz that timer is usually missed.
gesture_fsm.ActionExecutor had always sent two; this path disagreed.

THE TRAP THEY GUARD.  A rule with promote_double emits LEFT_CLICK for
the first gesture and DOUBLE_CLICK for the second.  Sending two presses
for that second event would put three clicks on the desktop, which reads
as a triple-click.

NO CAMERA, NO MODEL, NO REAL MOUSE, NO CONFIG FILE.

    python test_action_dispatch.py
"""
from __future__ import annotations

import unittest

import hand_cursor_2
from hand_cursor_2 import ActionDispatcher
from gesture_fsm import (DOUBLE_CLICK, DRAG_START, DRAG_STOP, GestureFSM,
                         LEFT_CLICK, RIGHT_CLICK)

# Invented labels: nothing in the dispatch path may know a gesture name.
AIM, CLOSED, SPLIT = "zeta_wave", "quux_grip", "delta_fan"


class RecordingCursor:
    """Records backend calls instead of touching the desktop."""

    name = "recording"

    def __init__(self):
        self.calls = []

    def move(self, x, y):
        self.calls.append(("move", x, y))

    def click(self, button="left", count=1):
        self.calls.append(("click", button, count))

    def press(self, button="left"):
        self.calls.append(("press", button))

    def release(self, button="left"):
        self.calls.append(("release", button))

    def clicks(self):
        """(button, count) for every click call, in order."""
        return [(b, c) for kind, b, c in self.calls if kind == "click"]

    def presses(self):
        """Total button presses the OS would see."""
        return sum(c for _b, c in self.clicks())


def dispatcher():
    cursor = RecordingCursor()
    return ActionDispatcher(cursor), cursor


def rule(action, **extra):
    out = {"id": extra.pop("id", "r-%s" % action.lower()),
           "trigger": "transition", "from_state": AIM, "to_state": CLOSED,
           "action": action, "enabled": True, "cooldown_sec": 0.0,
           "source": "any", "hand": "any", "max_time_sec": 5.0}
    out.update(extra)
    return out


def config(*rules):
    return {"settings": {}, "geometry_bindings": list(rules),
            "yolo_bindings": [], "deleted_bindings": []}


class Gesture:
    """Drives a real FSM and feeds every event to a real dispatcher."""

    def __init__(self, *rules, **settings):
        self.fsm = GestureFSM(config(*rules), stability_threshold=1,
                              window_size=1, transition_memory=6,
                              **settings)
        self.dispatcher, self.cursor = dispatcher()
        self.now = 100.0
        self.events = []

    def perform(self, *poses, step=0.21):
        for pose in poses:
            self.now += step
            action = self.fsm.update(pose, self.now, source="semantic")
            if action is not None:
                self.events.append(str(action))
                self.dispatcher.dispatch(action, self.now)
        return self


# ─── 1-3: what each action puts on the wire ─────────────────────────────

class ClickCountTests(unittest.TestCase):

    def test_1_left_click_is_one_click(self):
        disp, cursor = dispatcher()
        disp.dispatch(LEFT_CLICK, 1.0)
        self.assertEqual(cursor.clicks(), [("left", 1)])

    def test_2_right_click_is_one_right_click(self):
        """RIGHT_CLICK never reaches the count branch at all.

        It is not in (LEFT_CLICK, DOUBLE_CLICK), so it falls through to
        ActionExecutor, which has always sent exactly one.
        """
        disp, cursor = dispatcher()
        executed = []

        class Executor:
            def dispatch(self, action, now):
                executed.append(str(action))
                return True

        disp._executor = Executor()
        disp.dispatch(RIGHT_CLICK, 1.0)
        self.assertEqual(executed, [RIGHT_CLICK])
        self.assertEqual(cursor.clicks(), [],
                         "a right click went down the left-button path")

    def test_3_double_click_is_two_left_clicks(self):
        disp, cursor = dispatcher()
        disp.dispatch(DOUBLE_CLICK, 1.0)
        self.assertEqual(cursor.clicks(), [("left", 2)])
        self.assertEqual(cursor.presses(), 2)

    def test_3_the_two_presses_are_one_backend_operation(self):
        """One call, not two — so nothing can interleave between them."""
        disp, cursor = dispatcher()
        disp.dispatch(DOUBLE_CLICK, 1.0)
        self.assertEqual(len(cursor.clicks()), 1)

    def test_a_failed_click_reports_nothing_delivered(self):
        class Broken(RecordingCursor):
            def click(self, button="left", count=1):
                raise OSError("no session")

        disp = ActionDispatcher(Broken())
        disp.dispatch(DOUBLE_CLICK, 1.0)
        self.assertEqual(disp.click_count, 0)


# ─── 4: the promotion must not be duplicated ────────────────────────────

class PromotionTests(unittest.TestCase):
    """Two gestures must put exactly two clicks on the desktop."""

    RULE = rule(LEFT_CLICK, promote_double=True, id="promoter")

    def promoted(self):
        run = Gesture(self.RULE)
        run.perform(AIM, CLOSED)          # first gesture  -> LEFT_CLICK
        run.perform(AIM, CLOSED)          # second         -> DOUBLE_CLICK
        return run

    def test_4_the_engine_really_does_promote(self):
        run = self.promoted()
        self.assertEqual(run.events, [LEFT_CLICK, DOUBLE_CLICK])

    def test_4_a_promoted_double_delivers_exactly_two_presses(self):
        run = self.promoted()
        self.assertEqual(run.cursor.presses(), 2,
                         "the desktop would have seen a triple-click")
        self.assertEqual(run.cursor.clicks(), [("left", 1), ("left", 1)])

    def test_4_a_standalone_double_still_delivers_two(self):
        """A rule bound directly to DOUBLE_CLICK is not a promotion."""
        run = Gesture(rule(DOUBLE_CLICK, id="explicit"))
        run.perform(AIM, CLOSED)
        self.assertEqual(run.events, [DOUBLE_CLICK])
        self.assertEqual(run.cursor.presses(), 2)

    def test_4_an_explicit_double_cancels_a_pending_promotion(self):
        """Documented engine behaviour, asserted at the backend.

        GestureFSM._fire clears every pending single when an explicit
        DOUBLE_CLICK fires, so the two cannot overlap into a triple.  The
        promoter's next gesture is therefore a fresh single, and the
        press counts must follow: 1 + 2 + 1.
        """
        promoter = rule(LEFT_CLICK, promote_double=True, id="promoter")
        explicit = rule(DOUBLE_CLICK, id="explicit",
                        from_state=SPLIT, to_state=CLOSED)
        run = Gesture(promoter, explicit)

        run.perform(AIM, CLOSED, step=0.05)        # LEFT_CLICK
        self.assertEqual(run.cursor.presses(), 1)

        run.perform(SPLIT, CLOSED, step=0.05)      # standalone double
        self.assertEqual(run.cursor.presses(), 3)

        run.perform(AIM, CLOSED, step=0.05)        # a fresh single
        self.assertEqual(run.events,
                         [LEFT_CLICK, DOUBLE_CLICK, LEFT_CLICK])
        self.assertEqual(run.cursor.presses(), 4)

    def test_4_a_promotion_survives_another_rule_clicking_between(self):
        """Rule identity, not recency, decides what a DOUBLE_CLICK means."""
        promoter = rule(LEFT_CLICK, promote_double=True, id="promoter")
        other = rule(LEFT_CLICK, id="other", from_state=SPLIT,
                     to_state=CLOSED)
        run = Gesture(promoter, other)

        run.perform(AIM, CLOSED, step=0.05)        # promoter single
        run.perform(SPLIT, CLOSED, step=0.05)      # a different rule
        run.perform(AIM, CLOSED, step=0.05)        # promoter completes

        self.assertEqual(run.events,
                         [LEFT_CLICK, LEFT_CLICK, DOUBLE_CLICK])
        self.assertEqual(run.cursor.presses(), 3,
                         "the promotion was scored as a standalone double")
        self.assertEqual(run.cursor.clicks(),
                         [("left", 1), ("left", 1), ("left", 1)])

    def test_4_a_third_gesture_starts_a_new_pair(self):
        run = self.promoted()
        run.perform(AIM, CLOSED)                   # a fresh single
        run.perform(AIM, CLOSED)                   # promotes again
        self.assertEqual(run.events, [LEFT_CLICK, DOUBLE_CLICK,
                                      LEFT_CLICK, DOUBLE_CLICK])
        self.assertEqual(run.cursor.presses(), 4)

    def test_4_a_lapsed_window_is_two_ordinary_clicks(self):
        run = Gesture(self.RULE)
        run.perform(AIM, CLOSED)
        run.now += 5.0                             # past double_click_sec
        run.perform(AIM, CLOSED)
        self.assertEqual(run.events, [LEFT_CLICK, LEFT_CLICK])
        self.assertEqual(run.cursor.presses(), 2)

    def test_a_bare_action_string_is_a_standalone_double(self):
        """No rule attached means the caller asked for a double outright."""
        disp, cursor = dispatcher()
        disp.dispatch(LEFT_CLICK, 1.0)
        disp.dispatch(DOUBLE_CLICK, 1.1)
        self.assertEqual(cursor.presses(), 3)


# ─── Gesture speed ──────────────────────────────────────────────────────

class SpeedTests(unittest.TestCase):
    """The delivered double must not depend on how fast the hand moves.

    Before this change the two clicks had to land inside the desktop's
    own double-click timer, so a slower gesture silently degraded to two
    single clicks.  Now the pair is one backend operation whenever the
    engine promotes, and the engine's own window is the only gate.
    """

    RULE = rule(LEFT_CLICK, promote_double=True, id="promoter")

    def run_at(self, gap):
        run = Gesture(self.RULE)
        run.perform(AIM, CLOSED)
        run.now += gap
        run.perform(AIM, CLOSED)
        return run

    def test_a_brisk_double_delivers_two_presses(self):
        run = self.run_at(0.10)
        self.assertEqual(run.events, [LEFT_CLICK, DOUBLE_CLICK])
        self.assertEqual(run.cursor.presses(), 2)

    def test_a_slow_double_inside_the_engine_window_still_delivers_two(self):
        """0.6 s would have missed a 500 ms desktop timer outright."""
        run = self.run_at(0.30)
        self.assertEqual(run.events, [LEFT_CLICK, DOUBLE_CLICK])
        self.assertEqual(run.cursor.presses(), 2)

    def test_every_speed_inside_the_window_delivers_exactly_two(self):
        for gap in (0.0, 0.05, 0.1, 0.2, 0.25):
            run = self.run_at(gap)
            self.assertEqual(run.cursor.presses(), 2,
                             "gap=%.2fs delivered %d presses"
                             % (gap, run.cursor.presses()))


# ─── 5-6: everything else is untouched ──────────────────────────────────

class UnchangedBehaviourTests(unittest.TestCase):

    def test_5_drag_start_presses_and_holds(self):
        disp, cursor = dispatcher()
        disp.dispatch(DRAG_START, 1.0)
        self.assertEqual(cursor.calls, [("press", "left")])
        self.assertTrue(disp.dragging)
        self.assertEqual(disp.drag_count, 1)

    def test_5_drag_stop_releases_once(self):
        disp, cursor = dispatcher()
        disp.dispatch(DRAG_START, 1.0)
        disp.dispatch(DRAG_STOP, 1.5)
        self.assertEqual(cursor.calls,
                         [("press", "left"), ("release", "left")])
        self.assertFalse(disp.dragging)

    def test_5_a_repeated_drag_start_does_not_press_twice(self):
        disp, cursor = dispatcher()
        disp.dispatch(DRAG_START, 1.0)
        disp.dispatch(DRAG_START, 1.1)
        self.assertEqual(cursor.calls, [("press", "left")])

    def test_5_a_double_click_never_touches_the_drag_state(self):
        disp, cursor = dispatcher()
        disp.dispatch(DOUBLE_CLICK, 1.0)
        self.assertFalse(disp.dragging)
        self.assertEqual(disp.drag_count, 0)

    def test_6_cooldown_is_the_engine_s_and_is_unchanged(self):
        """The dispatcher has no cooldown; the FSM gates the events."""
        cooled = rule(LEFT_CLICK, cooldown_sec=5.0, id="cooled")
        run = Gesture(cooled)
        run.perform(AIM, CLOSED)
        run.perform(AIM, CLOSED)
        self.assertEqual(run.events, [LEFT_CLICK])
        self.assertEqual(run.cursor.presses(), 1)

    def test_6_a_cooldown_lapsing_lets_the_next_click_through(self):
        cooled = rule(LEFT_CLICK, cooldown_sec=0.5, id="cooled")
        run = Gesture(cooled)
        run.perform(AIM, CLOSED)
        run.now += 1.0
        run.perform(AIM, CLOSED)
        self.assertEqual(run.events, [LEFT_CLICK, LEFT_CLICK])
        self.assertEqual(run.cursor.presses(), 2)

    def test_click_count_still_counts_actions_not_presses(self):
        disp, _cursor = dispatcher()
        disp.dispatch(LEFT_CLICK, 1.0)
        disp.dispatch(DOUBLE_CLICK, 1.1)
        self.assertEqual(disp.click_count, 2)

    def test_last_action_is_still_recorded(self):
        disp, _cursor = dispatcher()
        disp.dispatch(DOUBLE_CLICK, 7.5)
        self.assertEqual(disp.last_action, DOUBLE_CLICK)
        self.assertEqual(disp.last_action_time, 7.5)


# ─── The backends already spoke `count` ─────────────────────────────────

class BackendCountTests(unittest.TestCase):
    """No backend needed changing; each already honours a count."""

    def test_the_ctypes_backends_loop_press_release(self):
        import inspect
        for backend in (hand_cursor_2.Win32Backend,
                        hand_cursor_2.X11Backend):
            body = inspect.getsource(backend.click)
            self.assertIn("for _ in range(count):", body)
            self.assertIn("self.press(button)", body)
            self.assertIn("self.release(button)", body)

    def test_the_pynput_backend_forwards_the_count(self):
        import inspect
        body = inspect.getsource(hand_cursor_2.PynputCursor.click)
        self.assertIn("count", body)

    def test_native_cursor_forwards_the_count(self):
        recorded = []

        class Backend:
            def click(self, button, count):
                recorded.append((button, count))

        cursor = hand_cursor_2.NativeCursor(Backend())
        cursor.click("left", 2)
        self.assertEqual(recorded, [("left", 2)])

    def test_the_two_output_paths_now_agree(self):
        """ActionExecutor always sent two; the dispatcher now does too."""
        import inspect
        import gesture_fsm
        body = inspect.getsource(gesture_fsm.ActionExecutor.dispatch)
        self.assertIn("self._mouse.click(self._Button.left, 2)", body)

        disp, cursor = dispatcher()
        disp.dispatch(DOUBLE_CLICK, 1.0)
        self.assertEqual(cursor.presses(), 2)


class NoHardcodedNamesTests(unittest.TestCase):

    FORBIDDEN = ("two_up", "fist", "one", "peace", "grip", "point",
                 "open", "palm", "three_gun", "stop")

    def test_the_dispatch_path_names_no_gesture(self):
        import inspect
        for fn in (ActionDispatcher.dispatch, ActionDispatcher._clicks_for,
                   ActionDispatcher._note_click):
            body = inspect.getsource(fn).lower()
            for label in self.FORBIDDEN:
                self.assertNotIn('"%s"' % label, body)
                self.assertNotIn("'%s'" % label, body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
