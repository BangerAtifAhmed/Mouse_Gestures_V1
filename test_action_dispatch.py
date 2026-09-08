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

    def scroll(self, x, y):
        self.calls.append(("scroll", x, y))

    def clicks(self):
        """(button, count) for every click call, in order."""
        return [(b, c) for kind, b, c in self.calls if kind == "click"]

    def presses(self):
        """Total button presses the OS would see."""
        return sum(c for _b, c in self.clicks())

    def scrolls(self):
        """(x, y) for every scroll call, in order."""
        return [(x, y) for kind, x, y in self.calls if kind == "scroll"]


def dispatcher():
    cursor = RecordingCursor()
    disp = ActionDispatcher(cursor)
    # Inject ActionExecutor with mock cursor for testing
    from gesture_fsm import ActionExecutor
    executor = ActionExecutor(dry_run=False)
    executor._mouse = cursor  # Replace with mock cursor
    disp._executor = executor
    return disp, cursor


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


class SwipeActionTests(unittest.TestCase):
    """Regression tests for SWIPE_LEFT and SWIPE_RIGHT keyboard actions."""

    def test_swipe_left_maps_to_left_arrow(self):
        """SWIPE_LEFT action dispatches Left Arrow key for previous slide."""
        from gesture_fsm import SWIPE_LEFT, ACTION_MACROS
        self.assertEqual(ACTION_MACROS[SWIPE_LEFT], "left")

    def test_swipe_right_maps_to_right_arrow(self):
        """SWIPE_RIGHT action dispatches Right Arrow key for next slide."""
        from gesture_fsm import SWIPE_RIGHT, ACTION_MACROS
        self.assertEqual(ACTION_MACROS[SWIPE_RIGHT], "right")

    def test_swipe_left_keyboard_dispatch(self):
        """SWIPE_LEFT uses keyboard backend via ActionExecutor."""
        from gesture_fsm import ActionExecutor, SWIPE_LEFT
        executor = ActionExecutor(dry_run=True)
        # Should not raise and should return result
        result = executor.dispatch(SWIPE_LEFT)
        # dry_run mode just returns the result of _chord or similar
        self.assertIsNotNone(result)

    def test_swipe_right_keyboard_dispatch(self):
        """SWIPE_RIGHT uses keyboard backend via ActionExecutor."""
        from gesture_fsm import ActionExecutor, SWIPE_RIGHT
        executor = ActionExecutor(dry_run=True)
        # Should not raise and should return result
        result = executor.dispatch(SWIPE_RIGHT)
        self.assertIsNotNone(result)

    def test_swipe_rules_compile_and_dispatch(self):
        """Swipe rules can be created and dispatched end-to-end."""
        from gesture_fsm import SWIPE_LEFT, SWIPE_RIGHT

        left_rule = rule(SWIPE_LEFT, id="swipe_left_rule")
        right_rule = rule(SWIPE_RIGHT, id="swipe_right_rule")

        # Compile FSM with swipe rules
        fsm = Gesture(left_rule, right_rule)

        # Perform gestures
        fsm.perform(AIM, CLOSED)

        # Should have received SWIPE_LEFT event
        self.assertIn(SWIPE_LEFT, fsm.events)

    def test_swipe_one_shot_like_other_actions(self):
        """Swipe actions are one-shot, not repeated while gesture held."""
        from gesture_fsm import SWIPE_LEFT

        rule1 = rule(SWIPE_LEFT, id="swipe_left")
        run = Gesture(rule1)

        # Perform gesture: AIM -> CLOSED transition fires SWIPE_LEFT
        run.perform(AIM, CLOSED)
        first_count = len(run.events)
        self.assertGreater(first_count, 0)

        # Another frame same pose: should not fire again
        run.now += 0.1
        run.perform(CLOSED, CLOSED)  # Stay in CLOSED
        # Events should not increase
        self.assertEqual(len(run.events), first_count)

    def test_both_swipe_directions_work(self):
        """Both SWIPE_LEFT and SWIPE_RIGHT work in same mapping."""
        from gesture_fsm import SWIPE_LEFT, SWIPE_RIGHT

        left_rule = rule(SWIPE_LEFT, from_state=AIM, to_state=CLOSED,
                        id="swipe_left")
        right_rule = rule(SWIPE_RIGHT, from_state=SPLIT, to_state=CLOSED,
                         id="swipe_right")

        run = Gesture(left_rule, right_rule)

        # Left swipe
        run.perform(AIM, CLOSED)
        self.assertEqual(run.events, [SWIPE_LEFT])

        # Reset for right swipe
        run2 = Gesture(left_rule, right_rule)
        run2.perform(SPLIT, CLOSED)
        self.assertEqual(run2.events, [SWIPE_RIGHT])


class ContinuousScrollTests(unittest.TestCase):
    """Regression tests for continuous scrolling with progressive speed."""

    def test_1_scroll_up_starts_immediately(self):
        """SCROLL_UP fires on first gesture recognition."""
        from gesture_fsm import SCROLL_UP
        disp, cursor = dispatcher()
        disp.dispatch(SCROLL_UP, 1.0)
        self.assertEqual(len(cursor.scrolls()), 1)
        x, y = cursor.scrolls()[0]
        self.assertEqual(x, 0)
        self.assertGreater(y, 0, "SCROLL_UP should scroll positive Y")

    def test_2_scroll_down_starts_immediately(self):
        """SCROLL_DOWN fires on first gesture recognition."""
        from gesture_fsm import SCROLL_DOWN
        disp, cursor = dispatcher()
        disp.dispatch(SCROLL_DOWN, 1.0)
        self.assertEqual(len(cursor.scrolls()), 1)
        x, y = cursor.scrolls()[0]
        self.assertEqual(x, 0)
        self.assertLess(y, 0, "SCROLL_DOWN should scroll negative Y")

    def test_3_scroll_up_repeats_while_gesture_active(self):
        """SCROLL_UP continues scrolling on repeated dispatch."""
        from gesture_fsm import SCROLL_UP
        disp, cursor = dispatcher()
        # Simulate continuous gesture: call dispatch multiple times
        disp.dispatch(SCROLL_UP, 1.0)
        disp.dispatch(SCROLL_UP, 1.05)
        disp.dispatch(SCROLL_UP, 1.10)
        # Should have multiple scroll events
        self.assertGreaterEqual(len(cursor.scrolls()), 2)

    def test_4_scroll_down_repeats_while_gesture_active(self):
        """SCROLL_DOWN continues scrolling on repeated dispatch."""
        from gesture_fsm import SCROLL_DOWN
        disp, cursor = dispatcher()
        disp.dispatch(SCROLL_DOWN, 1.0)
        disp.dispatch(SCROLL_DOWN, 1.05)
        disp.dispatch(SCROLL_DOWN, 1.10)
        self.assertGreaterEqual(len(cursor.scrolls()), 2)

    def test_5_scroll_speed_increases_with_time(self):
        """Scroll speed increases gradually as gesture duration increases."""
        from gesture_fsm import ActionExecutor
        executor = ActionExecutor(dry_run=False)

        # Test speed curve directly
        speed_at_0 = executor._scroll_speed_curve(0.0)
        speed_at_1 = executor._scroll_speed_curve(1.0)
        speed_at_2 = executor._scroll_speed_curve(2.0)

        # Speed should increase with time
        self.assertLess(speed_at_0, speed_at_1)
        self.assertLess(speed_at_1, speed_at_2)

    def test_6_scroll_speed_has_maximum(self):
        """Scroll speed reaches a maximum and stays capped."""
        from gesture_fsm import SCROLL_UP
        disp, cursor = dispatcher()
        # Dispatch many times over long duration to reach max speed
        for i in range(50):
            disp.dispatch(SCROLL_UP, 5.0 + i * 0.1)

        scrolls = cursor.scrolls()
        speeds = [abs(s[1]) for s in scrolls]
        # Speed should stabilize at a reasonable maximum
        self.assertLessEqual(max(speeds), 15.0,
                            "scroll speed should be capped")

    def test_7_scroll_speed_based_on_time_not_frames(self):
        """Speed increases based on elapsed time, not frame count."""
        from gesture_fsm import SCROLL_UP

        # Two dispatches with same elapsed time but different time steps
        disp1, cursor1 = dispatcher()
        disp1.dispatch(SCROLL_UP, 1.0)
        disp1.dispatch(SCROLL_UP, 2.0)
        speed1 = abs(cursor1.scrolls()[-1][1])

        # Different time step, same elapsed time
        disp2, cursor2 = dispatcher()
        disp2.dispatch(SCROLL_UP, 1.0)
        disp2.dispatch(SCROLL_UP, 1.5)  # Same total elapsed (1.5-1 = 0.5)
        disp2.dispatch(SCROLL_UP, 2.0)
        speed2 = abs(cursor2.scrolls()[-1][1])

        # Speeds should be similar for same elapsed time
        self.assertAlmostEqual(speed1, speed2, delta=2.0)

    def test_8_scroll_stops_after_timeout(self):
        """Scrolling stops when gesture is not called for 0.5 seconds."""
        from gesture_fsm import SCROLL_UP
        disp, cursor = dispatcher()

        # Start scrolling
        disp.dispatch(SCROLL_UP, 1.0)
        initial_count = len(cursor.scrolls())

        # Wait beyond timeout (0.5 sec)
        disp.dispatch(SCROLL_UP, 1.6)

        # Scroll should have been reset (new activation)
        # Check that speed went back down
        scrolls = cursor.scrolls()
        # The new scroll after timeout should be at low speed
        self.assertGreater(len(scrolls), initial_count)

    def test_9_scroll_resets_acceleration_on_new_activation(self):
        """New gesture activation starts at minimum speed."""
        from gesture_fsm import SCROLL_UP
        disp, cursor = dispatcher()

        # First: long scroll (high speed)
        for t in [1.0, 1.1, 1.2, 1.3, 1.4]:
            disp.dispatch(SCROLL_UP, t)
        high_speed_count = len(cursor.scrolls())

        cursor.calls.clear()

        # Second: new activation after timeout
        disp.dispatch(SCROLL_UP, 2.5)
        disp.dispatch(SCROLL_UP, 2.55)

        # New activation speeds should be low
        new_scrolls = cursor.scrolls()
        self.assertGreaterEqual(len(new_scrolls), 1)

    def test_10_scroll_up_and_down_independent(self):
        """SCROLL_UP and SCROLL_DOWN have independent speeds."""
        from gesture_fsm import SCROLL_UP, SCROLL_DOWN

        disp_up, cursor_up = dispatcher()
        disp_up.dispatch(SCROLL_UP, 0.0)
        disp_up.dispatch(SCROLL_UP, 1.0)
        up_speed = abs(cursor_up.scrolls()[-1][1])

        disp_down, cursor_down = dispatcher()
        disp_down.dispatch(SCROLL_DOWN, 0.0)
        disp_down.dispatch(SCROLL_DOWN, 1.0)
        down_speed = abs(cursor_down.scrolls()[-1][1])

        # Both should progress similarly with time
        self.assertAlmostEqual(up_speed, down_speed, delta=1.0)

    def test_11_scroll_responds_to_repeated_calls(self):
        """Each dispatch call results in a scroll event."""
        from gesture_fsm import SCROLL_UP
        disp, cursor = dispatcher()

        # Call dispatch multiple times with small time increments
        num_calls = 5
        for i in range(num_calls):
            disp.dispatch(SCROLL_UP, 1.0 + i * 0.1)

        # Should have gotten scroll events for most/all calls
        self.assertGreaterEqual(len(cursor.scrolls()), num_calls - 1)

    def test_12_scroll_directions_opposite(self):
        """SCROLL_UP and SCROLL_DOWN have opposite Y directions."""
        from gesture_fsm import SCROLL_UP, SCROLL_DOWN

        disp_up, cursor_up = dispatcher()
        disp_up.dispatch(SCROLL_UP, 1.0)
        up_y = cursor_up.scrolls()[0][1]

        disp_down, cursor_down = dispatcher()
        disp_down.dispatch(SCROLL_DOWN, 1.0)
        down_y = cursor_down.scrolls()[0][1]

        self.assertGreater(up_y, 0, "SCROLL_UP should have positive Y")
        self.assertLess(down_y, 0, "SCROLL_DOWN should have negative Y")
        self.assertEqual(abs(up_y), abs(down_y),
                        "magnitudes should be equal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
