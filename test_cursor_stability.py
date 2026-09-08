"""Regression tests for cursor stabilisation during gesture actions.

WHAT THIS COVERS.  CursorStabiliser — the class that decides whether the
cursor holds still while a gesture articulates — driven by the same
replay harness cursor_stability_probe uses, so the components under test
are the shipped ones: the real stabiliser, the real OneEuroFilter, and a
real GestureFSM answering armed_action_origin().

THE DEFECT THESE PIN DOWN.  The articulation hold used to release on the
frame the action fired, because the pose had by then become the
transition's DESTINATION and stopped being an armed origin.  Measured,
that lurched the cursor 34.7 px at the click and left it free to travel
38.5 px between the two clicks of a double — far outside the 4 px box
the OS needs to see them as one double-click.

NO CAMERA, NO MODEL, NO MOUSE, NO CONFIG FILE.

    python test_cursor_stability.py
"""
from __future__ import annotations

import inspect
import math
import unittest

import hand_cursor_2
from hand_cursor_2 import CursorStabiliser
from cursor_stability_probe import (DOUBLE_CLICK_BOX_PX, FOLD_FRAMES,
                                    Replay, Script, rule)

# Arbitrary invented labels.  Nothing in the stabiliser may know a pose
# name, so the whole suite is written in gestures that exist nowhere in
# the project's vocabulary.
AIM, CLOSED, SPLIT = "zeta_wave", "quux_grip", "delta_fan"

CLICK = rule(AIM, CLOSED, "LEFT_CLICK", promote_double=True,
             max_time_sec=2.0)
ALT = rule(SPLIT, CLOSED, "RIGHT_CLICK", max_time_sec=2.0)
DRAG = {"id": "drag", "trigger": "hold", "pose": AIM,
        "action": "DRAG_START", "hold_sec": 0.1, "cooldown_sec": 0.0,
        "enabled": True, "source": "any", "hand": "any"}


def aim_then_fold(phase=FOLD_FRAMES, rest=20):
    script = Script().still(rest, AIM).mark("fold")
    script.fold(AIM, CLOSED, frames=phase)
    script.still(10, CLOSED)
    return script


def double(phase=8, rest=20):
    script = Script().still(rest, AIM).mark("fold-1")
    script.fold(AIM, CLOSED, frames=phase).mark("unfold")
    script.unfold(CLOSED, AIM, frames=phase).mark("fold-2")
    script.fold(AIM, CLOSED, frames=phase).still(10, CLOSED)
    return script


def drift(replay, mark, span=FOLD_FRAMES):
    start = replay.script.marks[mark]
    return replay.drift_over(start, start + span)


class ActionSettleTests(unittest.TestCase):
    """The hold outlives the action that ends the transition."""

    def test_2_a_single_click_no_longer_lurches(self):
        after = Replay(aim_then_fold(), [CLICK]).run()
        before = Replay(aim_then_fold(), [CLICK], settle=False).run()
        self.assertGreater(drift(before, "fold"), 20.0,
                           "the defect did not reproduce")
        self.assertLess(drift(after, "fold"), 5.0)

    def test_4_a_double_lands_both_clicks_on_the_same_pixel(self):
        after = Replay(double(), [CLICK]).run()
        self.assertEqual(len(after.clicks), 2, after.clicks)
        spread, _gap = after.spread_between_clicks()
        self.assertLessEqual(spread, DOUBLE_CLICK_BOX_PX,
                             "the OS would see two singles, not a double")

    def test_4_the_defect_reproduces_without_the_settle(self):
        before = Replay(double(), [CLICK], settle=False).run()
        spread, _gap = before.spread_between_clicks()
        self.assertGreater(spread, DOUBLE_CLICK_BOX_PX)

    def test_4_the_cursor_is_still_across_the_whole_double(self):
        after = Replay(double(), [CLICK]).run()
        self.assertLess(drift(after, "unfold"), 3.0,
                        "the cursor moved while the hand re-opened")
        self.assertLess(drift(after, "fold-2"), 3.0)

    def test_3_a_second_mapping_gets_the_same_treatment(self):
        """Action-agnostic: RIGHT_CLICK is not special-cased anywhere."""
        script = Script().still(20, SPLIT).mark("fold")
        script.fold(SPLIT, CLOSED).still(10, CLOSED)
        after = Replay(script, [CLICK, ALT]).run()
        self.assertEqual([a for _i, a, _x, _y, _t in after.clicks],
                         ["RIGHT_CLICK"])
        self.assertLess(drift(after, "fold"), 5.0)

    def test_the_pin_does_not_move_when_the_hold_is_extended(self):
        stab = CursorStabiliser()
        for index in range(hand_cursor_2.MOTION_BASELINE_FRAMES):
            stab.update(100.0, 100.0, 10.0 + index * 0.033, None)
        stab.update(140.0, 100.0, 11.0, 1.0)
        stab.update(180.0, 100.0, 11.05, 1.0)
        pinned = stab.sem_pin
        self.assertIsNotNone(pinned)

        stab.note_action(11.1)
        self.assertEqual(stab.sem_pin, pinned,
                         "the settle re-pinned to the drifted point")


class HoldReleaseTests(unittest.TestCase):
    """The settle must not become a stuck cursor."""

    def test_1_normal_movement_is_never_held(self):
        script = Script().still(10, "unbound_pose").mark("travel")
        script.travel(30, 600.0, "unbound_pose")
        replay = Replay(script, [CLICK]).run()
        self.assertEqual(replay.frozen_frames(
            script.marks["travel"], script.marks["travel"] + 30), 0)

    def test_6_pointing_while_holding_an_origin_pose_stays_live(self):
        script = Script().still(10, AIM).mark("travel")
        script.travel(30, 600.0, AIM)
        replay = Replay(script, [CLICK]).run()
        held = replay.frozen_frames(script.marks["travel"],
                                    script.marks["travel"] + 30)
        self.assertLessEqual(held, 4, "deliberate pointing was blocked")

    def test_6_moving_away_after_a_click_releases_the_settle(self):
        script = Script().still(20, AIM).mark("fold")
        script.fold(AIM, CLOSED)
        script.mark("travel").travel(30, 600.0, CLOSED)
        replay = Replay(script, [CLICK]).run()

        start = script.marks["travel"]
        # The settle is 800 ms; the hand must break out well inside it.
        held = replay.frozen_frames(start, start + 30)
        self.assertLess(held, 12,
                        "the cursor stayed stuck after the click")
        tail = replay.cursor[-1]
        self.assertFalse(tail[3], "the cursor never came back")

    def test_the_settle_expires_on_its_own(self):
        stab = CursorStabiliser()
        for index in range(hand_cursor_2.MOTION_BASELINE_FRAMES):
            stab.update(100.0, 100.0, 10.0 + index * 0.033, None)
        stab.update(140.0, 100.0, 11.0, 1.0)
        stab.update(180.0, 100.0, 11.05, 1.0)
        stab.note_action(11.1)

        # Well inside MOTION_RELEASE_PX, so only the settle is in play.
        held = stab.update(150.0, 100.0, 11.2, None)
        self.assertTrue(stab.frozen, held)

        past = 11.1 + hand_cursor_2.ACTION_SETTLE_MS / 1000.0 + 0.05
        stab.update(150.0, 100.0, past, None)
        self.assertFalse(stab.frozen, "the settle never expired")

    def test_the_safety_ceiling_still_bounds_the_hold(self):
        stab = CursorStabiliser()
        for index in range(hand_cursor_2.MOTION_BASELINE_FRAMES):
            stab.update(100.0, 100.0, 10.0 + index * 0.033, None)
        stab.update(140.0, 100.0, 11.0, 1.0)
        stab.update(180.0, 100.0, 11.05, 1.0)

        ceiling = hand_cursor_2.SEMANTIC_FREEZE_MAX_MS / 1000.0
        stab.update(181.0, 100.0, 11.05 + ceiling + 0.1, 1.0)
        self.assertFalse(stab.frozen,
                         "the ceiling stopped bounding the hold")

    def test_note_action_without_a_pin_is_harmless(self):
        stab = CursorStabiliser()
        stab.note_action(10.0)
        self.assertIsNone(stab.sem_pin)
        self.assertEqual(stab.action_hold_until, 0.0)
        x, y = stab.update(500.0, 400.0, 10.1, None)
        self.assertEqual((x, y), (500.0, 400.0))
        self.assertFalse(stab.frozen)

    def test_reset_clears_the_settle(self):
        stab = CursorStabiliser()
        for index in range(hand_cursor_2.MOTION_BASELINE_FRAMES):
            stab.update(100.0, 100.0, 10.0 + index * 0.033, None)
        stab.update(140.0, 100.0, 11.0, 1.0)
        stab.update(180.0, 100.0, 11.05, 1.0)
        stab.note_action(11.1)
        stab.reset()
        self.assertEqual(stab.action_hold_until, 0.0)
        self.assertIsNone(stab.sem_pin)
        self.assertFalse(stab.frozen)


class JitterTests(unittest.TestCase):
    """Landmark noise must not move the cursor, or freeze it."""

    def test_7_a_stationary_hand_barely_moves_the_cursor(self):
        script = Script().still(40, AIM)
        replay = Replay(script, [CLICK]).run()
        self.assertLess(replay.drift_over(10, 40), 1.5,
                        "landmark jitter reached the cursor")

    def test_7_a_stationary_hand_is_not_pinned_by_the_action_hold(self):
        script = Script().still(40, AIM)
        replay = Replay(script, [CLICK]).run()
        held = sum(1 for _i, _x, _y, frozen, reason in replay.cursor
                   if frozen and reason == "action-settle")
        self.assertEqual(held, 0,
                         "a still hand triggered the action hold")

    def test_7_jitter_alone_never_arms_the_articulation_hold(self):
        stab = CursorStabiliser()
        now = 10.0
        for index in range(60):
            now += 0.033
            wander = 0.45 * math.sin(index * 2.1)
            stab.update(100.0 + wander, 100.0 + wander, now, 1.0)
        self.assertIsNone(stab.sem_pin,
                          "sub-pixel wander armed the freeze")


class DragTests(unittest.TestCase):
    """A drag needs the cursor live; the settle must not touch it."""

    def test_5_the_engine_skips_the_settle_while_dragging(self):
        """Both dispatch sites guard the settle behind the drag flag."""
        body = inspect.getsource(hand_cursor_2.HandTrackerEngine._run)
        self.assertEqual(body.count("stabiliser.note_action(now_ts)"), 2)
        self.assertEqual(body.count("if not actions.dragging:"), 2)
        # And never unguarded: every call sits under the guard.
        for chunk in body.split("stabiliser.note_action(now_ts)")[:-1]:
            self.assertTrue(chunk.rstrip().endswith(
                "if not actions.dragging:"), chunk[-120:])

    def test_5_a_held_drag_still_follows_the_hand(self):
        script = Script().still(20, AIM).mark("travel")
        script.travel(30, 600.0, AIM)
        replay = Replay(script, [DRAG]).run()
        # A hold rule fires DRAG_START; the replay never calls
        # note_action for it, so travel must remain unheld.
        self.assertEqual(replay.frozen_frames(
            script.marks["travel"], script.marks["travel"] + 30), 0)
        last = replay.cursor[-1]
        self.assertGreater(last[1], 1200.0, "the cursor did not travel")


class NoHardcodedNamesTests(unittest.TestCase):
    """The stabiliser must not know what any gesture is called."""

    FORBIDDEN = ("two_up", "fist", "one", "peace", "grip", "point",
                 "open", "palm", "three_gun", "stop", "double_click",
                 "left_click", "right_click")

    def test_10_the_stabiliser_names_no_gesture_and_no_action(self):
        body = inspect.getsource(CursorStabiliser).lower()
        for label in self.FORBIDDEN:
            self.assertNotIn('"%s"' % label, body)
            self.assertNotIn("'%s'" % label, body)

    def test_10_note_action_is_told_nothing_about_the_action(self):
        signature = inspect.signature(CursorStabiliser.note_action)
        self.assertEqual(list(signature.parameters), ["self", "now"])

    def test_10_the_whole_suite_ran_on_invented_gestures(self):
        for label in (AIM, CLOSED, SPLIT):
            self.assertNotIn(label, hand_cursor_2.GEOMETRY_VOCABULARY)
            self.assertNotIn(label, hand_cursor_2.YOLO_CLASS_NAMES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
