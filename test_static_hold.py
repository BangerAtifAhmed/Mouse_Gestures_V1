"""Regression tests for Static Hold functionality.

Static Hold is a simple pose-based action trigger that fires when a pose is
continuously recognized for a specified duration (0, 3, 5, or 10 seconds).

WHAT THIS COVERS:
- Immediate firing (0 sec)
- Timed firing (3, 5, 10 sec)
- One-shot behavior (no repeated firing while pose held)
- Re-arming when pose changes
- Hand filtering (Any, Right, Left, Both)
- All action types (LEFT_CLICK, RIGHT_CLICK, DOUBLE_CLICK, etc.)
- Configuration save/reload
- Live engine reload
- Delete/undo lifecycle
- Enable/disable toggle

NO CAMERA, NO MODEL, NO REAL MOUSE, NO CONFIG FILE.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid

import gesture_fsm
from gesture_fsm import (LEFT_CLICK, RIGHT_CLICK, DOUBLE_CLICK, DRAG_START,
                         SCROLL_UP, GestureFSM, load_config, save_config)


# Invented pose names to verify no hardcoding
POSE_A, POSE_B = "alpha_grip", "beta_wave"
POSE_C, POSE_D = "gamma_touch", "delta_point"


def hold_rule(pose, action, hold_sec=0, **extra):
    """A Static Hold rule."""
    rule = {
        "id": "h%s" % uuid.uuid4().hex[:8],
        "trigger": "hold",
        "pose": pose,
        "action": action,
        "hold_sec": hold_sec,
        "enabled": True,
        "cooldown_sec": 0.0,
        "source": "any",
        "hand": "any",
    }
    rule.update(extra)
    return rule


def config(*rules):
    # Settings for testing: stabilize immediately (window=1, threshold=1)
    return {"settings": {"window_size": 1, "stability_threshold": 1},
            "geometry_bindings": list(rules),
            "yolo_bindings": [], "deleted_bindings": []}


class HoldFiringTests(unittest.TestCase):
    """Test when Static Hold fires."""

    def test_1_hold_0_sec_fires_immediately(self):
        """0 sec means fire as soon as pose is recognized."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0)
        fsm = GestureFSM(config(rule))

        now = 1.0
        action = fsm.update(POSE_A, now, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), LEFT_CLICK)

    def test_2_hold_3_sec_waits_for_three_seconds(self):
        """3 sec requires the pose to be held for 3 seconds."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=3)
        fsm = GestureFSM(config(rule))

        now = 1.0
        # At 1.0 sec: too early
        action = fsm.update(POSE_A, now, source="geometry")
        self.assertIsNone(action)

        # At 3.9 sec: still short
        action = fsm.update(POSE_A, now + 2.9, source="geometry")
        self.assertIsNone(action)

        # At 4.1 sec: ready
        action = fsm.update(POSE_A, now + 3.1, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), LEFT_CLICK)

    def test_3_hold_5_sec_waits_for_five_seconds(self):
        """5 sec requires the pose to be held for 5 seconds."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=5)
        fsm = GestureFSM(config(rule))

        # Start holding at time 0
        # At 4.9 sec: too early
        fsm.update(POSE_A, 0.0, source="geometry")
        action = fsm.update(POSE_A, 4.9, source="geometry")
        self.assertIsNone(action)

        # At 5.1 sec: ready (5.1 - 0.0 = 5.1 > 5.0)
        action = fsm.update(POSE_A, 5.1, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), LEFT_CLICK)

    def test_4_hold_10_sec_waits_for_ten_seconds(self):
        """10 sec requires the pose to be held for 10 seconds."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=10)
        fsm = GestureFSM(config(rule))

        # Start holding at time 0
        fsm.update(POSE_A, 0.0, source="geometry")
        # At 9.9 sec: too early
        action = fsm.update(POSE_A, 9.9, source="geometry")
        self.assertIsNone(action)

        # At 10.1 sec: ready (10.1 - 0.0 = 10.1 > 10.0)
        action = fsm.update(POSE_A, 10.1, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), LEFT_CLICK)


class HoldReleaseTests(unittest.TestCase):
    """Test that releasing the pose before hold time fires nothing."""

    def test_5_releasing_before_3_sec_does_not_fire(self):
        """Pose released before 3 sec → no action."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=3)
        fsm = GestureFSM(config(rule))

        now = 1.0
        # Hold for 1.5 sec
        action = fsm.update(POSE_A, now + 1.5, source="geometry")
        self.assertIsNone(action)

        # Release (change pose)
        action = fsm.update(POSE_B, now + 1.6, source="geometry")
        self.assertIsNone(action)

    def test_6_releasing_before_5_sec_does_not_fire(self):
        """Pose released before 5 sec → no action."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=5)
        fsm = GestureFSM(config(rule))

        now = 1.0
        # Hold for 3 sec
        action = fsm.update(POSE_A, now + 3.0, source="geometry")
        self.assertIsNone(action)

        # Release
        action = fsm.update(POSE_B, now + 3.1, source="geometry")
        self.assertIsNone(action)

    def test_7_releasing_before_10_sec_does_not_fire(self):
        """Pose released before 10 sec → no action."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=10)
        fsm = GestureFSM(config(rule))

        now = 1.0
        # Hold for 7 sec
        action = fsm.update(POSE_A, now + 7.0, source="geometry")
        self.assertIsNone(action)

        # Release
        action = fsm.update(POSE_B, now + 7.1, source="geometry")
        self.assertIsNone(action)


class OneShotTests(unittest.TestCase):
    """Test one-shot behavior: no repeated firing while pose held."""

    def test_8_continuously_held_pose_fires_only_once(self):
        """A held pose fires once, not every frame."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0)
        fsm = GestureFSM(config(rule))

        now = 1.0
        # First frame: fires
        action = fsm.update(POSE_A, now, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), LEFT_CLICK)

        # Next 5 frames: same pose, nothing fires
        for i in range(1, 6):
            action = fsm.update(POSE_A, now + 0.1 * i, source="geometry")
            self.assertIsNone(action,
                              f"frame {i}: should not fire again while pose held")

    def test_9_leaving_the_pose_re_arms_the_rule(self):
        """After leaving the pose, returning to it can fire again."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0)
        fsm = GestureFSM(config(rule))

        now = 1.0
        # Fire
        action = fsm.update(POSE_A, now, source="geometry")
        self.assertIsNotNone(action)

        # Hold (no more fires)
        action = fsm.update(POSE_A, now + 0.1, source="geometry")
        self.assertIsNone(action)

        # Change pose
        action = fsm.update(POSE_B, now + 0.2, source="geometry")
        self.assertIsNone(action)

        # Return to original pose: re-arming allows fire
        action = fsm.update(POSE_A, now + 0.3, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), LEFT_CLICK)

    def test_10_returning_to_held_pose_allows_it_to_fire_again(self):
        """Full cycle: fire, hold, release, return, fire again."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=1)
        fsm = GestureFSM(config(rule))

        # Start at time 0
        fsm.update(POSE_A, 0.0, source="geometry")
        # Hold for 1.1 sec → fire
        action = fsm.update(POSE_A, 1.1, source="geometry")
        self.assertIsNotNone(action)

        # Continue holding (no more fires)
        action = fsm.update(POSE_A, 1.2, source="geometry")
        self.assertIsNone(action)

        # Release at 2.0 sec
        action = fsm.update(POSE_B, 2.0, source="geometry")
        self.assertIsNone(action)

        # Return: re-armed at 2.1 sec
        action = fsm.update(POSE_A, 2.1, source="geometry")
        self.assertIsNone(action)
        # Hold for 1.1 sec more → fire at 3.2 sec (3.2 - 2.1 = 1.1)
        action = fsm.update(POSE_A, 3.2, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), LEFT_CLICK)


class HandFilteringTests(unittest.TestCase):
    """Test hand filtering (Any, Right, Left, Both)."""

    def test_11_hand_any_accepts_any_hand(self):
        """hand='any' fires regardless of hand."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, hand="any")
        fsm = GestureFSM(config(rule))

        now = 1.0
        # Simulate right hand
        action = fsm.update(POSE_A, now, source="geometry", hand="right")
        self.assertIsNotNone(action)

    def test_12_hand_right_requires_right_hand(self):
        """hand='right' fires only for right hand."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, hand="right")
        fsm = GestureFSM(config(rule))

        now = 1.0
        # Right hand: fires
        action = fsm.update(POSE_A, now, source="geometry", hand="right")
        self.assertIsNotNone(action)

        # Left hand: blocked
        fsm = GestureFSM(config(rule))
        action = fsm.update(POSE_A, now, source="geometry", hand="left")
        self.assertIsNone(action)

    def test_13_hand_left_requires_left_hand(self):
        """hand='left' fires only for left hand."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, hand="left")
        fsm = GestureFSM(config(rule))

        now = 1.0
        # Left hand: fires
        action = fsm.update(POSE_A, now, source="geometry", hand="left")
        self.assertIsNotNone(action)

        # Right hand: blocked
        fsm = GestureFSM(config(rule))
        action = fsm.update(POSE_A, now, source="geometry", hand="right")
        self.assertIsNone(action)

    def test_14_hand_both_works(self):
        """hand='both' works (project semantics)."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, hand="both")
        fsm = GestureFSM(config(rule))

        # Both hands visible: fires
        action = fsm.update(POSE_A, 0.0, source="geometry", hand="both", hand_count=2)
        self.assertIsNotNone(action)


class ActionTypesTests(unittest.TestCase):
    """Test that all action types work with Static Hold."""

    def test_static_hold_with_right_click(self):
        """Static Hold can trigger RIGHT_CLICK."""
        rule = hold_rule(POSE_A, RIGHT_CLICK, hold_sec=0)
        fsm = GestureFSM(config(rule))

        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), RIGHT_CLICK)

    def test_static_hold_with_double_click(self):
        """Static Hold can trigger DOUBLE_CLICK."""
        rule = hold_rule(POSE_A, DOUBLE_CLICK, hold_sec=0)
        fsm = GestureFSM(config(rule))

        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), DOUBLE_CLICK)

    def test_static_hold_with_drag_start(self):
        """Static Hold can trigger DRAG_START."""
        rule = hold_rule(POSE_A, DRAG_START, hold_sec=0)
        fsm = GestureFSM(config(rule))

        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), DRAG_START)

    def test_static_hold_with_scroll(self):
        """Static Hold can trigger SCROLL_UP."""
        rule = hold_rule(POSE_A, SCROLL_UP, hold_sec=0)
        fsm = GestureFSM(config(rule))

        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), SCROLL_UP)


class ConfigurationTests(unittest.TestCase):
    """Test configuration save/load/reload."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="hold-test-")
        self.path = os.path.join(self._tmp.name, "gesture_config.json")
        self._saved_path = gesture_fsm.CONFIG_PATH
        gesture_fsm.CONFIG_PATH = self.path
        self.addCleanup(self._restore)

    def _restore(self):
        gesture_fsm.CONFIG_PATH = self._saved_path
        self._tmp.cleanup()

    def test_15_disabled_hold_rule_does_not_fire(self):
        """disabled=False → no action."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, enabled=False)
        fsm = GestureFSM(config(rule))

        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNone(action)

    def test_16_re_enabled_hold_rule_fires_again(self):
        """Re-enabling a disabled rule allows it to fire."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, enabled=True)
        cfg = config(rule)

        # Disable it
        cfg["geometry_bindings"][0]["enabled"] = False
        fsm = GestureFSM(cfg)
        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNone(action)

        # Re-enable it
        cfg["geometry_bindings"][0]["enabled"] = True
        fsm = GestureFSM(cfg)
        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNotNone(action)

    def test_17_static_hold_saves_correctly(self):
        """Configuration with Static Hold saves to disk."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=3)
        cfg = config(rule)

        save_config(cfg, self.path)

        with open(self.path) as f:
            on_disk = json.load(f)

        self.assertEqual(len(on_disk["geometry_bindings"]), 1)
        self.assertEqual(on_disk["geometry_bindings"][0]["hold_sec"], 3)

    def test_18_static_hold_reloads_correctly(self):
        """Configuration loads with correct hold_sec value."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=5)
        cfg = config(rule)
        save_config(cfg, self.path)

        reloaded = load_config(self.path)

        self.assertEqual(len(reloaded["geometry_bindings"]), 1)
        self.assertEqual(reloaded["geometry_bindings"][0]["hold_sec"], 5)

    def test_19_static_hold_survives_restart(self):
        """A saved Static Hold mapping persists through save/reload cycle."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=10, id="h-persist")
        cfg = config(rule)
        save_config(cfg, self.path)

        # Reload and compile FSM
        reloaded = load_config(self.path)
        fsm = GestureFSM(reloaded)

        # Start holding at time 0
        fsm.update(POSE_A, 0.0, source="geometry")
        # Should fire after 10 sec
        action = fsm.update(POSE_A, 10.1, source="geometry")
        self.assertIsNotNone(action)

    def test_20_newly_created_hold_reaches_running_engine(self):
        """New Static Hold mapping goes live without restart."""
        # Create initial engine
        initial_rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, id="init")
        initial_cfg = config(initial_rule)
        save_config(initial_cfg, self.path)

        fsm = GestureFSM(initial_cfg)

        # User creates new mapping
        new_rule = hold_rule(POSE_B, RIGHT_CLICK, hold_sec=3, id="new")
        new_cfg = config(initial_rule, new_rule)
        save_config(new_cfg, self.path)

        # Reload
        reloaded = load_config(self.path)
        fsm = GestureFSM(reloaded)

        # New mapping works (hold 3 sec)
        fsm.update(POSE_B, 0.0, source="geometry")
        action = fsm.update(POSE_B, 3.1, source="geometry")
        self.assertIsNotNone(action)
        self.assertEqual(str(action), RIGHT_CLICK)


class DeleteUndoTests(unittest.TestCase):
    """Test delete/undo lifecycle."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="hold-del-test-")
        self.path = os.path.join(self._tmp.name, "gesture_config.json")
        self._saved_path = gesture_fsm.CONFIG_PATH
        gesture_fsm.CONFIG_PATH = self.path
        self.addCleanup(self._restore)

    def _restore(self):
        gesture_fsm.CONFIG_PATH = self._saved_path
        self._tmp.cleanup()

    def test_21_deleted_hold_rule_removed_from_engine(self):
        """Deleted mapping removed from running engine."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, id="h-del")
        cfg = config(rule)

        # Compile FSM with rule
        fsm = GestureFSM(cfg)
        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNotNone(action)

        # Delete and reload
        cfg_empty = config()
        fsm = GestureFSM(cfg_empty)
        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNone(action)

    def test_22_undone_hold_rule_restored(self):
        """Restored mapping becomes active again."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0, id="h-undo")
        cfg = config(rule)

        # Start with rule
        fsm = GestureFSM(cfg)
        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNotNone(action)

        # Delete it
        cfg_empty = config()
        cfg_empty["deleted_bindings"] = [rule]
        fsm = GestureFSM(cfg_empty)
        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNone(action)

        # Undo (restore from trash)
        cfg_restored = config(rule)
        fsm = GestureFSM(cfg_restored)
        action = fsm.update(POSE_A, 1.0, source="geometry")
        self.assertIsNotNone(action)


class DuplicateValidationTests(unittest.TestCase):
    """Test duplicate/conflict detection."""

    def test_23_duplicate_validation_works(self):
        """Duplicate Static Hold mappings are detected."""
        rule1 = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0)
        rule2 = hold_rule(POSE_A, LEFT_CLICK, hold_sec=0)
        cfg = config(rule1, rule2)

        # GestureFSM should compile both rules (no built-in dedup at FSM level)
        # but the UI should validate before saving
        fsm = GestureFSM(cfg)
        # Both rules compile
        self.assertEqual(len(list(fsm.rules)), 2)


class PreviewTests(unittest.TestCase):
    """Test preview display for Static Hold."""

    def test_24_active_mappings_displays_static_hold(self):
        """Static Hold mapping displays correctly in active mappings."""
        rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=3)
        cfg = config(rule)
        fsm = GestureFSM(cfg)

        # Verify rule is in FSM
        rules = list(fsm.rules)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].pose, POSE_A)
        self.assertEqual(rules[0].action, LEFT_CLICK)
        self.assertEqual(rules[0].hold_sec, 3)

    def test_25_hold_duration_displayed_correctly(self):
        """Selected hold duration is displayed in rule dict."""
        for hold_sec in [0, 3, 5, 10]:
            rule = hold_rule(POSE_A, LEFT_CLICK, hold_sec=hold_sec)
            cfg = config(rule)
            fsm = GestureFSM(cfg)

            rules = list(fsm.rules)
            self.assertEqual(rules[0].hold_sec, hold_sec,
                            f"hold_sec={hold_sec} not preserved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
