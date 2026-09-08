"""Regression tests for SWIPE_LEFT and SWIPE_RIGHT actions.

Verifies that the new actions can be:
- Created through the GUI builder
- Saved to gesture_config.json
- Reloaded after restart
- Edited in mappings
- Toggled enable/disable
- Deleted and undone

No actual swipe detection or rendering is implemented yet - these tests
only verify the configuration pipeline and lifecycle.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid

import gesture_fsm
from gesture_fsm import (SWIPE_LEFT, SWIPE_RIGHT, ACTIONS, MOUSE_ACTIONS,
                         load_config, save_config, GestureFSM)


def new_rule(from_state, to_state, action, **extra):
    """A rule shaped as GestureStudio._read_builder emits one."""
    rule = {
        "id": "r%s" % uuid.uuid4().hex[:8],
        "trigger": "transition",
        "from_state": from_state,
        "to_state": to_state,
        "action": action,
        "enabled": True,
        "cooldown_sec": 0.0,
        "source": "any",
        "hand": "any",
    }
    rule.update(extra)
    return rule


def config(*rules):
    return {"settings": {}, "geometry_bindings": list(rules),
            "yolo_bindings": [], "deleted_bindings": []}


class SwipeActionAvailabilityTests(unittest.TestCase):
    """The new actions are in the catalog and exportable."""

    def test_swipe_left_is_a_valid_action(self):
        self.assertEqual(SWIPE_LEFT, "SWIPE_LEFT")
        self.assertIn(SWIPE_LEFT, ACTIONS)
        self.assertIn(SWIPE_LEFT, MOUSE_ACTIONS)

    def test_swipe_right_is_a_valid_action(self):
        self.assertEqual(SWIPE_RIGHT, "SWIPE_RIGHT")
        self.assertIn(SWIPE_RIGHT, ACTIONS)
        self.assertIn(SWIPE_RIGHT, MOUSE_ACTIONS)

    def test_both_actions_appear_in_selectable_dropdown(self):
        from app import SELECTABLE_ACTIONS
        self.assertIn(SWIPE_LEFT, SELECTABLE_ACTIONS)
        self.assertIn(SWIPE_RIGHT, SELECTABLE_ACTIONS)


class SwipeActionLifecycleTests(unittest.TestCase):
    """Test the full create→save→reload→edit→delete lifecycle."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mg-swipe-test-")
        self.path = os.path.join(self._tmp.name, "gesture_config.json")
        self._saved_path = gesture_fsm.CONFIG_PATH
        gesture_fsm.CONFIG_PATH = self.path
        self.addCleanup(self._restore)

    def _restore(self):
        gesture_fsm.CONFIG_PATH = self._saved_path
        self._tmp.cleanup()

    def save(self, *rules):
        """Save rules to the temp config file."""
        cfg = config(*rules)
        gesture_fsm.save_config(cfg, self.path)
        return json.load(open(self.path, encoding="utf-8"))

    def test_1_create_swipe_left_mapping(self):
        """Create and persist a SWIPE_LEFT mapping."""
        rule = new_rule("peace", "closed_fist", SWIPE_LEFT)
        on_disk = self.save(rule)

        self.assertEqual(len(on_disk["geometry_bindings"]), 1)
        self.assertEqual(on_disk["geometry_bindings"][0]["action"],
                         SWIPE_LEFT)

    def test_2_create_swipe_right_mapping(self):
        """Create and persist a SWIPE_RIGHT mapping."""
        rule = new_rule("open", "grab", SWIPE_RIGHT)
        on_disk = self.save(rule)

        self.assertEqual(len(on_disk["geometry_bindings"]), 1)
        self.assertEqual(on_disk["geometry_bindings"][0]["action"],
                         SWIPE_RIGHT)

    def test_3_both_actions_coexist(self):
        """Two rules with different swipe actions can coexist."""
        left = new_rule("peace", "closed_fist", SWIPE_LEFT)
        right = new_rule("open", "grab", SWIPE_RIGHT)
        on_disk = self.save(left, right)

        self.assertEqual(len(on_disk["geometry_bindings"]), 2)
        actions = {r["action"] for r in on_disk["geometry_bindings"]}
        self.assertEqual(actions, {SWIPE_LEFT, SWIPE_RIGHT})

    def test_4_swipe_rules_compile_in_fsm(self):
        """The FSM accepts swipe actions without error."""
        left = new_rule("peace", "closed_fist", SWIPE_LEFT)
        right = new_rule("open", "grab", SWIPE_RIGHT)
        cfg = config(left, right)

        # Should not raise
        fsm = GestureFSM(cfg)

        # Verify both rules compiled
        self.assertEqual(len(list(fsm.rules)), 2)
        actions = {r.action for r in fsm.rules}
        self.assertEqual(actions, {SWIPE_LEFT, SWIPE_RIGHT})

    def test_5_swipe_rules_survive_save_reload_cycle(self):
        """A swipe mapping persists through save/reload."""
        original = new_rule("peace", "closed_fist", SWIPE_LEFT, id="swipe1")

        # Save
        first = self.save(original)
        saved_id = first["geometry_bindings"][0]["id"]

        # Reload from disk
        reloaded = load_config(self.path)
        self.assertEqual(len(reloaded["geometry_bindings"]), 1)
        self.assertEqual(reloaded["geometry_bindings"][0]["action"],
                         SWIPE_LEFT)
        self.assertEqual(reloaded["geometry_bindings"][0]["id"], saved_id)

    def test_6_swipe_rules_can_be_edited(self):
        """A swipe mapping can be edited and re-saved."""
        original = new_rule("peace", "closed_fist", SWIPE_LEFT, id="swipe1")
        self.save(original)

        # Edit: change target pose
        edited = new_rule("peace", "grab", SWIPE_LEFT, id="swipe1")
        on_disk = self.save(edited)

        self.assertEqual(on_disk["geometry_bindings"][0]["to_state"], "grab")

    def test_7_swipe_rules_can_be_toggled(self):
        """A swipe mapping can be toggled enable/disable."""
        rule = new_rule("peace", "closed_fist", SWIPE_LEFT)
        first = self.save(rule)

        # Disable it
        rule["enabled"] = False
        second = self.save(rule)
        self.assertFalse(second["geometry_bindings"][0]["enabled"])

        # Re-enable
        rule["enabled"] = True
        third = self.save(rule)
        self.assertTrue(third["geometry_bindings"][0]["enabled"])

    def test_8_swipe_rules_can_be_deleted(self):
        """A swipe mapping can be deleted."""
        original = new_rule("peace", "closed_fist", SWIPE_LEFT)
        first = self.save(original)
        self.assertEqual(len(first["geometry_bindings"]), 1)

        # Delete by saving without it
        on_disk = self.save()
        self.assertEqual(len(on_disk["geometry_bindings"]), 0)

    def test_9_swipe_rules_can_be_undone(self):
        """A deleted swipe mapping can be restored via undo."""
        original = new_rule("peace", "closed_fist", SWIPE_LEFT, id="swipe-del")
        active = self.save(original)
        original_id = active["geometry_bindings"][0]["id"]

        # Simulate deletion: move from geometry to deleted bindings
        deleted_rule = original.copy()
        on_disk = self.save()  # Clear active
        on_disk["deleted_bindings"] = [deleted_rule]
        gesture_fsm.save_config(on_disk, self.path)

        # Verify it's in trash
        trashed = load_config(self.path)
        self.assertEqual(len(trashed["geometry_bindings"]), 0)
        self.assertEqual(len(trashed["deleted_bindings"]), 1)
        self.assertEqual(trashed["deleted_bindings"][0]["id"], original_id)

        # Undo by restoring from trash
        restored_rule = trashed["deleted_bindings"][0]
        restored_config = {"settings": {}, "geometry_bindings": [restored_rule],
                           "yolo_bindings": [], "deleted_bindings": []}
        gesture_fsm.save_config(restored_config, self.path)
        restored = load_config(self.path)
        self.assertEqual(len(restored["geometry_bindings"]), 1)
        self.assertEqual(restored["geometry_bindings"][0]["id"], original_id)

    def test_10_swipe_rules_respect_hand_gating(self):
        """SWIPE_LEFT and SWIPE_RIGHT support hand selection."""
        left_hand = new_rule("peace", "grab", SWIPE_LEFT, hand="left")
        right_hand = new_rule("peace", "grab", SWIPE_RIGHT, hand="right")
        on_disk = self.save(left_hand, right_hand)

        self.assertEqual(on_disk["geometry_bindings"][0]["hand"], "left")
        self.assertEqual(on_disk["geometry_bindings"][1]["hand"], "right")


class SwipeActionExecutionTests(unittest.TestCase):
    """Test that ActionDispatcher/Executor accepts swipe actions."""

    def test_executor_accepts_swipe_left(self):
        """ActionDispatcher.dispatch handles SWIPE_LEFT gracefully."""
        from hand_cursor_2 import ActionDispatcher

        cursor = type('obj', (object,), {
            'move': lambda *a, **k: None,
            'click': lambda *a, **k: None,
            'press': lambda *a, **k: None,
            'release': lambda *a, **k: None,
        })()

        dispatcher = ActionDispatcher(cursor)
        # Should not crash; detection not yet implemented so passes through
        # to ActionExecutor, which currently has placeholder (no-op).
        try:
            dispatcher.dispatch(SWIPE_LEFT, 1.0)
        except Exception as e:
            self.fail("dispatch(SWIPE_LEFT) raised %s" % e)

    def test_executor_accepts_swipe_right(self):
        """ActionDispatcher.dispatch handles SWIPE_RIGHT gracefully."""
        from hand_cursor_2 import ActionDispatcher

        cursor = type('obj', (object,), {
            'move': lambda *a, **k: None,
            'click': lambda *a, **k: None,
            'press': lambda *a, **k: None,
            'release': lambda *a, **k: None,
        })()

        dispatcher = ActionDispatcher(cursor)
        # Should not crash; detection not yet implemented so passes through
        # to ActionExecutor, which currently has placeholder (no-op).
        try:
            dispatcher.dispatch(SWIPE_RIGHT, 1.0)
        except Exception as e:
            self.fail("dispatch(SWIPE_RIGHT) raised %s" % e)


if __name__ == "__main__":
    unittest.main(verbosity=2)
