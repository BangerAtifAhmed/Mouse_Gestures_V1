"""Regression tests for the mapping lifecycle: GUI -> disk -> runtime.

WHAT THIS COVERS.  The path a NEW mapping takes from the builder to the
rule set the tracker is actually matching against, and the two defects
that used to break it:

  * GestureStudio.save_config() wrote the file and told nobody, and
    HandTrackerEngine.reload_config() had no caller anywhere in the
    project.  The engine kept the rule set it compiled in __init__.

  * HandTrackerEngine._run() hoists self.fsm and self.semantic_fsm into
    locals before its loop, so even a reload left the running thread
    feeding the old machines.

WHAT IT NEVER TOUCHES.  Your gesture_config.json.  Every test writes to
a throwaway temp file and monkeypatches app.CONFIG_PATH at the module
level for the duration of the test, restoring it afterwards.

No camera, no model, no mouse, no window: the GUI and engine objects are
built with object.__new__ and given only the attributes the methods
under test actually read.

    python test_config_lifecycle.py
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid

import gesture_fsm
from gesture_fsm import GestureFSM, load_config

# The live semantic machine, exactly as HandTrackerEngine._apply_config
# builds it.
SEMANTIC = {"stability_threshold": 1, "window_size": 1,
            "transition_memory": 6}
FRAME = 0.21


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


def bare_studio(rules=(), deleted=(), engine=None):
    """A GestureStudio with no window, for the pure-model methods.

    tk.Tk.__getattr__ delegates to self.tk, which recurses forever on an
    uninitialised instance; binding it to None turns that into the
    AttributeError the hasattr guards already expect.
    """
    import app
    studio = object.__new__(app.GestureStudio)
    studio.tk = None
    studio.rules = [dict(r) for r in rules]
    studio.deleted = [dict(r) for r in deleted]
    studio.settings = dict(gesture_fsm.DEFAULT_SETTINGS)
    studio.engine = engine
    studio._undo_stack = []
    studio.toasts = []
    studio.notes = []
    studio._dirty = True
    studio._clear_dirty = lambda: setattr(studio, "_dirty", False)
    studio._toast = lambda message, *a, **k: studio.toasts.append(message)
    studio._camera_note = lambda message, *a, **k: studio.notes.append(
        message)
    studio._settings_to_widgets = lambda *a, **k: None
    studio._forget_undo = lambda *a, **k: None
    studio._render_bin = lambda *a, **k: None
    studio._render_table = lambda *a, **k: None
    return studio


def build_machines(cfg):
    """The two FSMs, as _apply_config builds them."""
    return GestureFSM(cfg), GestureFSM(cfg, **SEMANTIC)


def drive(fsm, script, hand=None, source="semantic"):
    actions, now = [], 100.0
    for pose in script:
        now += FRAME
        action = fsm.update(pose, now, source=source, hand=hand,
                            hand_count=1)
        if action is not None:
            actions.append(str(action))
    return actions


def compiled(fsm):
    return sorted("%s->%s=%s" % (r.from_state, r.to_state, r.action)
                  for r in fsm.rules)


class LifecycleCase(unittest.TestCase):
    """Base: a temp config path, and app.CONFIG_PATH pointed at it."""

    def setUp(self):
        import app
        self.app = app
        self._tmp = tempfile.TemporaryDirectory(prefix="mg-lifecycle-")
        self.path = os.path.join(self._tmp.name, "gesture_config.json")
        self._saved_path = app.CONFIG_PATH
        app.CONFIG_PATH = self.path
        self.addCleanup(self._restore)

    def _restore(self):
        self.app.CONFIG_PATH = self._saved_path
        self._tmp.cleanup()

    def save(self, studio):
        ok = self.app.GestureStudio.save_config(studio)
        self.assertTrue(ok, "save_config reported failure")
        return json.load(open(self.path, encoding="utf-8"))

    def reload(self):
        """A restart: rebuild both machines from the file."""
        return build_machines(load_config(self.path))

    def engine(self, real_actions=False):
        """A HandTrackerEngine carrying only what these methods read.

        object.__new__ so no camera, model or cursor backend is built.

        load_config is patched on hand_cursor_2 rather than by moving
        CONFIG_PATH: gesture_fsm.load_config binds its default path at
        import time, so reassigning gesture_fsm.CONFIG_PATH afterwards
        does not redirect a no-argument call — it would silently read the
        user's real config instead of the temp one.

        `real_actions` swaps the counting stub for the project's own
        ActionDispatcher over a recording cursor, which is what the drag
        tests need: the button state that matters lives in that class.
        """
        import hand_cursor_2
        import adaptive_timing

        eng = object.__new__(hand_cursor_2.HandTrackerEngine)
        eng.logs = []
        eng._log = lambda message: eng.logs.append(message)
        eng.geometry_learner = adaptive_timing.TimingLearner()
        eng.semantic_learner = adaptive_timing.TimingLearner()
        self.addCleanup(eng.geometry_learner.stop)
        self.addCleanup(eng.semantic_learner.stop)
        eng._ai_confidence = 0.6
        eng._config_generation = 0

        if real_actions:
            eng.cursor = RecordingCursor()
            eng.actions = hand_cursor_2.ActionDispatcher(eng.cursor)
        else:
            class Actions:
                released = 0

                def release(self):
                    Actions.released += 1
                    return False

            eng.actions = Actions()

        path = self.path
        saved = hand_cursor_2.load_config
        hand_cursor_2.load_config = lambda *a, **k: load_config(path)
        self.addCleanup(setattr, hand_cursor_2, "load_config", saved)
        return eng, hand_cursor_2

    def gui_for(self, engine, rules=(), deleted=()):
        """A GUI model wired to `engine`, so Save Config resynchronises."""
        return bare_studio(rules, deleted=deleted, engine=engine)


# ─── 1-3, 5, 7: new mappings reach the runtime ──────────────────────────

class NewMappingTests(LifecycleCase):

    def test_1_a_new_right_click_mapping_works(self):
        right = new_rule("two_up", "fist", "RIGHT_CLICK")
        self.save(bare_studio([right]))
        _geo, semantic = self.reload()
        self.assertEqual(compiled(semantic), ["two_up->fist=RIGHT_CLICK"])
        self.assertEqual(drive(semantic, ["two_up", "fist"]),
                         ["RIGHT_CLICK"])

    def test_2_a_new_left_click_mapping_works(self):
        left = new_rule("one", "fist", "LEFT_CLICK")
        self.save(bare_studio([left]))
        _geo, semantic = self.reload()
        self.assertEqual(drive(semantic, ["one", "fist"]), ["LEFT_CLICK"])

    def test_3_both_new_mappings_coexist(self):
        left = new_rule("one", "fist", "LEFT_CLICK")
        right = new_rule("two_up", "fist", "RIGHT_CLICK")
        self.save(bare_studio([left, right]))

        _geo, semantic = self.reload()
        self.assertEqual(compiled(semantic),
                         ["one->fist=LEFT_CLICK",
                          "two_up->fist=RIGHT_CLICK"])
        self.assertEqual(drive(semantic, ["two_up", "fist"]),
                         ["RIGHT_CLICK"])
        _geo, semantic = self.reload()
        self.assertEqual(drive(semantic, ["one", "fist"]), ["LEFT_CLICK"])

    def test_5_saving_and_reloading_preserves_active_mappings(self):
        rules = [new_rule("one", "fist", "LEFT_CLICK"),
                 new_rule("two_up", "fist", "RIGHT_CLICK"),
                 new_rule("peace", "grip", "MIDDLE_CLICK",
                          source="semantic")]
        first = self.save(bare_studio(rules))

        # Reload into a fresh GUI model, then save again unchanged.
        studio = bare_studio()
        self.app.GestureStudio._config_to_rules(studio, first)
        self.assertEqual(len(studio.rules), 3)
        second = self.save(studio)

        self.assertEqual(
            sorted(r["id"] for r in first["geometry_bindings"]
                   + first["yolo_bindings"]),
            sorted(r["id"] for r in second["geometry_bindings"]
                   + second["yolo_bindings"]),
            "a save/load/save round trip changed the active set")

    def test_7_gui_active_count_equals_runtime_compiled_count(self):
        rules = [new_rule("one", "fist", "LEFT_CLICK"),
                 new_rule("two_up", "fist", "RIGHT_CLICK"),
                 new_rule("peace", "grip", "MIDDLE_CLICK",
                          source="semantic")]
        binned = [new_rule("open", "grip", "DRAG_START")]
        studio = bare_studio(rules, deleted=binned)
        self.save(studio)

        geometry, semantic = self.reload()
        self.assertEqual(len(tuple(semantic.rules)), len(studio.rules))
        self.assertEqual(len(tuple(geometry.rules)), len(studio.rules))


# ─── 4: the bin never compiles ──────────────────────────────────────────

class DeletedMappingTests(LifecycleCase):

    def test_4_deleted_mappings_never_compile(self):
        active = new_rule("one", "fist", "LEFT_CLICK")
        binned = new_rule("two_up", "fist", "RIGHT_CLICK")
        on_disk = self.save(bare_studio([active], deleted=[binned]))

        self.assertEqual(len(on_disk["deleted_bindings"]), 1)
        _geo, semantic = self.reload()
        self.assertEqual(compiled(semantic), ["one->fist=LEFT_CLICK"])
        self.assertEqual(drive(semantic, ["two_up", "fist"]), [])

    def test_a_binned_rule_does_not_reappear_after_a_round_trip(self):
        active = new_rule("one", "fist", "LEFT_CLICK")
        binned = new_rule("two_up", "fist", "RIGHT_CLICK")
        first = self.save(bare_studio([active], deleted=[binned]))

        studio = bare_studio()
        self.app.GestureStudio._config_to_rules(studio, first)
        self.assertEqual([r["id"] for r in studio.rules], [active["id"]])
        self.assertEqual([r["id"] for r in studio.deleted], [binned["id"]])


# ─── 6: ids ─────────────────────────────────────────────────────────────

class RuleIdTests(LifecycleCase):

    def test_6_a_new_rule_is_not_lost_to_a_duplicate_id(self):
        shared = "rCOLLIDE"
        active = new_rule("two_up", "fist", "RIGHT_CLICK", id=shared)
        binned = new_rule("one", "fist", "LEFT_CLICK", id=shared)
        on_disk = self.save(bare_studio([active], deleted=[binned]))

        _geo, semantic = self.reload()
        self.assertEqual(compiled(semantic), ["two_up->fist=RIGHT_CLICK"])

        studio = bare_studio()
        self.app.GestureStudio._config_to_rules(studio, on_disk)
        self.assertEqual([r["action"] for r in studio.rules],
                         ["RIGHT_CLICK"])
        self.assertEqual(studio.deleted, [])

    def test_generated_ids_do_not_repeat(self):
        seen = {new_rule("a", "b", "LEFT_CLICK")["id"]
                for _ in range(20000)}
        self.assertEqual(len(seen), 20000)

    def test_two_active_rules_sharing_an_id_keep_the_first(self):
        shared = "rSAME"
        first = new_rule("one", "fist", "LEFT_CLICK", id=shared)
        second = new_rule("two_up", "fist", "RIGHT_CLICK", id=shared)
        on_disk = self.save(bare_studio([first, second]))

        studio = bare_studio()
        self.app.GestureStudio._config_to_rules(studio, on_disk)
        self.assertEqual([r["action"] for r in studio.rules], ["LEFT_CLICK"])


# ─── The synchronisation fix itself ─────────────────────────────────────

class RecordingEngine:
    """Stands in for HandTrackerEngine: records the resync call."""

    def __init__(self, raises=None):
        self.reloads = 0
        self._raises = raises

    def reload_config(self):
        self.reloads += 1
        if self._raises is not None:
            raise self._raises


class ResyncTests(LifecycleCase):

    def test_saving_notifies_the_running_engine(self):
        engine = RecordingEngine()
        studio = bare_studio([new_rule("two_up", "fist", "RIGHT_CLICK")],
                             engine=engine)
        self.save(studio)
        self.assertEqual(engine.reloads, 1,
                         "Save Config did not resynchronise the tracker")

    def test_saving_without_an_engine_is_harmless(self):
        studio = bare_studio([new_rule("one", "fist", "LEFT_CLICK")],
                             engine=None)
        self.save(studio)          # asserts save_config returned True

    def test_a_failing_reload_does_not_fail_the_save(self):
        engine = RecordingEngine(raises=RuntimeError("camera busy"))
        studio = bare_studio([new_rule("one", "fist", "LEFT_CLICK")],
                             engine=engine)
        self.save(studio)
        self.assertTrue(any("old mappings" in note
                            for note in studio.notes),
                        "a failed resync was not reported")

    def test_the_save_still_lands_on_disk_when_the_reload_fails(self):
        engine = RecordingEngine(raises=RuntimeError("camera busy"))
        studio = bare_studio([new_rule("two_up", "fist", "RIGHT_CLICK")],
                             engine=engine)
        on_disk = self.save(studio)
        self.assertEqual(len(on_disk["geometry_bindings"]), 1)


class RecordingCursor:
    """A cursor backend that records instead of touching the desktop."""

    def __init__(self):
        self.pressed = []
        self.released = []
        self.clicks = []

    def press(self, button="left"):
        self.pressed.append(button)

    def release(self, button="left"):
        self.released.append(button)

    def click(self, button="left", count=1):
        self.clicks.append((button, count))

    def move(self, x, y):
        pass


class EngineReloadTests(LifecycleCase):
    """_apply_config / reload_config on a real engine object."""

    def test_reload_config_rebuilds_both_machines(self):
        eng, _module = self.engine()
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK")]))
        eng._apply_config()
        self.assertEqual(compiled(eng.semantic_fsm),
                         ["one->fist=LEFT_CLICK"])
        first_generation = eng._config_generation

        # The user adds a mapping in the GUI and presses Save.
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK"),
                               new_rule("two_up", "fist",
                                        "RIGHT_CLICK")]))
        eng.reload_config()

        self.assertEqual(compiled(eng.semantic_fsm),
                         ["one->fist=LEFT_CLICK",
                          "two_up->fist=RIGHT_CLICK"])
        self.assertEqual(compiled(eng.fsm),
                         ["one->fist=LEFT_CLICK",
                          "two_up->fist=RIGHT_CLICK"])
        self.assertGreater(eng._config_generation, first_generation,
                           "the generation stamp did not advance")

    def test_the_new_rule_actually_fires_after_a_reload(self):
        eng, _module = self.engine()
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK")]))
        eng._apply_config()
        self.assertEqual(drive(eng.semantic_fsm,
                               ["one", "two_up", "fist"]), ["LEFT_CLICK"])

        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK"),
                               new_rule("two_up", "fist",
                                        "RIGHT_CLICK")]))
        eng.reload_config()
        self.assertEqual(drive(eng.semantic_fsm,
                               ["one", "two_up", "fist"]), ["RIGHT_CLICK"])

    def test_reload_drops_a_held_button_before_swapping(self):
        eng, _module = self.engine()
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK")]))
        eng._apply_config()
        before = eng.actions.released
        eng.reload_config()
        self.assertEqual(eng.actions.released, before + 1,
                         "the mouse button was not released")

    def test_the_learners_survive_a_reload(self):
        """Rebuilding the machines must not discard what was learnt."""
        eng, _module = self.engine()
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK",
                                        adaptive_timing=True)]))
        eng._apply_config()
        eng.semantic_learner.seed("one", "fist", [0.4] * 25)
        before = eng.semantic_learner.samples("one", "fist")
        self.assertGreater(before, 0)

        eng.reload_config()
        self.assertEqual(eng.semantic_learner.samples("one", "fist"),
                         before, "a reload threw away the timing history")


# ─── 8, 9: unchanged behaviour ──────────────────────────────────────────

class UnchangedBehaviourTests(LifecycleCase):

    def test_8_adaptive_timing_is_unchanged_across_the_lifecycle(self):
        import adaptive_timing
        rule = new_rule("one", "fist", "LEFT_CLICK", adaptive_timing=True)
        self.save(bare_studio([rule]))
        cfg = load_config(self.path)

        learner = adaptive_timing.TimingLearner()
        self.addCleanup(learner.stop)
        fsm = GestureFSM(cfg, learner=learner, **SEMANTIC)

        compiled_rule = fsm.rules[0]
        self.assertTrue(compiled_rule.adaptive_timing)
        # Below MIN_SAMPLES the baseline stands, exactly as before.
        self.assertEqual(learner.tolerance("one", "fist"),
                         adaptive_timing.BASELINE_SEC)
        self.assertEqual(drive(fsm, ["one", "fist"]), ["LEFT_CLICK"])

        # A deleted mapping forgets its timing, unchanged.
        learner.seed("one", "fist", [0.5] * 25)
        self.assertGreater(learner.samples("one", "fist"), 0)
        self.assertTrue(learner.forget("one", "fist"))
        self.assertEqual(learner.tolerance("one", "fist"),
                         adaptive_timing.BASELINE_SEC)

    def test_8_an_explicit_window_survives_a_round_trip(self):
        rule = new_rule("one", "fist", "LEFT_CLICK", max_time_sec=0.2)
        self.save(bare_studio([rule]))
        _geo, semantic = self.reload()
        self.assertAlmostEqual(semantic.rules[0].max_time_sec, 0.2)
        self.assertFalse(semantic.rules[0].adaptive_timing)

    def test_9_hand_gating_survives_the_lifecycle(self):
        rules = [new_rule("one", "fist", "LEFT_CLICK", hand="left"),
                 new_rule("two_up", "fist", "RIGHT_CLICK", hand="right")]
        self.save(bare_studio(rules))
        _geo, semantic = self.reload()

        self.assertEqual(drive(semantic, ["two_up", "fist"], hand="right"),
                         ["RIGHT_CLICK"])
        _geo, semantic = self.reload()
        self.assertEqual(drive(semantic, ["two_up", "fist"], hand="left"),
                         [])
        _geo, semantic = self.reload()
        self.assertEqual(drive(semantic, ["one", "fist"], hand="left"),
                         ["LEFT_CLICK"])
        _geo, semantic = self.reload()
        self.assertEqual(drive(semantic, ["one", "fist"], hand="right"), [])

    def test_9_the_hand_field_is_normalised_on_the_way_through(self):
        rule = new_rule("one", "fist", "LEFT_CLICK", hand="Right_Hand")
        self.save(bare_studio([rule]))
        _geo, semantic = self.reload()
        self.assertEqual(semantic.rules[0].hand, "right")


# ─── 10: no gesture names in the changed code ───────────────────────────

class NoHardcodedNamesTests(unittest.TestCase):
    """The lifecycle must be blind to what a gesture is called."""

    # Every label the project's own vocabulary knows about, plus the two
    # from the bug report.  None may appear in the code that was changed.
    FORBIDDEN = ("two_up", "fist", "one", "peace", "grip", "point",
                 "open", "palm", "three_gun", "stop")

    def changed_functions(self):
        import inspect
        import app
        import hand_cursor_2
        return {
            "GestureStudio._resync_engine":
                inspect.getsource(app.GestureStudio._resync_engine),
            "GestureStudio.save_config":
                inspect.getsource(app.GestureStudio.save_config),
            "GestureStudio._rules_to_config":
                inspect.getsource(app.GestureStudio._rules_to_config),
            "HandTrackerEngine.reload_config":
                inspect.getsource(
                    hand_cursor_2.HandTrackerEngine.reload_config),
            "HandTrackerEngine._apply_config":
                inspect.getsource(
                    hand_cursor_2.HandTrackerEngine._apply_config),
        }

    def test_10_no_gesture_name_appears_in_the_changed_code(self):
        for name, body in self.changed_functions().items():
            lowered = body.lower()
            for label in self.FORBIDDEN:
                self.assertNotIn(
                    '"%s"' % label, lowered,
                    "%s hardcodes the gesture name %r" % (name, label))
                self.assertNotIn(
                    "'%s'" % label, lowered,
                    "%s hardcodes the gesture name %r" % (name, label))

    def test_10_the_lifecycle_works_with_invented_names(self):
        """The same journey, with labels that exist nowhere in the code."""
        with tempfile.TemporaryDirectory(prefix="mg-names-") as tmp:
            import app
            path = os.path.join(tmp, "gesture_config.json")
            saved = app.CONFIG_PATH
            app.CONFIG_PATH = path
            try:
                rules = [new_rule("zeta_wave", "quux_grip", "RIGHT_CLICK"),
                         new_rule("alpha_ray", "quux_grip", "LEFT_CLICK")]
                studio = bare_studio(rules)
                self.assertTrue(app.GestureStudio.save_config(studio))
            finally:
                app.CONFIG_PATH = saved

            _geo, semantic = build_machines(load_config(path))
            self.assertEqual(len(tuple(semantic.rules)), 2)
            self.assertEqual(
                drive(semantic, ["zeta_wave", "quux_grip"]),
                ["RIGHT_CLICK"])
            _geo, semantic = build_machines(load_config(path))
            self.assertEqual(
                drive(semantic, ["alpha_ray", "quux_grip"]),
                ["LEFT_CLICK"])


# ─── D: the complete lifecycle, on ONE running engine, no restart ───────

class RunningEngineLifecycleTests(LifecycleCase):
    """Steps 1-11 of the reported user journey, in order, no restart.

    ONE engine object throughout.  Nothing here rebuilds it, and no test
    reaches past reload_config() to poke a machine into place: every
    change arrives the way a user's would, through Save Config.
    """

    def setUp(self):
        super().setUp()
        self.eng, _module = self.engine()
        self.left = new_rule("one", "fist", "LEFT_CLICK")

        # 1-2.  The engine starts with one mapping.
        self.save(bare_studio([self.left]))
        self.eng._apply_config()
        self.assertEqual(compiled(self.eng.semantic_fsm),
                         ["one->fist=LEFT_CLICK"])

    def running(self):
        """The machine the loop would be using right now."""
        refreshed = self.eng._machines_for_frame(-1)
        self.assertIsNotNone(refreshed)
        _generation, _geometry, semantic = refreshed
        return semantic

    def test_3_to_7_a_new_mapping_goes_live_without_a_restart(self):
        before = self.eng._config_generation
        right = new_rule("two_up", "fist", "RIGHT_CLICK")

        # 3-4.  Created in the GUI and saved.  Nothing else is called:
        # save_config is what has to reach the engine.
        self.save(self.gui_for(self.eng, [self.left, right]))

        # 5.  The RUNNING engine now holds both.
        self.assertEqual(compiled(self.eng.semantic_fsm),
                         ["one->fist=LEFT_CLICK",
                          "two_up->fist=RIGHT_CLICK"])
        self.assertGreater(self.eng._config_generation, before)

        # 6-7.  Both fire, each with its own action.
        self.assertEqual(drive(self.running(), ["two_up", "fist"]),
                         ["RIGHT_CLICK"])
        self.assertEqual(drive(self.running(), ["one", "fist"]),
                         ["LEFT_CLICK"])

    def test_8_mappings_survive_a_disconnect_and_reconnect(self):
        right = new_rule("two_up", "fist", "RIGHT_CLICK")
        self.save(self.gui_for(self.eng, [self.left, right]))
        expected = compiled(self.eng.semantic_fsm)

        # Disconnect.  stop() resets FSM STATE; it must not touch rules.
        import threading
        self.eng._stop_event = threading.Event()
        self.eng._thread = None
        self.eng.stream = None
        self.eng._latest_frame = None
        self.eng.status = {"running": True, "camera_index": 0, "fps": 30.0}
        self.eng.stop()

        self.assertEqual(compiled(self.eng.semantic_fsm), expected,
                         "a disconnect discarded the rule set")
        self.assertIsNone(self.eng.semantic_fsm.stable_gesture,
                          "a disconnect should still clear pose state")

        # Reconnect: the loop re-enters and takes the current pair.
        self.assertEqual(drive(self.running(), ["two_up", "fist"]),
                         ["RIGHT_CLICK"])

    def test_9_a_second_new_mapping_also_goes_live(self):
        right = new_rule("two_up", "fist", "RIGHT_CLICK")
        self.save(self.gui_for(self.eng, [self.left, right]))

        third = new_rule("peace", "fist", "MIDDLE_CLICK")
        self.save(self.gui_for(self.eng, [self.left, right, third]))

        self.assertEqual(compiled(self.eng.semantic_fsm),
                         ["one->fist=LEFT_CLICK",
                          "peace->fist=MIDDLE_CLICK",
                          "two_up->fist=RIGHT_CLICK"])
        self.assertEqual(drive(self.running(), ["peace", "fist"]),
                         ["MIDDLE_CLICK"])

    def test_10_a_deleted_mapping_leaves_the_running_machine(self):
        right = new_rule("two_up", "fist", "RIGHT_CLICK")
        self.save(self.gui_for(self.eng, [self.left, right]))
        self.assertEqual(drive(self.running(), ["two_up", "fist"]),
                         ["RIGHT_CLICK"])

        # Deleted in the GUI: gone from self.rules, moved to the bin.
        self.save(self.gui_for(self.eng, [self.left], deleted=[right]))

        self.assertEqual(compiled(self.eng.semantic_fsm),
                         ["one->fist=LEFT_CLICK"])
        self.assertEqual(drive(self.running(), ["two_up", "fist"]), [],
                         "a deleted mapping still fired")

    def test_11_recreating_the_mapping_as_a_brand_new_rule_works(self):
        right = new_rule("two_up", "fist", "RIGHT_CLICK")
        self.save(self.gui_for(self.eng, [self.left, right]))
        self.save(self.gui_for(self.eng, [self.left], deleted=[right]))
        self.assertEqual(drive(self.running(), ["two_up", "fist"]), [])

        # A NEW rule with a NEW id — no recycle-bin restore involved.
        recreated = new_rule("two_up", "fist", "RIGHT_CLICK")
        self.assertNotEqual(recreated["id"], right["id"])
        self.save(self.gui_for(self.eng, [self.left, recreated],
                               deleted=[right]))

        self.assertEqual(compiled(self.eng.semantic_fsm),
                         ["one->fist=LEFT_CLICK",
                          "two_up->fist=RIGHT_CLICK"])
        self.assertEqual(drive(self.running(), ["two_up", "fist"]),
                         ["RIGHT_CLICK"])
        self.assertEqual(
            [r.id for r in self.eng.semantic_fsm.rules
             if r.from_state == "two_up"], [recreated["id"]],
            "the running machine used the binned rule, not the new one")


# ─── The loop's pickup point ────────────────────────────────────────────

class MachinesForFrameTests(LifecycleCase):
    """_machines_for_frame is the loop's only view of a reload."""

    def prepared(self):
        eng, _module = self.engine()
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK")]))
        eng._apply_config()
        return eng

    def test_an_unchanged_generation_returns_none(self):
        eng = self.prepared()
        self.assertIsNone(
            eng._machines_for_frame(eng._config_generation),
            "the loop would re-read the machines on every frame")

    def test_a_changed_generation_returns_the_new_pair(self):
        eng = self.prepared()
        seen = eng._config_generation

        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK"),
                               new_rule("two_up", "fist",
                                        "RIGHT_CLICK")]))
        eng.reload_config()

        refreshed = eng._machines_for_frame(seen)
        self.assertIsNotNone(refreshed, "the loop would keep the old pair")
        generation, geometry, semantic = refreshed
        self.assertEqual(generation, eng._config_generation)
        self.assertIs(geometry, eng.fsm)
        self.assertIs(semantic, eng.semantic_fsm)
        self.assertEqual(compiled(semantic),
                         ["one->fist=LEFT_CLICK",
                          "two_up->fist=RIGHT_CLICK"])

    def test_the_pair_is_never_half_swapped(self):
        """Both machines always come from the same rebuild.

        _apply_config builds both into locals, publishes them, and only
        then advances the stamp — so a stamp that has moved proves both
        assignments landed.  Asserted by rebuilding repeatedly and
        checking the two always agree on the rule set.
        """
        eng = self.prepared()
        seen = eng._config_generation
        for count in range(2, 8):
            rules = [new_rule("p%d" % index, "fist", "LEFT_CLICK")
                     for index in range(count)]
            self.save(bare_studio(rules))
            eng.reload_config()
            refreshed = eng._machines_for_frame(seen)
            self.assertIsNotNone(refreshed)
            seen, geometry, semantic = refreshed
            self.assertEqual(len(tuple(geometry.rules)),
                             len(tuple(semantic.rules)))
            self.assertEqual(compiled(geometry), compiled(semantic))

    def test_a_failed_rebuild_leaves_the_stamp_and_the_pair_alone(self):
        """A reload that raises must not advance the stamp.

        The loop keys entirely off the stamp, so an unmoved stamp is what
        keeps it on the rule set it already had rather than on half of a
        rebuild that never completed.
        """
        import hand_cursor_2
        eng = self.prepared()
        before_generation = eng._config_generation
        before_rules = compiled(eng.semantic_fsm)

        saved = hand_cursor_2.load_config

        def explode(*_a, **_k):
            raise OSError("config unreadable")

        hand_cursor_2.load_config = explode
        try:
            with self.assertRaises(OSError):
                eng.reload_config()
        finally:
            hand_cursor_2.load_config = saved

        self.assertEqual(eng._config_generation, before_generation)
        self.assertEqual(compiled(eng.semantic_fsm), before_rules)
        self.assertIsNone(eng._machines_for_frame(before_generation))

    def test_the_loop_actually_calls_it(self):
        """The pickup is wired into _run, not merely available."""
        import inspect
        import hand_cursor_2
        body = inspect.getsource(hand_cursor_2.HandTrackerEngine._run)
        self.assertIn("self._machines_for_frame(config_generation)", body)
        self.assertIn("config_generation, fsm, semantic_fsm = refreshed",
                      body)
        # And the pair is never captured once and kept: the only other
        # reads of self.fsm/self.semantic_fsm in the loop are the initial
        # hoist, which the refresh above replaces.
        self.assertEqual(body.count("semantic_fsm = self.semantic_fsm"), 1)


# ─── F: drag safety across a reload ─────────────────────────────────────

class DragSafetyTests(LifecycleCase):
    """A held button must never survive the rule set that pressed it."""

    DRAG = {"trigger": "hold", "pose": "open", "action": "DRAG_START",
            "hold_sec": 0.05, "cooldown_sec": 0.0, "enabled": True,
            "source": "any", "hand": "any"}

    def dragging_engine(self):
        """An engine holding the button, pressed by a real mapping."""
        eng, _module = self.engine(real_actions=True)
        drag = dict(self.DRAG, id="r%s" % uuid.uuid4().hex[:8])
        self.save(bare_studio([drag]))
        eng._apply_config()

        # Drive the real FSM until the hold matures, then dispatch what
        # it produced through the real ActionDispatcher.
        now = 100.0
        fired = None
        for _ in range(10):
            now += 0.05
            action = eng.semantic_fsm.update("open", now, source="semantic")
            if action is not None:
                fired = action
                eng.actions.dispatch(action, now)
        self.assertEqual(str(fired), "DRAG_START",
                         "the mapping never started a drag")
        self.assertTrue(eng.actions.dragging)
        self.assertEqual(eng.cursor.pressed, ["left"])
        return eng, drag

    def test_a_reload_releases_a_held_button(self):
        eng, _drag = self.dragging_engine()

        # The user edits their mappings and saves while still holding.
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK")]))
        eng.reload_config()

        self.assertEqual(eng.cursor.released, ["left"],
                         "the button was not released")
        self.assertFalse(eng.actions.dragging, "a stuck drag remains")

    def test_the_release_is_reported(self):
        eng, _drag = self.dragging_engine()
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK")]))
        eng.reload_config()
        self.assertTrue(any("DRAG_STOP" in line for line in eng.logs),
                        "the release was not logged")

    def test_the_new_rule_set_is_live_after_the_release(self):
        eng, _drag = self.dragging_engine()
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK")]))
        eng.reload_config()

        self.assertEqual(compiled(eng.semantic_fsm),
                         ["one->fist=LEFT_CLICK"])
        self.assertEqual(drive(eng.semantic_fsm, ["one", "fist"]),
                         ["LEFT_CLICK"])

    def test_releasing_is_not_repeated_when_nothing_is_held(self):
        eng, _module = self.engine(real_actions=True)
        self.save(bare_studio([new_rule("one", "fist", "LEFT_CLICK")]))
        eng._apply_config()
        eng.reload_config()
        self.assertEqual(eng.cursor.released, [],
                         "a release was sent with no button held")

    def test_a_drag_survives_a_reload_that_keeps_its_mapping(self):
        """The button is dropped whatever the new rule set contains.

        Deliberate: the drag belongs to a machine that no longer exists,
        and the replacement has no memory of the press.  Recorded here so
        the choice is visible rather than incidental.
        """
        eng, drag = self.dragging_engine()
        self.save(bare_studio([drag]))          # the SAME mapping
        eng.reload_config()
        self.assertEqual(eng.cursor.released, ["left"])
        self.assertFalse(eng.actions.dragging)


# ─── H: the whole chain, arrow by arrow ─────────────────────────────────

class EndToEndChainTests(LifecycleCase):
    """Every link the brief names, asserted in order on one engine."""

    def test_gui_rule_reaches_the_running_loop_and_dispatches(self):
        eng, _module = self.engine()
        left = new_rule("one", "fist", "LEFT_CLICK")
        self.save(bare_studio([left]))
        eng._apply_config()

        # The loop is running and has taken its pair.
        loop_generation = eng._config_generation
        loop_semantic = eng.semantic_fsm
        self.assertEqual(compiled(loop_semantic), ["one->fist=LEFT_CLICK"])

        # GUI creates a rule  ->  self.rules
        right = new_rule("two_up", "fist", "RIGHT_CLICK")
        studio = self.gui_for(eng, [left, right])
        self.assertIn(right["id"], [r["id"] for r in studio.rules])

        # -> gesture_config.json
        on_disk = self.save(studio)
        self.assertIn(right["id"],
                      [r["id"] for r in on_disk["geometry_bindings"]])

        # -> reload_config() -> new FSM objects
        self.assertIsNot(eng.semantic_fsm, loop_semantic)

        # -> _config_generation changes
        self.assertGreater(eng._config_generation, loop_generation)

        # -> the running loop notices
        refreshed = eng._machines_for_frame(loop_generation)
        self.assertIsNotNone(refreshed)
        loop_generation, _geometry, loop_semantic = refreshed

        # -> the running semantic FSM contains the new rule
        self.assertIn("two_up->fist=RIGHT_CLICK", compiled(loop_semantic))

        # -> the gesture dispatches the correct action
        self.assertEqual(drive(loop_semantic, ["two_up", "fist"]),
                         ["RIGHT_CLICK"])

        # and the loop settles: no further pickup until the next save.
        self.assertIsNone(eng._machines_for_frame(loop_generation))


# ─── G: diagnostics are opt-in ──────────────────────────────────────────

class DiagnosticsDefaultTests(unittest.TestCase):
    """The lifecycle traces stay quiet unless asked for."""

    def test_config_debug_is_off_by_default(self):
        import app
        import hand_cursor_2
        self.assertFalse(app.CONFIG_DEBUG)
        self.assertFalse(hand_cursor_2.CONFIG_DEBUG)

    def test_transition_debug_is_off_by_default(self):
        import hand_cursor_2
        self.assertFalse(hand_cursor_2.SEMANTIC_TRANSITION_DEBUG)
        self.assertFalse(gesture_fsm.DEFAULT_SETTINGS["transition_debug"])

    def test_the_env_vars_still_turn_them_on(self):
        """The switch still exists; only its default moved.

        Asserted against the accepted-values table rather than by
        reloading the module: importing hand_cursor_2 again would re-run
        cv2 and mediapipe at module scope and hand every other test in
        this file a different set of class objects.
        """
        import hand_cursor_2
        for value in ("1", "true", "True", "yes", "on"):
            self.assertIn(value, hand_cursor_2.DEBUG_ON)
        for value in ("", "0", "false", "off", "no"):
            self.assertNotIn(value, hand_cursor_2.DEBUG_ON)

    def test_a_quiet_reload_prints_nothing(self):
        import contextlib
        import io as _io
        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            GestureFSM({"settings": {}, "geometry_bindings": [
                new_rule("one", "fist", "LEFT_CLICK")]})
        self.assertEqual(buffer.getvalue(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
