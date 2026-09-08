"""Automated tests for the semantic-transition engine in gesture_fsm.py.

WHAT THIS IS.  A headless unit suite.  It needs no camera, no model, no
GUI and no mouse: it drives GestureFSM directly with scripted label
streams, which is the only way to reproduce a stabiliser race
deterministically.  gesture_test.py remains the live instrument for the
real pipeline; this is the bench for the logic underneath it.

WHAT IT NEVER TOUCHES.  gesture_config.json.  Every test builds its own
config dict in memory and hands it to GestureFSM directly, so no test
can read, seed, create or overwrite a user mapping.

WHAT IT ASSERTS.  Two kinds of test, kept apart on purpose:

  * BEHAVIOUR THAT IS CORRECT and must not regress — noise resistance,
    cooldown, adaptive timing, hand gating, stream separation.

  * CHARACTERISATION of behaviour that is under investigation, recorded
    as what the engine does TODAY rather than as what it should do.
    Every such test says so in its own docstring.  They exist so that a
    later change has a documented baseline to move, and so that moving
    it is a deliberate edit here rather than a silent surprise.

The live misfire this file was opened for is NOT reproduced by any test
here, because it is not an engine fault: see transition_probe.py.

    python test_gesture_fsm.py

Run it directly rather than through unittest discovery: this repository
also contains test.py and testmoniter.py, which are applications rather
than test modules and open windows when imported.
"""
from __future__ import annotations

import io
import contextlib
import unittest

import adaptive_timing
from gesture_fsm import (DOUBLE_CLICK, GestureFSM, LEFT_CLICK, MIDDLE_CLICK,
                         RIGHT_CLICK, format_transition_report)


# ─── Harness ────────────────────────────────────────────────────────────────

def rule(from_state, to_state, action, **extra):
    """One transition binding, in the shape the config file uses."""
    out = {
        "id": extra.pop("id", f"{from_state}->{to_state}"),
        "trigger": "transition",
        "from_state": from_state,
        "to_state": to_state,
        "action": action,
        "enabled": True,
        # Off unless a test is about cooldown, so an unrelated test cannot
        # fail for a reason it never meant to exercise.
        "cooldown_sec": extra.pop("cooldown_sec", 0.0),
    }
    out.update(extra)
    return out


def config(*rules, **settings):
    """A config dict with the diagnostic log off unless a test wants it."""
    merged = {"transition_log": False}
    merged.update(settings)
    return {
        "settings": merged,
        "geometry_bindings": list(rules),
        "yolo_bindings": [],
        "deleted_bindings": [],
    }


class Driver:
    """Feeds an FSM one label per frame and collects what comes back.

    The clock is the driver's, not the test's, so a test reads as the
    sequence of poses the hand made rather than as arithmetic on
    timestamps.
    """

    def __init__(self, fsm, step=0.033, source=None):
        self.fsm = fsm
        self.step = step
        self.source = source
        self.now = 100.0

    def feed(self, *labels, **kwargs):
        """Push labels one frame apart.  Returns the actions they produced."""
        step = kwargs.pop("step", None) or self.step
        hand = kwargs.pop("hand", None)
        hand_count = kwargs.pop("hand_count", 0)
        got = []
        for label in labels:
            self.now += step
            action = self.fsm.update(label, self.now, source=self.source,
                                     hand=hand, hand_count=hand_count)
            if action is not None:
                got.append(action)
        return got

    def wait(self, seconds):
        """Advance the clock without observing anything."""
        self.now += seconds


class RecordingLearner:
    """A TimingLearner stand-in that records instead of computing.

    Substituted so the timing assertions are about WHICH observations the
    engine offers for learning, which is the part this change could
    plausibly have altered.  The estimator's own maths is exercised
    against the real class in AdaptiveTimingTests.
    """

    def __init__(self, tolerance=1.2):
        self.submitted = []
        self._tolerance = tolerance

    def submit(self, from_state, to_state, duration_sec):
        self.submitted.append((from_state, to_state, duration_sec))
        return True

    def tolerance(self, from_state, to_state):
        return self._tolerance

    def snapshot(self):
        return {}

    def pairs(self):
        return [(a, b) for a, b, _ in self.submitted]


def captured(fn, *args, **kwargs):
    """Run `fn`, returning (result, everything it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


# The two mappings the reported failure was about.  Named once so every
# test that uses them shows the same pair.
LEFT_RULE = rule("one", "fist", LEFT_CLICK, id="left-rule")
RIGHT_RULE = rule("two_up", "fist", RIGHT_CLICK, id="right-rule")


# ─── The reported failure ───────────────────────────────────────────────────

class TransitionAttributionTests(unittest.TestCase):
    """Which of two rules sharing a target gets credited for an arrival."""

    def machine(self, **settings):
        return GestureFSM(config(LEFT_RULE, RIGHT_RULE, **settings))

    def test_a_single_frame_origin_under_a_majority_vote(self):
        """CHARACTERISATION — what the engine does today, not a verdict.

        `one` is stable; the classifier reports two_up once, then fist
        twice.  Under a 3/2 vote `two_up` never wins the threshold, so it
        never enters _recent at all, and `one` is still the nearest
        stabilised pose when `fist` arrives.  LEFT_CLICK fires.

        Whether that is right is a policy question about single-frame
        evidence, and this test takes no position on it.  Note that the
        LIVE semantic stream runs window_size=1, where this shape cannot
        arise — see StreamSeparationTests.
        """
        drive = Driver(self.machine())
        self.assertEqual(drive.feed("one", "one", "one"), [])
        self.assertEqual(drive.fsm.stable_gesture, "one")

        fired = drive.feed("two_up", "fist", "fist")

        self.assertEqual(drive.fsm.stable_gesture, "fist")
        self.assertEqual([str(a) for a in fired], [LEFT_CLICK])

    def test_the_same_shape_on_the_live_stream_settings(self):
        """The live semantic machine has no vote, so it sees `two_up`.

        window_size=1 makes every classifier frame a state, which puts
        two_up in _recent as the immediate predecessor and gives the
        RIGHT_CLICK rule the nearest origin.
        """
        fsm = GestureFSM(config(LEFT_RULE, RIGHT_RULE), window_size=1,
                         stability_threshold=1, transition_memory=6)
        drive = Driver(fsm, step=0.21)
        drive.feed("one")
        fired = drive.feed("two_up", "fist")
        self.assertEqual([str(a) for a in fired], [RIGHT_CLICK])

    def test_genuine_two_up_source_fires_right_click(self):
        """two_up held long enough to be accepted, then fist: RIGHT_CLICK."""
        drive = Driver(self.machine())
        drive.feed("one", "one", "one")
        self.assertEqual(drive.fsm.stable_gesture, "one")

        drive.feed("two_up", "two_up")
        self.assertEqual(drive.fsm.stable_gesture, "two_up",
                         "two_up should have been accepted as a state")

        fired = drive.feed("fist", "fist")
        self.assertEqual([str(a) for a in fired], [RIGHT_CLICK])

    def test_genuine_one_source_fires_left_click(self):
        """The mapping that was misfiring still fires when it should."""
        drive = Driver(self.machine())
        fired = drive.feed("one", "one", "one", "fist", "fist")
        self.assertEqual([str(a) for a in fired], [LEFT_CLICK])

    def test_the_two_rules_cannot_fire_each_other(self):
        """Both mappings, alternated, each keeping its own action."""
        drive = Driver(self.machine())

        drive.feed("one", "one", "one")
        self.assertEqual([str(a) for a in drive.feed("fist", "fist")],
                         [LEFT_CLICK])

        drive.feed("two_up", "two_up", "two_up")
        self.assertEqual([str(a) for a in drive.feed("fist", "fist")],
                         [RIGHT_CLICK])

        drive.feed("one", "one", "one")
        self.assertEqual([str(a) for a in drive.feed("fist", "fist")],
                         [LEFT_CLICK])

        drive.feed("two_up", "two_up", "two_up")
        self.assertEqual([str(a) for a in drive.feed("fist", "fist")],
                         [RIGHT_CLICK])

    def test_rule_id_matches_the_action(self):
        """The event carries the rule that produced it, not just a string."""
        drive = Driver(self.machine())
        fired = drive.feed("one", "one", "one", "fist", "fist")
        self.assertEqual(fired[0].rule_id, "left-rule")

        drive.feed("two_up", "two_up", "two_up")
        fired = drive.feed("fist", "fist")
        self.assertEqual(fired[0].rule_id, "right-rule")


# ─── The noise resistance that had to survive ───────────────────────────────

class NoiseResistanceTests(unittest.TestCase):
    """The majority vote still absorbs stray frames.

    The fix narrows attribution, and the thing it must not do is make
    every classifier frame authoritative — that would trade a wrong
    action for a different wrong action.
    """

    def test_stray_unmapped_pose_does_not_break_a_transition(self):
        """A frame of a pose no rule starts from is still just noise."""
        drive = Driver(GestureFSM(config(LEFT_RULE, RIGHT_RULE)))
        drive.feed("one", "one", "one")
        fired = drive.feed("palm", "fist", "fist")
        self.assertEqual([str(a) for a in fired], [LEFT_CLICK])

    def test_stray_frame_cannot_invent_a_transition(self):
        """One frame of an origin never becomes a state on its own."""
        drive = Driver(GestureFSM(config(LEFT_RULE)))
        drive.feed("palm", "palm", "palm")
        fired = drive.feed("one", "fist", "fist")
        self.assertEqual(fired, [])

    def test_an_intermediate_stable_pose_is_still_tolerated(self):
        """one -> peace -> fist still clicks: peace starts no rule.

        This is the curl a real hand makes, and the walk back through the
        history exists to survive it.
        """
        drive = Driver(GestureFSM(config(LEFT_RULE)))
        drive.feed("one", "one", "one")
        drive.feed("peace", "peace")
        self.assertEqual(drive.fsm.stable_gesture, "peace")
        fired = drive.feed("fist", "fist")
        self.assertEqual([str(a) for a in fired], [LEFT_CLICK])

    def test_a_competing_origin_in_the_middle_wins_the_walk(self):
        """The pose in the middle IS another rule's origin, and it is
        stabilised.  The backward walk stops at the nearest origin, so
        the newer rule wins."""
        drive = Driver(GestureFSM(config(LEFT_RULE, RIGHT_RULE)))
        drive.feed("one", "one", "one")
        drive.feed("two_up", "two_up")
        fired = drive.feed("fist", "fist")
        self.assertEqual([str(a) for a in fired], [RIGHT_CLICK],
                         "the nearest origin owns the transition")

    def test_an_unbound_pose_in_the_middle_is_walked_past(self):
        """THE MECHANISM BEHIND THE LIVE MISFIRE, in isolation.

        `two_up` is stabilised and is the immediate predecessor — but no
        COMPILED rule starts from it, so it is not an origin, and the
        documented single-hop tolerance walks straight past it to `one`.

        This is the engine behaving as designed.  It only looks like a
        bug when the user believes a two_up rule exists and it does not:
        see transition_probe.py.
        """
        drive = Driver(GestureFSM(config(LEFT_RULE)))     # no two_up rule
        drive.feed("one", "one", "one")
        drive.feed("two_up", "two_up")
        fired = drive.feed("fist", "fist")
        self.assertEqual([str(a) for a in fired], [LEFT_CLICK])

    def test_stabilizer_settings_are_unchanged(self):
        """The vote is still a vote, with the documented defaults."""
        fsm = GestureFSM(config(LEFT_RULE))
        self.assertEqual(fsm.stabilizer.window_size, 3)
        self.assertEqual(fsm.stabilizer.threshold, 2)

    def test_single_frame_poses_never_become_states(self):
        """The whole point of the window, asserted directly."""
        fsm = GestureFSM(config(LEFT_RULE))
        drive = Driver(fsm)
        drive.feed("one", "one")
        self.assertEqual(fsm.stable_gesture, "one")
        drive.feed("two_up")
        self.assertEqual(fsm.stable_gesture, "one",
                         "one frame outvoted a stabilised pose")


# ─── Generality ─────────────────────────────────────────────────────────────

class ArbitraryGestureNameTests(unittest.TestCase):
    """Nothing in the fix knows a gesture name.

    The same two shapes as StaleOriginTests, driven with labels that
    exist nowhere in the codebase.  If any part of the attribution were
    special-cased to one, two_up or fist, these would pass while the real
    ones failed, or the reverse.
    """

    RULES = (rule("alpha", "charlie", LEFT_CLICK, id="a"),
             rule("bravo", "charlie", MIDDLE_CLICK, id="b"))

    def test_single_frame_origin_for_invented_names(self):
        """CHARACTERISATION, matching the real-name case exactly."""
        drive = Driver(GestureFSM(config(*self.RULES)))
        drive.feed("alpha", "alpha", "alpha")
        fired = drive.feed("bravo", "charlie", "charlie")
        self.assertEqual([str(a) for a in fired], [LEFT_CLICK])

    def test_accepted_origin_fires_for_invented_names(self):
        drive = Driver(GestureFSM(config(*self.RULES)))
        drive.feed("alpha", "alpha", "alpha")
        drive.feed("bravo", "bravo")
        fired = drive.feed("charlie", "charlie")
        self.assertEqual([str(a) for a in fired], [MIDDLE_CLICK])

    def test_three_rules_sharing_one_target(self):
        """The gate is a set membership test, so arity is not special."""
        rules = (rule("aa", "zz", LEFT_CLICK, id="1"),
                 rule("bb", "zz", RIGHT_CLICK, id="2"),
                 rule("cc", "zz", MIDDLE_CLICK, id="3"))
        drive = Driver(GestureFSM(config(*rules)))
        drive.feed("aa", "aa", "aa")
        # CHARACTERISATION: cc never won the vote, so aa is still nearest.
        self.assertEqual([str(a) for a in drive.feed("cc", "zz", "zz")],
                         [LEFT_CLICK])

        drive.feed("cc", "cc", "cc")
        self.assertEqual([str(a) for a in drive.feed("zz", "zz")],
                         [MIDDLE_CLICK])

    def test_unicode_and_spaces_in_names(self):
        """Labels are opaque strings to the engine, and stay that way."""
        rules = (rule("hand up", "hand closed", LEFT_CLICK, id="u1"),
                 rule("hånd_né", "hand closed", RIGHT_CLICK, id="u2"))
        drive = Driver(GestureFSM(config(*rules)))
        drive.feed("hand up", "hand up", "hand up")
        # Labels round-trip untouched; the attribution is the same shape
        # as every other case, and so is the characterised outcome.
        self.assertEqual([str(a) for a in drive.feed(
            "hånd_né", "hand closed", "hand closed")], [LEFT_CLICK])


# ─── Diagnostics ────────────────────────────────────────────────────────────

class TransitionDebugTests(unittest.TestCase):
    """The [transition-debug] instrument reports, and changes nothing."""

    def machine(self, *rules, **settings):
        return GestureFSM(config(*rules, **settings))

    def report_for(self, *rules, **kwargs):
        script = kwargs.pop("script")
        fsm = self.machine(*rules, **kwargs)
        drive = Driver(fsm, source="semantic")
        drive.feed(*script)
        return fsm.last_transition_report

    def test_the_report_lists_every_rule_targeting_the_destination(self):
        report = self.report_for(LEFT_RULE, RIGHT_RULE,
                                 script=("one", "one", "one", "fist",
                                         "fist"))
        self.assertEqual(len(report["candidates"]), 2)
        self.assertEqual({row["from_state"] for row in report["candidates"]},
                         {"one", "two_up"})

    def test_the_report_carries_every_requested_field(self):
        report = self.report_for(LEFT_RULE, RIGHT_RULE,
                                 script=("one", "one", "one", "fist",
                                         "fist"))
        for key in ("stream", "state_before", "state_after", "destination",
                    "recent", "candidates", "selected", "walk_origins",
                    "walk_skipped", "walk_stopped_at"):
            self.assertIn(key, report)
        row = report["candidates"][0]
        for key in ("rule_id", "from_state", "to_state", "action", "hand",
                    "source", "elapsed", "timing_ok", "hand_ok",
                    "source_ok", "cooldown_ok", "selected"):
            self.assertIn(key, row)

    def test_recent_history_carries_timestamps(self):
        report = self.report_for(LEFT_RULE, RIGHT_RULE,
                                 script=("one", "one", "one", "fist",
                                         "fist"))
        self.assertTrue(all(isinstance(t, float)
                            for _state, t in report["recent"]))
        self.assertEqual([s for s, _t in report["recent"]], ["one", "fist"])

    def test_the_walk_records_what_it_skipped(self):
        """The decisive field for the live misfire."""
        fsm = GestureFSM(config(LEFT_RULE), window_size=1,
                         stability_threshold=1, transition_memory=6)
        drive = Driver(fsm, step=0.21, source="semantic")
        drive.feed("one", "two_up", "fist")
        report = fsm.last_transition_report
        self.assertEqual(report["walk_origins"], ["one"])
        self.assertEqual(report["walk_skipped"], ["two_up"])
        self.assertEqual(report["walk_stopped_at"], "one")

    def test_a_report_is_produced_when_nothing_is_selected(self):
        fsm = GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, max_time_sec=0.05)))
        drive = Driver(fsm, source="semantic")
        drive.feed("one", "one", "one")
        drive.wait(1.0)
        self.assertEqual(drive.feed("fist", "fist"), [])
        report = fsm.last_transition_report
        self.assertEqual(report["selected"], [])
        self.assertEqual(len(report["candidates"]), 1)
        self.assertFalse(report["candidates"][0]["timing_ok"])

    def test_no_report_for_a_pose_no_rule_targets(self):
        fsm = GestureFSM(config(LEFT_RULE))
        Driver(fsm, source="semantic").feed("palm", "palm", "palm")
        self.assertIsNone(fsm.last_transition_report)

    def test_the_instrument_is_silent_by_default(self):
        drive = Driver(GestureFSM(config(LEFT_RULE)), source="semantic")
        _fired, out = captured(drive.feed, "one", "one", "one", "fist",
                               "fist")
        self.assertEqual(out, "")

    def test_the_instrument_prints_when_asked(self):
        drive = Driver(self.machine(LEFT_RULE, RIGHT_RULE,
                                    transition_debug=True),
                       source="semantic")
        _fired, out = captured(drive.feed, "one", "one", "one", "fist",
                               "fist")
        self.assertIn("[transition-debug]", out)
        self.assertIn("selected=one->fist:LEFT_CLICK", out)
        self.assertIn("rule_id=left-rule", out)

    def test_the_formatter_merges_caller_supplied_fields(self):
        report = self.report_for(LEFT_RULE, RIGHT_RULE,
                                 script=("one", "one", "one", "fist",
                                         "fist"))
        text = format_transition_report(report, raw="fist",
                                        labelled="fist_right",
                                        confidence="0.83",
                                        dispatch="LEFT_CLICK")
        self.assertIn("raw=fist", text)
        self.assertIn("labelled=fist_right", text)
        self.assertIn("confidence=0.83", text)
        self.assertIn("dispatch=LEFT_CLICK", text)

    def test_the_cooldown_column_reads_pre_fire(self):
        """Built before _fire(), so the winner is not shown as cooling."""
        fsm = GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, cooldown_sec=5.0)))
        drive = Driver(fsm, source="semantic")
        drive.feed("one", "one", "one", "fist", "fist")
        self.assertTrue(fsm.last_transition_report["candidates"][0]
                        ["cooldown_ok"])


# ─── Cooldown ───────────────────────────────────────────────────────────────

class CooldownTests(unittest.TestCase):
    """Cooldown is unchanged, and remains independent of the transition
    window that decides whether a rule matched at all."""

    def gesture(self, drive):
        """One full one -> fist, from a clean origin."""
        drive.feed("one", "one", "one")
        return drive.feed("fist", "fist")

    def test_second_attempt_inside_the_cooldown_is_suppressed(self):
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, cooldown_sec=1.0))))
        self.assertEqual([str(a) for a in self.gesture(drive)], [LEFT_CLICK])
        self.assertEqual(self.gesture(drive), [])

    def test_the_same_attempt_fires_once_the_cooldown_lapses(self):
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, cooldown_sec=1.0))))
        self.gesture(drive)
        drive.wait(1.5)
        self.assertEqual([str(a) for a in self.gesture(drive)], [LEFT_CLICK])

    def test_zero_cooldown_fires_every_time(self):
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, cooldown_sec=0.0))))
        for _ in range(3):
            self.assertEqual([str(a) for a in self.gesture(drive)],
                             [LEFT_CLICK])

    def test_cooldown_is_per_rule_not_per_target(self):
        """A cooling rule does not silence its neighbour."""
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, cooldown_sec=5.0,
                 id="left-rule"),
            rule("two_up", "fist", RIGHT_CLICK, cooldown_sec=0.0,
                 id="right-rule"))))
        self.assertEqual([str(a) for a in self.gesture(drive)], [LEFT_CLICK])

        drive.feed("two_up", "two_up", "two_up")
        self.assertEqual([str(a) for a in drive.feed("fist", "fist")],
                         [RIGHT_CLICK])

        self.assertEqual(self.gesture(drive), [], "left rule still cooling")

    def test_cooldown_does_not_widen_or_narrow_the_transition_window(self):
        """Two rules with identical windows and different cooldowns reach
        the window test identically."""
        wide = GestureFSM(config(rule("one", "fist", LEFT_CLICK,
                                      cooldown_sec=0.0, max_time_sec=0.2)))
        cooled = GestureFSM(config(rule("one", "fist", LEFT_CLICK,
                                        cooldown_sec=9.0,
                                        max_time_sec=0.2)))
        for fsm in (wide, cooled):
            drive = Driver(fsm)
            drive.feed("one", "one", "one")
            drive.wait(0.5)                    # past max_time_sec
            self.assertEqual(drive.feed("fist", "fist"), [],
                             "the window, not the cooldown, rejects this")

    def test_a_cooling_rule_does_not_hand_its_trigger_to_a_sibling(self):
        """The cooldown test runs AFTER ownership is decided, so a
        cooling rule blocks its trigger rather than yielding it."""
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, cooldown_sec=5.0, hand="right",
                 id="left-rule"),
            rule("one", "fist", RIGHT_CLICK, hand="left", id="other"))))
        self.assertEqual([str(a) for a in self.gesture(drive)], [LEFT_CLICK])
        # Unknown handedness satisfies both; the first rule owns the
        # trigger and is cooling, so nothing fires.
        self.assertEqual(self.gesture(drive), [])

    def test_promote_double_still_works(self):
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, cooldown_sec=0.05,
                 promote_double=True))))
        self.assertEqual([str(a) for a in self.gesture(drive)], [LEFT_CLICK])
        self.assertEqual([str(a) for a in self.gesture(drive)],
                         [DOUBLE_CLICK])


# ─── Adaptive timing ────────────────────────────────────────────────────────

class AdaptiveTimingTests(unittest.TestCase):
    """Timing is unchanged: per pose pair, no learning from lost hands,
    forgotten on delete, and never mixed into recognition."""

    def test_a_genuine_transition_is_sampled_for_its_own_pair(self):
        learner = RecordingLearner()
        drive = Driver(GestureFSM(
            config(rule("one", "fist", LEFT_CLICK, adaptive_timing=True)),
            learner=learner))
        drive.feed("one", "one", "one", "fist", "fist")
        self.assertEqual(learner.pairs(), [("one", "fist")])
        self.assertGreater(learner.submitted[0][2], 0.0)

    def test_only_adaptive_rules_are_sampled(self):
        learner = RecordingLearner()
        drive = Driver(GestureFSM(
            config(rule("one", "fist", LEFT_CLICK)), learner=learner))
        drive.feed("one", "one", "one", "fist", "fist")
        self.assertEqual(learner.submitted, [])

    def test_a_lost_hand_on_the_path_is_not_a_sample(self):
        """Unchanged: `one -> hand lost -> fist` measures the gap, not
        the movement, so it teaches the estimator nothing."""
        learner = RecordingLearner()
        drive = Driver(GestureFSM(
            config(rule("one", "fist", LEFT_CLICK, adaptive_timing=True)),
            learner=learner))
        drive.feed("one", "one", "one")
        drive.feed("none", "none")
        fired = drive.feed("fist", "fist")

        self.assertEqual(learner.submitted, [],
                         "a lost hand became training data")
        self.assertEqual([str(a) for a in fired], [LEFT_CLICK],
                         "the gate is about learning, not recognition")

    def test_a_sample_is_taken_before_the_window_filters_it(self):
        """Unchanged: an arrival too slow to fire is still learnt from,
        or the estimate can never grow past the gate tuning it."""
        learner = RecordingLearner(tolerance=0.1)
        drive = Driver(GestureFSM(
            config(rule("one", "fist", LEFT_CLICK, adaptive_timing=True)),
            learner=learner))
        drive.feed("one", "one", "one")
        drive.wait(1.0)
        fired = drive.feed("fist", "fist")

        self.assertEqual(fired, [], "1.0s is outside a 0.1s tolerance")
        self.assertEqual(learner.pairs(), [("one", "fist")],
                         "the rejected arrival was not sampled")

    def test_the_pair_that_is_sampled_is_the_pair_that_matched(self):
        """CHARACTERISATION.  The sample follows the attribution.

        Under a 3/2 vote the single two_up frame is invisible, `one`
        wins the walk, and the elapsed time is booked against the
        `one -> fist` pair.  If the attribution is ever changed, this
        expectation has to move with it — which is why it is written
        down rather than left implicit.
        """
        learner = RecordingLearner()
        drive = Driver(GestureFSM(
            config(rule("one", "fist", LEFT_CLICK, adaptive_timing=True),
                   rule("two_up", "fist", RIGHT_CLICK,
                        adaptive_timing=True)),
            learner=learner))
        drive.feed("one", "one", "one")
        drive.feed("two_up", "fist", "fist")
        self.assertEqual(learner.pairs(), [("one", "fist")])

    def test_the_two_pairs_keep_separate_buckets(self):
        learner = RecordingLearner()
        drive = Driver(GestureFSM(
            config(rule("one", "fist", LEFT_CLICK, adaptive_timing=True),
                   rule("two_up", "fist", RIGHT_CLICK,
                        adaptive_timing=True)),
            learner=learner))
        drive.feed("one", "one", "one", "fist", "fist")
        drive.feed("two_up", "two_up", "two_up")
        drive.feed("fist", "fist")
        self.assertEqual(learner.pairs(),
                         [("one", "fist"), ("two_up", "fist")])

    def test_an_explicit_window_is_never_replaced_by_the_estimator(self):
        learner = RecordingLearner(tolerance=2.5)
        drive = Driver(GestureFSM(
            config(rule("one", "fist", LEFT_CLICK, max_time_sec=0.2)),
            learner=learner))
        drive.feed("one", "one", "one")
        drive.wait(1.0)
        self.assertEqual(drive.feed("fist", "fist"), [],
                         "a non-adaptive rule kept its own max_time_sec")

    def test_an_adaptive_rule_uses_the_estimator(self):
        learner = RecordingLearner(tolerance=2.0)
        drive = Driver(GestureFSM(
            config(rule("one", "fist", LEFT_CLICK, adaptive_timing=True,
                        max_time_sec=0.2)),
            learner=learner))
        drive.feed("one", "one", "one")
        drive.wait(1.0)
        self.assertEqual([str(a) for a in drive.feed("fist", "fist")],
                         [LEFT_CLICK],
                         "1.0s is inside the estimator's 2.0s tolerance")

    def test_estimator_starts_at_the_baseline_and_only_grows(self):
        estimator = adaptive_timing.AdaptiveTiming()
        self.assertEqual(estimator.tolerance("one", "fist"),
                         adaptive_timing.BASELINE_SEC)
        for _ in range(adaptive_timing.MIN_SAMPLES + 5):
            estimator.observe("one", "fist", 0.05)
        self.assertEqual(estimator.tolerance("one", "fist"),
                         adaptive_timing.BASELINE_SEC,
                         "fast samples must not tighten the window")

    def test_a_deleted_mapping_forgets_its_timing(self):
        estimator = adaptive_timing.AdaptiveTiming()
        for _ in range(adaptive_timing.MIN_SAMPLES + 5):
            estimator.observe("one", "fist", 1.8)
            estimator.observe("two_up", "fist", 1.8)
        self.assertGreater(estimator.tolerance("one", "fist"),
                           adaptive_timing.BASELINE_SEC)

        self.assertTrue(estimator.forget("one", "fist"))
        self.assertEqual(estimator.tolerance("one", "fist"),
                         adaptive_timing.BASELINE_SEC)
        self.assertEqual(estimator.samples("one", "fist"), 0)

        self.assertGreater(estimator.tolerance("two_up", "fist"),
                           adaptive_timing.BASELINE_SEC,
                           "forgetting one pair touched another")
        self.assertFalse(estimator.forget("one", "fist"))

    def test_the_real_learner_still_round_trips(self):
        learner = adaptive_timing.TimingLearner()
        try:
            for _ in range(adaptive_timing.MIN_SAMPLES + 5):
                learner.submit("one", "fist", 1.8)
            self.assertTrue(learner.drain(timeout=5.0))
            self.assertGreaterEqual(learner.samples("one", "fist"),
                                    adaptive_timing.MIN_SAMPLES)
            self.assertTrue(learner.forget("one", "fist"))
            self.assertEqual(learner.tolerance("one", "fist"),
                             adaptive_timing.BASELINE_SEC)
        finally:
            learner.stop()


# ─── Hand gating ────────────────────────────────────────────────────────────

class HandGateTests(unittest.TestCase):
    """The hand requirement is applied exactly as before the change."""

    def test_a_right_hand_rule_ignores_a_left_hand(self):
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, hand="right"))))
        drive.feed("one", "one", "one", hand="left")
        self.assertEqual(drive.feed("fist", "fist", hand="left"), [])

    def test_a_right_hand_rule_fires_for_a_right_hand(self):
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, hand="right"))))
        drive.feed("one", "one", "one", hand="right")
        self.assertEqual([str(a) for a in drive.feed("fist", "fist",
                                                     hand="right")],
                         [LEFT_CLICK])

    def test_two_rules_differing_only_by_hand(self):
        cfg = config(rule("one", "fist", LEFT_CLICK, hand="right", id="r"),
                     rule("one", "fist", RIGHT_CLICK, hand="left", id="l"))
        drive = Driver(GestureFSM(cfg))
        drive.feed("one", "one", "one", hand="left")
        fired = drive.feed("fist", "fist", hand="left")
        self.assertEqual([str(a) for a in fired], [RIGHT_CLICK])
        self.assertEqual(fired[0].rule_id, "l")

    def test_unknown_handedness_fires_exactly_one_rule(self):
        """A blank side satisfies both, and config order decides."""
        cfg = config(rule("one", "fist", LEFT_CLICK, hand="right", id="r"),
                     rule("one", "fist", RIGHT_CLICK, hand="left", id="l"))
        drive = Driver(GestureFSM(cfg))
        drive.feed("one", "one", "one", hand="")
        fired = drive.feed("fist", "fist", hand="")
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].rule_id, "r")

    def test_the_hand_gate_runs_before_the_stale_check(self):
        """A rule the hand excludes is not a competing origin.

        `two_up -> fist` is a left-hand rule, so on a right hand it is not
        in play at all and cannot make `one -> fist` ambiguous.
        """
        cfg = config(rule("one", "fist", LEFT_CLICK, id="left-rule"),
                     rule("two_up", "fist", RIGHT_CLICK, hand="left",
                          id="right-rule"))
        drive = Driver(GestureFSM(cfg))
        drive.feed("one", "one", "one", hand="right")
        fired = drive.feed("two_up", "fist", "fist", hand="right")
        self.assertEqual([str(a) for a in fired], [LEFT_CLICK])

    def test_both_hands_is_a_count_test(self):
        drive = Driver(GestureFSM(config(
            rule("one", "fist", LEFT_CLICK, hand="both"))))
        drive.feed("one", "one", "one", hand_count=1)
        self.assertEqual(drive.feed("fist", "fist", hand_count=1), [])

        drive.feed("one", "one", "one", hand_count=2)
        self.assertEqual([str(a) for a in drive.feed("fist", "fist",
                                                     hand_count=2)],
                         [LEFT_CLICK])


# ─── Stream separation ──────────────────────────────────────────────────────

class StreamSeparationTests(unittest.TestCase):
    """Two machines, two vocabularies, no shared state."""

    def test_the_two_machines_do_not_share_evidence(self):
        cfg = config(LEFT_RULE, RIGHT_RULE)
        geometry = GestureFSM(cfg)
        semantic = GestureFSM(cfg)

        # The semantic stream sees the pose that makes `one` stale.
        Driver(semantic, source="semantic").feed("one", "one", "one",
                                                 "two_up")

        # The geometric stream never saw it, so its own clean gesture
        # is unaffected by the other machine's evidence.
        drive = Driver(geometry, source="geometry")
        drive.feed("one", "one", "one")
        self.assertEqual([str(a) for a in drive.feed("fist", "fist")],
                         [LEFT_CLICK])

    def test_a_source_tagged_rule_ignores_the_other_stream(self):
        cfg = config(rule("one", "fist", LEFT_CLICK, source="semantic"))
        drive = Driver(GestureFSM(cfg), source="geometry")
        self.assertEqual(drive.feed("one", "one", "one", "fist", "fist"), [])

    def test_a_stabilizer_of_one_still_attributes_correctly(self):
        """The semantic machine runs window=1: every frame is a state.

        The gate must be a no-op there — the raw stream and the accepted
        stream are the same stream — and the correct rule must still win.
        """
        cfg = config(LEFT_RULE, RIGHT_RULE)
        fsm = GestureFSM(cfg, window_size=1, stability_threshold=1,
                         transition_memory=6)
        drive = Driver(fsm, step=0.21, source="semantic")
        drive.feed("one")
        self.assertEqual(fsm.stable_gesture, "one")
        self.assertEqual([str(a) for a in drive.feed("two_up", "fist")],
                         [RIGHT_CLICK])


# ─── Reset and recovery ─────────────────────────────────────────────────────

class ResetTests(unittest.TestCase):
    """The evidence trail is cleared wherever the stabiliser is."""

    def test_reset_forgets_the_raw_trail(self):
        fsm = GestureFSM(config(LEFT_RULE, RIGHT_RULE))
        drive = Driver(fsm)
        drive.feed("one", "one", "one", "two_up")
        fsm.reset(drive.now)

        # After a reset nothing is remembered, so a clean gesture is clean.
        drive.feed("one", "one", "one")
        self.assertEqual([str(a) for a in drive.feed("fist", "fist")],
                         [LEFT_CLICK])

    def test_reset_clears_the_stabilizer_too(self):
        fsm = GestureFSM(config(LEFT_RULE))
        drive = Driver(fsm)
        drive.feed("one", "one", "one")
        fsm.reset(drive.now)
        self.assertIsNone(fsm.stable_gesture)

    def test_an_empty_rule_set_is_quiet_and_harmless(self):
        # Building it prints the "no bindings configured" advisory, which
        # belongs in a user's console and not in a test report.
        fsm, _notice = captured(GestureFSM, config())
        drive = Driver(fsm)
        self.assertEqual(drive.feed("one", "one", "fist", "fist"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
