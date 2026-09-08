"""Synthetic reproduction of the semantic-transition misfire.

READ-ONLY INSTRUMENT.  No camera, no model, no mouse, no config file:
every scenario builds its rule set in memory and drives GestureFSM
directly, so the only thing under test is the transition engine.

It runs each scenario twice, against two rule sets:

  as-intended     one -> fist = LEFT_CLICK  and  two_up -> fist = RIGHT_CLICK
                  the mappings as described in the bug report.

  as-configured   ONLY one -> fist = LEFT_CLICK, copied field for field
                  from geometry_bindings in the live gesture_config.json.
                  The RIGHT_CLICK rule is in deleted_bindings there, which
                  GestureFSM._compile() deliberately never reads.

Running both is the whole point.  A scenario that passes as-intended and
fails as-configured is not an engine bug.

    python transition_probe.py            # summary
    python transition_probe.py -v         # every [transition-debug] block
"""
from __future__ import annotations

import sys

from gesture_fsm import GestureFSM, format_transition_report

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

# The live semantic machine, exactly as hand_cursor_2.py builds it.
SEMANTIC = {"stability_threshold": 1, "window_size": 1,
            "transition_memory": 6}

# ~4.7 Hz, the measured YOLO cadence.
FRAME = 0.21

# Copied field for field from geometry_bindings in gesture_config.json.
LIVE_LEFT = {
    "id": "r382d69fe", "trigger": "transition", "action": "LEFT_CLICK",
    "enabled": True, "cooldown_sec": 0.16, "from_state": "one",
    "to_state": "fist", "adaptive_timing": True, "promote_double": True,
    "name": "Left_Click", "source": "any", "hand": "any",
}
# Copied field for field from deleted_bindings in gesture_config.json.
LIVE_RIGHT = {
    "id": "r7cb46d31", "trigger": "transition", "action": "RIGHT_CLICK",
    "enabled": True, "cooldown_sec": 0.35, "from_state": "two_up",
    "to_state": "fist", "adaptive_timing": True, "name": "Right_Click",
    "source": "any", "hand": "any",
}

AS_INTENDED = [LIVE_LEFT, LIVE_RIGHT]
AS_CONFIGURED = [LIVE_LEFT]


def run(rules, script, settings=None, hand=None, hand_count=1,
        label=""):
    """Drive one scripted stream.  Returns the actions it produced."""
    cfg = {"settings": {}, "geometry_bindings": [dict(r) for r in rules],
           "yolo_bindings": [], "deleted_bindings": []}
    fsm = GestureFSM(cfg, **dict(SEMANTIC, **(settings or {})))

    actions, now = [], 100.0
    for pose in script:
        now += FRAME
        action = fsm.update(pose, now, source="semantic", hand=hand,
                            hand_count=hand_count)
        report = fsm.last_transition_report
        if VERBOSE and report is not None:
            print(format_transition_report(
                report, scenario=label, fed_pose=pose,
                dispatch=str(action) if action else "NONE"))
            print()
        if action is not None:
            actions.append(str(action))
    return actions


class Scenario:
    """One lettered test, with what it expects and what it got."""

    def __init__(self, letter, title, script, expected, **kwargs):
        self.letter = letter
        self.title = title
        self.script = script
        self.expected = expected          # per rule-set name
        self.kwargs = kwargs
        self.results = {}

    def execute(self):
        for name, rules in (("as-intended", AS_INTENDED),
                            ("as-configured", AS_CONFIGURED)):
            self.results[name] = run(
                rules, self.script,
                label="%s/%s" % (self.letter, name), **self.kwargs)
        return self


SCENARIOS = [
    Scenario("A", "one -> fist",
             ["one", "fist"],
             {"as-intended": ["LEFT_CLICK"],
              "as-configured": ["LEFT_CLICK"]}),

    Scenario("B", "two_up -> fist",
             ["two_up", "fist"],
             {"as-intended": ["RIGHT_CLICK"],
              "as-configured": []}),

    Scenario("C", "one -> two_up -> fist",
             ["one", "two_up", "fist"],
             {"as-intended": ["RIGHT_CLICK"],
              "as-configured": ["LEFT_CLICK"]}),

    Scenario("D", "one -> noisy two_up -> fist (live 1/1 vote)",
             ["one", "one", "one", "two_up", "fist", "fist"],
             {"as-intended": ["RIGHT_CLICK"],
              "as-configured": ["LEFT_CLICK"]}),

    Scenario("D2", "one -> noisy two_up -> fist (3/2 majority vote)",
             ["one", "one", "one", "two_up", "fist", "fist"],
             {"as-intended": ["LEFT_CLICK"],
              "as-configured": ["LEFT_CLICK"]},
             settings={"window_size": 3, "stability_threshold": 2,
                       "transition_memory": 2}),

    Scenario("E", "two_up -> fist, both mappings present",
             ["two_up", "fist"],
             {"as-intended": ["RIGHT_CLICK"],
              "as-configured": []}),

    Scenario("F", "one -> fist, both mappings present",
             ["one", "fist"],
             {"as-intended": ["LEFT_CLICK"],
              "as-configured": ["LEFT_CLICK"]}),

    Scenario("G", "two_up -> fist with the right hand",
             ["two_up", "fist"],
             {"as-intended": ["RIGHT_CLICK"],
              "as-configured": []},
             hand="right"),
]


def hand_gating():
    """TEST H: the same transition under per-hand rules."""
    left_only = dict(LIVE_LEFT, hand="left")
    right_only = dict(LIVE_RIGHT, hand="right")
    rules = [left_only, right_only]

    rows = [
        ("two_up -> fist, RIGHT hand", ["two_up", "fist"], "right",
         ["RIGHT_CLICK"]),
        ("two_up -> fist, LEFT hand", ["two_up", "fist"], "left", []),
        ("one -> fist, LEFT hand", ["one", "fist"], "left",
         ["LEFT_CLICK"]),
        ("one -> fist, RIGHT hand", ["one", "fist"], "right", []),
    ]
    print("TEST H  per-hand rules: one->fist[left]=LEFT, "
          "two_up->fist[right]=RIGHT")
    passed = True
    for title, script, hand, expected in rows:
        got = run(rules, script, hand=hand, label="H/" + title)
        ok = got == expected
        passed = passed and ok
        print("   %-28s hand=%-5s -> %-16s expected %-16s  %s"
              % (title, hand, got or "[]", expected or "[]",
                 "PASS" if ok else "FAIL"))
    return passed


def main():
    print(__doc__.strip().splitlines()[0])
    print()
    print("%-4s %-42s %-14s %-16s %-16s %s"
          % ("", "scenario", "rule set", "expected", "got", ""))
    print("-" * 108)

    failures = []
    for scenario in SCENARIOS:
        scenario.execute()
        for name in ("as-intended", "as-configured"):
            got = scenario.results[name]
            expected = scenario.expected[name]
            ok = got == expected
            if not ok:
                failures.append((scenario.letter, name, expected, got))
            print("%-4s %-42s %-14s %-16s %-16s %s"
                  % (scenario.letter, scenario.title, name,
                     expected or "[]", got or "[]",
                     "PASS" if ok else "FAIL"))
        print()

    ok_h = hand_gating()
    if not ok_h:
        failures.append(("H", "per-hand", "see above", "see above"))

    print()
    print("=" * 108)
    if failures:
        print("MISMATCHES: %d" % len(failures))
        for letter, name, expected, got in failures:
            print("   %s/%s expected %s, got %s"
                  % (letter, name, expected, got))
    else:
        print("Every scenario matched its expectation for BOTH rule sets.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
