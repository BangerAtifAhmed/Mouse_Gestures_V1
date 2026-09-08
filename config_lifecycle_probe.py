"""End-to-end trace of a NEW mapping, GUI builder -> runtime FSM.

READ-ONLY with respect to your real config: every scenario writes to a
throwaway file in the system temp directory.  gesture_config.json is
never opened for writing, and never read except to report its path.

No camera, no model, no mouse.  The GUI's own serialiser is exercised
directly — GestureStudio._rules_to_config is called on a bare instance,
so the routing under test is the real one and not a copy of it.

    python config_lifecycle_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid

import gesture_fsm
from gesture_fsm import GestureFSM, load_config

# The live semantic machine, exactly as HandTrackerEngine._apply_config
# builds it.
SEMANTIC = {"stability_threshold": 1, "window_size": 1,
            "transition_memory": 6}
FRAME = 0.21

PASS, FAIL = "PASS", "FAIL"
results = []


def check(label, got, expected):
    ok = got == expected
    results.append((label, ok))
    print("   %-52s got %-18s expected %-18s %s"
          % (label, repr(got), repr(expected), PASS if ok else FAIL))
    return ok


def new_rule(from_state, to_state, action, **extra):
    """A rule shaped exactly as GestureStudio._read_builder emits one."""
    rule = {
        "id": "r%s" % uuid.uuid4().hex[:8],
        "trigger": "transition",
        "from_state": from_state,
        "to_state": to_state,
        "action": action,
        "enabled": True,
        "cooldown_sec": 0.35,
        "source": "any",
        "hand": "any",
    }
    rule.update(extra)
    return rule


def gui_serialise(rules, deleted=(), settings=None):
    """Run the REAL GestureStudio._rules_to_config on a bare instance.

    Built with object.__new__ so no window is created: the method only
    touches self.rules, self.deleted and self.settings, and its call to
    _widgets_to_settings returns immediately when the panel is absent.
    """
    import app
    studio = object.__new__(app.GestureStudio)
    # tk.Tk.__getattr__ delegates to self.tk, which recurses forever on an
    # uninitialised instance.  Binding it to None makes the delegation
    # raise AttributeError instead, so hasattr() answers False and
    # _widgets_to_settings takes its documented "no panel" early return.
    studio.tk = None
    studio.rules = [dict(r) for r in rules]
    studio.deleted = [dict(r) for r in deleted]
    studio.settings = dict(settings or gesture_fsm.DEFAULT_SETTINGS)
    return app.GestureStudio._rules_to_config(studio)


def build_machines(cfg):
    """The two FSMs, exactly as _apply_config builds them."""
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


# ═══ INVESTIGATION 6 — a fresh config with ONLY the new mapping ═════════

def investigation_6(tmp):
    print("\nINVESTIGATION 6 — fresh config, ONLY two_up -> fist "
          "= RIGHT_CLICK")
    right = new_rule("two_up", "fist", "RIGHT_CLICK")
    payload = gui_serialise([right])

    path = os.path.join(tmp, "only_right.json")
    gesture_fsm.save_config(payload, path)
    on_disk = json.load(open(path, encoding="utf-8"))

    check("written to an ACTIVE list",
          [k for k in ("geometry_bindings", "yolo_bindings")
           if on_disk.get(k)], ["geometry_bindings"])
    check("deleted_bindings empty", on_disk.get("deleted_bindings"), [])

    cfg = load_config(path)
    geometry, semantic = build_machines(cfg)
    check("semantic FSM rule count", len(tuple(semantic.rules)), 1)
    check("semantic compiled rule", compiled(semantic),
          ["two_up->fist=RIGHT_CLICK"])
    check("perform two_up -> fist", drive(semantic, ["two_up", "fist"]),
          ["RIGHT_CLICK"])
    check("no LEFT_CLICK anywhere",
          "LEFT_CLICK" in str(drive(semantic, ["two_up", "fist"])), False)


# ═══ INVESTIGATION 7 — build up two mappings across save/reload ═════════

def investigation_7(tmp):
    print("\nINVESTIGATION 7 — create, save, restart, add, save, restart")
    path = os.path.join(tmp, "lifecycle.json")

    # A. blank config
    gesture_fsm.save_config(gui_serialise([]), path)
    _geo, semantic = build_machines(load_config(path))
    check("A. blank config compiles nothing", len(tuple(semantic.rules)), 0)

    # B/C. create two_up -> fist = RIGHT_CLICK and save
    right = new_rule("two_up", "fist", "RIGHT_CLICK")
    gesture_fsm.save_config(gui_serialise([right]), path)

    # D/E. restart == rebuild the machines from the file
    _geo, semantic = build_machines(load_config(path))
    check("E. compiled after restart", compiled(semantic),
          ["two_up->fist=RIGHT_CLICK"])
    # F.
    check("F. two_up -> fist", drive(semantic, ["two_up", "fist"]),
          ["RIGHT_CLICK"])

    # G/H. add one -> fist = LEFT_CLICK and save
    left = new_rule("one", "fist", "LEFT_CLICK")
    gesture_fsm.save_config(gui_serialise([right, left]), path)

    # I. reload
    _geo, semantic = build_machines(load_config(path))
    check("I. both rules compiled", compiled(semantic),
          ["one->fist=LEFT_CLICK", "two_up->fist=RIGHT_CLICK"])
    # J.
    check("J. two_up -> fist", drive(semantic, ["two_up", "fist"]),
          ["RIGHT_CLICK"])
    # K.  A fresh machine, so the first gesture's cooldown is irrelevant.
    _geo, semantic = build_machines(load_config(path))
    check("K. one -> fist", drive(semantic, ["one", "fist"]),
          ["LEFT_CLICK"])


# ═══ INVESTIGATION 8 — ids ══════════════════════════════════════════════

def investigation_8(tmp):
    print("\nINVESTIGATION 8 — id uniqueness and dedup")
    ids = {new_rule("a", "b", "LEFT_CLICK")["id"] for _ in range(5000)}
    check("5000 generated ids are unique", len(ids), 5000)

    # A new active rule whose id collides with a binned one must win:
    # _config_to_rules takes the active lists first and skips any binned
    # entry whose id is already seen.
    shared = "rCOLLIDE"
    active = new_rule("two_up", "fist", "RIGHT_CLICK", id=shared)
    binned = new_rule("one", "fist", "LEFT_CLICK", id=shared)
    payload = gui_serialise([active], deleted=[binned])
    path = os.path.join(tmp, "collide.json")
    gesture_fsm.save_config(payload, path)

    _geo, semantic = build_machines(load_config(path))
    check("active rule survives an id collision", compiled(semantic),
          ["two_up->fist=RIGHT_CLICK"])

    import app
    studio = object.__new__(app.GestureStudio)
    studio.tk = None
    studio._undo_stack = []
    studio.settings = dict(gesture_fsm.DEFAULT_SETTINGS)
    studio.rules, studio.deleted = [], []
    for method in ("_settings_to_widgets", "_forget_undo", "_render_bin"):
        setattr(studio, method, lambda *a, **k: None)
    app.GestureStudio._config_to_rules(
        studio, json.load(open(path, encoding="utf-8")))
    check("GUI keeps the active rule, not the binned one",
          [r["action"] for r in studio.rules], ["RIGHT_CLICK"])
    check("GUI drops the colliding bin entry", len(studio.deleted), 0)


# ═══ INVESTIGATION 9 — placement vs routing ════════════════════════════

def investigation_9(tmp):
    print("\nINVESTIGATION 9 — which list a rule lands in, and what routes it")
    any_rule = new_rule("two_up", "fist", "RIGHT_CLICK")
    sem_rule = new_rule("peace", "fist", "MIDDLE_CLICK", source="semantic")
    payload = gui_serialise([any_rule, sem_rule])

    check("source='any' is written to geometry_bindings",
          [r["id"] for r in payload["geometry_bindings"]], [any_rule["id"]])
    check("source='semantic' is written to yolo_bindings",
          [r["id"] for r in payload["yolo_bindings"]], [sem_rule["id"]])

    path = os.path.join(tmp, "routing.json")
    gesture_fsm.save_config(payload, path)
    cfg = load_config(path)
    geometry, semantic = build_machines(cfg)

    # The decisive point: _compile merges BOTH lists into ONE rule set
    # that BOTH machines receive.  Placement does not route.
    check("geometry FSM compiled both entries",
          len(tuple(geometry.rules)), 2)
    check("semantic FSM compiled both entries",
          len(tuple(semantic.rules)), 2)

    # What DOES route is the rule's own `source`, tested per frame.
    accepts_geo = [r.id for r in semantic.rules if r.accepts_source(
        "geometry")]
    accepts_sem = [r.id for r in semantic.rules if r.accepts_source(
        "semantic")]
    check("source='any' accepts the geometry stream",
          any_rule["id"] in accepts_geo, True)
    check("source='semantic' refuses the geometry stream",
          sem_rule["id"] in accepts_geo, False)
    check("both accept the semantic stream",
          sorted(accepts_sem), sorted([any_rule["id"], sem_rule["id"]]))

    # And so a semantic gesture bound with source='any' reaches the
    # semantic machine regardless of the list it was written to.
    check("the geometry-listed rule fires on the semantic stream",
          drive(semantic, ["two_up", "fist"]), ["RIGHT_CLICK"])


# ═══ THE REPORTED SYMPTOM, reproduced ══════════════════════════════════

def reproduce(tmp):
    print("\nREPRODUCTION — save a new mapping WITHOUT reloading the engine")
    path = os.path.join(tmp, "repro.json")

    # The engine starts with one mapping and builds its machines ONCE.
    left = new_rule("one", "fist", "LEFT_CLICK")
    gesture_fsm.save_config(gui_serialise([left]), path)
    _geo, running = build_machines(load_config(path))
    check("engine started with", compiled(running), ["one->fist=LEFT_CLICK"])

    # The user now creates the new mapping in the GUI and saves it.
    right = new_rule("two_up", "fist", "RIGHT_CLICK")
    gesture_fsm.save_config(gui_serialise([left, right]), path)
    on_disk = load_config(path)
    check("the file now holds both",
          sorted(r["action"] for r in on_disk["geometry_bindings"]),
          ["LEFT_CLICK", "RIGHT_CLICK"])

    # But nothing rebuilt the machines, so the running one is unchanged.
    check("the RUNNING machine still holds one rule",
          compiled(running), ["one->fist=LEFT_CLICK"])
    check("performing two_up -> fist on the running machine",
          drive(running, ["one", "two_up", "fist"]), ["LEFT_CLICK"])

    # Rebuilding is all it takes.
    _geo, rebuilt = build_machines(on_disk)
    check("after a rebuild, the same gesture",
          drive(rebuilt, ["one", "two_up", "fist"]), ["RIGHT_CLICK"])


def main():
    print(__doc__.strip().splitlines()[0])
    print("real config (never written here): %s" % gesture_fsm.CONFIG_PATH)

    with tempfile.TemporaryDirectory(prefix="mousegesture-probe-") as tmp:
        for stage in (investigation_6, investigation_7, investigation_8,
                      investigation_9, reproduce):
            stage(tmp)

    failed = [label for label, ok in results if not ok]
    print("\n" + "=" * 100)
    print("%d checks, %d failed" % (len(results), len(failed)))
    for label in failed:
        print("   FAILED: %s" % label)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
