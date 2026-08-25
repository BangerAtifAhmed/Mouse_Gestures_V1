"""Gesture test harness — a 50-attempt bench for one existing mapping.

WHAT THIS IS.  A read-only instrument.  It attaches a GestureProbe to the
running HandTrackerEngine, watches the REAL recognition pipeline —
camera, MediaPipe, YOLO, the semantic GestureFSM, the real
ActionExecutor — and reports which stage each attempt reached.  Nothing
here synthesises a gesture, replays a recording, or stands in for a
model.  The user performs the gesture; this only watches.

WHAT IT DELIBERATELY DOES NOT DO.  It never writes gesture_config.json,
never creates or edits a mapping, never changes a threshold or a
setting, and never retries, debounces or suppresses an attempt.  A
failed attempt is recorded as failed.  The export path is checked
against the config path so a report can never overwrite the mapping
file.

ONE THING TO KNOW BEFORE RUNNING IT.  Because this is the real pipeline
with the real dispatcher, a successful attempt delivers a REAL mouse
click to whatever window has focus.  That is the point — "was the click
actually delivered" is one of the stages being measured — but it means
the test window should be the focused one, and 50 attempts means up to
50 real clicks.  The panel says so before it starts.
"""
from __future__ import annotations

import collections
import json
import os
import time
import tkinter as tk
from tkinter import filedialog, ttk

import hand_cursor_2
from gesture_fsm import CONFIG_PATH, HAND_ANY

# How long an attempt may stay unresolved before it is scored a failure.
# Generous: a slow deliberate fold at 4.7 Hz plus the FSM's own timing
# gates is well under this, so a timeout means the attempt really did
# stall rather than merely being slow.
ATTEMPT_TIMEOUT_SEC = 2.5

# Poll cadence for draining the probe.  Fast enough that the log feels
# live, slow enough that it costs the GUI thread nothing measurable.
DRAIN_MS = 100

FAIL_KINDS = ("YOLO", "Stabilizer", "FSM transition", "Hand gate",
              "Cooldown", "Rule matching", "Mouse dispatch",
              "Test instrumentation", "Other")

# The classification the report is written against.  H and I are kept
# apart deliberately: "the backend refused" and "the bench could not see
# it" are different findings, and collapsing them is what made the first
# 30-attempt report blame Windows for a bug in this file.
CLASSES = {
    "A": "YOLO miss",
    "B": "stabilizer/semantic miss",
    "C": "FSM transition not generated",
    "D": "transition generated but rule did not match",
    "E": "hand gate rejected",
    "F": "cooldown rejected",
    "G": "action not dispatched",
    "H": "dispatched but mouse backend failed",
    "I": "backend succeeded but the test did not observe the click",
}
CLASS_TO_KIND = {"A": "YOLO", "B": "Stabilizer", "C": "FSM transition",
                 "D": "Rule matching", "E": "Hand gate", "F": "Cooldown",
                 "G": "Mouse dispatch", "H": "Mouse dispatch",
                 "I": "Test instrumentation"}

# How close an observed OS click must be to a dispatch to count as the
# same event.  One YOLO frame is ~213 ms; this is wide enough to absorb
# listener thread scheduling without swallowing the next attempt.
CLICK_MATCH_SEC = 0.40

# Frames kept around a failure.  A failed attempt is worth reconstructing
# exactly; a successful one is not, and keeping every frame of fifty
# successes would bloat the export for no diagnostic gain.
FRAME_BUFFER = 600          # rolling, session-wide
FRAME_LEAD_SEC = 1.0        # context retained before the attempt armed


class ClickWatcher:
    """Independent OS-level observation of injected clicks.

    The mouse backends report nothing: pynput's click() returns None and
    WINAPI mouse_event is void, so "did the OS take it" cannot be read
    from the call.  A listener is a genuinely separate channel — it sees
    the event after the OS has processed it, rather than asking the
    injector whether it thinks it succeeded.

    Optional by design.  If pynput has no listener backend here, the
    bench says the observation is unavailable rather than scoring every
    click as a failure.
    """

    def __init__(self) -> None:
        self.available = False
        self.error = ""
        self.presses = []
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import mouse as _mouse
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}: {exc}"
            return

        def on_click(_x, _y, button, pressed):
            if pressed and str(button).endswith("left"):
                self.presses.append(time.perf_counter())

        try:
            self._listener = _mouse.Listener(on_click=on_click)
            self._listener.start()
            self.available = True
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}: {exc}"

    def saw_click_near(self, when: float) -> bool:
        """Was a left press observed within the match window of `when`?"""
        return any(abs(t - when) <= CLICK_MATCH_SEC for t in self.presses)

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None


def _ms(a, b):
    """b - a in milliseconds, or None if either end never happened."""
    if a is None or b is None:
        return None
    return round((b - a) * 1000.0, 1)


class Attempt:
    """One observed origin -> destination cycle, and where it got to."""

    def __init__(self, index: int, armed_at: float) -> None:
        self.index = index
        self.armed_at = armed_at

        # Stage flags
        self.yolo_saw_dest = False
        self.accepted = False
        self.transition = False
        self.rule_matched = False
        self.hand_gate = None          # None = not reached
        self.cooldown_ok = None
        self.dispatched = False
        self.delivered = None

        # Observations
        self.dest_score = None
        self.side = ""
        self.semantic_label = ""
        self.state_before = None
        self.state_after = None
        self.recent = []
        self.action = None

        # Timestamps (engine monotonic clock)
        self.one_detected_time = armed_at
        self.fist_detected_time = None
        self.fist_accepted_time = None
        self.fist_left_time = None
        self.transition_time = None
        self.dispatch_time = None
        self.mouse_click_time = None

        # Backend evidence, gathered from the counters rather than from
        # dispatch()'s return value, which is None on every path.
        # Cooldown is a property of the INSTANT the destination arrived,
        # not of when the bench got round to scoring the attempt.  Sampled
        # there and kept, so _diagnose cannot ask the question late and
        # get a different answer -- which is what filed three real
        # cooldown rejections under the generic FSM bucket.
        self.since_last_fire = None
        self.cooldown_remaining = None

        # Raw probe rows, kept so a failure can be reconstructed frame by
        # frame instead of argued about from aggregates.
        self.frames = []
        self.unexplained = False

        self.backend = ""
        self.handler_entered = False
        self.backend_invoked = False
        self.backend_ok = None         # None = no evidence either way
        self.clicks_delta = 0
        self.os_observed = None        # None = watcher unavailable

        self.result = None             # "SUCCESS" / "FAILED"
        self.reason = ""
        self.fail_kind = ""
        self.klass = ""

    # ── derived timings ────────────────────────────────────────────────
    @property
    def one_to_fist_ms(self):
        return _ms(self.one_detected_time, self.fist_accepted_time)

    @property
    def fist_duration_ms(self):
        return _ms(self.fist_accepted_time,
                   self.fist_left_time or self.transition_time)

    @property
    def detect_to_accept_ms(self):
        return _ms(self.fist_detected_time, self.fist_accepted_time)

    @property
    def accept_to_dispatch_ms(self):
        return _ms(self.fist_accepted_time, self.dispatch_time)

    @property
    def total_ms(self):
        return _ms(self.one_detected_time,
                   self.mouse_click_time or self.transition_time)

    def as_dict(self) -> dict:
        return {
            "attempt": self.index,
            "result": self.result,
            "reason": self.reason,
            "fail_kind": self.fail_kind,
            "class": self.klass,
            "class_meaning": CLASSES.get(self.klass, ""),
            "unexplained": self.unexplained,
            "stages": {
                "yolo_detected_dest": self.yolo_saw_dest,
                "yolo_confidence": self.dest_score,
                "semantic_label": self.semantic_label,
                "side": self.side,
                "stabilizer_accepted": self.accepted,
                "state_before": self.state_before,
                "state_after": self.state_after,
                "transition_detected": self.transition,
                "rule_matched": self.rule_matched,
                "hand_gate_passed": self.hand_gate,
                "cooldown_passed": self.cooldown_ok,
                "since_last_fire_ms": (round(self.since_last_fire * 1000, 1)
                                       if self.since_last_fire is not None
                                       else None),
                "cooldown_remaining_ms": (
                    round(self.cooldown_remaining * 1000, 1)
                    if self.cooldown_remaining is not None else None),
                "action_dispatched": self.dispatched,
                "left_click_handler_entered": self.handler_entered,
                "mouse_backend": self.backend,
                "mouse_backend_invoked": self.backend_invoked,
                "mouse_backend_succeeded": self.backend_ok,
                "click_count_delta": self.clicks_delta,
                "os_click_observed": self.os_observed,
                "mouse_click_delivered": self.delivered,
                "recent_states": list(self.recent),
                "action": self.action,
            },
            "timings_ms": {
                "one_to_fist": self.one_to_fist_ms,
                "fist_detect_to_accept": self.detect_to_accept_ms,
                "fist_duration": self.fist_duration_ms,
                "accept_to_dispatch": self.accept_to_dispatch_ms,
                "total": self.total_ms,
            },
            "raw_frames": self.frames,
            "timestamps": {
                "one_detected": self.one_detected_time,
                "fist_detected": self.fist_detected_time,
                "fist_accepted": self.fist_accepted_time,
                "transition": self.transition_time,
                "dispatch": self.dispatch_time,
                "mouse_click": self.mouse_click_time,
            },
        }

    def render(self, origin: str, dest: str) -> str:
        """The per-attempt block shown live in the log."""
        tick, cross = "✓", "✗"
        out = [f"Attempt #{self.index}"]
        if self.yolo_saw_dest:
            score = ("%.2f" % self.dest_score) if self.dest_score else "n/a"
            out.append(f"YOLO: {origin} → {dest} ({score})")
        else:
            out.append(f"YOLO: {origin} → (never saw {dest})")
        out.append(f"Side: {self.side or 'unknown'}")
        out.append(f"Semantic: {self.semantic_label or '-'}")
        out.append(f"FSM: {origin} → {dest} "
                   f"{tick if self.transition else cross}")
        if self.transition:
            out.append(f"Rule: matched {tick}")
            if self.hand_gate is not None:
                out.append(f"Hand gate: {tick if self.hand_gate else cross}")
            if self.cooldown_ok is not None:
                out.append(f"Cooldown: {tick if self.cooldown_ok else cross}")
            out.append(f"Action: {self.action} "
                       f"{tick if self.delivered else cross}")
        else:
            out.append(f"Reason: {self.reason}")
        timing = self.one_to_fist_ms
        if timing is not None:
            out.append(f"Timing: one→fist {timing} ms"
                       + (f", total {self.total_ms} ms"
                          if self.total_ms is not None else ""))
        out.append(f"Result: {self.result}")
        return "\n".join(out) + "\n\n"


class TestSession:
    """Segments the probe stream into attempts and scores them.

    Kept free of Tk so the scoring logic can be exercised headlessly —
    the panel below is only a view onto this.
    """

    def __init__(self, rule: dict, target: int = 50) -> None:
        self.rule = rule
        self.origin = str(rule.get("from_state", ""))
        self.dest = str(rule.get("to_state", ""))
        self.action = str(rule.get("action", ""))
        self.hand = str(rule.get("hand", HAND_ANY) or HAND_ANY)
        self.cooldown = float(rule.get("cooldown_sec", 0.0) or 0.0)
        self.max_time = float(rule.get("max_time_sec", 0.0) or 0.0)
        self.target = int(target)

        self.watcher = None            # set by the panel; may be absent
        self._rows = collections.deque(maxlen=FRAME_BUFFER)
        self.attempts = []
        self.open = None
        self.left_origin = False
        self.last_fire_t = None
        self.rows_seen = 0

    @property
    def done(self) -> bool:
        return len(self.attempts) >= self.target

    def feed(self, rows) -> list:
        """Consume probe rows; return the attempts closed by them."""
        closed = []
        for row in rows:
            self.rows_seen += 1
            closed.extend(self._one(row))
        return closed

    def _one(self, row) -> list:
        closed = []
        self._rows.append(row)
        now = row["t"]
        state = row["state_after"]
        pose = row["sem_pose"]

        # ── timeout on an attempt that stalled ──────────────────────────
        if self.open is not None and \
                (now - self.open.armed_at) > ATTEMPT_TIMEOUT_SEC:
            closed.append(self._close(self.open, now))
            self.open = None
            self.left_origin = False

        # ── arm on the origin becoming the state ────────────────────────
        # Note this does NOT return early.  A destination the stabiliser
        # rejected never becomes a state, so the stream stays on the
        # origin throughout — returning here would make exactly the
        # failure this bench exists to expose invisible to it.
        if state == self.origin:
            if self.open is None:
                self.open = Attempt(len(self.attempts) + 1, now)
                self.left_origin = False
            elif self.left_origin:
                # Came back to origin without firing: that attempt is over.
                closed.append(self._close(self.open, now))
                self.open = Attempt(len(self.attempts) + 1, now)
                self.left_origin = False

        if self.open is None:
            return closed
        att = self.open

        # "Engaged" means the user visibly tried: either the state left
        # the origin, or the classifier named the destination even though
        # the state did not follow.  Either way the attempt is now
        # resolvable, and returning to the origin will score it.
        if state not in (self.origin, None, "", "none"):
            self.left_origin = True
        if pose == self.dest:
            self.left_origin = True

        # ── stage: YOLO reported the destination ────────────────────────
        if pose == self.dest:
            if not att.yolo_saw_dest:
                att.yolo_saw_dest = True
                att.fist_detected_time = now
            att.dest_score = row.get("score") or att.dest_score
            att.side = row.get("side") or att.side
            att.semantic_label = row.get("yolo_label") or att.semantic_label

        # ── stage: the stabiliser accepted it as a state ────────────────
        if state == self.dest and not att.accepted:
            att.accepted = True
            att.fist_accepted_time = now
            att.state_before = row.get("state_before")
            att.state_after = state
            att.recent = list(row.get("recent") or ())
            if self.last_fire_t is not None:
                att.since_last_fire = now - self.last_fire_t
                att.cooldown_remaining = max(
                    0.0, self.cooldown - att.since_last_fire)
        if att.accepted and state != self.dest and att.fist_left_time is None:
            att.fist_left_time = now

        # ── stage: the rule fired ───────────────────────────────────────
        if row.get("action") == self.action and \
                row.get("rule_id") == self.rule.get("id"):
            att.transition = True
            att.rule_matched = True
            att.hand_gate = True
            att.cooldown_ok = True
            att.dispatched = True
            att.action = row["action"]
            att.transition_time = now
            att.dispatch_time = now

            # [action] LEFT_CLICK has been printed by now.  What follows
            # is the backend question, answered from the counters: they
            # advance only after the cursor call returned without
            # raising.  A zero delta on a click action therefore means
            # the backend refused it, which is a different finding from
            # having no evidence at all.
            att.handler_entered = True
            att.backend_invoked = True
            att.backend = row.get("backend", "") or ""
            att.clicks_delta = (int(row.get("clicks_after", 0))
                                - int(row.get("clicks_before", 0)))
            att.backend_ok = att.clicks_delta > 0
            if self.watcher is not None and self.watcher.available:
                att.os_observed = self.watcher.saw_click_near(now)
            att.delivered = bool(att.backend_ok)
            if att.delivered:
                att.mouse_click_time = now
            self.last_fire_t = now
            closed.append(self._close(att, now))
            self.open = None
            self.left_origin = False
        return closed

    def _frame(self, row, att) -> dict:
        """One probe row, enriched with what the FSM would have weighed."""
        t = row["t"]
        # The probe records `recent` as bare state names, so the origin's
        # entry time comes from the attempt, which is the same instant:
        # both are "when the state became the origin".
        origin_t = round(att.armed_at, 4)
        elapsed = _ms(att.armed_at, t)
        cool_left = None
        if self.cooldown > 0 and self.last_fire_t is not None:
            cool_left = round(max(0.0, self.cooldown
                                  - (t - self.last_fire_t)) * 1000.0, 1)
        acted = row.get("action")
        if acted and row.get("rule_id") == self.rule.get("id"):
            verdict = f"FIRED {acted}"
        elif acted:
            verdict = f"fired {acted} for another rule"
        elif row.get("state_after") == self.dest:
            verdict = "destination is current, rule produced nothing"
        else:
            verdict = "no action"
        return {
            "timestamp": round(t, 4),
            "yolo_label": row.get("yolo_raw"),
            "yolo_confidence": row.get("score"),
            "semantic_label": row.get("yolo_label"),
            "hand": row.get("sem_hand") or row.get("side"),
            "hand_count": row.get("hand_count"),
            "fsm_state_before": row.get("state_before"),
            "fsm_state_after": row.get("state_after"),
            "fsm_recent": list(row.get("recent") or ()),
            "origin_timestamp": origin_t,
            "destination_timestamp": (att.fist_accepted_time
                                      if row.get("state_after") == self.dest
                                      else None),
            "elapsed_since_origin_ms": elapsed,
            "cooldown_remaining_ms": cool_left,
            "rule_evaluation": verdict,
        }

    def _close(self, att, now: float):
        """Score a finished attempt.  Never edits it into a pass."""
        if att.dispatched and att.backend_ok:
            att.result, att.klass = "SUCCESS", ""
            att.reason = ""
        elif att.dispatched and att.backend_ok is False:
            att.result, att.klass = "FAILED", "H"
            att.reason = ("cursor backend refused the click "
                          "(click_count did not advance)")
        elif att.dispatched:
            att.result, att.klass = "FAILED", "I"
            att.reason = ("dispatched, but this bench gathered no "
                          "evidence either way")
        elif not att.yolo_saw_dest:
            att.result, att.klass = "FAILED", "A"
            att.reason = (f"'{self.dest}' never detected above the "
                          f"confidence floor")
        elif not att.accepted:
            att.result, att.klass = "FAILED", "B"
            att.reason = "detected but not accepted as a semantic state"
        else:
            att.result = "FAILED"
            att.klass, att.reason = self._diagnose(att, now)
        att.fail_kind = CLASS_TO_KIND.get(att.klass, "") if att.klass else ""

        # Keep the frame trace for failures only.  A success needs no
        # reconstruction, and fifty of them would bloat the export.
        if att.result != "SUCCESS":
            lo = att.armed_at - FRAME_LEAD_SEC
            att.frames = [self._frame(r, att) for r in self._rows
                          if lo <= r["t"] <= now]
        self.attempts.append(att)
        return att

    def _diagnose(self, att, now):
        """Why did an accepted destination not produce a transition?"""
        if self.origin not in (att.recent or ()):
            return ("C",
                    f"'{self.origin}' was no longer in the transition "
                    f"memory when '{self.dest}' arrived")

        gap = att.one_to_fist_ms
        if self.max_time > 0 and gap is not None and \
                gap > self.max_time * 1000.0:
            return ("C",
                    f"origin was {gap:.0f} ms old, past the rule's "
                    f"{self.max_time:.2f}s window")

        if self.hand not in (HAND_ANY, "", None) and \
                att.side and att.side != self.hand:
            att.hand_gate = False
            return ("E",
                    f"rule is bound to '{self.hand}' but the hand was "
                    f"'{att.side}'")

        # Measured when the destination ARRIVED.  `now` is the close time
        # and can be hundreds of ms later, by which point a real cooldown
        # block has silently expired and the attempt looks like a generic
        # FSM failure instead of the throttle doing its job.
        if self.cooldown > 0 and att.since_last_fire is not None and \
                att.since_last_fire < self.cooldown:
            att.cooldown_ok = False
            return ("F",
                    f"only {att.since_last_fire * 1000:.0f} ms since the "
                    f"last fire when '{self.dest}' arrived, cooldown is "
                    f"{self.cooldown * 1000:.0f} ms")

        att.unexplained = True
        return ("C", "transition not generated -- no gate accounts for "
                     "this; see the frame trace")

    # ── summary ────────────────────────────────────────────────────────
    def summary(self) -> dict:
        n = len(self.attempts)
        ok = [a for a in self.attempts if a.result == "SUCCESS"]
        fails = {k: 0 for k in FAIL_KINDS}
        classes = {k: 0 for k in CLASSES}
        for a in self.attempts:
            if a.result != "SUCCESS":
                fails[a.fail_kind if a.fail_kind in fails else "Other"] += 1
                if a.klass in classes:
                    classes[a.klass] += 1
        timings = [a.one_to_fist_ms for a in ok if a.one_to_fist_ms]
        within = [a for a in self.attempts
                  if a.one_to_fist_ms is not None and self.max_time > 0
                  and a.one_to_fist_ms <= self.max_time * 1000.0]
        return {
            "total": n,
            "yolo_detected": sum(1 for a in self.attempts if a.yolo_saw_dest),
            "semantic_accepted": sum(1 for a in self.attempts if a.accepted),
            "transitions": sum(1 for a in self.attempts if a.transition),
            "within_window": len(within),
            "rules_matched": sum(1 for a in self.attempts if a.rule_matched),
            "hand_gate_passed": sum(1 for a in self.attempts
                                    if a.hand_gate is True),
            "cooldown_passed": sum(1 for a in self.attempts
                                   if a.cooldown_ok is True),
            "dispatched": sum(1 for a in self.attempts if a.dispatched),
            "backend_invoked": sum(1 for a in self.attempts
                                   if a.backend_invoked),
            "backend_errors": sum(1 for a in self.attempts
                                  if a.backend_ok is False),
            "os_observed": sum(1 for a in self.attempts
                               if a.os_observed is True),
            "delivered": sum(1 for a in self.attempts if a.delivered),
            "successful": len(ok),
            "rate": (len(ok) / n) if n else 0.0,
            "failures": fails,
            "classes": classes,
            "backend": next((a.backend for a in self.attempts if a.backend),
                            "n/a"),
            "os_observation": ("available" if (self.watcher is not None
                                               and self.watcher.available)
                               else "UNAVAILABLE"),
            "median_one_to_fist_ms": (sorted(timings)[len(timings) // 2]
                                      if timings else None),
        }

    def render_summary(self) -> str:
        s = self.summary()
        pct = 100.0 * s["rate"]
        watch = s["os_observation"]
        lines = [
            "=" * 52,
            f"{s['total']} attempts",
            "",
            f"YOLO:                    {s['yolo_detected']}",
            f"semantic:                {s['semantic_accepted']}",
            f"transitions:             {s['transitions']}",
            f"transitions within {self.max_time:.1f}s:  "
            f"{s['within_window']}",
            f"rules matched:           {s['rules_matched']}",
            f"hand gate passed:        {s['hand_gate_passed']}",
            f"cooldown passed:         {s['cooldown_passed']}",
            f"{self.action} dispatched: {s['dispatched']}",
            f"mouse backend invoked:   {s['backend_invoked']}"
            f"   (backend={s['backend']})",
            f"backend exceptions:      {s['backend_errors']}",
            "",
            f"OS-level observation:    {watch}",
        ]
        if watch == "available":
            lines.append(f"OS clicks observed:      {s['os_observed']}")
        else:
            lines.append("OS delivery cannot be directly confirmed "
                         "by this test")
        lines += [
            "",
            f"Successful: {s['successful']}/{s['total']}",
            f"Success rate: {pct:.1f}%",
        ]
        if s["median_one_to_fist_ms"] is not None:
            lines.append(f"Median {self.origin}->{self.dest}: "
                         f"{s['median_one_to_fist_ms']:.0f} ms")
        lines += ["", "Missed clicks:"]
        for label, key in (("FSM", "FSM transition"),
                           ("Rule matching", "Rule matching"),
                           ("Hand gate", "Hand gate"),
                           ("Cooldown", "Cooldown"),
                           ("Dispatch", "Mouse dispatch"),
                           ("Mouse backend", "Mouse dispatch"),
                           ("Test instrumentation",
                            "Test instrumentation")):
            if label == "Dispatch":
                v = s["classes"]["G"]
            elif label == "Mouse backend":
                v = s["classes"]["H"]
            else:
                v = s["failures"][key]
            lines.append(f"  {label} = {v}")
        lines += ["", "by class:"]
        for k in sorted(CLASSES):
            if s["classes"][k]:
                lines.append(f"  {k}  {CLASSES[k]}: {s['classes'][k]}")
        lines += ["", ("PASS" if pct >= 95.0 else "NEEDS INVESTIGATION")]
        lines.append("=" * 52)
        return "\n".join(lines)

    def report(self) -> dict:
        return {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": ("Observational only. No mapping, config value or "
                     "threshold was modified by this run."),
            "rule_under_test": {
                "id": self.rule.get("id"),
                "from_state": self.origin,
                "to_state": self.dest,
                "action": self.action,
                "hand": self.hand,
                "cooldown_sec": self.cooldown,
                "max_time_sec": self.max_time,
            },
            "summary": self.summary(),
            "verdict": ("PASS" if self.summary()["rate"] >= 0.95
                        else "NEEDS INVESTIGATION"),
            "probe_rows_observed": self.rows_seen,
            "unexplained_failures": [a.as_dict() for a in self.attempts
                                     if a.unexplained],
            "mouse_pipeline": {
                "backend": self.summary()["backend"],
                "os_level_observation": self.summary()["os_observation"],
                "note": ("dispatch() returns None on every path, so "
                         "delivery is judged from click_count, and "
                         "confirmed independently only when a listener "
                         "is available"),
            },
            "class_legend": CLASSES,
            "attempts": [a.as_dict() for a in self.attempts],
        }


# ── Panel ──────────────────────────────────────────────────────────────
# The palette is duplicated rather than imported: app.py imports this
# module, so importing back would be circular.  A few colour literals is
# a cheaper price than a lazy import that breaks on load order.
BG = "#18181b"
CARD = "#27272a"
BORDER = "#3f3f46"
TEXT = "#f4f4f5"
FAINT = "#71717a"
ACCENT = "#10b981"
DANGER = "#f43f5e"
PAD = 14


class GestureTestPanel(tk.Toplevel):
    """The Test Gesture window.  Owns no configuration state."""

    def __init__(self, studio) -> None:
        super().__init__(studio)
        self.studio = studio
        self.session = None
        self._job = None
        self._rules = []

        self.title("Gesture Testing")
        self.configure(bg=BG)
        self.geometry("660x780")
        self.minsize(560, 620)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._build()
        self._load_rules()

    # ── construction ───────────────────────────────────────────────────
    def _build(self) -> None:
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=PAD, pady=(PAD, 0))
        tk.Label(head, text="GESTURE TESTING", bg=BG, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(head, text="Runs against the live recognition pipeline. "
                            "Reads your mappings; never edits them.",
                 bg=BG, fg=FAINT, font=("Segoe UI", 8),
                 wraplength=610, justify="left").pack(anchor="w")

        pick = tk.Frame(self, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        pick.pack(fill="x", padx=PAD, pady=(PAD, 0))
        tk.Label(pick, text="MAPPING UNDER TEST", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(
            anchor="w", padx=PAD, pady=(PAD, 2))
        self.rule_var = tk.StringVar()
        self.rule_box = ttk.Combobox(pick, textvariable=self.rule_var,
                                     state="readonly", width=52)
        self.rule_box.pack(anchor="w", padx=PAD, pady=(0, 8))

        row = tk.Frame(pick, bg=CARD)
        row.pack(fill="x", padx=PAD, pady=(0, PAD))
        tk.Label(row, text="ATTEMPTS", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        self.count_var = tk.StringVar(value="50")
        ttk.Spinbox(row, from_=1, to=500, increment=1, width=6,
                    textvariable=self.count_var).pack(side="left",
                                                      padx=(8, 0))
        self.start_button = tk.Label(row, text="  Start 50-Test Run  ",
                                     bg=ACCENT, fg="#052e16", cursor="hand2",
                                     font=("Segoe UI", 9, "bold"))
        self.start_button.pack(side="right")
        self.start_button.bind("<Button-1>", lambda _e: self._start())
        self.stop_button = tk.Label(row, text="  Stop  ", bg=CARD, fg=TEXT,
                                    cursor="hand2", font=("Segoe UI", 9))
        self.stop_button.pack(side="right", padx=(0, 8))
        self.stop_button.bind("<Button-1>", lambda _e: self._stop("stopped"))

        tk.Label(
            self,
            text="A successful attempt delivers a REAL mouse click to "
                 "whatever window has focus. That is deliberate — whether "
                 "the click was delivered is one of the measured stages — "
                 "so keep this window focused while testing.",
            bg=BG, fg=DANGER, font=("Segoe UI", 8), wraplength=610,
            justify="left").pack(anchor="w", padx=PAD, pady=(8, 0))

        prog = tk.Frame(self, bg=BG)
        prog.pack(fill="x", padx=PAD, pady=(8, 0))
        self.progress_label = tk.Label(prog, text="Idle.", bg=BG, fg=TEXT,
                                       font=("Segoe UI", 9, "bold"))
        self.progress_label.pack(side="left")
        self.bar = ttk.Progressbar(prog, mode="determinate", length=240)
        self.bar.pack(side="right")

        foot = tk.Frame(self, bg=BG)
        foot.pack(fill="x", side="bottom", padx=PAD, pady=(0, PAD))
        self.export_button = tk.Label(foot, text="  Export Test Report  ",
                                      bg=CARD, fg=FAINT, cursor="hand2",
                                      font=("Segoe UI", 9))
        self.export_button.pack(side="right")
        self.export_button.bind("<Button-1>", lambda _e: self._export())
        self.verdict = tk.Label(foot, text="", bg=BG, fg=TEXT,
                                font=("Segoe UI", 10, "bold"))
        self.verdict.pack(side="left")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=PAD, pady=PAD)
        self.log = tk.Text(body, bg=CARD, fg=TEXT, insertbackground=TEXT,
                           font=("Consolas", 9), wrap="none", height=18,
                           relief="flat", padx=10, pady=8)
        scroll = ttk.Scrollbar(body, orient="vertical",
                               command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(fill="y", side="right")
        self.log.pack(fill="both", expand=True, side="left")
        self.log.tag_configure("ok", foreground=ACCENT)
        self.log.tag_configure("bad", foreground=DANGER)
        self.log.tag_configure("dim", foreground=FAINT)

    def _load_rules(self) -> None:
        """Read the studio's rules.  Read-only: nothing is written back."""
        self._rules = [r for r in getattr(self.studio, "rules", [])
                       if r.get("trigger") == "transition"
                       and r.get("enabled", True)]
        labels = [f"{r.get('from_state')} -> {r.get('to_state')} = "
                  f"{r.get('action')}" for r in self._rules]
        self.rule_box.configure(values=labels)
        if labels:
            self.rule_box.current(0)
        else:
            self.rule_var.set("no transition mappings to test")

    # ── run control ────────────────────────────────────────────────────
    def _note(self, text, tag="dim"):
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")

    def _start(self) -> None:
        if self.session is not None and self._job is not None:
            return
        engine = getattr(self.studio, "engine", None)
        if engine is None or not engine.running:
            self._note("Camera is not connected. Connect it on the "
                       "Live Camera tab first.", "bad")
            return
        if not self._rules:
            self._note("There are no transition mappings to test.", "bad")
            return

        rule = self._rules[max(0, self.rule_box.current())]
        try:
            target = max(1, int(self.count_var.get()))
        except (TypeError, ValueError):
            target = 50

        self.log.delete("1.0", "end")
        self.verdict.configure(text="")
        self.session = TestSession(rule, target)

        # Independent OS-level observation.  If it cannot start, the run
        # still goes ahead and the report says the observation was
        # unavailable -- which is the honest answer, and not the same as
        # every click having failed.
        watcher = ClickWatcher()
        watcher.start()
        self.session.watcher = watcher
        engine.probe = hand_cursor_2.GestureProbe()

        self._note(f"Testing {self.session.origin} -> {self.session.dest}"
                   f" = {self.session.action}   ({target} attempts)")
        self._note(f"cooldown {self.session.cooldown:.2f}s | "
                   f"within {self.session.max_time:.2f}s | "
                   f"hand {self.session.hand}")
        backend = getattr(getattr(self.studio, "engine", None), "cursor",
                          None)
        self._note(f"backend={getattr(backend, 'name', '?')}   "
                   f"OS observation="
                   f"{'available' if watcher.available else 'UNAVAILABLE'}"
                   + (f" ({watcher.error})" if watcher.error else ""))
        self._note("Perform the gesture. Each completed attempt is scored"
                   " below.\n")
        self.bar.configure(maximum=target, value=0)
        self.start_button.configure(bg=BORDER, fg=FAINT)
        self._poll()

    def _poll(self) -> None:
        self._job = None
        engine = getattr(self.studio, "engine", None)
        session = self.session
        if session is None:
            return
        if engine is None or not engine.running or engine.probe is None:
            self._stop("camera stopped")
            return

        for att in session.feed(engine.probe.drain()):
            tag = "ok" if att.result == "SUCCESS" else "bad"
            self.log.insert("end", att.render(session.origin, session.dest),
                            tag)
            self.log.see("end")

        done = len(session.attempts)
        self.progress_label.configure(
            text=f"Testing: {done} / {session.target}")
        self.bar.configure(value=done)
        if session.done:
            self._stop("complete")
            return
        self._job = self.after(DRAIN_MS, self._poll)

    def _stop(self, why: str) -> None:
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        engine = getattr(self.studio, "engine", None)
        if engine is not None:
            engine.probe = None            # detach: pipeline back to normal
        if self.session is not None and self.session.watcher is not None:
            self.session.watcher.stop()
        self.start_button.configure(bg=ACCENT, fg="#052e16")
        session = self.session
        if session is None:
            return
        if session.attempts:
            self.log.insert("end", "\n" + session.render_summary() + "\n",
                            "dim")
            self.log.see("end")
            passed = session.summary()["rate"] >= 0.95
            self.verdict.configure(
                text=("PASS" if passed else "NEEDS INVESTIGATION"),
                fg=(ACCENT if passed else DANGER))
            self.export_button.configure(fg=TEXT)
        self.progress_label.configure(
            text=f"{why.capitalize()} - {len(session.attempts)} attempt(s).")

    # ── export ─────────────────────────────────────────────────────────
    def _export(self) -> None:
        session = self.session
        if session is None or not session.attempts:
            self._note("Nothing to export yet.", "bad")
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = filedialog.asksaveasfilename(
            parent=self, title="Export Test Report",
            defaultextension=".json",
            initialfile=f"gesture_test_{session.origin}_to_"
                        f"{session.dest}_{stamp}.json",
            filetypes=[("JSON report", "*.json"), ("Text report", "*.txt")])
        if not path:
            return
        # The config file is never a legal destination, whatever the user
        # types into the dialog.
        if os.path.abspath(path) == os.path.abspath(CONFIG_PATH):
            self._note("Refusing to write the report over "
                       "gesture_config.json.", "bad")
            return
        try:
            if path.lower().endswith(".txt"):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(f"Gesture test report - {session.origin} -> "
                             f"{session.dest} = {session.action}\n\n")
                    for att in session.attempts:
                        fh.write(att.render(session.origin, session.dest))
                    fh.write(session.render_summary() + "\n")
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(session.report(), fh, indent=2)
        except OSError as exc:
            self._note(f"Could not write the report: {exc}", "bad")
            return
        self._note(f"Report written to {path}", "ok")

    def _close(self) -> None:
        self._stop("closed")
        self.destroy()
