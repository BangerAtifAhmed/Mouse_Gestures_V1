"""Universal, config-driven gesture state machine.

Nothing in this module knows the name of a single gesture.  Every pose,
every transition and every action pairing arrives from gesture_config.json
(or an equivalent dict), so adding a gesture to the vocabulary is a change
to a JSON file and never a change to this file.  The engine's only
assumptions are that gestures are strings and that time moves forward.

The config it eats:

    {
      "settings": { ... optional tuning ... },
      "transitions": [
        {"from_state": "point", "to_state": "grip",
         "action": "LEFT_CLICK", "max_time_sec": 0.8}
      ],
      "holds": [
        {"pose": "timeout", "action": "SHOW_DESKTOP",
         "hold_sec": 0.4, "cooldown_sec": 2.0}
      ]
    }

Rules may also arrive grouped by which classifier feeds them, which is the
shape app.py writes.  "geometry_bindings", "yolo_bindings", "bindings" and
"mappings" are all read, and each entry is routed to the right rule class
by its own "trigger" key (or by carrying a "pose", which only a hold does):

    {"settings": {...},
     "geometry_bindings": [ {..., "trigger": "transition"}, ... ],
     "yolo_bindings":     [ {..., "trigger": "hold"}, ... ],
     "deleted_bindings":  [ ... ]}

"deleted_bindings" is the GUI's recycle bin and is deliberately NOT read:
a binned rule must never fire, so it is the one list this engine ignores.

Four failure modes drive the design, and each one is a separate mechanism
rather than a knob on a shared one:

  * A classifier that flickers for a frame must not break a chain, so raw
    labels pass through a sliding-window majority vote before the rules
    ever see them, and declared abstentions never enter the window at all.

  * A held pose must not fire a macro thirty times a second, so holds are
    edge-triggered — armed on entry, disarmed only by leaving the pose —
    with a cooldown as a second, independent guard.

  * Two deliberate single clicks must not merge into a double, and a
    deliberate double must not arrive as two singles, so click timing is
    discriminated explicitly rather than left to the OS.

  * A mouse button held down outlives the process that pressed it.  Any
    path that loses the hand, sees an unmapped pose, or simply runs out of
    patience emits the matching DRAG_STOP rather than assuming one is
    coming.
"""

from __future__ import annotations

import json
import os
from collections import Counter, deque

import adaptive_timing

__all__ = [
    # Action vocabulary
    "LEFT_CLICK", "RIGHT_CLICK", "MIDDLE_CLICK", "DOUBLE_CLICK",
    "DRAG_START", "DRAG_STOP", "SCROLL_UP", "SCROLL_DOWN",
    "SHOW_DESKTOP", "TASK_VIEW", "MINIMISE_ALL", "LOCK_SCREEN",
    "SWITCH_WINDOW", "CLOSE_WINDOW", "COPY", "PASTE", "SCREENSHOT",
    "VOLUME_UP", "VOLUME_DOWN", "MUTE", "MEDIA_PLAY_PAUSE",
    "KEYBOARD_MACRO",
    "ACTIONS", "MOUSE_ACTIONS", "MACRO_ACTIONS", "ACTION_MACROS",
    "NO_GESTURE",
    # Hand requirement
    "HAND_ANY", "HAND_RIGHT", "HAND_LEFT", "HAND_BOTH",
    "HANDS", "normalise_hand", "hands_can_coincide",
    # Engine
    "GestureStabilizer", "MajorityStabilizer",
    "TransitionRule", "HoldRule", "ActionEvent",
    "GestureFSM", "ActionExecutor",
    # Config plumbing
    "CONFIG_PATH", "DEFAULT_SETTINGS",
    "SENSITIVITY_MIN", "SENSITIVITY_MAX", "SENSITIVITY_STEP",
    "AI_CONFIDENCE_MIN", "AI_CONFIDENCE_MAX", "AI_CONFIDENCE_DEFAULT",
    "load_config", "save_config", "empty_config",
    "normalise_gesture",
]

# ─── Action vocabulary ──────────────────────────────────────────────────────
# Plain strings on purpose: they cross a JSON boundary in both directions
# and appear verbatim in the GUI, so an enum would only add a translation
# layer at every edge.

LEFT_CLICK = "LEFT_CLICK"
RIGHT_CLICK = "RIGHT_CLICK"
MIDDLE_CLICK = "MIDDLE_CLICK"
DOUBLE_CLICK = "DOUBLE_CLICK"
DRAG_START = "DRAG_START"
DRAG_STOP = "DRAG_STOP"
SCROLL_UP = "SCROLL_UP"
SCROLL_DOWN = "SCROLL_DOWN"

SHOW_DESKTOP = "SHOW_DESKTOP"
TASK_VIEW = "TASK_VIEW"
MINIMISE_ALL = "MINIMISE_ALL"
LOCK_SCREEN = "LOCK_SCREEN"
SWITCH_WINDOW = "SWITCH_WINDOW"
CLOSE_WINDOW = "CLOSE_WINDOW"
COPY = "COPY"
PASTE = "PASTE"
SCREENSHOT = "SCREENSHOT"
VOLUME_UP = "VOLUME_UP"
VOLUME_DOWN = "VOLUME_DOWN"
MUTE = "MUTE"
MEDIA_PLAY_PAUSE = "MEDIA_PLAY_PAUSE"

KEYBOARD_MACRO = "KEYBOARD_MACRO"

# Mouse actions are the ones the cursor backend performs directly.
MOUSE_ACTIONS = (
    LEFT_CLICK, RIGHT_CLICK, MIDDLE_CLICK, DOUBLE_CLICK,
    DRAG_START, DRAG_STOP, SCROLL_UP, SCROLL_DOWN,
)

# Named macros expand to a chord.  Keeping the expansion here rather than
# in the executor means the GUI can show a user what a name will actually
# press without importing pynput.
ACTION_MACROS = {
    SHOW_DESKTOP: "win+d",
    TASK_VIEW: "win+tab",
    MINIMISE_ALL: "win+m",
    LOCK_SCREEN: "win+l",
    SWITCH_WINDOW: "alt+tab",
    CLOSE_WINDOW: "alt+f4",
    COPY: "ctrl+c",
    PASTE: "ctrl+v",
    SCREENSHOT: "win+shift+s",
    VOLUME_UP: "volume_up",
    VOLUME_DOWN: "volume_down",
    MUTE: "volume_mute",
    MEDIA_PLAY_PAUSE: "media_play_pause",
}

MACRO_ACTIONS = tuple(ACTION_MACROS) + (KEYBOARD_MACRO,)

ACTIONS = MOUSE_ACTIONS + MACRO_ACTIONS

# The sentinel for "no gesture was supplied".  It is a real state as far as
# the engine is concerned — losing the hand is exactly the event that has
# to release a stuck drag — so it is not in the ignored list.
NO_GESTURE = "none"

# ─── Hand requirement ───────────────────────────────────────────────────────
# Which hand a rule demands.  Stored on the rule and checked at match time,
# deliberately NOT folded into the pose name: a pose is a shape, and which
# hand made it is a separate fact about the same frame.  Encoding it in the
# label instead ("three_gun_right") doubles the vocabulary and makes a rule
# that genuinely does not care impossible to express.
HAND_ANY = "any"
HAND_RIGHT = "right"
HAND_LEFT = "left"
HAND_BOTH = "both"

HANDS = (HAND_ANY, HAND_RIGHT, HAND_LEFT, HAND_BOTH)


def normalise_hand(value) -> str:
    """Any spelling of a hand requirement -> one of HANDS.  Never raises.

    Unrecognised input becomes HAND_ANY rather than an error, because this
    reads a config file a human may have edited by hand: a typo should cost
    the constraint, not the whole rule.
    """
    text = str(value or "").strip().lower().replace(" ", "_")
    if text in ("r", "right", "right_hand", "righthand"):
        return HAND_RIGHT
    if text in ("l", "left", "left_hand", "lefthand"):
        return HAND_LEFT
    if text in ("both", "both_hands", "bothhands", "two", "two_hands", "2"):
        return HAND_BOTH
    return HAND_ANY


def hands_can_coincide(a, b) -> bool:
    """True when two hand requirements can BOTH be met on the same frame.

    This is the duplicate test.  Two rules on the same trigger are only a
    conflict if there is a frame that satisfies both of them — otherwise
    they are alternatives and may coexist, which is the whole point of
    per-hand mappings:

        right vs left   -> False.  A hand is one or the other, so the two
                           rules are mutually exclusive by construction.
        any   vs x      -> True.   ANY matches whatever x matches.
        both  vs right  -> True.   A two-hand frame still has a primary
                           side, so a "both hands" rule and a "right hand"
                           rule can fire on the very same frame.
        x     vs x      -> True.
    """
    left, right = normalise_hand(a), normalise_hand(b)
    if left == right:
        return True
    if HAND_ANY in (left, right):
        return True
    if {left, right} == {HAND_RIGHT, HAND_LEFT}:
        return False
    return True          # BOTH paired with a specific side


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gesture_config.json")

DEFAULT_SETTINGS = {
    # Sliding-window majority vote.  3/2 is 100 ms at 30 Hz and absorbs any
    # single-frame misclassification; widen both together for a noisier
    # classifier at the cost of latency on every rule.
    "window_size": 3,
    "stability_threshold": 2,

    # Labels that mean "the classifier is abstaining", not "the hand is in
    # this pose".  They never enter the vote and never become a state, so a
    # half-curled hand between two poses cannot break a transition chain.
    # Anything listed here is invisible to the rules.
    "ignored_states": ["idle", ""],

    # Two clicks closer together than this are one double-click; anything
    # further apart is two singles.  0.8 s because a pose-cycle double is
    # four stabilised poses and measures 0.20–0.67 s end to end.
    "double_click_sec": 0.8,

    # How far back a transition may look for its origin pose.  With this at
    # 2 a chain tolerates exactly one unmapped pose in the middle, which is
    # what a real hand produces when curling; raising it trades false
    # negatives for false positives.
    "transition_memory": 2,

    # Failsafes.  A drag with no hand to steer it is released after this
    # long, and no drag survives drag_timeout_sec whatever the hand does.
    "stuck_release_sec": 0.5,
    "drag_timeout_sec": 30.0,

    # Applied to any rule that does not name its own.
    "default_cooldown_sec": 0.35,
    "default_hold_sec": 0.4,
    "default_max_time_sec": 0.8,

    # ── Cursor settings: read by hand_cursor_2.py, ignored by this engine ──
    # They live here so one file is the whole configuration and the GUI has
    # somewhere to put them; the FSM simply carries them through.
    "cursor_sensitivity": 1.4,

    # These two are the mirror pair, and only ONE may be on at a time.
    # Mirroring the preview already puts the landmarks in display space, so
    # reflecting the control maths as well cancels out and the cursor runs
    # backwards under a picture that looks correct.
    "is_mirrored": True,
    "invert_cursor_x": False,

    # Shared detection floor for MediaPipe and YOLO.  Worth tuning per
    # gesture set, which is why the GUI puts it on a slider.
    "ai_confidence": 0.6,
}

# Bounds the GUI enforces and hand_cursor clamps to.
SENSITIVITY_MIN = 1.0
SENSITIVITY_MAX = 3.0
SENSITIVITY_STEP = 0.1

AI_CONFIDENCE_MIN = 0.1
AI_CONFIDENCE_MAX = 0.9
AI_CONFIDENCE_DEFAULT = 0.6


def normalise_gesture(raw_gesture) -> str:
    """Fold a classifier label into the form the rules are matched against."""
    if raw_gesture is None:
        return NO_GESTURE
    label = str(raw_gesture).strip().lower()
    return label if label else NO_GESTURE


def _as_float(value, fallback: float) -> float:
    """Coerce to float, never raising — not even on a bad fallback.

    The fallback often comes from the config too ("default_cooldown_sec"),
    so `float(fallback)` was itself a crash path when a hand-edited file
    put a string there.  0.0 is the floor of last resort.
    """
    try:
        out = float(value)
        if out == out:                                 # reject NaN
            return out
    except (TypeError, ValueError):
        pass
    try:
        out = float(fallback)
        return out if out == out else 0.0
    except (TypeError, ValueError):
        return 0.0


def _as_int(value, fallback: int, minimum: int = 1) -> int:
    """Coerce to an int at or above `minimum`, never raising."""
    for candidate in (value, fallback):
        try:
            return max(minimum, int(float(candidate)))
        except (TypeError, ValueError):
            continue
    return minimum


def _as_list(value) -> list:
    """Anything iterable becomes a list; anything else becomes empty.

    Strings are deliberately NOT exploded into characters — a config that
    says `"ignored_states": "idle"` means one label, not four.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return []


# ─── Stabiliser ─────────────────────────────────────────────────────────────

class GestureStabilizer:
    """Sliding window over raw labels.  Subclasses decide what 'stable' means.

    The last stable verdict is retained when the window is inconclusive, so
    a moment of genuine ambiguity holds the previous state rather than
    dropping to None and re-arming every rule that was waiting on it.
    """

    def __init__(self, window_size: int = 3, threshold: int = 2) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        self._window_size = int(window_size)
        self._threshold = min(int(threshold), self._window_size)
        self._window = deque(maxlen=self._window_size)
        self._stable = None

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def window(self) -> tuple:
        return tuple(self._window)

    @property
    def stable(self):
        return self._stable

    def push(self, label: str):
        self._window.append(label)
        candidate = self._resolve()
        if candidate is not None:
            self._stable = candidate
        return self._stable

    def reset(self) -> None:
        self._window.clear()
        self._stable = None

    def _resolve(self):
        raise NotImplementedError


class MajorityStabilizer(GestureStabilizer):
    """Majority vote.  A label must win `threshold` of `window_size` frames."""

    def _resolve(self):
        if not self._window:
            return None
        label, count = Counter(self._window).most_common(1)[0]
        return label if count >= self._threshold else None


# ─── Rules ──────────────────────────────────────────────────────────────────

class _Rule:
    """Fields shared by both trigger kinds."""

    __slots__ = ("id", "name", "enabled", "action", "keys",
                 "cooldown_sec", "source", "hand")

    def __init__(self, raw: dict, index: int, settings: dict) -> None:
        self.id = str(raw.get("id") or f"rule-{index}")
        self.enabled = bool(raw.get("enabled", True))
        self.action = str(raw.get("action", "")).strip().upper()
        if not self.action:
            raise ValueError("rule has no action")

        # Free-text chord, used when action is KEYBOARD_MACRO and ignored
        # otherwise.  Named actions carry their own chord in ACTION_MACROS.
        self.keys = str(raw.get("keys", "")).strip()
        if self.action == KEYBOARD_MACRO and not self.keys:
            raise ValueError("KEYBOARD_MACRO rule has no keys")

        self.cooldown_sec = max(0.0, _as_float(
            raw.get("cooldown_sec"), settings["default_cooldown_sec"]))

        # Optional: which classifier a rule listens to.  Left at "any" the
        # engine is fully source-agnostic, which is the documented default;
        # a project running two classifiers at different rates can tag its
        # rules and keep the two vocabularies from colliding.
        self.source = str(raw.get("source", "any")).strip().lower() or "any"

        # Which hand must perform this gesture.  HAND_ANY is the default and
        # the meaning of an absent key, so every rule written before this
        # field existed keeps working untouched.
        self.hand = normalise_hand(raw.get("hand"))

        # Filled in by _finalise(), which subclasses call once their own
        # fields exist — _describe() reads them.
        self.name = str(raw.get("name") or "")

    def _finalise(self) -> None:
        if not self.name:
            self.name = self._describe()

    def _describe(self) -> str:
        return self.action

    def accepts_source(self, source) -> bool:
        return self.source == "any" or source is None or source == self.source

    def trigger_key(self):
        """What this rule listens for, IGNORING its hand requirement.

        Two rules sharing a trigger_key are the same physical event seen
        through different hand filters.  At most one of them may fire on
        any given frame — see GestureFSM._one_per_trigger().
        """
        return (self.__class__.__name__, self._describe())

    def accepts_hand(self, hand, hand_count=0) -> bool:
        """True when the hand on screen satisfies this rule's requirement.

        `hand` is the side of the hand that produced the pose ("left" /
        "right" / "" when unknown); `hand_count` is how many hands the
        tracker can currently see.

        Two deliberate choices about missing information:

          * An UNKNOWN side satisfies a left/right rule.  Handedness comes
            from MediaPipe and is occasionally blank — refusing the rule
            then would make a bound gesture fail silently and look broken,
            which is worse than firing on the wrong hand once.

          * HAND_BOTH is a count test, not a side test.  It asks for two
            hands in frame, which is the only part of "both hands" the
            tracker can actually verify; the pose itself is still read from
            the primary hand.
        """
        requirement = self.hand
        if requirement == HAND_ANY:
            return True
        if requirement == HAND_BOTH:
            return int(hand_count or 0) >= 2
        side = str(hand or "").strip().lower()
        if not side:
            return True
        return side == requirement

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "action": self.action,
            "cooldown_sec": round(self.cooldown_sec, 3),
        }
        # Written only when it constrains something, so a rule that does not
        # care about the hand stays as short in the file as it always was.
        if self.hand != HAND_ANY:
            out["hand"] = self.hand
        if self.action == KEYBOARD_MACRO:
            out["keys"] = self.keys
        if self.source != "any":
            out["source"] = self.source
        return out


class TransitionRule(_Rule):
    """Fires when the stable pose becomes `to_state` having been `from_state`.

    `max_time_sec` bounds the gap between the two, which is what separates
    "the user curled point into grip" from "the user was pointing a minute
    ago and has now closed their hand for an unrelated reason".
    """

    __slots__ = ("from_state", "to_state", "max_time_sec", "promote_double",
                 "adaptive_timing")

    def __init__(self, raw: dict, index: int, settings: dict) -> None:
        super().__init__(raw, index, settings)
        self.from_state = normalise_gesture(raw.get("from_state"))
        self.to_state = normalise_gesture(raw.get("to_state"))
        if self.from_state == NO_GESTURE or self.to_state == NO_GESTURE:
            raise ValueError("transition needs both from_state and to_state")
        if self.from_state == self.to_state:
            raise ValueError("transition endpoints must differ")

        self.max_time_sec = max(0.0, _as_float(
            raw.get("max_time_sec"), settings["default_max_time_sec"]))

        # Opt-in per rule, and deliberately a separate key rather than a
        # sentinel value of max_time_sec: zero already means "no limit
        # at all", so overloading it would make "let the engine decide"
        # indistinguishable from "never expire".  A rule that carries an
        # explicit max_time_sec and no marker keeps that number exactly,
        # which is what stops an existing config being converted behind
        # the user's back.
        self.adaptive_timing = bool(raw.get("adaptive_timing", False))

        # Opt-in: repeat the same trigger inside the double-click window and
        # get one DOUBLE_CLICK instead of two LEFT_CLICKs.  Off by default,
        # because a user who wants a double-click has a gesture for it.
        self.promote_double = bool(raw.get("promote_double", False))
        self._finalise()

    def _describe(self) -> str:
        return f"{self.from_state} → {self.to_state}"

    def to_dict(self) -> dict:
        out = super().to_dict()
        out.update({
            "trigger": "transition",
            "from_state": self.from_state,
            "to_state": self.to_state,
            "max_time_sec": round(self.max_time_sec, 3),
        })
        if self.promote_double:
            out["promote_double"] = True
        return out


class HoldRule(_Rule):
    """Fires once when `pose` has been stable for `hold_sec`.

    Edge-triggered: the rule arms on entering the pose, fires when the hold
    matures, and cannot fire again until the pose has been something else.
    `repeat` opts into level-triggering for the handful of actions that are
    useless without it — scrolling, volume — and stays off for everything
    else so a held hand cannot spam a macro.
    """

    __slots__ = ("pose", "hold_sec", "repeat", "repeat_sec")

    def __init__(self, raw: dict, index: int, settings: dict) -> None:
        super().__init__(raw, index, settings)
        self.pose = normalise_gesture(raw.get("pose") or raw.get("state"))
        if self.pose == NO_GESTURE:
            raise ValueError("hold needs a pose")

        # hold_ms is accepted because it is the natural unit in a GUI.
        if raw.get("hold_ms") is not None:
            self.hold_sec = max(
                0.0, _as_float(raw.get("hold_ms"), 0.0) / 1000.0)
        else:
            self.hold_sec = max(0.0, _as_float(
                raw.get("hold_sec"), settings["default_hold_sec"]))

        self.repeat = bool(raw.get("repeat", False))
        self.repeat_sec = max(0.05, _as_float(raw.get("repeat_sec"), 0.25))
        self._finalise()

    def _describe(self) -> str:
        return f"hold {self.pose}"

    def to_dict(self) -> dict:
        out = super().to_dict()
        out.update({
            "trigger": "hold",
            "pose": self.pose,
            "hold_sec": round(self.hold_sec, 3),
        })
        if self.repeat:
            out["repeat"] = True
            out["repeat_sec"] = round(self.repeat_sec, 3)
        return out


# ─── Emitted events ─────────────────────────────────────────────────────────

class ActionEvent(str):
    """The action string, with the rule that produced it attached.

    Subclassing str rather than wrapping it keeps every caller that only
    wants the name working unchanged — `event == LEFT_CLICK`, `event in
    MOUSE_ACTIONS` and f-string interpolation all behave — while an
    executor that needs the chord can read `.keys` off the same object.
    """

    __slots__ = ("rule_id", "rule_name", "keys", "at")

    def __new__(cls, action, rule_id=None, rule_name=None, keys="", at=0.0):
        self = super().__new__(cls, action)
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.keys = keys or ACTION_MACROS.get(str(action), "")
        self.at = float(at)
        return self

    @property
    def action(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"<ActionEvent {str(self)} from {self.rule_name!r}>"


# ─── Config I/O ─────────────────────────────────────────────────────────────

def empty_config() -> dict:
    """The one and only default state: three settings and three empty lists.

    THERE IS NO OTHER CONFIG FACTORY IN THIS CODEBASE.  There used to be
    two — a fallback that injected point -> grip on a missing file, and a
    default_config() behind the GUI's Load Defaults button that installed
    five rules — and both have been deleted.  Nothing in this program can
    produce a binding any more; only the user can, through the GUI or by
    editing the JSON.

    The shape is exact and deliberately minimal.  The FSM's tuning knobs
    are NOT written here: they have working values in DEFAULT_SETTINGS and
    are merged in at construction, so a fresh file stays readable instead
    of opening with a dozen numbers nobody asked about.
    """
    return {
        "settings": {
            "cursor_sensitivity": 1.4,
            # Mirroring ON, inversion OFF — exactly one of the two.  They
            # are both reflections of the same axis, so turning on both
            # (or neither) cancels out and the cursor runs backwards under
            # a preview that looks correct.
            "is_mirrored": True,
            "invert_cursor_x": False,
            "ai_confidence": 0.6,
        },
        "geometry_bindings": [],
        "yolo_bindings": [],
        "deleted_bindings": [],
    }


def load_config(path: str = CONFIG_PATH, *, quiet: bool = False) -> dict:
    """Read a config, never raising.

    Every failure downgrades to an EMPTY config with a console warning: a
    gesture controller that refuses to start because a JSON file has a
    trailing comma is worse than one that starts with nothing bound, and
    nothing bound is exactly what the file said.
    """
    if not os.path.exists(path):
        if not quiet:
            print(f"[config] {os.path.basename(path)} not found — "
                  f"starting with no bindings")
        return empty_config()

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        if not quiet:
            print(f"[config] {os.path.basename(path)} unreadable ({exc}) — "
                  f"starting with no bindings")
        return empty_config()

    if not isinstance(data, dict):
        if not quiet:
            print(f"[config] {os.path.basename(path)} is not an object — "
                  f"starting with no bindings")
        return empty_config()

    # Every list the callers index into is guaranteed present, so a config
    # written by an older version — or hand-edited down to just "settings"
    # — still answers .get()/[] for all of them rather than raising.
    data.setdefault("settings", {})
    for key in ("transitions", "holds", "geometry_bindings",
                "yolo_bindings", "deleted_bindings"):
        value = data.get(key)
        if not isinstance(value, list):
            data[key] = []
    return data


def save_config(config: dict, path: str = CONFIG_PATH) -> None:
    """Write atomically, so a crash mid-write cannot destroy the config."""
    payload = json.dumps(config, indent=2, ensure_ascii=False)
    temp = f"{path}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


# ─── The state machine ──────────────────────────────────────────────────────

class GestureFSM:
    """Turns a stream of gesture labels into a stream of action strings.

    Feed it every frame:

        action = fsm.update(label, time.perf_counter())
        if action is not None:
            executor.dispatch(action)

    `update` returns at most one action per call.  When two rules mature on
    the same frame the second is queued and returned on the next call — at
    30 Hz that is a 33 ms deferral, and it keeps the contract to one
    action per call rather than making every caller handle a list.
    """

    def __init__(self, config=None, stabilizer=None, learner=None,
                 **overrides) -> None:
        if config is None:
            config = load_config()
        elif isinstance(config, str):
            config = load_config(config)
        elif not isinstance(config, dict):
            raise TypeError("config must be a path, a dict, or None")

        # Every value below is coerced rather than trusted.  A hand-edited
        # config can put a string where a number belongs, or a number where
        # a list belongs, and none of that may stop the tracker starting.
        settings = dict(DEFAULT_SETTINGS)
        try:
            settings.update(config.get("settings") or {})
        except (TypeError, ValueError):
            print("[config] 'settings' is not an object — using defaults")
        settings.update(overrides)
        self._settings = settings

        self._ignored = frozenset(
            normalise_gesture(s)
            for s in _as_list(settings.get("ignored_states"))
            if str(s).strip() != ""
        ) | {""}

        self._double_click_sec = max(0.0, _as_float(
            settings.get("double_click_sec"), 0.8))
        self._stuck_release_sec = max(0.0, _as_float(
            settings.get("stuck_release_sec"), 0.5))
        self._drag_timeout_sec = max(0.0, _as_float(
            settings.get("drag_timeout_sec"), 30.0))

        window = _as_int(settings.get("window_size"), 3)
        threshold = _as_int(settings.get("stability_threshold"), 2)
        self._stabilizer = stabilizer or MajorityStabilizer(
            window, min(threshold, window))

        # The frame context the rules are matched against.  Initialised
        # here so any entry point into _tick() finds them defined, even one
        # that never calls update().
        self._hand = None
        self._hand_count = 0

        # Never raises: a fault here would leave the engine half-built.
        self._fault_reported = False

        raw_count = sum(len(_as_list(config.get(key))) for key in
                        ("transitions", "holds", "mappings",
                         "geometry_bindings", "yolo_bindings"))
        self._transitions, self._holds = self._compile(config, settings)

        # An empty rule set is a supported state, not an error to paper
        # over: the tracker still moves the cursor, it just has nothing
        # bound.  Nothing is injected here — the two cases are only told
        # apart so the message can say which one happened, because "I wrote
        # rules and none loaded" needs a different fix from "I wrote none".
        if not self._transitions and not self._holds:
            if raw_count:
                print(f"[config] all {raw_count} rule(s) in the config were "
                      f"unusable — see the reasons above. No gesture is "
                      f"bound; the cursor will still move.")
            else:
                print("[config] no bindings configured — the cursor will "
                      "move but no gesture will act. Add mappings in "
                      "app.py, then press Save Config.")

        # A rule cannot promote a double it is not allowed to fire.  The
        # two guards are independently sensible and silently incompatible,
        # so the conflict is reported rather than resolved behind the
        # user's back — lowering the cooldown here would weaken the jitter
        # protection they asked for.
        for rule in self._transitions:
            if rule.promote_double and \
                    rule.cooldown_sec > self._double_click_sec * 0.25:
                print(f"[config] '{rule.name}': a {rule.cooldown_sec:.2f}s "
                      f"cooldown blocks the second click of a double "
                      f"(window {self._double_click_sec:.2f}s). Lower it to "
                      f"~{self._double_click_sec * 0.2:.2f}s or turn off "
                      f"promote_double.")

        # Every pose any rule mentions.  A stable pose outside this set is
        # "unmapped", which is one of the conditions that releases a drag.
        self._known = {r.from_state for r in self._transitions}
        self._known |= {r.to_state for r in self._transitions}
        self._known |= {r.pose for r in self._holds}

        memory = _as_int(settings.get("transition_memory"), 2,
                         minimum=1) + 1
        self._recent = deque(maxlen=memory)

        # One learner per FSM.  The geometric and semantic machines run
        # at rates 6x apart, so pooling their observations for the same
        # pose pair would average two unrelated distributions.  Separate
        # instances also make pose-pair isolation structural rather than
        # something the estimator has to promise.
        #
        # Injected when the caller has one to share -- the engine keeps
        # its learners across config reloads, so editing an unrelated
        # mapping does not throw away what has been learnt.  A private
        # one is built otherwise, which keeps this class usable on its
        # own in a test.
        self._timing = learner or adaptive_timing.TimingLearner()

        self._current = None
        self._previous = None
        self._entered_at = 0.0
        self._queue = deque()

        self._last_fired = {}          # rule id -> time
        self._hold_armed = {}          # rule id -> time it fired
        self._hold_next = {}           # rule id -> next repeat due
        self._pending_click = {}       # rule id -> time of unmatched click

        self._dragging = False
        self._drag_rule = None
        self._drag_started_at = 0.0
        self._drag_lost_since = None

        self.click_count = 0
        self.double_click_count = 0
        self.drag_start_count = 0
        self.drag_stop_count = 0
        self.macro_count = 0

    # ── construction helpers ────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: str = CONFIG_PATH, **kwargs) -> "GestureFSM":
        return cls(load_config(path), **kwargs)

    @staticmethod
    def _compile(config: dict, settings: dict):
        """Build rule objects, dropping the bad ones rather than the file.

        One malformed rule in a hand-edited config should cost that rule
        and nothing else — the user still gets the other fourteen bindings
        they wrote, plus a line telling them which one to fix.
        """
        transitions, holds = [], []

        raw_transitions = _as_list(config.get("transitions"))
        raw_holds = _as_list(config.get("holds"))

        # Accepted aliases.  A hand-written config that groups rules by
        # which classifier feeds them reads naturally, and refusing it would
        # be a silent empty-config start rather than an error anyone sees.
        # "deleted_bindings" is deliberately NOT among these: the GUI's
        # recycle bin lives in the same file and must never compile.
        for key in ("geometry_bindings", "yolo_bindings", "bindings"):
            for entry in _as_list(config.get(key)):
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("trigger", "")).lower() == "hold" \
                        or entry.get("pose"):
                    raw_holds.append(entry)
                else:
                    raw_transitions.append(entry)

        # A single "mappings" list with a "trigger" discriminator is also
        # accepted, because that is the shape a GUI naturally produces.
        for entry in _as_list(config.get("mappings")):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("trigger", "")).lower() == "hold":
                raw_holds = list(raw_holds) + [entry]
            else:
                raw_transitions = list(raw_transitions) + [entry]

        for index, raw in enumerate(raw_transitions):
            if not isinstance(raw, dict):
                continue
            try:
                rule = TransitionRule(raw, index, settings)
            except Exception as exc:
                print(f"[config] skipping transition #{index}: "
                      f"{exc.__class__.__name__}: {exc}")
                continue
            if rule.enabled:
                transitions.append(rule)

        for index, raw in enumerate(raw_holds):
            if not isinstance(raw, dict):
                continue
            try:
                rule = HoldRule(raw, index, settings)
            except Exception as exc:
                print(f"[config] skipping hold #{index}: "
                      f"{exc.__class__.__name__}: {exc}")
                continue
            if rule.enabled:
                holds.append(rule)

        return transitions, holds

    # ── introspection ───────────────────────────────────────────────────

    @property
    def stabilizer(self):
        return self._stabilizer

    @property
    def stable_gesture(self):
        """What the engine currently believes, after the majority vote."""
        return self._current

    @property
    def previous_gesture(self):
        return self._previous

    @property
    def is_dragging(self) -> bool:
        return self._dragging

    @property
    def rules(self) -> tuple:
        return tuple(self._transitions) + tuple(self._holds)

    def describe(self) -> str:
        return (f"{len(self._transitions)} transition(s), "
                f"{len(self._holds)} hold(s), "
                f"{self._stabilizer.threshold}/{self._stabilizer.window_size} "
                f"window")

    # ── the loop entry point ────────────────────────────────────────────

    def update(self, gesture, timestamp: float, source=None,
               hand=None, hand_count=0):
        """One frame.  Returns an action string, or None.

        The return is an ActionEvent, which *is* a str — compare it to the
        action constants directly.  `.keys` carries the chord for macros.

        `hand` and `hand_count` describe the frame, not the pose: which side
        produced it and how many hands are visible.  Both are optional, so
        a caller that does not track handedness behaves exactly as before.
        """
        self._hand = hand
        self._hand_count = hand_count
        try:
            events = self._advance(gesture, _as_float(timestamp, 0.0), source)
        except Exception as exc:                      # pragma: no cover
            events = self._recover(exc, timestamp)
        if events:
            self._queue.extend(events)
        return self._queue.popleft() if self._queue else None

    def update_pair(self, gesture, timestamp: float, source=None,
                    hand=None, hand_count=0):
        """`(stable_gesture, action)` for callers that display both."""
        action = self.update(gesture, timestamp, source, hand, hand_count)
        return self._current, action

    def poll(self, gesture, timestamp: float, source=None,
             hand=None, hand_count=0) -> list:
        """Every action this frame produced, in order.  Drains the queue."""
        self._hand = hand
        self._hand_count = hand_count
        try:
            events = self._advance(gesture, _as_float(timestamp, 0.0), source)
        except Exception as exc:                      # pragma: no cover
            events = self._recover(exc, timestamp)
        if self._queue:
            events = list(self._queue) + events
            self._queue.clear()
        return events

    def _recover(self, exc, timestamp) -> list:
        """Fall back to IDLE after an internal fault, without stranding a drag.

        A custom gesture or a half-written rule should never take the
        tracking thread down, so every path into the rule engine funnels
        through here.  The state machine is reset — which is the IDLE
        state — and the reset's own cleanup is RETURNED rather than
        dropped: if a drag was in progress, the caller still gets its
        DRAG_STOP and the mouse button is not left held.

        Reported once.  The failure mode is per-frame, so printing every
        time would bury the first occurrence under thousands of copies.
        """
        if not self._fault_reported:
            self._fault_reported = True
            print(f"[fsm] internal fault, falling back to idle "
                  f"({exc.__class__.__name__}: {exc})")

        when = _as_float(timestamp, 0.0)
        cleanup = []

        # Release FIRST, using the least machinery that can produce the
        # event.  reset() touches the stabiliser and several collections,
        # any of which the fault may have broken — and if reset() then
        # raises, a DRAG_STOP built inside it is lost and the mouse button
        # stays physically held.  Emitting it here means the caller gets it
        # whatever happens next.
        try:
            if self._dragging:
                cleanup.append(self._emit_drag_stop(self._drag_rule, when))
        except Exception:
            self._dragging = False
            self._drag_rule = None

        try:
            cleanup.extend(self.reset(when))
        except Exception:
            self._hard_reset()
        return cleanup

    def _hard_reset(self) -> None:
        """Last-resort state clear that calls into nothing that can fail."""
        self._current = None
        self._previous = None
        self._entered_at = 0.0
        self._dragging = False
        self._drag_rule = None
        self._drag_lost_since = None
        self._recent = deque(maxlen=3)
        self._queue = deque()
        self._last_fired = {}
        self._hold_armed = {}
        self._hold_next = {}
        self._pending_click = {}
        try:
            self._stabilizer.reset()
        except Exception:
            self._stabilizer = MajorityStabilizer(3, 2)

    def reset(self, timestamp: float = 0.0) -> list:
        """Forget everything, releasing anything still held.

        Returns the cleanup actions.  Callers that maintain their own
        button state may ignore the return value; callers that do not must
        dispatch it, or a drag interrupted by a lost hand stays pressed.
        """
        cleanup = []
        if self._dragging:
            cleanup.append(self._emit_drag_stop(self._drag_rule,
                                                float(timestamp)))
        cleanup.extend(self._queue)

        self._stabilizer.reset()
        self._recent.clear()
        self._queue.clear()
        self._current = None
        self._previous = None
        self._entered_at = 0.0
        self._hold_armed.clear()
        self._hold_next.clear()
        self._pending_click.clear()
        self._drag_lost_since = None
        return cleanup

    # ── engine ──────────────────────────────────────────────────────────

    def _advance(self, gesture, now: float, source) -> list:
        label = normalise_gesture(gesture)
        events = []

        if label in self._ignored:
            # An abstention is not a pose.  It must not enter the vote,
            # because a stabilised "idle" between point and grip would
            # become the predecessor of grip and silently break the chain
            # every transition depends on.
            pass
        else:
            stable = self._stabilizer.push(label)
            if stable is not None and stable != self._current:
                self._previous = self._current
                self._current = stable
                self._entered_at = now
                self._recent.append((stable, now))
                events.extend(self._on_state_change(now, source))

        events.extend(self._tick(now, source))
        return events

    def _on_state_change(self, now: float, source) -> list:
        # Safety before features: a drag whose pose has ended is released
        # before any new rule gets a chance to fire on the same frame.
        events = self._drag_safety(now)

        candidates = [rule for rule in self._transitions
                      if rule.to_state == self._current
                      and rule.accepts_source(source)
                      and rule.accepts_hand(self._hand, self._hand_count)]
        matched = self._match_origin(candidates, now)

        if matched:
            for rule in matched:
                events.extend(self._fire(rule, now))
            # Consume the history the match was drawn from, so one arrival
            # at a target cannot be claimed twice by a later frame.
            self._recent.clear()
            self._recent.append((self._current, now))

        return events

    @staticmethod
    def _one_per_trigger(rules: list) -> list:
        """Keep the first rule for each trigger, drop later duplicates.

        This is what makes an UNKNOWN handedness safe.  A blank side
        satisfies both a left rule and a right rule — deliberately, so a
        momentary gap in MediaPipe's handedness does not silently break a
        binding — but without this the two would both fire and one hand
        movement would produce two actions.

        Config order decides the winner, so the choice is deterministic
        rather than dictionary order.  When the side IS known only one of
        the pair passes accepts_hand() in the first place and this is a
        no-op.
        """
        if len(rules) < 2:
            return rules
        seen, kept = set(), []
        for rule in rules:
            key = rule.trigger_key()
            if key in seen:
                continue
            seen.add(key)
            kept.append(rule)
        return kept

    def _match_origin(self, candidates: list, now: float) -> list:
        """Pick the rules whose origin is the pose the hand actually came from.

        Walking back through recent poses tolerates an intermediate hop,
        which is what lets a chain survive the brief stable pose a real
        hand makes while curling.  Left unbounded, though, that tolerance
        is actively wrong: with point→grip bound to click and open→grip
        bound to drag, the sequence open→point→grip satisfies *both*, and
        one deliberate click also presses and holds the mouse button.

        So the search stops at the first pose that is an origin for
        anything targeting this state.  The nearest origin is the one the
        user just left, and only rules starting there are eligible — a
        farther one lost the race and does not get a second look, even if
        the nearer rule turns out to be cooling down.
        """
        if not candidates:
            return []

        origins = {rule.from_state for rule in candidates}

        # index -1 is the pose just entered; an origin has to precede it.
        for index in range(len(self._recent) - 2, -1, -1):
            state, entered = self._recent[index]
            if state in origins:
                elapsed = now - entered

                # SAMPLE BEFORE FILTERING.  An estimator fed only the
                # transitions that passed the current window is trained on
                # a population truncated by the very gate it is tuning:
                # under an 800 ms window the longest surviving sample was
                # 792 ms, a ceiling that looks natural and is pure
                # artefact.  Learning from that can never widen the
                # window, so the rejected ones are recorded here, one line
                # above the test that will throw them away.
                #
                # Submitting is also asynchronous, which settles a
                # question the synchronous version got awkwardly: the
                # sample that trips a recompute is filtered against the
                # OLD tolerance, because the worker publishes the new one
                # afterwards.  No arrival can widen the window it is
                # itself being measured against.
                # LEARNING ONLY.  If the path from the origin to here
                # crossed a frame with no hand in it, the elapsed time
                # measures how long the hand was missing, not how long
                # the movement takes.  `one -> hand lost -> fist` is not
                # evidence about `one -> fist`, so it is not learnt from.
                #
                # This gates the sample and NOTHING else: the rule below
                # still matches, still fires, and still respects its
                # window exactly as before.  Recognition is unchanged;
                # only what counts as training data is narrower.
                crossed_lost = any(
                    adaptive_timing.is_lost(seen)
                    for seen, _ in tuple(self._recent)[index + 1:-1])

                # submit(), not observe(): one bounded put_nowait, then
                # the recognition thread walks away.  Sorting fifty
                # samples is the learning worker's problem.
                if not crossed_lost:
                    for rule in candidates:
                        if rule.from_state == state and rule.adaptive_timing:
                            self._timing.submit(rule.from_state,
                                                rule.to_state, elapsed)

                eligible = [rule for rule in candidates
                            if rule.from_state == state
                            and self._within_window(rule, elapsed)]
                # Ownership first, cooldown second — same reasoning as the
                # hold loop.  Deciding the winner among rules that differ
                # only by hand BEFORE the cooldown test stops a cooling
                # rule from quietly handing its trigger to a sibling.
                return [rule for rule in self._one_per_trigger(eligible)
                        if not self._cooling(rule, now)]
            if state == self._current:
                # Already visited the target; the chain restarted there.
                return []
        return []

    def _within_window(self, rule, elapsed: float) -> bool:
        """Is this arrival recent enough for the rule to claim it?

        Two sources, never mixed.  A rule that names its own
        max_time_sec keeps that number for its whole life -- an existing
        config must behave tomorrow exactly as it does today.  Only a
        rule that opted in gets the estimator's value, which is bounded
        by adaptive_timing's baseline and ceiling and so can never leave
        a transition armed indefinitely.
        """
        if rule.adaptive_timing:
            return elapsed <= self._timing.tolerance(rule.from_state,
                                                     rule.to_state)
        return rule.max_time_sec <= 0.0 or elapsed <= rule.max_time_sec

    def armed_action_origin(self, now: float, source=None, hand=None,
                            hand_count: int = 0):
        """When the current pose became an armed action-transition origin.

        Returns the timestamp the origin was entered while at least one
        transition could still fire from it, and None otherwise.  Answers
        "is an action in flight from this pose right now", which is what
        the cursor path needs in order to stop dragging the pointer
        around while the user is mid-gesture.

        DERIVED, NOT LATCHED, and read-only.  Nothing is stored, so there
        is no flag for a missed cleanup path to leak: the moment the pose
        changes, the window lapses, the rule is deleted, or the machine
        is reset, the next call simply returns None.  That is what makes
        "the cursor can never stay frozen" structural rather than a
        promise made by six separate release paths.

        The window consulted here is the rule's own -- adaptive or
        explicit -- so this reports the real transition context and
        invents no timing of its own.  A caller that wants to stop
        holding the cursor sooner applies its own bound to the returned
        timestamp; that is an interaction choice and deliberately not
        this engine's business.
        """
        current = self._current
        if current is None or current in self._ignored \
                or current == NO_GESTURE:
            return None

        elapsed = now - self._entered_at
        for rule in self._transitions:
            if rule.from_state != current:
                continue
            if not rule.accepts_source(source):
                continue
            if not rule.accepts_hand(hand, hand_count):
                continue
            if self._within_window(rule, elapsed):
                return self._entered_at
        return None

    def timing_snapshot(self) -> dict:
        """What the estimator currently believes.  Diagnostics only."""
        return self._timing.snapshot()

    def _tick(self, now: float, source) -> list:
        events = self._drag_safety(now)

        fired_triggers = set()
        for rule in self._holds:
            if not rule.accepts_source(source):
                continue

            # Checked before the pose test so a hold whose hand requirement
            # has stopped being satisfied is DISARMED rather than frozen
            # armed — otherwise dropping the second hand mid-hold would
            # leave the rule primed to fire the instant it came back.
            if not rule.accepts_hand(self._hand, self._hand_count):
                self._hold_armed.pop(rule.id, None)
                self._hold_next.pop(rule.id, None)
                continue

            if self._current != rule.pose:
                # Leaving the pose is what re-arms the rule.  This is the
                # edge in "edge-triggered": without it a held hand would
                # satisfy the hold test on every frame forever.
                self._hold_armed.pop(rule.id, None)
                self._hold_next.pop(rule.id, None)
                continue

            if (now - self._entered_at) < rule.hold_sec:
                continue

            if rule.id in self._hold_armed:
                if not rule.repeat:
                    continue
                if now < self._hold_next.get(rule.id, 0.0):
                    continue

            # One action per trigger, and the loser is ARMED rather than
            # merely skipped.  Skipping alone would only postpone it: the
            # winner arms itself and goes quiet, and on the very next frame
            # the untouched sibling would find nothing in its way and fire
            # the second action a frame late.  Arming both keeps them in
            # lockstep until the pose is left, which is the edge that
            # re-arms the pair together.
            #
            # Ownership is claimed BEFORE the cooldown test on purpose.  A
            # rule that is merely cooling still owns its trigger for this
            # frame; handing it to a sibling whose only difference is a
            # hand requirement an unknown side happens to satisfy too would
            # turn the cooldown into a way of reaching the other action.
            key = rule.trigger_key()
            if key in fired_triggers:
                self._hold_armed[rule.id] = now
                self._hold_next[rule.id] = now + rule.repeat_sec
                continue
            fired_triggers.add(key)

            if self._cooling(rule, now):
                continue

            events.extend(self._fire(rule, now))
            self._hold_armed[rule.id] = now
            self._hold_next[rule.id] = now + rule.repeat_sec

        # Expire click halves that never found a partner, so a pending
        # single from a minute ago cannot promote an unrelated later click.
        if self._pending_click:
            stale = [rid for rid, when in self._pending_click.items()
                     if (now - when) > self._double_click_sec]
            for rid in stale:
                self._pending_click.pop(rid, None)

        return events

    def _drag_safety(self, now: float) -> list:
        """Release a drag that nothing is going to end on its own.

        Three independent ways out, because the failure they prevent — a
        mouse button still held after the process exits — is the one bug
        in this system that makes the whole desktop unusable:

          1. the hand is gone, or the pose is one no rule mentions, for
             longer than stuck_release_sec;
          2. the drag has simply run too long;
          3. (elsewhere) an explicit DRAG_STOP rule fired.
        """
        if not self._dragging:
            self._drag_lost_since = None
            return []

        if self._drag_timeout_sec > 0.0 and \
                (now - self._drag_started_at) > self._drag_timeout_sec:
            print("[fsm] drag exceeded its time limit — releasing")
            return [self._emit_drag_stop(self._drag_rule, now)]

        adrift = self._current is None \
            or self._current == NO_GESTURE \
            or self._current not in self._known

        if not adrift:
            self._drag_lost_since = None
            return []

        if self._drag_lost_since is None:
            self._drag_lost_since = now
            return []

        if (now - self._drag_lost_since) >= self._stuck_release_sec:
            print("[fsm] hand lost or pose unmapped while dragging — "
                  "releasing")
            return [self._emit_drag_stop(self._drag_rule, now)]

        return []

    def _cooling(self, rule: _Rule, now: float) -> bool:
        if rule.cooldown_sec <= 0.0:
            return False
        last = self._last_fired.get(rule.id)
        return last is not None and (now - last) < rule.cooldown_sec

    def _fire(self, rule: _Rule, now: float) -> list:
        """Run a rule's action through the per-action bookkeeping."""
        self._last_fired[rule.id] = now
        action = rule.action

        if action == DRAG_START:
            if self._dragging:
                # Already down.  Re-pressing would desynchronise the
                # engine's idea of the button from the OS's.
                return []
            self._dragging = True
            self._drag_rule = rule
            self._drag_started_at = now
            self._drag_lost_since = None
            self.drag_start_count += 1
            return [self._event(rule, DRAG_START, now)]

        if action == DRAG_STOP:
            if not self._dragging:
                return []
            return [self._emit_drag_stop(rule, now)]

        if action == DOUBLE_CLICK:
            # An explicit double cancels any single waiting to be promoted,
            # so the two cannot overlap into a triple.
            self._pending_click.clear()
            self.double_click_count += 1
            return [self._event(rule, DOUBLE_CLICK, now)]

        if action in (LEFT_CLICK, RIGHT_CLICK, MIDDLE_CLICK):
            return [self._register_click(rule, now)]

        if action in MACRO_ACTIONS:
            self.macro_count += 1
            return [self._event(rule, action, now)]

        return [self._event(rule, action, now)]

    def _register_click(self, rule: _Rule, now: float) -> ActionEvent:
        """Single or double, decided by the gap since this rule last fired.

        Only a repeat of the *same* rule can promote.  Two different
        gestures bound to LEFT_CLICK are two deliberate clicks that happen
        to land close together, and merging them would be the accidental
        overlap this is here to prevent.
        """
        promote = getattr(rule, "promote_double", False)
        if promote:
            pending = self._pending_click.get(rule.id)
            gap = now - pending if pending is not None else None
            if gap is not None and 0.0 <= gap <= self._double_click_sec:
                self._pending_click.pop(rule.id, None)
                self.double_click_count += 1
                return self._event(rule, DOUBLE_CLICK, now)
            self._pending_click[rule.id] = now

        self.click_count += 1
        return self._event(rule, rule.action, now)

    def _emit_drag_stop(self, rule, now: float) -> ActionEvent:
        self._dragging = False
        self._drag_rule = None
        self._drag_lost_since = None
        self.drag_stop_count += 1
        return self._event(rule, DRAG_STOP, now)

    @staticmethod
    def _event(rule, action: str, now: float) -> ActionEvent:
        return ActionEvent(
            action,
            rule_id=getattr(rule, "id", None),
            rule_name=getattr(rule, "name", None),
            keys=getattr(rule, "keys", "") if action == KEYBOARD_MACRO else "",
            at=now,
        )


# ─── Output ─────────────────────────────────────────────────────────────────

_KEY_ALIASES = {
    "win": "cmd", "super": "cmd", "meta": "cmd", "windows": "cmd",
    "control": "ctrl", "escape": "esc", "return": "enter",
    "del": "delete", "ins": "insert", "pgup": "page_up",
    "pgdn": "page_down", "pagedown": "page_down", "pageup": "page_up",
}


class ActionExecutor:
    """Performs action strings against the OS.

    pynput is imported on construction rather than at module scope so
    gesture_fsm stays importable — and unit-testable — on a machine with no
    input stack at all.  A missing package costs the output and nothing
    else.

    Held buttons are tracked here because this object is the only thing
    that knows whether a press actually reached the OS.  `release_all` is
    idempotent and safe to call from a finally block or an atexit hook,
    which is where it belongs.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._held = set()
        self._error_shown = False
        self._mouse = None
        self._keyboard = None
        self._Button = None
        self._Key = None

        if dry_run:
            return

        from pynput.mouse import Button, Controller as MouseController
        from pynput.keyboard import Key, Controller as KeyController
        self._mouse = MouseController()
        self._keyboard = KeyController()
        self._Button = Button
        self._Key = Key

    # ── public ──────────────────────────────────────────────────────────

    def dispatch(self, action, now: float = 0.0) -> bool:
        """Perform one action.  Never raises; returns True if it happened."""
        name = str(action)
        keys = getattr(action, "keys", "") or ACTION_MACROS.get(name, "")

        # Checked before the dry-run branch so a rehearsal reports the same
        # refusal a real run would; otherwise dry_run would claim success
        # for an action nothing knows how to perform.
        if name not in MOUSE_ACTIONS and not keys:
            return False

        if self.dry_run:
            print(f"[dry-run] {name}" + (f" ({keys})" if keys else ""))
            return True

        try:
            if name == LEFT_CLICK:
                self._mouse.click(self._Button.left, 1)
            elif name == RIGHT_CLICK:
                self._mouse.click(self._Button.right, 1)
            elif name == MIDDLE_CLICK:
                self._mouse.click(self._Button.middle, 1)
            elif name == DOUBLE_CLICK:
                self._mouse.click(self._Button.left, 2)
            elif name == DRAG_START:
                self._mouse.press(self._Button.left)
                self._held.add("left")
            elif name == DRAG_STOP:
                self._mouse.release(self._Button.left)
                self._held.discard("left")
            elif name == SCROLL_UP:
                self._mouse.scroll(0, 2)
            elif name == SCROLL_DOWN:
                self._mouse.scroll(0, -2)
            elif keys:
                return self._chord(keys)
            else:
                return False
        except Exception as exc:
            self._warn(f"output unavailable: {exc}")
            return False
        return True

    def release_all(self) -> None:
        """Drop anything still held.  Safe to call repeatedly."""
        if self.dry_run or not self._held:
            self._held.clear()
            return
        for name in tuple(self._held):
            try:
                self._mouse.release(getattr(self._Button, name))
            except Exception:
                pass
            self._held.discard(name)

    def close(self) -> None:
        self.release_all()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release_all()
        return False

    # ── internals ───────────────────────────────────────────────────────

    def _chord(self, spec: str) -> bool:
        """Press a chord and release it, whatever happens in between.

        The release runs from a finally over the keys actually pressed, so
        a failure halfway through cannot leave the Windows key latched —
        the keyboard's version of the stuck mouse button, and considerably
        harder to escape.
        """
        tokens = [t.strip().lower() for t in str(spec).split("+") if t.strip()]
        if not tokens:
            return False

        resolved = []
        for token in tokens:
            key = self._resolve(token)
            if key is None:
                self._warn(f"unknown key {token!r} in macro {spec!r}")
                return False
            resolved.append(key)

        pressed = []
        try:
            for key in resolved:
                self._keyboard.press(key)
                pressed.append(key)
            return True
        except Exception as exc:
            self._warn(f"keyboard unavailable: {exc}")
            return False
        finally:
            for key in reversed(pressed):
                try:
                    self._keyboard.release(key)
                except Exception:
                    pass

    def _resolve(self, token: str):
        token = _KEY_ALIASES.get(token, token)
        if len(token) == 1:
            return token
        return getattr(self._Key, token, None)

    def _warn(self, message: str) -> None:
        if not self._error_shown:
            self._error_shown = True
            print(f"[action] {message}")


def validate_macro(spec: str) -> bool:
    """True if `spec` names keys pynput can resolve.  Import-safe."""
    tokens = [t.strip().lower() for t in str(spec).split("+") if t.strip()]
    if not tokens:
        return False
    try:
        from pynput.keyboard import Key
    except Exception:
        return True          # cannot check; assume the user is right
    for token in tokens:
        token = _KEY_ALIASES.get(token, token)
        if len(token) == 1:
            continue
        if getattr(Key, token, None) is None:
            return False
    return True


if __name__ == "__main__":
    import time

    fsm = GestureFSM()
    print(f"[fsm] {fsm.describe()}")
    for rule in fsm.rules:
        print(f"       {rule.name:<28} -> {rule.action}")

    # A scripted hand: point, a stray frame, the curl through an unmapped
    # pose, then grip.  The click must survive all of it.
    script = ["point", "point", "point", "peace", "idle", "idle",
              "grip", "grip", "grip"]
    clock = time.perf_counter()
    for index, label in enumerate(script):
        action = fsm.update(label, clock + index * 0.033)
        if action is not None:
            print(f"  frame {index:>2} {label:<8} -> {action}")
