from __future__ import annotations

from collections import Counter, deque

__all__ = [
    "LEFT_CLICK",
    "DOUBLE_CLICK",
    "DRAG_START",
    "DRAG_STOP",
    "NO_GESTURE",
    "GestureStabilizer",
    "MajorityStabilizer",
    "ConsecutiveStabilizer",
    "GestureFSM",
    "normalise_gesture",
]

LEFT_CLICK = "LEFT_CLICK"
DOUBLE_CLICK = "DOUBLE_CLICK"
DRAG_START = "DRAG_START"
DRAG_STOP = "DRAG_STOP"
NO_GESTURE = "none"

DEFAULT_WINDOW_SIZE = 5
DEFAULT_STABILITY_THRESHOLD = 3
DEFAULT_DOUBLE_CLICK_THRESHOLD = 0.4
DEFAULT_CLICK_TRANSITION = ("one", "fist")
DEFAULT_DRAG_TRANSITION = None


def normalise_gesture(raw_gesture: str | None) -> str:
    if raw_gesture is None:
        return NO_GESTURE
    label = str(raw_gesture).strip().lower()
    return label if label else NO_GESTURE


class GestureStabilizer:

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        threshold: int = DEFAULT_STABILITY_THRESHOLD,
    ) -> None:
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
    def stable(self) -> str | None:
        return self._stable

    def push(self, label: str) -> str | None:
        self._window.append(label)
        candidate = self._resolve()
        if candidate is not None:
            self._stable = candidate
        return self._stable

    def reset(self) -> None:
        self._window.clear()
        self._stable = None

    def _resolve(self) -> str | None:
        raise NotImplementedError


class MajorityStabilizer(GestureStabilizer):

    def _resolve(self) -> str | None:
        if not self._window:
            return None
        label, count = Counter(self._window).most_common(1)[0]
        return label if count >= self._threshold else None


class ConsecutiveStabilizer(GestureStabilizer):

    def _resolve(self) -> str | None:
        if len(self._window) < self._threshold:
            return None
        recent = list(self._window)[-self._threshold:]
        first = recent[0]
        for label in recent:
            if label != first:
                return None
        return first


class GestureFSM:

    def __init__(
        self,
        stabilizer: GestureStabilizer | None = None,
        double_click_threshold: float = DEFAULT_DOUBLE_CLICK_THRESHOLD,
        click_transition: tuple = DEFAULT_CLICK_TRANSITION,
        drag_transition: tuple | None = DEFAULT_DRAG_TRANSITION,
        drag_release: str | None = None,
    ) -> None:
        if double_click_threshold < 0.0:
            raise ValueError("double_click_threshold must be non-negative")

        source, target = click_transition
        if source == target:
            raise ValueError("click_transition endpoints must differ")

        self._stabilizer = stabilizer if stabilizer is not None else MajorityStabilizer()
        self._double_click_threshold = float(double_click_threshold)
        self._source_state = normalise_gesture(source)
        self._target_state = normalise_gesture(target)

        if drag_transition is None:
            self._drag_source = None
            self._drag_target = None
        else:
            drag_source, drag_target = drag_transition
            if drag_source == drag_target:
                raise ValueError("drag_transition endpoints must differ")
            self._drag_source = normalise_gesture(drag_source)
            self._drag_target = normalise_gesture(drag_target)
            if (self._drag_source == self._source_state
                    and self._drag_target == self._target_state):
                raise ValueError(
                    "click_transition and drag_transition are identical; "
                    "the two would be indistinguishable"
                )

        self._drag_release = (None if drag_release is None
                              else normalise_gesture(drag_release))

        self._previous_stable_state = None
        self._current_stable_state = None
        self._pending_click_time = None
        self._is_dragging = False
        self._click_count = 0
        self._double_click_count = 0
        self._drag_start_count = 0
        self._drag_stop_count = 0

    @property
    def stabilizer(self) -> GestureStabilizer:
        return self._stabilizer

    @property
    def previous_stable_state(self) -> str | None:
        return self._previous_stable_state

    @property
    def current_stable_state(self) -> str | None:
        return self._current_stable_state

    @property
    def double_click_threshold(self) -> float:
        return self._double_click_threshold

    @property
    def click_transition(self) -> tuple:
        return (self._source_state, self._target_state)

    @property
    def drag_transition(self) -> tuple | None:
        if self._drag_source is None:
            return None
        return (self._drag_source, self._drag_target)

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    @property
    def click_count(self) -> int:
        return self._click_count

    @property
    def double_click_count(self) -> int:
        return self._double_click_count

    @property
    def drag_start_count(self) -> int:
        return self._drag_start_count

    @property
    def drag_stop_count(self) -> int:
        return self._drag_stop_count

    def is_double_click_window_open(self, current_time: float) -> bool:
        if self._pending_click_time is None:
            return False
        elapsed = current_time - self._pending_click_time
        return 0.0 <= elapsed <= self._double_click_threshold

    def time_remaining(self, current_time: float) -> float:
        if self._pending_click_time is None:
            return 0.0
        remaining = self._double_click_threshold - (current_time - self._pending_click_time)
        return remaining if remaining > 0.0 else 0.0

    def update(
        self, raw_gesture: str | None, current_time: float
    ) -> tuple[str | None, str | None]:
        label = normalise_gesture(raw_gesture)
        stable = self._stabilizer.push(label)

        if stable is None or stable == self._current_stable_state:
            return stable, None

        self._previous_stable_state = self._current_stable_state
        self._current_stable_state = stable

        return stable, self._evaluate_transition(current_time)

    def reset(self) -> None:
        self._stabilizer.reset()
        self._previous_stable_state = None
        self._current_stable_state = None
        self._pending_click_time = None
        self._is_dragging = False

    def _evaluate_transition(self, current_time: float) -> str | None:
        previous = self._previous_stable_state
        current = self._current_stable_state

        if self._is_dragging and self._releases_drag(current):
            self._is_dragging = False
            self._drag_stop_count += 1
            return DRAG_STOP

        if self._starts_drag(previous, current):
            self._is_dragging = True
            self._drag_start_count += 1
            self._pending_click_time = None
            return DRAG_START

        if previous == self._source_state and current == self._target_state:
            return self._register_click(current_time)

        return None

    def _starts_drag(self, previous: str | None, current: str | None) -> bool:
        if self._drag_source is None or self._is_dragging:
            return False
        return previous == self._drag_source and current == self._drag_target

    def _releases_drag(self, current: str | None) -> bool:
        if self._drag_target is None:
            return False
        if self._drag_release is not None:
            return current in (self._drag_release, NO_GESTURE)
        return current != self._drag_target

    def _register_click(self, current_time: float) -> str:
        if self.is_double_click_window_open(current_time):
            self._pending_click_time = None
            self._double_click_count += 1
            return DOUBLE_CLICK

        self._pending_click_time = current_time
        self._click_count += 1
        return LEFT_CLICK
