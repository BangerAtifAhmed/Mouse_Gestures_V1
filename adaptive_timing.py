"""Internal, per-pose-pair transition tolerance.

WHAT THIS SOLVES.  `max_time_sec` bounds the gap between a transition's
origin and its destination.  One fixed number cannot serve every mapping:
measured on this project, `one -> fist` on the ~4.7 Hz semantic branch has
a median duration of 814 ms, while a geometric pair at 30 Hz completes in
a fraction of that.  Asking the user to know and tune that per mapping is
the wrong division of labour, so the engine estimates it instead.

THE SHAPE OF THE ESTIMATE.  A fixed baseline that applies from the first
frame, extended -- never contracted below the baseline -- when enough
observations justify it, and clamped by a hard ceiling that is neither
learned nor configurable:

    tolerance = clamp(max(BASELINE, p95(recent) * SAFETY), BASELINE, CEILING)

Extension-only is what makes a bad estimate safe.  The worst a wrong
estimate can do is widen the window and cost precision; it can never
narrow one and newly break a mapping that used to work.

THE TRAP THIS AVOIDS.  The estimator MUST be fed durations that the
current tolerance rejects, not only ones that fired.  Learning from
successes alone samples a population truncated by the very gate being
tuned: measured under the old 800 ms window, the longest *successful*
`one -> fist` was 792 ms -- a ceiling that looks like a natural limit and
is purely an artefact of the filter.  An estimator trained on that can
never grow past it.  See GestureFSM._match_origin, which samples before
it filters.

WHAT IS DELIBERATELY NOT ADAPTIVE.  Cooldown encodes user intent -- how
fast repeats are wanted -- not a measurable property of the hand, so
there is no ground truth to learn toward.  Confidence and stabilisation
are likewise left alone.  This module changes one number and nothing
else.

STATE LIFETIME.  Session-scoped and in-memory by design.  Nothing is
written to disk, so there is no file to corrupt, migrate, or explain, and
a fresh process starts from a known-good baseline rather than from
whatever the last session happened to see.
"""
from __future__ import annotations

import collections
import math
import queue
import threading
import time

# Applied from the first frame, before any observation exists.  Chosen
# above the measured median (814 ms) so a cold start is already usable,
# and below the old hand-tuned 2.0 s so the ceiling has somewhere to go.
BASELINE_SEC = 1.2

# Absolute cap.  Not learned, not configurable, not reachable by any
# amount of evidence.  A transition can never stay armed longer than this.
CEILING_SEC = 2.5

# Evidence gates.  Below MIN_SAMPLES the baseline stands unchanged; past
# it the value is recomputed only once per RECOMPUTE_EVERY new samples, so
# the tolerance cannot twitch from frame to frame.
MIN_SAMPLES = 20
RECOMPUTE_EVERY = 10

# Bounded history: adapts to a user whose pace changes, without letting
# observations from an hour ago outvote the last minute.
HISTORY_SIZE = 50

# Rounding step.  Coarse enough that ordinary sample churn produces no
# visible change in behaviour.
QUANTUM_MS = 50

# States that mean "there was no hand to measure", as opposed to "the
# hand was in some other pose".  A transition whose path crosses one of
# these is not evidence about how long the movement takes -- the clock
# kept running while the classifier had nothing at all -- so it must not
# become learning data, in calibration or at runtime.
#
# Spelled literally rather than imported from gesture_fsm, which imports
# this module.  NO_GESTURE is "none"; the empty string is what an absent
# or unreadable label normalises to.
LOST_STATES = frozenset({"none", "", None})


def is_lost(state) -> bool:
    """True when this state means the hand was missing, not posed."""
    return state in LOST_STATES


# Headroom over the observed p95.  The p95 is what a deliberate gesture
# costs; the margin covers the slower tail without reaching for the max,
# which is noisy on a sample of fifty.
SAFETY_FACTOR = 1.25


def quantize(seconds: float) -> float:
    """Round UP to the next QUANTUM_MS.

    Up, not nearest: rounding down would shave the estimate below the
    evidence that produced it, which is the one direction this module is
    not allowed to move on its own.  Done in integer milliseconds so the
    result is exact -- 1.2 stays 1.2 rather than 1.2000000000000002.
    """
    ms = math.ceil(seconds * 1000.0 / QUANTUM_MS) * QUANTUM_MS
    return ms / 1000.0


def percentile(values, q: float) -> float:
    """Nearest-rank percentile.  No interpolation, so every result is a
    duration that was actually observed rather than a synthetic midpoint.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, min(len(ordered),
                      int(math.ceil(q / 100.0 * len(ordered)))))
    return ordered[rank - 1]


class _PairStats:
    """Everything known about one (from_state, to_state) pair."""

    __slots__ = ("durations", "count", "tolerance", "last_calc")

    def __init__(self) -> None:
        self.durations = collections.deque(maxlen=HISTORY_SIZE)
        self.count = 0
        self.tolerance = BASELINE_SEC
        self.last_calc = 0


class AdaptiveTiming:
    """Per-pose-pair transition tolerance, keyed by the movement itself.

    THE KEY IS THE MOVEMENT, AND ONLY THE MOVEMENT.  Keyed by
    (from_state, to_state) -- never by rule id, never by action, and
    never pooled globally.  Three reasons, in order of how badly each
    alternative would fail:

      * The action is irrelevant to timing.  `one -> fist = LEFT_CLICK`
        and `peace -> grip = LEFT_CLICK` are different movements that
        happen to share an outcome; merging them on the action would
        average two unrelated distributions.
      * Rule ids churn.  Editing a mapping in the GUI reissues its id --
        observed three times in this project's own history -- so an
        id-keyed history is thrown away on every edit.
      * A global pool would let a slow mapping widen a fast one's
        window, which is precisely the stale-trigger failure the ceiling
        exists to prevent.

    Every bucket is an independent dict entry with its own history,
    count and tolerance.  Nothing derives one pair's value from another.

    HANDEDNESS IS DELIBERATELY *NOT* IN THE KEY.  `one -> fist [right]`
    and `one -> fist [left]` share one timing bucket.  This is a decision,
    not an oversight:

      * The tolerance answers "how long does this movement take", and a
        hand folding from one to fist takes about as long either side.
      * Sharing halves the setup burden -- one 20-example calibration
        covers both variants instead of two.
      * Extension-only makes the merged estimate safe in the direction
        that matters.  If a user is slower with their non-dominant hand
        the shared estimate converges on the slower of the two, and the
        faster hand gets a slightly generous but still ceiling-bounded
        window.  The reverse -- a shared estimate that is too TIGHT for
        one hand -- cannot happen, because the value only ever grows.

    Crucially this shares TIMING and nothing else.  Which hand a rule
    accepts is decided entirely by accepts_hand() on the rule itself, so
    a shared bucket can never make a left-hand mapping fire for a right
    hand.  If per-hand histories are ever wanted, the key becomes a
    three-tuple here and nothing else in the system needs to change.
    """

    __slots__ = ("_pairs",)

    def __init__(self) -> None:
        self._pairs = {}

    # ── observation ────────────────────────────────────────────────────
    def observe(self, from_state: str, to_state: str,
                duration_sec: float) -> None:
        """Record how long one origin -> destination arrival took.

        Call this for EVERY arrival, including ones the current tolerance
        is about to reject.  See the module docstring: sampling only the
        accepted ones freezes the estimate at whatever the filter already
        allows.
        """
        try:
            duration = float(duration_sec)
        except (TypeError, ValueError):
            return
        # A negative or non-finite gap means the clock moved oddly, not
        # that the user gestured strangely.  Nothing to learn from it.
        if not (duration == duration) or duration in (float("inf"),
                                                      float("-inf")):
            return
        if duration < 0.0:
            return

        stats = self._pairs.get((from_state, to_state))
        if stats is None:
            stats = _PairStats()
            self._pairs[(from_state, to_state)] = stats

        stats.durations.append(duration)
        stats.count += 1

        if stats.count < MIN_SAMPLES:
            return
        if (stats.count - stats.last_calc) < RECOMPUTE_EVERY:
            return
        stats.last_calc = stats.count
        stats.tolerance = self._compute(stats.durations)

    @staticmethod
    def _compute(durations) -> float:
        estimate = percentile(durations, 95) * SAFETY_FACTOR
        # max() before clamp is what makes this extension-only: a run of
        # fast gestures can never pull the window under the baseline.
        return min(CEILING_SEC, max(BASELINE_SEC, quantize(estimate)))

    # ── query ──────────────────────────────────────────────────────────
    def tolerance(self, from_state: str, to_state: str) -> float:
        """Seconds this pair may take.  Always within [BASELINE, CEILING]."""
        stats = self._pairs.get((from_state, to_state))
        if stats is None:
            return BASELINE_SEC
        return stats.tolerance

    def samples(self, from_state: str, to_state: str) -> int:
        stats = self._pairs.get((from_state, to_state))
        return stats.count if stats is not None else 0

    def snapshot(self) -> dict:
        """Diagnostics only -- nothing reads this to make a decision."""
        return {
            f"{a} -> {b}": {
                "tolerance_sec": s.tolerance,
                "samples": s.count,
                "held": len(s.durations),
                "is_baseline": s.tolerance == BASELINE_SEC,
            }
            for (a, b), s in sorted(self._pairs.items())
        }

    def forget(self, from_state: str, to_state: str) -> bool:
        """Drop everything known about one pair.  True if it existed.

        Deleting a mapping must leave no trace of it: history, count and
        learned tolerance all go, so a pair recreated later starts from
        the baseline like any other new mapping rather than inheriting
        an estimate the user cannot see or reason about.
        """
        return self._pairs.pop((from_state, to_state), None) is not None

    def reset(self) -> None:
        self._pairs.clear()


# ── Asynchronous front end ─────────────────────────────────────────────
# How long the worker waits for a sample before looping round to check
# whether it has been asked to stop.  Short enough to shut down promptly,
# long enough that an idle worker costs nothing measurable.
WORKER_POLL_SEC = 0.25

# Bounded on purpose.  If learning ever falls behind the recognition loop
# the correct failure is to drop samples, not to grow without limit:
# timing estimates are statistical, and losing a handful of observations
# changes nothing, whereas an unbounded queue eventually takes the
# process down.
QUEUE_MAX = 512


class TimingLearner:
    """Thread-safe front for AdaptiveTiming.

    THE SPLIT.  The recognition thread must never do statistics.  All it
    does here is `submit()`, which is one bounded `put_nowait` and a
    dict lookup -- no sorting, no allocation beyond a tuple, no lock it
    can contend on, and no path that can raise into the camera loop.  A
    background worker owns the estimator outright and is the only thread
    that ever touches it.

    HOW READS STAY CHEAP.  The worker never mutates the dict the hot
    path reads.  It builds a new one and rebinds the attribute, which is
    atomic under the GIL, so a reader gets either the old mapping or the
    new one and never a half-updated one.  Same trick the YOLO worker
    uses for its (gesture, score, latency) triple, and for the same
    reason: it removes the lock from the path that runs most often.

    OVERFLOW.  A full queue drops the sample and counts it.  That is a
    deliberate quality-of-estimate cost paid to protect the frame rate,
    and `stats()` reports it so the trade is visible rather than silent.
    """

    __slots__ = ("_estimator", "_queue", "_thread", "_stop", "_tolerances",
                 "_dropped", "_processed", "_worker_sec", "_lock", "_epochs",
                 "_stale")

    def __init__(self, estimator=None) -> None:
        self._estimator = estimator or AdaptiveTiming()
        self._queue = queue.Queue(maxsize=QUEUE_MAX)
        self._thread = None
        self._stop = threading.Event()

        # Read by the recognition thread, written only by rebinding.
        self._tolerances = {}

        # One counter per pair, bumped by forget().  A queued sample
        # carries the epoch it was submitted under, so anything already
        # in flight for a deleted mapping is recognised as stale and
        # thrown away instead of landing after the deletion.  This is
        # what stops a deleted mapping finishing its learning.
        self._epochs = {}
        self._stale = 0

        self._dropped = 0
        self._processed = 0
        self._worker_sec = 0.0
        # Serialises every write to the estimator -- the worker's batch
        # updates and the GUI's calibration seed alike -- so two pairs
        # can never be mutated concurrently.  Reads do not take it: they
        # go through _tolerances, which is replaced rather than mutated.
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="timing-learner")
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── hot path ───────────────────────────────────────────────────────
    def submit(self, from_state: str, to_state: str,
               duration_sec: float) -> bool:
        """Hand one observation to the worker.  Never blocks, never raises.

        Called from the recognition thread, so everything expensive is
        deliberately on the other side of the queue.
        """
        # Started on first use rather than in __init__: a GestureFSM with
        # no adaptive rules should not cost a thread, and a great many
        # FSMs get built over a session and in tests.
        if self._thread is None:
            self.start()
        try:
            self._queue.put_nowait((from_state, to_state, duration_sec,
                                    self._epochs.get((from_state,
                                                      to_state), 0)))
            return True
        except queue.Full:
            self._dropped += 1
            return False
        except Exception:
            return False

    def tolerance(self, from_state: str, to_state: str) -> float:
        """The current window for a pair.  One dict lookup, no lock."""
        return self._tolerances.get((from_state, to_state), BASELINE_SEC)

    # ── worker ─────────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=WORKER_POLL_SEC)
            except queue.Empty:
                continue
            batch = [item]
            # Drain whatever else is waiting: recomputing once for a
            # burst beats recomputing per sample.
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._apply(batch)

    def _apply(self, batch) -> None:
        started = time.perf_counter()
        try:
            with self._lock:
                for from_state, to_state, duration, epoch in batch:
                    # The context has to still be the one this sample was
                    # taken under.  Checked HERE, under the lock, rather
                    # than at submit time, because the mapping can be
                    # deleted while the sample sits in the queue.
                    if epoch != self._epochs.get((from_state, to_state), 0):
                        self._stale += 1
                        continue
                    self._estimator.observe(from_state, to_state, duration)
                self._publish()
        except Exception:
            # A learning fault must cost learning and nothing else; the
            # recognition loop keeps whatever tolerances it already has.
            pass
        self._processed += len(batch)
        self._worker_sec += time.perf_counter() - started

    def _publish(self) -> None:
        """Rebind, never mutate -- see the class docstring."""
        self._tolerances = {
            pair: stats.tolerance
            for pair, stats in self._estimator._pairs.items()
        }

    # ── calibration hand-off ───────────────────────────────────────────
    def seed(self, from_state: str, to_state: str, durations) -> None:
        """Install observations gathered before the mapping existed.

        Synchronous by design: this runs once, from the GUI thread, at
        the end of calibration, and the caller needs the tolerance to be
        live before it saves the mapping.
        """
        with self._lock:
            for duration in durations:
                self._estimator.observe(from_state, to_state, duration)
            self._publish()

    def forget(self, from_state: str, to_state: str) -> bool:
        """Retire a pair: history, tolerance and anything still queued.

        Synchronous and under the lock, so by the time this returns the
        pair reads as baseline again and no in-flight sample can revive
        it.  Called from the GUI thread when a mapping is deleted.
        """
        pair = (from_state, to_state)
        with self._lock:
            existed = self._estimator.forget(from_state, to_state)
            # Bump first, publish second: any sample already queued for
            # this pair now carries a stale epoch and will be discarded
            # by the worker rather than applied after the deletion.
            self._epochs[pair] = self._epochs.get(pair, 0) + 1
            self._publish()
        return existed

    # ── diagnostics ────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "queue_depth": self._queue.qsize(),
            "queue_max": QUEUE_MAX,
            "dropped": self._dropped,
            "stale_discarded": self._stale,
            "processed": self._processed,
            "worker_total_ms": round(self._worker_sec * 1000.0, 3),
            "worker_avg_ms": (round(self._worker_sec * 1000.0
                                    / self._processed, 4)
                              if self._processed else 0.0),
            "running": self.running,
        }

    def snapshot(self) -> dict:
        return self._estimator.snapshot()

    def samples(self, from_state: str, to_state: str) -> int:
        return self._estimator.samples(from_state, to_state)

    def drain(self, timeout: float = 2.0) -> bool:
        """Block until the queue is empty.  Tests and shutdown only."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self._queue.empty():
                return True
            time.sleep(0.005)
        return self._queue.empty()


class CalibrationSession:
    """Counts valid examples of ONE pose pair, before any mapping exists.

    Deliberately free of Tk so the acceptance rules can be tested
    headlessly.  It reads the same probe rows the gesture test bench
    reads, and it is strict about what counts: a sample is only taken
    when the state genuinely moved from the origin to the destination,
    with a hand the mapping would accept, inside the ceiling.  A YOLO
    miss, an unrelated pose pair, an unknown state or a wrong hand
    advances nothing, because a calibration padded with those would
    teach the estimator the wrong distribution on its very first
    contact with the user.
    """

    __slots__ = ("from_state", "to_state", "hand", "needed", "durations",
                 "rejected", "_armed_at", "_seen_origin", "_discarded")

    def __init__(self, from_state: str, to_state: str, hand: str = "any",
                 needed: int = MIN_SAMPLES) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.hand = (hand or "any").lower()
        self.needed = int(needed)
        self.durations = []
        self.rejected = {"hand": 0, "no_origin": 0, "too_slow": 0,
                         "hand_lost": 0}
        self._armed_at = None
        self._seen_origin = False
        # Set when a lost hand throws an attempt away, so the destination
        # that arrives afterwards is recognised as the tail of THAT
        # attempt rather than tallied a second time as a fresh one with
        # no origin.  One discarded attempt, one rejection.
        self._discarded = False

    @property
    def collected(self) -> int:
        return len(self.durations)

    @property
    def done(self) -> bool:
        return len(self.durations) >= self.needed

    def _hand_ok(self, row) -> bool:
        if self.hand in ("any", "", "both"):
            return True
        side = (row.get("sem_hand") or row.get("side") or "").lower()
        # An unknown side is not evidence of the WRONG side, and refusing
        # it would stall calibration whenever handedness drops for a
        # frame.  Same tolerance the rules themselves apply.
        return side in ("", self.hand)

    def feed(self, rows) -> int:
        """Consume probe rows; return how many samples were accepted."""
        before = len(self.durations)
        for row in rows:
            if self.done:
                break
            state = row.get("state_after")
            now = row.get("t")
            if now is None:
                continue

            # The hand went missing.  Whatever was in flight is not a
            # measurement of a gesture any more -- the clock ran on while
            # the classifier had nothing -- so the attempt is abandoned
            # and the user simply performs another one.
            if is_lost(state):
                if self._seen_origin:
                    self.rejected["hand_lost"] += 1
                    self._discarded = True
                self._armed_at = None
                self._seen_origin = False
                continue

            if state == self.from_state:
                self._armed_at = now
                self._seen_origin = True
                self._discarded = False
                continue

            if state != self.to_state:
                continue

            if not self._seen_origin or self._armed_at is None:
                if self._discarded:
                    # Already counted as hand_lost when the hand went.
                    self._discarded = False
                else:
                    self.rejected["no_origin"] += 1
                continue
            if not self._hand_ok(row):
                self.rejected["hand"] += 1
                self._armed_at = None
                self._seen_origin = False
                continue

            duration = now - self._armed_at
            if duration <= 0.0 or duration > CEILING_SEC:
                self.rejected["too_slow"] += 1
            else:
                self.durations.append(duration)
            self._armed_at = None
            self._seen_origin = False
        return len(self.durations) - before

    def summary(self) -> dict:
        return {
            "pair": f"{self.from_state} -> {self.to_state}",
            "collected": self.collected,
            "needed": self.needed,
            "rejected": dict(self.rejected),
            "median_ms": (round(sorted(self.durations)[len(self.durations)
                                                       // 2] * 1000.0, 1)
                          if self.durations else None),
        }
