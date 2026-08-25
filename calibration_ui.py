"""The one moment adaptive timing is visible to the user.

A new mapping asks for twenty ordinary examples of its gesture, and that
is the whole of the user's involvement.  Everything after it -- every
runtime observation, every recompute, every tolerance change -- happens
silently in the learning worker and is never surfaced.  The intended
shape of the experience is:

    create -> give 20 examples -> use normally

WHY IT ASKS AT ALL.  Without it a brand new mapping spends its first
twenty real attempts on the 1.2 s baseline, and a user whose gesture
naturally runs longer would experience those as random misses with no
explanation.  Twenty deliberate examples up front turn that into a
thirty-second setup step that says what it is doing.

WHY THE MAPPING DOES NOT EXIST YET.  Calibration watches the raw
recognition stream through a GestureProbe rather than through a live
rule.  Nothing is written to gesture_config.json until the twentieth
sample lands, so cancelling -- or closing the window, or losing the
camera -- leaves no half-initialised mapping behind.  There is no
partially-created state to clean up because there is no created state.

LAYOUT.  The window has no fixed size.  An earlier version pinned it to
460x340 with resizing off, which clipped the buttons the moment a camera
picker had to appear.  Instead the footer is packed FIRST against the
bottom so it always reserves its height, the body expands into what is
left, and _fit() grows the window to whatever the content actually asks
for after anything is shown or hidden.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import adaptive_timing
import hand_cursor_2

# Matches the test bench: fast enough to feel live, cheap enough that the
# GUI thread never notices.
POLL_MS = 100

BG = "#18181b"
CARD = "#27272a"
BORDER = "#3f3f46"
TEXT = "#f4f4f5"
FAINT = "#71717a"
ACCENT = "#10b981"
DANGER = "#f43f5e"
PAD = 14

# Floor only.  The window is sized from its content and grows past this
# whenever the content needs more; this simply stops it collapsing to
# something unusable if a theme reports odd metrics.
MIN_W, MIN_H = 470, 430


def _button(parent, text, command, primary=False):
    """A flat clickable label, styled like the rest of the studio."""
    widget = tk.Label(parent, text=f"  {text}  ",
                      bg=(ACCENT if primary else CARD),
                      fg=("#052e16" if primary else TEXT),
                      cursor="hand2",
                      font=("Segoe UI", 9, "bold" if primary else "normal"))
    widget.bind("<Button-1>", lambda _e: command())
    return widget


class CalibrationDialog(tk.Toplevel):
    """Collects MIN_SAMPLES valid examples, then hands them to the engine.

    `on_done(durations)` is called only on success, from the GUI thread.
    Cancel, close and camera loss all end the dialog without calling it.
    """

    def __init__(self, studio, rule, on_done) -> None:
        super().__init__(studio)
        self.studio = studio
        self.rule = rule
        self.on_done = on_done
        self._job = None
        self._finished = False
        # Whether collection ever actually started.  A camera event that
        # arrives before it did must not be reported as an interruption.
        self._collecting = False
        # And whether an interruption has already been reported, so a
        # second notification does not overwrite the explanation with a
        # blander one.  camera_lost() is reached twice on a deliberate
        # disconnect: once directly, once via the state watcher.
        self._interrupted = False

        self.session = adaptive_timing.CalibrationSession(
            rule.get("from_state", ""), rule.get("to_state", ""),
            rule.get("hand", "any"),
            needed=adaptive_timing.MIN_SAMPLES)

        self.title("Set up gesture")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.transient(studio)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._build()

        # The dialog follows the camera rather than sampling it once.
        # That is what lets "Connect Camera" continue this same session
        # instead of making the user close it and start again.
        try:
            studio.watch_camera(self._on_camera_state)
        except Exception:
            pass
        self._start()
        self._fit()

    # ── construction ───────────────────────────────────────────────────
    def _build(self) -> None:
        # FOOTER FIRST.  Packed against the bottom before anything that
        # expands, so it always owns its strip of the window.  Pack it
        # after the body and a body that grows pushes it off the edge --
        # which is exactly how the buttons came to be clipped.
        footer = tk.Frame(self, bg=BG)
        footer.pack(side="bottom", fill="x", padx=PAD, pady=(8, PAD))

        self.cancel_button = _button(footer, "Cancel", self._cancel)
        self.cancel_button.pack(side="right")

        # There is deliberately no Start button anywhere: collection
        # begins on its own the moment frames are available, whether
        # that is now or after Connect Camera.
        self.connect_button = _button(footer, "Connect Camera",
                                      self._connect, primary=True)
        self.change_button = _button(footer, "Change Camera",
                                     self._change_camera)
        self.disconnect_button = _button(footer, "Disconnect Camera",
                                         self._disconnect)

        body = tk.Frame(self, bg=BG)
        body.pack(side="top", fill="both", expand=True, padx=PAD,
                  pady=(PAD, 0))

        tk.Label(body, text="ONE LAST STEP", bg=BG, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(body, text=f"Perform this gesture "
                            f"{adaptive_timing.MIN_SAMPLES} times.",
                 bg=BG, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w",
                                                     pady=(2, 0))

        card = tk.Frame(body, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        card.pack(fill="x", pady=(10, 0))
        tk.Label(card, text=f"{self.session.from_state}  →  "
                            f"{self.session.to_state}",
                 bg=CARD, fg=TEXT,
                 font=("Segoe UI", 13)).pack(anchor="w", padx=PAD,
                                             pady=(10, 2))
        tk.Label(card, text=f"{self.rule.get('action', '')}"
                            f"    hand: {self.session.hand}",
                 bg=CARD, fg=FAINT,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=PAD,
                                            pady=(0, 10))

        # ── camera section ──────────────────────────────────────────
        # Present in both states: a picker when there is no camera, and
        # what is running when there is.
        self.camera_row = tk.Frame(body, bg=BG)
        tk.Label(self.camera_row, text="CAMERA", bg=BG, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")

        self.camera_pick = tk.Frame(self.camera_row, bg=BG)
        self.camera_var = tk.StringVar(value="")
        self.camera_box = ttk.Combobox(
            self.camera_pick, textvariable=self.camera_var,
            state="readonly", values=[], font=("Segoe UI", 9))
        self.camera_box.pack(fill="x")
        self.camera_box.bind("<<ComboboxSelected>>",
                             lambda _e: self._pick_camera())
        self.refresh_button = _button(self.camera_pick, "Refresh Cameras",
                                      self._refresh_cameras)
        self.refresh_button.pack(anchor="w", pady=(6, 0))

        self.camera_info = tk.Label(self.camera_row, text="", bg=BG,
                                    fg=TEXT, anchor="w", justify="left",
                                    font=("Segoe UI", 10))

        self.hint = tk.Label(
            body,
            text="Do it the way you normally would — no need to hold the "
                 "pose. Only clean examples are counted.",
            bg=BG, fg=FAINT, font=("Segoe UI", 8), wraplength=420,
            justify="left")
        self.hint.pack(anchor="w", pady=(12, 0))

        self.count = tk.Label(body,
                              text=f"0 / {adaptive_timing.MIN_SAMPLES}",
                              bg=BG, fg=ACCENT,
                              font=("Segoe UI", 22, "bold"))
        self.count.pack(anchor="w", pady=(10, 2))

        self.bar = ttk.Progressbar(body, mode="determinate",
                                   maximum=adaptive_timing.MIN_SAMPLES)
        self.bar.pack(fill="x")

        self.status = tk.Label(body, text="", bg=BG, fg=FAINT,
                               font=("Segoe UI", 8), wraplength=420,
                               justify="left", anchor="w")
        self.status.pack(fill="x", pady=(8, 0))

    def _fit(self) -> None:
        """Grow the window to whatever the content now asks for.

        Called after anything is shown or hidden.  Only ever grows: a
        window that shrank back would jump about as the camera section
        came and went.
        """
        try:
            self.update_idletasks()
            want_w = max(MIN_W, self.winfo_reqwidth())
            want_h = max(MIN_H, self.winfo_reqheight())
            self.minsize(MIN_W, MIN_H)
            if want_w > self.winfo_width() or want_h > self.winfo_height():
                self.geometry(f"{want_w}x{want_h}")
        except tk.TclError:                     # pragma: no cover
            pass

    def _footer_for(self, connected: bool) -> None:
        """Show the buttons that belong to the current camera state."""
        for widget in (self.connect_button, self.change_button,
                       self.disconnect_button):
            widget.pack_forget()
        if connected:
            self.disconnect_button.pack(side="left")
            self.change_button.pack(side="left", padx=(8, 0))
        else:
            self.connect_button.pack(side="left")

    # ── run ────────────────────────────────────────────────────────────
    def _start(self) -> None:
        """Begin collecting, or ask for a camera and wait for one."""
        if not self._camera_live():
            self._await_camera()
            return
        self._begin()

    def _camera_live(self) -> bool:
        try:
            return bool(self.studio.camera_is_live())
        except Exception:
            engine = getattr(self.studio, "engine", None)
            return engine is not None and getattr(engine, "running", False)

    def _await_camera(self) -> None:
        """No camera yet: offer to pick one and start it, session intact."""
        self.count.configure(text="Camera not connected")
        self.status.configure(
            text="Connect the camera to continue the "
                 f"{adaptive_timing.MIN_SAMPLES}-sample gesture setup.",
            fg=DANGER)
        self.camera_info.pack_forget()
        self.camera_row.pack(fill="x", pady=(12, 0), before=self.hint)
        self.camera_pick.pack(fill="x", pady=(2, 0))
        self._sync_camera_list()
        self._footer_for(connected=False)
        self._fit()

    def _show_connected(self) -> None:
        """Camera is live: name it, and offer to change or release it."""
        self.camera_pick.pack_forget()
        self.camera_row.pack(fill="x", pady=(12, 0), before=self.hint)
        try:
            label = self.studio.selected_camera_label()
        except Exception:
            label = "camera"
        self.camera_info.configure(text=f"{label}\nCamera connected")
        self.camera_info.pack(fill="x", pady=(2, 0))
        self._footer_for(connected=True)
        self._fit()

    def _sync_camera_list(self) -> None:
        """Mirror the studio's list and selection into this dropdown."""
        try:
            cams = list(getattr(self.studio, "cameras", []))
            chosen = self.studio.selected_camera()
        except Exception:
            return
        self.camera_box.configure(values=[c.label for c in cams])
        if not cams:
            self.camera_var.set("")
            # APPEND, never replace.  The caller has already explained
            # why this dialog is open -- the 20-sample setup, or a
            # camera that vanished mid-collection -- and overwriting
            # that with a generic line loses the reason.
            existing = self.status.cget("text")
            detail = ("No camera detected. Connect one, then Refresh "
                      "Cameras.")
            self.status.configure(
                text=(f"{existing} {detail}" if existing else detail),
                fg=DANGER)
            return
        for cam in cams:
            if cam.index == chosen:
                self.camera_var.set(cam.label)
                return
        self.camera_var.set(cams[0].label)

    def _pick_camera(self) -> None:
        """Selection here IS the studio's selection -- one state, not two."""
        label = self.camera_var.get()
        for cam in getattr(self.studio, "cameras", []):
            if cam.label == label:
                self.studio.camera_index = cam.index
                try:
                    self.studio.camera_var.set(cam.label)
                except Exception:
                    pass
                break

    def _refresh_cameras(self) -> None:
        """Re-enumerate through the studio, so both lists stay identical."""
        self.status.configure(text="Looking for cameras…", fg=FAINT)
        try:
            self.studio.refresh_cameras()
        except Exception as exc:
            self.status.configure(
                text=f"Could not list cameras "
                     f"({exc.__class__.__name__}).", fg=DANGER)

    def _connect(self) -> None:
        """Ask the studio to start the camera; do not wait for it here.

        Connecting imports mediapipe and opens a device, which is why the
        studio does it on a worker.  Blocking the Tk thread on that would
        freeze this dialog mid-click, so the answer arrives through the
        camera-state watcher instead.
        """
        self.count.configure(text="Connecting…")
        self.status.configure(text="Starting the camera…", fg=FAINT)
        try:
            self.studio._connect()
        except Exception as exc:
            self.status.configure(
                text=f"Could not start the camera "
                     f"({exc.__class__.__name__}).", fg=DANGER)
            self._await_camera()

    def _change_camera(self) -> None:
        """Release the current device and offer the picker again."""
        try:
            self.studio._change_camera()
        except Exception:
            pass

    def _disconnect(self) -> None:
        try:
            self.studio._disconnect()
        except Exception:
            pass

    def _on_camera_state(self, state) -> None:
        """Camera state changed under us."""
        if self._finished:
            return
        if state == "CONNECTED":
            if self._job is None:
                self.status.configure(text="", fg=FAINT)
                self._begin()
            else:
                self._show_connected()
        elif state in ("DISCONNECTED", "ERROR"):
            self.camera_lost()
        try:
            if self.camera_pick.winfo_manager():
                self._sync_camera_list()
        except tk.TclError:
            pass

    def _begin(self) -> None:
        """Attach the probe and start counting.  No user action needed."""
        engine = getattr(self.studio, "engine", None)
        if engine is None:
            self._await_camera()
            return
        self._collecting = True
        self._interrupted = False
        self.count.configure(text=f"0 / {self.session.needed}")
        self.bar.configure(value=0)
        self._show_connected()
        engine.probe = hand_cursor_2.GestureProbe()
        self._poll()

    def camera_lost(self) -> None:
        """The camera went away.  Stop cleanly and say so.

        Nothing is created and nothing is seeded, so there is no partial
        mapping to unwind -- the session simply stops counting and offers
        the camera again.
        """
        if self._finished:
            return
        if not self._collecting:
            if self._interrupted:
                # Already explained.  A repeat notification must not
                # replace "the camera disconnected" with the generic
                # opening prompt.
                return
            # Never started, so nothing was interrupted.  Reporting a
            # pause here would replace the "connect a camera" prompt
            # with a count that was never running -- which is what a
            # background camera scan firing the watcher used to do.
            self._await_camera()
            return
        self._collecting = False
        self._interrupted = True
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        engine = getattr(self.studio, "engine", None)
        if engine is not None:
            engine.probe = None
        self.count.configure(text=f"{self.session.collected} / "
                                  f"{self.session.needed}  (paused)")
        self.status.configure(
            text="The camera disconnected, so collection stopped. "
                 "Nothing was saved. Reconnect to carry on.", fg=DANGER)
        self.camera_info.pack_forget()
        self.camera_row.pack(fill="x", pady=(12, 0), before=self.hint)
        self.camera_pick.pack(fill="x", pady=(2, 0))
        self._sync_camera_list()
        self._footer_for(connected=False)
        self._fit()

    def _poll(self) -> None:
        self._job = None
        if self._finished:
            return
        engine = getattr(self.studio, "engine", None)
        if engine is None or not engine.running or engine.probe is None:
            # Camera lost mid-way.  Nothing was written, so there is
            # nothing to undo -- but the probe still has to come off, or
            # a dead session leaves the engine recording into a buffer
            # nobody will ever drain.
            self.camera_lost()
            return

        self.session.feed(engine.probe.drain())
        got = self.session.collected
        self.count.configure(text=f"{got} / {self.session.needed}")
        self.bar.configure(value=got)

        # Say why the counter did not move.  Silence here reads as the
        # gesture not being recognised at all, when the usual cause is
        # simply that the hand left the frame part-way through.
        rejected = self.session.rejected
        if rejected["hand_lost"]:
            self.status.configure(
                text=f"{rejected['hand_lost']} attempt(s) discarded — the "
                     f"hand left the frame part-way through. Keep it in "
                     f"view for the whole movement.", fg=FAINT)
        elif rejected["hand"]:
            self.status.configure(
                text=f"{rejected['hand']} example(s) used the wrong hand "
                     f"and were not counted.", fg=FAINT)
        elif rejected["too_slow"]:
            self.status.configure(
                text=f"{rejected['too_slow']} example(s) took too long "
                     f"and were not counted.", fg=FAINT)

        if self.session.done:
            self._succeed()
            return
        self._job = self.after(POLL_MS, self._poll)

    def _succeed(self) -> None:
        self._finished = True
        self._detach()
        durations = list(self.session.durations)
        self.count.configure(text=f"{len(durations)} samples collected")
        self.status.configure(text="Timing initialised. The mapping is "
                                   "ready to use.", fg=ACCENT)
        try:
            self.on_done(durations)
        finally:
            self.after(900, self.destroy)

    def _cancel(self) -> None:
        """Abandon without creating anything."""
        self._finished = True
        self._detach()
        self.destroy()

    def _detach(self) -> None:
        # Stop following the camera first: a watcher left registered on a
        # destroyed dialog would be called on the next state change.
        try:
            self.studio.unwatch_camera(self._on_camera_state)
        except Exception:
            pass
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        engine = getattr(self.studio, "engine", None)
        if engine is not None:
            engine.probe = None
