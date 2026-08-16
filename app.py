"""Gesture Studio — visual gesture-to-action binding for the hand cursor.

A standalone tkinter tool that reads ./gestures/*.png, lets you bind any
pose or any pair of poses to any OS action, and writes the result to
gesture_config.json for gesture_fsm.GestureFSM to execute.

Run it with:  python app.py

Nothing here talks to the camera.  The GUI's only job is to produce a
config file; the engine's only job is to consume one.  That split is why
you can redesign a whole control scheme without restarting the tracker.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
import uuid
from tkinter import font as tkfont
from tkinter import messagebox, ttk

try:
    from PIL import (Image, ImageDraw, ImageFont, ImageOps, ImageStat,
                     ImageTk)
except ImportError:                                       # pragma: no cover
    sys.exit("Pillow is required:  pip install Pillow")

import gesture_fsm
from gesture_fsm import (ACTION_MACROS, ACTIONS, CONFIG_PATH, DOUBLE_CLICK,
                         DRAG_START, DRAG_STOP, KEYBOARD_MACRO, LEFT_CLICK,
                         MIDDLE_CLICK, MOUSE_ACTIONS, RIGHT_CLICK,
                         SENSITIVITY_MAX, SENSITIVITY_MIN, SENSITIVITY_STEP,
                         ActionExecutor, default_config, load_config,
                         save_config, validate_macro)

# ─── Palette ────────────────────────────────────────────────────────────────

BG = "#18181b"          # window
CARD = "#27272a"        # panel cards
FIELD = "#1f1f23"       # entry / list interiors
BORDER = "#3f3f46"
HOVER = "#323238"
TEXT = "#f4f4f5"
MUTED = "#a1a1aa"
FAINT = "#71717a"
INK = "#e4e4e7"         # gesture line art, re-inked for the dark theme
ACCENT = "#10b981"      # green — primary
CYAN = "#06b6d4"        # secondary / semantic path
DANGER = "#f43f5e"          # destructive text / status
DANGER_FILL = "#ef4444"     # destructive button face
DANGER_HOVER = "#dc2626"
ON_DANGER = "#fef2f2"
AMBER = "#f59e0b"
ON_ACCENT = "#0b0f0d"

PAD = 14
THUMB = 76
SLOT = 84

GESTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gestures")

# ─── Classifier vocabularies ────────────────────────────────────────────────
# Purely informational: the engine accepts any string, but a rule bound to a
# pose no classifier emits can never fire, and silently producing one of
# those is the single most confusing thing this tool could do.  So the
# catalog labels where each pose comes from.
#
# GEOMETRIC is detect_gesture()'s output — one classification per captured
# frame, which is why clicks hang off it.  SEMANTIC is the YOLO branch's
# class list, an order of magnitude slower and reserved for macros.

GEOMETRIC = ("point", "peace", "grip", "open")

SEMANTIC = (
    "grabbing", "grip", "holy", "point", "call", "three3", "timeout",
    "xsign", "hand_heart", "hand_heart2", "little_finger", "middle_finger",
    "take_picture", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted", "three_gun",
    "thumb_index", "thumb_index2", "no_gesture",
)

# The gestures/ folder spells three poses with "_inverse" where the model
# reports "_inverted".  A config carrying the folder's spelling would never
# match a prediction, so the catalog maps filenames onto the model's names
# and shows the canonical one.
_SPELLING = {
    "peace_inverse": "peace_inverted",
    "stop_inverse": "stop_inverted",
    "two_up_inverse": "two_up_inverted",
}

TRANSITION = "transition"
HOLD = "hold"


def pretty(label: str) -> str:
    """'little_finger' -> 'Little Finger'."""
    return " ".join(part.capitalize() for part in str(label).split("_"))


def canonical(stem: str) -> str:
    """Fold a filename stem onto the label a classifier actually emits."""
    stem = stem.strip().lower()
    return _SPELLING.get(stem, stem)


def source_of(label: str) -> str:
    """Which classifier can produce this pose."""
    in_geo = label in GEOMETRIC
    in_sem = label in SEMANTIC
    if in_geo and in_sem:
        return "any"
    if in_geo:
        return "geometry"
    if in_sem:
        return "semantic"
    return "unknown"


def rule_source(labels) -> str:
    """The source a rule should declare, given the poses it uses."""
    kinds = {source_of(l) for l in labels if l}
    if kinds == {"geometry"} or kinds == {"geometry", "any"}:
        return "geometry"
    if kinds == {"semantic"} or kinds == {"semantic", "any"}:
        return "semantic"
    return "any"


# ─── Gesture catalog ────────────────────────────────────────────────────────

class GestureLibrary:
    """Scans gestures/ and decodes thumbnails off the main thread.

    Thirty large PNGs take a noticeable moment to decode, and doing it
    inline would freeze the window before it had finished drawing.  The
    worker hands back PIL images; the Tk side turns them into PhotoImages,
    because those may only be created on the thread running the mainloop.
    """

    def __init__(self, directory: str = GESTURE_DIR) -> None:
        self.directory = directory
        self.labels = []              # canonical, sorted
        self._stem_for = {}           # canonical -> file stem
        self._path_for = {}           # canonical -> full path
        self._results = queue.Queue()
        self._scan()

    def _scan(self) -> None:
        found = {}
        if os.path.isdir(self.directory):
            for name in sorted(os.listdir(self.directory)):
                stem, ext = os.path.splitext(name)
                if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                label = canonical(stem)
                found[label] = os.path.join(self.directory, name)
                self._stem_for[label] = stem

        # Every pose either classifier can emit belongs in the catalog,
        # with or without artwork.  The folder ships 25 PNGs against a
        # 34-class model plus the geometric set, and seeding from the
        # vocabularies rather than the directory is what keeps the missing
        # nine — "timeout" among them, which the default config binds —
        # bindable instead of invisible.  They draw as lettered tiles.
        for label in GEOMETRIC + SEMANTIC:
            if label == "no_gesture":
                continue          # the absence sentinel, not a pose
            found.setdefault(label, None)

        self._path_for = found
        self.labels = sorted(found)

    def start_loading(self, size: int = THUMB) -> None:
        worker = threading.Thread(target=self._load_all, args=(size,),
                                  daemon=True)
        worker.start()

    def _load_all(self, size: int) -> None:
        for label in self.labels:
            path = self._path_for.get(label)
            try:
                image = (self._decode(path, size) if path
                         else placeholder(label, size))
            except Exception:
                image = placeholder(label, size)
            self._results.put((label, image))
        self._results.put((None, None))

    @staticmethod
    def _decode(path: str, size: int) -> "Image.Image":
        with Image.open(path) as raw:
            raw.load()
            image = raw.convert("RGBA")
        image.thumbnail((size, size), Image.LANCZOS)

        if image.getchannel("A").getextrema()[0] > 250:
            grey = image.convert("L")
            # The catalog art is black line work on white paper.  Dropped
            # onto a dark panel unchanged it reads as a glaring white card
            # with the drawing lost inside it, so the luminance becomes an
            # alpha mask and the strokes are re-inked in the theme's
            # foreground: the hand survives, the paper does not.
            if ImageStat.Stat(grey).mean[0] > 160:
                mask = ImageOps.autocontrast(ImageOps.invert(grey))
                inked = Image.new("RGBA", image.size, INK)
                inked.putalpha(mask)
                image = inked

        # Returned with alpha intact so Tk composites it against whichever
        # surface it lands on — tile, socket, or hover highlight — rather
        # than baking in one background that is wrong on the other two.
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(image, ((size - image.width) // 2,
                             (size - image.height) // 2))
        return canvas

    def drain(self):
        """Yield (label, PIL image) pairs decoded since the last call."""
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                return
            yield item


def placeholder(label: str, size: int) -> "Image.Image":
    """A labelled tile for poses with no PNG, so the grid stays uniform."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([2, 2, size - 3, size - 3], radius=10,
                           outline=BORDER, width=2)
    initials = "".join(part[0] for part in label.split("_")[:2]).upper()
    try:
        font = ImageFont.truetype("segoeui.ttf", int(size * 0.34))
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), initials, font=font)
    draw.text(((size - box[2] + box[0]) / 2, (size - box[3] + box[1]) / 2),
              initials, fill=MUTED, font=font)
    return image


# ─── Small themed widgets ───────────────────────────────────────────────────

class Card(tk.Frame):
    """A panel with a hairline border, the unit the layout is built from."""

    def __init__(self, master, **kwargs):
        super().__init__(master, bg=CARD, highlightbackground=BORDER,
                         highlightcolor=BORDER, highlightthickness=1,
                         bd=0, **kwargs)


class FlatButton(tk.Label):
    """A label that behaves like a button.

    ttk.Button under clam still draws a themed bevel that reads as a
    different era from the rest of this window, and the amount of styling
    needed to hide it exceeds the amount of code needed to skip it.
    """

    def __init__(self, master, text, command, *, kind="ghost", width=None):
        palettes = {
            "accent": (ACCENT, ON_ACCENT, "#0ea472"),
            "cyan":   (CYAN, ON_ACCENT, "#0596b4"),
            "ghost":  (CARD, TEXT, HOVER),
            "danger": (CARD, DANGER, "#3a1620"),
            # Filled rather than outlined: this is the only irreversible
            # control on the panel, and it should not read as a peer of the
            # four reversible ones sitting beside it.
            "danger-solid": (DANGER_FILL, ON_DANGER, DANGER_HOVER),
        }
        self._bg, self._fg, self._hover = palettes.get(kind, palettes["ghost"])
        super().__init__(master, text=text, bg=self._bg, fg=self._fg,
                         padx=14, pady=7, cursor="hand2",
                         font=("Segoe UI", 9, "bold"))
        if kind in ("ghost", "danger"):
            self.configure(highlightbackground=BORDER, highlightthickness=1)
        if width:
            self.configure(width=width)
        self._command = command
        self._enabled = True
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", self._click)

    def _enter(self, _event=None):
        if self._enabled:
            self.configure(bg=self._hover)

    def _leave(self, _event=None):
        self.configure(bg=self._bg if self._enabled else CARD)

    def _click(self, _event=None):
        if self._enabled:
            self._command()

    def set_enabled(self, state: bool) -> None:
        self._enabled = bool(state)
        self.configure(fg=self._fg if state else FAINT,
                       bg=self._bg if state else CARD,
                       cursor="hand2" if state else "arrow")


class Segmented(tk.Frame):
    """Two-option segmented control — the trigger-type switch."""

    def __init__(self, master, options, command):
        super().__init__(master, bg=FIELD, highlightbackground=BORDER,
                         highlightthickness=1)
        self._command = command
        self._buttons = {}
        self.value = options[0][0]
        for key, text in options:
            label = tk.Label(self, text=text, bg=FIELD, fg=MUTED,
                             padx=18, pady=6, cursor="hand2",
                             font=("Segoe UI", 9, "bold"))
            label.pack(side="left")
            label.bind("<Button-1>", lambda _e, k=key: self.select(k))
            self._buttons[key] = label
        self.select(self.value, notify=False)

    def select(self, key, notify=True):
        self.value = key
        for name, label in self._buttons.items():
            active = name == key
            label.configure(bg=ACCENT if active else FIELD,
                            fg=ON_ACCENT if active else MUTED)
        if notify:
            self._command(key)


class Toggle(tk.Frame):
    """Checkbox with a drawn indicator, so it matches the rest of the theme."""

    def __init__(self, master, text, value=False, command=None):
        super().__init__(master, bg=CARD)
        self._value = bool(value)
        self._command = command
        self._box = tk.Label(self, width=2, bg=CARD, fg=ON_ACCENT,
                             font=("Segoe UI", 9, "bold"), cursor="hand2",
                             highlightbackground=BORDER, highlightthickness=1)
        self._box.pack(side="left")
        self._text = tk.Label(self, text=text, bg=CARD, fg=MUTED,
                              font=("Segoe UI", 9), padx=8, cursor="hand2")
        self._text.pack(side="left")
        for widget in (self._box, self._text):
            widget.bind("<Button-1>", self._flip)
        self._render()

    def _flip(self, _event=None):
        self._value = not self._value
        self._render()
        if self._command:
            self._command(self._value)

    def _render(self):
        self._box.configure(text="✓" if self._value else " ",
                            bg=ACCENT if self._value else FIELD)
        self._text.configure(fg=TEXT if self._value else MUTED)

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        self._value = bool(value)
        self._render()


class Slot(tk.Frame):
    """A pose socket in the rule builder.

    Clicking one arms it; the next gesture picked from the catalog lands
    here.  Arming is shown with the accent border rather than a separate
    "which field am I editing" control, because the socket is already the
    thing being pointed at.
    """

    def __init__(self, master, title, on_arm):
        super().__init__(master, bg=CARD)
        self.label = None
        self._on_arm = on_arm
        self._armed = False

        tk.Label(self, text=title, bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")

        self._box = tk.Frame(self, bg=FIELD, width=SLOT, height=SLOT,
                             highlightbackground=BORDER, highlightthickness=2,
                             cursor="hand2")
        self._box.pack_propagate(False)
        self._box.pack(pady=(4, 0))

        self._image = tk.Label(self._box, bg=FIELD, fg=FAINT, text="＋",
                               font=("Segoe UI", 20), cursor="hand2")
        self._image.pack(expand=True)

        self._caption = tk.Label(self, text="pick a pose", bg=CARD, fg=FAINT,
                                 font=("Segoe UI", 8), wraplength=SLOT + 16)
        self._caption.pack(pady=(4, 0))

        for widget in (self._box, self._image, self._caption):
            widget.bind("<Button-1>", lambda _e: self._on_arm(self))

    def set_armed(self, armed: bool) -> None:
        self._armed = armed
        self._box.configure(highlightbackground=ACCENT if armed else BORDER)

    def set_pose(self, label, photo=None) -> None:
        self.label = label
        if label is None:
            self._image.configure(image="", text="＋", fg=FAINT)
            self._image.image = None
            self._caption.configure(text="pick a pose", fg=FAINT)
            return
        if photo is not None:
            self._image.configure(image=photo, text="")
            self._image.image = photo
        else:
            self._image.configure(image="", text=pretty(label)[:2].upper(),
                                  fg=TEXT)
            self._image.image = None
        kind = source_of(label)
        colour = {"geometry": ACCENT, "semantic": CYAN,
                  "any": ACCENT}.get(kind, AMBER)
        self._caption.configure(text=pretty(label), fg=colour)


# ─── The application ────────────────────────────────────────────────────────

class GestureStudio(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title("Gesture Studio — gesture to action binding")
        self.configure(bg=BG)
        # Opens at a comfortable size but clamps to the work area, because a
        # 900-px-tall default on a 1536×793 desktop puts the mapping panel's
        # buttons below the screen edge.
        self.update_idletasks()
        width = min(1340, self.winfo_screenwidth() - 80)
        height = min(900, self.winfo_screenheight() - 120)
        self.geometry(f"{width}x{height}")
        # The table absorbs the squeeze down to this; below it the builder's
        # field grid starts clipping rather than reflowing.
        self.minsize(1120, 620)

        self.library = GestureLibrary()
        self.thumbs = {}              # label -> PhotoImage (kept alive here)
        self.rules = []               # list of rule dicts, GUI's source of truth
        self.settings = dict(gesture_fsm.DEFAULT_SETTINGS)
        self.editing_id = None
        self.armed_slot = None
        self._executor = None
        self._test_job = None
        self._dirty = False
        self._loading = False         # suppresses builder auto-defaults
        self._booting = True          # suppresses dirty-marking until shown
        self._undo_stack = []         # deletions, newest last

        self._init_fonts()
        self._init_styles()
        self._build()

        self.library.start_loading()
        self.after(60, self._pump_thumbnails)

        self._load_from_disk()
        self._booting = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── chrome ──────────────────────────────────────────────────────────

    def _init_fonts(self) -> None:
        family = "Segoe UI" if "Segoe UI" in tkfont.families() else "TkDefaultFont"
        self.f_title = (family, 17, "bold")
        self.f_sub = (family, 9)
        self.f_head = (family, 10, "bold")
        self.f_body = (family, 9)
        self.f_mono = ("Consolas", 9)

    def _init_styles(self) -> None:
        style = ttk.Style(self)
        # clam is the only bundled theme that honours background overrides
        # on Windows; without this the fields stay system-grey.
        style.theme_use("clam")

        style.configure("Dark.TCombobox", fieldbackground=FIELD,
                        background=FIELD, foreground=TEXT, arrowcolor=MUTED,
                        bordercolor=BORDER, lightcolor=FIELD,
                        darkcolor=FIELD, insertcolor=TEXT, padding=6)
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", FIELD)],
                  foreground=[("readonly", TEXT)],
                  bordercolor=[("focus", ACCENT)])
        # The dropdown is a native Listbox and is styled through the option
        # database rather than the ttk style.
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", ON_ACCENT)
        self.option_add("*TCombobox*Listbox.font", self.f_body)

        style.configure("Dark.TEntry", fieldbackground=FIELD, foreground=TEXT,
                        bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD,
                        insertcolor=TEXT, padding=6)
        style.map("Dark.TEntry", bordercolor=[("focus", ACCENT)])

        style.configure("Dark.TSpinbox", fieldbackground=FIELD,
                        background=FIELD, foreground=TEXT, arrowcolor=MUTED,
                        bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD,
                        insertcolor=TEXT, padding=5)
        style.map("Dark.TSpinbox", bordercolor=[("focus", ACCENT)])

        # clam draws the Scale's slider with `background` and the groove with
        # `troughcolor`; the light/dark pair removes the bevel that would
        # otherwise read as a raised 3-D nub against the flat theme.
        style.configure("Dark.Horizontal.TScale", background=ACCENT,
                        troughcolor=FIELD, bordercolor=BORDER,
                        lightcolor=ACCENT, darkcolor=ACCENT)
        style.map("Dark.Horizontal.TScale",
                  background=[("active", "#0ea472")])

        style.configure("Dark.Vertical.TScrollbar", background=BORDER,
                        troughcolor=BG, bordercolor=BG, arrowcolor=MUTED,
                        lightcolor=BORDER, darkcolor=BORDER)
        style.map("Dark.Vertical.TScrollbar",
                  background=[("active", FAINT)])

        style.configure("Dark.Treeview", background=FIELD,
                        fieldbackground=FIELD, foreground=TEXT,
                        bordercolor=BORDER, lightcolor=FIELD,
                        darkcolor=FIELD, rowheight=30, font=self.f_body)
        style.configure("Dark.Treeview.Heading", background=CARD,
                        foreground=MUTED, relief="flat", font=self.f_body,
                        padding=6)
        style.map("Dark.Treeview.Heading",
                  background=[("active", HOVER)], foreground=[("active", TEXT)])
        style.map("Dark.Treeview",
                  background=[("selected", "#14532d")],
                  foreground=[("selected", TEXT)])
        style.layout("Dark.Treeview", [
            ("Dark.Treeview.treearea", {"sticky": "nswe"}),
        ])

    def _build(self) -> None:
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        self._build_header()

        left = tk.Frame(self, bg=BG)
        left.grid(row=1, column=0, sticky="ns", padx=(PAD, 0), pady=(0, 0))
        # Settings first, against the bottom: the catalog is the expanding
        # half, and anything packed after an expanding widget loses its
        # space on a short window.
        self._build_settings(left)
        self._build_catalog(left)

        right = tk.Frame(self, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=PAD)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_builder(right)
        self._build_table(right)

        self._build_hotkeys()
        self._build_status()

    # ── general settings ────────────────────────────────────────────────

    def _build_settings(self, master) -> None:
        card = Card(master)
        card.pack(side="bottom", fill="x", pady=(PAD, 0))

        tk.Label(card, text="CURSOR SETTINGS", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(anchor="w", padx=PAD, pady=(PAD, 2))
        tk.Label(card, text="Read by hand_cursor_2.py at startup.",
                 bg=CARD, fg=FAINT, font=("Segoe UI", 8)).pack(
            anchor="w", padx=PAD)

        # ── sensitivity ────────────────────────────────────────────────
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=PAD, pady=(10, 0))
        tk.Label(row, text="SPEED / SENSITIVITY", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        self.sens_readout = tk.Label(row, text="1.4×", bg=CARD, fg=ACCENT,
                                     font=("Consolas", 10, "bold"))
        self.sens_readout.pack(side="right")

        self.sensitivity_var = tk.DoubleVar(value=1.4)
        self.sens_scale = ttk.Scale(
            card, from_=SENSITIVITY_MIN, to=SENSITIVITY_MAX,
            variable=self.sensitivity_var, orient="horizontal",
            style="Dark.Horizontal.TScale", command=self._on_sensitivity)
        self.sens_scale.pack(fill="x", padx=PAD, pady=(4, 0))

        scale_ends = tk.Frame(card, bg=CARD)
        scale_ends.pack(fill="x", padx=PAD)
        tk.Label(scale_ends, text=f"{SENSITIVITY_MIN:.1f}", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 7)).pack(side="left")
        tk.Label(scale_ends, text=f"{SENSITIVITY_MAX:.1f}", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 7)).pack(side="right")

        # ── direction ──────────────────────────────────────────────────
        self.mirror_toggle = Toggle(card, "Mirror camera  (m)", value=True,
                                    command=lambda _v: self._on_direction())
        self.mirror_toggle.pack(anchor="w", padx=PAD, pady=(12, 0))

        self.invert_toggle = Toggle(card, "Invert X-axis  (i)", value=False,
                                    command=lambda _v: self._on_direction())
        self.invert_toggle.pack(anchor="w", padx=PAD, pady=(4, 0))

        self.direction_note = tk.Label(
            card, text="", bg=CARD, fg=AMBER, font=("Segoe UI", 8),
            wraplength=232, justify="left", anchor="w")
        self.direction_note.pack(fill="x", padx=PAD, pady=(6, PAD))

        self._on_direction()

    def _on_sensitivity(self, raw) -> None:
        """Snap the slider to the 0.1 step and mirror it into the readout.

        Re-setting the variable re-enters this callback exactly once: the
        second pass sees an already-rounded value, changes nothing, and the
        recursion stops there.
        """
        value = round(float(raw), 1)
        if abs(value - self.sensitivity_var.get()) > 1e-9:
            self.sensitivity_var.set(value)
        if hasattr(self, "sens_readout"):
            self.sens_readout.configure(text=f"{value:.1f}×")
            self._mark_dirty()

    def _on_direction(self) -> None:
        """Warn when the two reflections cancel each other out.

        Mirroring the preview already puts the landmarks in display space,
        so inverting the control maths as well is a second reversal: the
        picture looks right and the cursor runs backwards.  Both off is the
        same trap.  The panel says so rather than silently correcting it,
        because which of the two you want depends on the camera.
        """
        if not hasattr(self, "direction_note"):
            return
        mirrored = self.mirror_toggle.get()
        inverted = self.invert_toggle.get()

        if mirrored == inverted:
            self.direction_note.configure(
                text="⚠  Both on (or both off) cancel out — the cursor will "
                     "run backwards. Turn exactly one on.", fg=AMBER)
        elif mirrored:
            self.direction_note.configure(
                text="✓  Mirrored preview, cursor follows your hand.",
                fg=ACCENT)
        else:
            self.direction_note.configure(
                text="✓  Raw preview, cursor follows your hand.", fg=ACCENT)
        self._mark_dirty()

    # ── hotkey footer ───────────────────────────────────────────────────

    def _build_hotkeys(self) -> None:
        """The same hotkeys the camera overlay prints, mirrored here.

        Reference only — these keys are read by cv2.waitKey() in the
        tracker's preview window, so they do nothing while this window has
        focus.  The panel above is how you set the same things from here.
        """
        bar = tk.Frame(self, bg=CARD, highlightbackground=BORDER,
                       highlightthickness=1)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew",
                 padx=PAD, pady=(PAD, 0))

        inner = tk.Frame(bar, bg=CARD)
        inner.pack(padx=PAD, pady=8)

        tk.Label(inner, text="CAMERA WINDOW HOTKEYS", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 16))

        for caps, text in ((("+", "−"), "Speed"),
                           (("M",), "Mirror"),
                           (("I",), "Invert X"),
                           (("Q",), "Quit")):
            group = tk.Frame(inner, bg=CARD)
            group.pack(side="left", padx=(0, 22))
            for cap in caps:
                tk.Label(group, text=cap, bg=FIELD, fg=TEXT,
                         font=("Consolas", 9, "bold"), width=3,
                         padx=2, pady=2, highlightbackground=BORDER,
                         highlightthickness=1).pack(side="left", padx=(0, 3))
            tk.Label(group, text=text, bg=CARD, fg=MUTED,
                     font=("Segoe UI", 9)).pack(side="left", padx=(5, 0))

    def _build_header(self) -> None:
        bar = tk.Frame(self, bg=BG)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=PAD, pady=(PAD, 10))
        bar.columnconfigure(0, weight=1)

        titles = tk.Frame(bar, bg=BG)
        titles.grid(row=0, column=0, sticky="w")
        tk.Label(titles, text="Gesture Studio", bg=BG, fg=TEXT,
                 font=self.f_title).pack(anchor="w")
        tk.Label(titles,
                 text="Bind any pose, or any pair of poses, to any action. "
                      "Saved to gesture_config.json.",
                 bg=BG, fg=MUTED, font=self.f_sub).pack(anchor="w")

        buttons = tk.Frame(bar, bg=BG)
        buttons.grid(row=0, column=1, sticky="e")
        FlatButton(buttons, "Load Defaults", self._load_defaults).pack(
            side="left", padx=(0, 8))
        FlatButton(buttons, "Reload File", self._load_from_disk).pack(
            side="left", padx=(0, 8))
        self.save_button = FlatButton(buttons, "Save Config", self._save,
                                      kind="accent")
        self.save_button.pack(side="left")

    # ── catalog ─────────────────────────────────────────────────────────

    def _build_catalog(self, master) -> None:
        card = Card(master)
        card.pack(fill="both", expand=True)

        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x", padx=PAD, pady=(PAD, 6))
        tk.Label(head, text="GESTURE CATALOG", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(anchor="w")
        self.catalog_count = tk.Label(head, text="scanning…", bg=CARD,
                                      fg=FAINT, font=self.f_sub)
        self.catalog_count.pack(anchor="w")

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._render_catalog())
        search = ttk.Entry(card, textvariable=self.search_var,
                           style="Dark.TEntry", font=self.f_body, width=26)
        search.pack(fill="x", padx=PAD, pady=(0, 8))

        # The per-tile dots are unlabelled by design now — the source is
        # still spelled out in the rule preview line and the table's Path
        # column, so the legend was the third place saying the same thing.
        holder = tk.Frame(card, bg=CARD)
        holder.pack(side="top", fill="both", expand=True,
                    padx=(PAD, 4), pady=(0, PAD))

        self.canvas = tk.Canvas(holder, bg=CARD, highlightthickness=0,
                                width=250, height=180, bd=0)
        scroll = ttk.Scrollbar(holder, orient="vertical",
                               style="Dark.Vertical.TScrollbar",
                               command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.grid_frame = tk.Frame(self.canvas, bg=CARD)
        self._grid_window = self.canvas.create_window(
            (0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self._grid_window,
                                                width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_wheel(self, event) -> None:
        widget = self.winfo_containing(event.x_root, event.y_root)
        # Only scroll the catalog when the pointer is actually over it;
        # otherwise the wheel would hijack the mappings table too.
        while widget is not None:
            if widget is self.canvas or widget is self.grid_frame:
                self.canvas.yview_scroll(-int(event.delta / 120), "units")
                return
            widget = getattr(widget, "master", None)

    def _pump_thumbnails(self) -> None:
        arrived = False
        for label, image in self.library.drain():
            if label is None:
                self.catalog_count.configure(
                    text=f"{len(self.library.labels)} gestures")
                self._render_catalog()
                return
            self.thumbs[label] = ImageTk.PhotoImage(image)
            arrived = True
        if arrived:
            self._render_catalog()
            self.catalog_count.configure(
                text=f"{len(self.thumbs)} / {len(self.library.labels)} loaded")
        self.after(80, self._pump_thumbnails)

    def _render_catalog(self) -> None:
        for child in self.grid_frame.winfo_children():
            child.destroy()

        needle = self.search_var.get().strip().lower()
        labels = [l for l in self.library.labels
                  if not needle or needle in l.replace("_", " ")
                  or needle in l]

        columns = 2
        for index, label in enumerate(labels):
            tile = self._make_tile(self.grid_frame, label)
            tile.grid(row=index // columns, column=index % columns,
                      padx=4, pady=4, sticky="nsew")
        for column in range(columns):
            self.grid_frame.columnconfigure(column, weight=1)

        if not labels:
            tk.Label(self.grid_frame, text="no match", bg=CARD, fg=FAINT,
                     font=self.f_body).grid(row=0, column=0, pady=20)

    def _make_tile(self, master, label: str) -> tk.Frame:
        tile = tk.Frame(master, bg=FIELD, highlightbackground=BORDER,
                        highlightthickness=1, cursor="hand2", padx=4, pady=6)

        photo = self.thumbs.get(label)
        image = tk.Label(tile, bg=FIELD, image=photo,
                         text="" if photo else "…",
                         fg=FAINT, width=0 if photo else 8,
                         height=0 if photo else 4)
        if photo:
            image.image = photo
        image.pack()

        name = tk.Label(tile, text=pretty(label), bg=FIELD, fg=TEXT,
                        font=("Segoe UI", 8), wraplength=THUMB + 24)
        name.pack(pady=(4, 0))

        kind = source_of(label)
        colour = {"geometry": ACCENT, "semantic": CYAN,
                  "any": ACCENT}.get(kind, AMBER)
        dot = tk.Label(tile, text="●", bg=FIELD, fg=colour,
                       font=("Segoe UI", 7))
        dot.pack()

        widgets = (tile, image, name, dot)
        for widget in widgets:
            widget.bind("<Button-1>", lambda _e, l=label: self._pick(l))
            widget.bind("<Enter>", lambda _e, w=widgets: [
                x.configure(bg=HOVER) for x in w])
            widget.bind("<Leave>", lambda _e, w=widgets: [
                x.configure(bg=FIELD) for x in w])
        return tile

    def _pick(self, label: str) -> None:
        """A catalog click lands in the armed socket, then advances.

        Membership in _visible_slots() decides which socket is eligible,
        not winfo_ismapped(): the trigger switch repacks these widgets, and
        Tk does not update the mapped flag until the geometry manager next
        runs, so asking the widget would mis-route every pick made between
        flipping the trigger and the next idle cycle.
        """
        visible = self._visible_slots()
        slot = self.armed_slot if self.armed_slot in visible else visible[0]
        slot.set_pose(label, self.thumbs.get(label))
        self._arm(visible[(visible.index(slot) + 1) % len(visible)])
        self._refresh_preview()

    # ── rule builder ────────────────────────────────────────────────────

    def _build_builder(self, master) -> None:
        card = Card(master)
        card.grid(row=0, column=0, sticky="ew", pady=(0, PAD))

        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x", padx=PAD, pady=(PAD, 0))
        self.builder_title = tk.Label(head, text="CREATE MAPPING", bg=CARD,
                                      fg=TEXT, font=self.f_head)
        self.builder_title.pack(side="left")
        self.trigger = Segmented(head, [(TRANSITION, "Transition"),
                                        (HOLD, "Static Hold")],
                                 self._on_trigger_change)
        self.trigger.pack(side="right")

        body = tk.Frame(card, bg=CARD)
        body.pack(fill="x", padx=PAD, pady=PAD)

        # Sockets ------------------------------------------------------
        sockets = tk.Frame(body, bg=CARD)
        sockets.pack(side="left", padx=(0, PAD))

        self.slot_from = Slot(sockets, "FROM POSE", self._arm)
        self.slot_arrow = tk.Label(sockets, text="→", bg=CARD, fg=ACCENT,
                                   font=("Segoe UI", 18, "bold"))
        self.slot_to = Slot(sockets, "TO POSE", self._arm)
        self.slot_pose = Slot(sockets, "POSE", self._arm)

        self.slot_from.pack(side="left")
        self.slot_arrow.pack(side="left", padx=10, pady=(18, 0))
        self.slot_to.pack(side="left")
        self.slot_pose.pack(side="left")

        # Fields -------------------------------------------------------
        fields = tk.Frame(body, bg=CARD)
        fields.pack(side="left", fill="both", expand=True)
        fields.columnconfigure(1, weight=1)
        fields.columnconfigure(3, weight=1)

        def caption(text, row, column):
            tk.Label(fields, text=text, bg=CARD, fg=FAINT,
                     font=("Segoe UI", 8, "bold")).grid(
                row=row, column=column, sticky="w", padx=(0, 8), pady=(0, 2))

        caption("ACTION", 0, 0)
        self.action_var = tk.StringVar(value=LEFT_CLICK)
        self.action_box = ttk.Combobox(
            fields, textvariable=self.action_var, values=list(ACTIONS),
            state="readonly", style="Dark.TCombobox", font=self.f_body,
            width=22)
        self.action_box.grid(row=1, column=0, columnspan=2, sticky="ew",
                             pady=(0, 10))
        self.action_box.bind("<<ComboboxSelected>>",
                             lambda _e: self._on_action_change())

        caption("NAME (OPTIONAL)", 0, 2)
        self.name_var = tk.StringVar()
        ttk.Entry(fields, textvariable=self.name_var, style="Dark.TEntry",
                  font=self.f_body).grid(row=1, column=2, columnspan=2,
                                         sticky="ew", padx=(12, 0),
                                         pady=(0, 10))

        self.timing_caption = tk.Label(fields, text="WITHIN (SEC)", bg=CARD,
                                       fg=FAINT, font=("Segoe UI", 8, "bold"))
        self.timing_caption.grid(row=2, column=0, sticky="w", pady=(0, 2))
        self.timing_var = tk.StringVar(value="0.80")
        # A Spinbox's `command` fires for the arrows only, so a typed value
        # would leave the preview showing the previous number.  The trace
        # catches both routes.
        self.timing_var.trace_add("write", lambda *_: self._refresh_preview())
        self.timing_box = ttk.Spinbox(fields, from_=0.0, to=10.0, increment=0.05,
                                      textvariable=self.timing_var, width=8,
                                      style="Dark.TSpinbox", font=self.f_body)
        self.timing_box.grid(row=3, column=0, sticky="w")

        tk.Label(fields, text="COOLDOWN (SEC)", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).grid(row=2, column=1,
                                                    sticky="w", pady=(0, 2))
        self.cooldown_var = tk.StringVar(value="0.35")
        self.cooldown_var.trace_add("write",
                                    lambda *_: self._refresh_preview())
        ttk.Spinbox(fields, from_=0.0, to=30.0, increment=0.05,
                    textvariable=self.cooldown_var, width=8,
                    style="Dark.TSpinbox", font=self.f_body).grid(
            row=3, column=1, sticky="w")

        self.keys_caption = tk.Label(fields, text="KEY COMBINATION", bg=CARD,
                                     fg=FAINT, font=("Segoe UI", 8, "bold"))
        self.keys_caption.grid(row=2, column=2, columnspan=2, sticky="w",
                               padx=(12, 0), pady=(0, 2))
        self.keys_var = tk.StringVar(value="win+d")
        self.keys_var.trace_add("write", lambda *_: self._refresh_preview())
        self.keys_box = ttk.Entry(fields, textvariable=self.keys_var,
                                  style="Dark.TEntry", font=self.f_mono)
        self.keys_box.grid(row=3, column=2, columnspan=2, sticky="ew",
                           padx=(12, 0))

        toggles = tk.Frame(fields, bg=CARD)
        toggles.grid(row=4, column=0, columnspan=4, sticky="w", pady=(12, 0))
        self.promote_toggle = Toggle(
            toggles, "Repeat inside the double-click window fires DOUBLE_CLICK",
            command=self._on_promote_toggle)
        self.promote_toggle.pack(anchor="w")
        self.repeat_toggle = Toggle(
            toggles, "Repeat while held (for scroll and volume)",
            command=lambda _v: self._refresh_preview())

        # Preview + actions --------------------------------------------
        footer = tk.Frame(card, bg=CARD)
        footer.pack(fill="x", padx=PAD, pady=(0, PAD))

        self.preview = tk.Label(footer, text="", bg=CARD, fg=MUTED,
                                font=self.f_mono, anchor="w",
                                justify="left", wraplength=940)
        self.preview.pack(side="left", fill="x", expand=True)

        self.clear_button = FlatButton(footer, "Clear", self._clear_builder)
        self.clear_button.pack(side="right", padx=(8, 0))
        self.commit_button = FlatButton(footer, "Add Mapping", self._commit,
                                        kind="accent")
        self.commit_button.pack(side="right")

        self._arm(self.slot_from)
        self._on_trigger_change(TRANSITION)

    def _visible_slots(self):
        if self.trigger.value == TRANSITION:
            return [self.slot_from, self.slot_to]
        return [self.slot_pose]

    def _arm(self, slot) -> None:
        self.armed_slot = slot
        for candidate in (self.slot_from, self.slot_to, self.slot_pose):
            candidate.set_armed(candidate is slot)

    def _on_trigger_change(self, kind) -> None:
        if kind == TRANSITION:
            self.slot_pose.pack_forget()
            self.slot_from.pack(side="left")
            self.slot_arrow.pack(side="left", padx=10, pady=(18, 0))
            self.slot_to.pack(side="left")
            self.timing_caption.configure(text="WITHIN (SEC)")
        else:
            self.slot_from.pack_forget()
            self.slot_arrow.pack_forget()
            self.slot_to.pack_forget()
            self.slot_pose.pack(side="left")
            self.timing_caption.configure(text="HOLD FOR (SEC)")

        # The box means two different things in the two modes, so carrying
        # a value across the switch would silently reinterpret it.  Skipped
        # while loading a saved rule, which brings its own value.
        if not self._loading:
            self.timing_var.set("0.80" if kind == TRANSITION else "0.40")

        self._arm(self._visible_slots()[0])
        self._on_action_change()

    def _on_action_change(self) -> None:
        action = self.action_var.get()
        is_transition = self.trigger.value == TRANSITION

        if action == KEYBOARD_MACRO:
            self.keys_box.configure(state="normal")
            self.keys_caption.configure(text="KEY COMBINATION", fg=FAINT)
        else:
            chord = ACTION_MACROS.get(action)
            self.keys_box.configure(state="disabled")
            self.keys_caption.configure(
                text=f"KEY COMBINATION  ({chord})" if chord
                else "KEY COMBINATION  (n/a)", fg=FAINT)

        # promote_double only means anything for a single-click action on a
        # transition; showing it elsewhere would imply it does something.
        clickable = action in (LEFT_CLICK, RIGHT_CLICK, MIDDLE_CLICK)
        if is_transition and clickable:
            self.promote_toggle.pack(anchor="w")
        else:
            self.promote_toggle.pack_forget()

        if not is_transition:
            self.repeat_toggle.pack(anchor="w")
        else:
            self.repeat_toggle.pack_forget()

        self._refresh_preview()

    def _on_promote_toggle(self, enabled: bool) -> None:
        """Keep the cooldown short enough that a double can actually land.

        The cooldown is the floor on the gap between two clicks and the
        double-click window is the ceiling; set the floor above the ceiling
        and the second click is swallowed, so the option the user just
        switched on would do nothing at all.
        """
        window = float(self.settings.get("double_click_sec", 0.8))
        try:
            cooldown = float(self.cooldown_var.get())
        except (TypeError, ValueError):
            cooldown = 0.35

        if enabled and cooldown > window * 0.25:
            self.cooldown_var.set(f"{window * 0.2:.2f}")
            self._toast(f"Cooldown lowered to {window * 0.2:.2f}s — above "
                        f"{window * 0.25:.2f}s it would swallow the second "
                        f"click.", AMBER)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        # The variable traces are live before the footer that owns the
        # preview label exists, and attaching a textvariable to a Spinbox
        # writes to it, so the first few callbacks arrive mid-construction.
        if not hasattr(self, "preview"):
            return

        rule = self._read_builder(validate=False)
        if rule is None:
            self.preview.configure(text="", fg=MUTED)
            return

        action = rule["action"]
        chord = rule.get("keys") or ACTION_MACROS.get(action, "")
        suffix = f"  [{chord}]" if chord else ""

        if rule["trigger"] == TRANSITION:
            window = rule["max_time_sec"]
            body = (f"{rule['from_state']} → {rule['to_state']} "
                    + (f"within {window:.2f}s" if window > 0
                       else "with no time limit"))
        else:
            delay = rule["hold_sec"]
            body = (f"hold {rule['pose']} for {delay:.2f}s" if delay > 0
                    else f"enter {rule['pose']}")

        note = ""
        source = rule.get("source", "any")
        if source == "semantic":
            note = "   • YOLO path (~4.7 Hz)"
        elif source == "any" and rule["trigger"] == TRANSITION:
            poses = [rule.get("from_state"), rule.get("to_state")]
            if any(source_of(p) == "unknown" for p in poses if p):
                note = "   • pose is not in a known vocabulary"

        self.preview.configure(
            text=f"{body}  ⇒  {action}{suffix}"
                 f"   • cooldown {rule['cooldown_sec']:.2f}s{note}",
            fg=AMBER if "not in a known" in note else MUTED)

    def _read_builder(self, validate=True):
        """Assemble a rule dict from the form, or None if it is incomplete."""
        kind = self.trigger.value
        action = self.action_var.get()

        def number(var, fallback):
            try:
                return max(0.0, float(var.get()))
            except (TypeError, ValueError):
                return fallback

        rule = {
            "id": self.editing_id or f"r{uuid.uuid4().hex[:8]}",
            "trigger": kind,
            "action": action,
            "enabled": True,
            "cooldown_sec": number(self.cooldown_var, 0.35),
        }

        if kind == TRANSITION:
            src, dst = self.slot_from.label, self.slot_to.label
            if not src or not dst:
                if validate:
                    self._toast("Pick both a FROM and a TO pose.", DANGER)
                return None
            if src == dst:
                if validate:
                    self._toast("A transition needs two different poses.",
                                DANGER)
                return None
            rule.update({"from_state": src, "to_state": dst,
                         "max_time_sec": number(self.timing_var, 0.8)})
            if action in (LEFT_CLICK, RIGHT_CLICK, MIDDLE_CLICK) \
                    and self.promote_toggle.get():
                rule["promote_double"] = True
            poses = [src, dst]
        else:
            pose = self.slot_pose.label
            if not pose:
                if validate:
                    self._toast("Pick a pose to hold.", DANGER)
                return None
            rule.update({"pose": pose, "hold_sec": number(self.timing_var, 0.4)})
            if self.repeat_toggle.get():
                rule["repeat"] = True
                rule["repeat_sec"] = 0.25
            poses = [pose]

        if action == KEYBOARD_MACRO:
            chord = self.keys_var.get().strip()
            if not chord:
                if validate:
                    self._toast("KEYBOARD_MACRO needs a key combination, "
                                "e.g. ctrl+shift+esc.", DANGER)
                return None
            if validate and not validate_macro(chord):
                self._toast(f"pynput does not recognise '{chord}'.", DANGER)
                return None
            rule["keys"] = chord

        name = self.name_var.get().strip()
        if name:
            rule["name"] = name

        rule["source"] = rule_source(poses)
        return rule

    def _commit(self) -> None:
        rule = self._read_builder()
        if rule is None:
            return

        clash = self._conflict(rule)
        if clash is not None:
            self._toast(f"That trigger is already bound to "
                        f"{clash['action']}.", DANGER)
            return

        if self.editing_id:
            for index, existing in enumerate(self.rules):
                if existing["id"] == self.editing_id:
                    rule["enabled"] = existing.get("enabled", True)
                    self.rules[index] = rule
                    break
            self._toast("Mapping updated.", ACCENT)
        else:
            self.rules.append(rule)
            self._toast("Mapping added.", ACCENT)

        self._mark_dirty()
        self._clear_builder()
        self._render_table()

    def _conflict(self, rule):
        """Two rules on the same trigger would both fire.  Refuse the second."""
        for existing in self.rules:
            if existing["id"] == rule["id"]:
                continue
            if existing["trigger"] != rule["trigger"]:
                continue
            if rule["trigger"] == TRANSITION:
                if (existing.get("from_state") == rule.get("from_state")
                        and existing.get("to_state") == rule.get("to_state")):
                    return existing
            elif existing.get("pose") == rule.get("pose"):
                return existing
        return None

    def _clear_builder(self) -> None:
        self.editing_id = None
        self.slot_from.set_pose(None)
        self.slot_to.set_pose(None)
        self.slot_pose.set_pose(None)
        self.name_var.set("")
        self.action_var.set(LEFT_CLICK)
        self.cooldown_var.set("0.35")
        self.timing_var.set("0.80" if self.trigger.value == TRANSITION
                            else "0.40")
        self.keys_var.set("win+d")
        self.promote_toggle.set(False)
        self.repeat_toggle.set(False)
        self.builder_title.configure(text="CREATE MAPPING")
        self.commit_button.configure(text="Add Mapping")
        self._arm(self._visible_slots()[0])
        self._on_action_change()

    # ── mapping table ───────────────────────────────────────────────────

    def _build_table(self, master) -> None:
        card = Card(master)
        card.grid(row=1, column=0, sticky="nsew")

        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x", padx=PAD, pady=(PAD, 8))
        tk.Label(head, text="ACTIVE MAPPINGS", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(side="left")
        self.table_count = tk.Label(head, text="", bg=CARD, fg=FAINT,
                                    font=self.f_sub)
        self.table_count.pack(side="left", padx=8)

        # The button row is packed BEFORE the table and against the bottom
        # edge.  Pack hands out space in packing order, so an expanding
        # widget declared first claims the whole cavity and anything packed
        # after it is pushed outside the frame and silently unmapped — which
        # is exactly what hid these buttons on a short window.  Reserving
        # their strip first makes the table absorb the remainder instead.
        buttons = tk.Frame(card, bg=CARD)
        buttons.pack(side="bottom", fill="x", padx=PAD, pady=PAD)

        FlatButton(buttons, "Edit", self._edit_selected).pack(side="left")
        FlatButton(buttons, "Toggle", self._toggle_selected).pack(
            side="left", padx=(8, 0))
        FlatButton(buttons, "Duplicate", self._duplicate_selected).pack(
            side="left", padx=(8, 0))
        FlatButton(buttons, "Test", self._test_selected, kind="cyan").pack(
            side="left", padx=(8, 0))

        FlatButton(buttons, "Delete Selected", self.delete_selected_mapping,
                   kind="danger-solid").pack(side="right")
        self.undo_button = FlatButton(buttons, "Undo Delete", self.undo_delete)
        self.undo_button.pack(side="right", padx=(0, 8))
        self.undo_button.set_enabled(False)

        holder = tk.Frame(card, bg=CARD)
        holder.pack(side="top", fill="both", expand=True, padx=PAD)

        columns = ("on", "trigger", "gestures", "action", "timing",
                   "cooldown", "path")
        # "extended" so Ctrl/Shift-click can gather several rules for one
        # deletion; the single-row operations below act on the anchor row.
        # height=6 keeps the natural request modest so the card can shrink
        # on a small screen without squeezing anything off the bottom.
        self.table = ttk.Treeview(holder, columns=columns, show="headings",
                                  style="Dark.Treeview", selectmode="extended",
                                  height=6)
        headings = {
            "on": ("", 34), "trigger": ("Trigger", 90),
            "gestures": ("Gestures", 210), "action": ("Action", 215),
            "timing": ("Timing", 100), "cooldown": ("Cooldown", 85),
            "path": ("Path", 70),
        }
        for key, (title, width) in headings.items():
            self.table.heading(key, text=title)
            self.table.column(key, width=width,
                              anchor="center" if key in ("on", "path")
                              else "w",
                              stretch=key == "gestures")

        scroll = ttk.Scrollbar(holder, orient="vertical",
                               style="Dark.Vertical.TScrollbar",
                               command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.table.tag_configure("odd", background="#232327")
        self.table.tag_configure("off", foreground=FAINT)
        self.table.tag_configure("semantic", foreground=CYAN)
        self.table.bind("<Double-1>", lambda _e: self._edit_selected())

        # Bound to the table, not the toplevel: at window scope BackSpace
        # would delete a mapping every time the user corrected a typo in
        # the name or key-combination fields.
        for sequence in ("<Delete>", "<BackSpace>"):
            self.table.bind(sequence, self._on_delete_key)
        self.table.bind("<Control-z>", self._on_undo_key)

    def _render_table(self) -> None:
        self.table.delete(*self.table.get_children())
        for index, rule in enumerate(self.rules):
            tags = ["odd"] if index % 2 else []
            if not rule.get("enabled", True):
                tags.append("off")

            if rule["trigger"] == TRANSITION:
                gestures = (f"{pretty(rule['from_state'])}  →  "
                            f"{pretty(rule['to_state'])}")
                window = rule.get("max_time_sec", 0.0)
                # Zero is the engine's "no limit", which reads as an
                # impossible 0.00 s deadline if printed as a number.
                timing = f"within {window:.2f}s" if window > 0 else "no limit"
            else:
                gestures = f"hold {pretty(rule['pose'])}"
                delay = rule.get("hold_sec", 0.0)
                timing = f"{delay:.2f}s" if delay > 0 else "instant"
                if rule.get("repeat"):
                    timing += " ↻"

            action = rule["action"]
            if action == KEYBOARD_MACRO:
                action = f"{action}  {rule.get('keys', '')}"
            elif rule.get("promote_double"):
                action = f"{action}  ⇄2"

            path = {"geometry": "30 Hz", "semantic": "YOLO"}.get(
                rule.get("source", "any"), "any")

            self.table.insert(
                "", "end", iid=rule["id"],
                values=("✓" if rule.get("enabled", True) else "○",
                        rule["trigger"].capitalize(), gestures, action,
                        timing, f"{rule.get('cooldown_sec', 0.0):.2f}s", path),
                tags=tuple(tags))

        active = sum(1 for r in self.rules if r.get("enabled", True))
        self.table_count.configure(
            text=f"{active} active of {len(self.rules)}")

    def _selected(self):
        selection = self.table.selection()
        if not selection:
            self._toast("Select a mapping first.", AMBER)
            return None
        for rule in self.rules:
            if rule["id"] == selection[0]:
                return rule
        return None

    def _edit_selected(self) -> None:
        rule = self._selected()
        if rule is None:
            return

        self.editing_id = rule["id"]
        self._loading = True
        self.trigger.select(rule["trigger"], notify=True)
        self._loading = False

        if rule["trigger"] == TRANSITION:
            self.slot_from.set_pose(rule["from_state"],
                                    self.thumbs.get(rule["from_state"]))
            self.slot_to.set_pose(rule["to_state"],
                                  self.thumbs.get(rule["to_state"]))
            self.timing_var.set(f"{rule.get('max_time_sec', 0.8):.2f}")
            self.promote_toggle.set(bool(rule.get("promote_double")))
        else:
            self.slot_pose.set_pose(rule["pose"], self.thumbs.get(rule["pose"]))
            self.timing_var.set(f"{rule.get('hold_sec', 0.4):.2f}")
            self.repeat_toggle.set(bool(rule.get("repeat")))

        self.action_var.set(rule["action"])
        self.cooldown_var.set(f"{rule.get('cooldown_sec', 0.35):.2f}")
        self.keys_var.set(rule.get("keys", "win+d"))
        self.name_var.set(rule.get("name", ""))

        self.builder_title.configure(text="EDIT MAPPING")
        self.commit_button.configure(text="Save Changes")
        self._on_action_change()

    def _toggle_selected(self) -> None:
        rule = self._selected()
        if rule is None:
            return
        rule["enabled"] = not rule.get("enabled", True)
        self._mark_dirty()
        self._render_table()
        self.table.selection_set(rule["id"])

    def _duplicate_selected(self) -> None:
        rule = self._selected()
        if rule is None:
            return
        clone = dict(rule)
        clone["id"] = f"r{uuid.uuid4().hex[:8]}"
        clone["name"] = f"{rule.get('name', rule['action'])} copy"
        clone["enabled"] = False        # a duplicate trigger must not fire
        self.rules.append(clone)
        self._mark_dirty()
        self._render_table()
        self.table.selection_set(clone["id"])
        self._toast("Duplicated, disabled — change its trigger, then enable.",
                    AMBER)

    def _selected_rules(self) -> list:
        """Every highlighted rule, in table order rather than click order."""
        chosen = set(self.table.selection())
        return [rule for rule in self.rules if rule["id"] in chosen]

    def _on_delete_key(self, _event=None):
        self.delete_selected_mapping()
        return "break"          # keep Treeview's own key handling out of it

    def _on_undo_key(self, _event=None):
        self.undo_delete()
        return "break"

    def delete_selected_mapping(self) -> None:
        """Remove every selected mapping from the table and from self.rules.

        `self.rules` is the session's single source of truth and the only
        thing `_rules_to_config()` serialises, so dropping the entries here
        is what makes a deletion survive Save Config.  The Treeview is then
        rebuilt from that list rather than edited in parallel with it,
        which is what stops the two from drifting apart.
        """
        doomed = self._selected_rules()
        if not doomed:
            self._toast("Select a mapping to delete.", AMBER)
            return

        if len(doomed) > 1 and not messagebox.askokcancel(
                "Delete mappings?",
                f"Delete these {len(doomed)} mappings?\n\n"
                f"Undo Delete puts them back until you close the window.",
                parent=self):
            return

        order = [rule["id"] for rule in self.rules]
        doomed_ids = {rule["id"] for rule in doomed}

        # Leave the highlight on the first survivor below the deletion, so
        # holding Delete walks down the list instead of stranding focus.
        successor = next(
            (rule_id for rule_id in order[order.index(doomed[-1]["id"]) + 1:]
             if rule_id not in doomed_ids), None)

        # Positions are recorded alongside the rules: undo should restore
        # the list as it was, not append the survivors to the bottom.
        self._undo_stack.append(
            [(order.index(rule["id"]), dict(rule)) for rule in doomed])
        self.undo_button.set_enabled(True)

        self.rules = [r for r in self.rules if r["id"] not in doomed_ids]
        if self.editing_id in doomed_ids:
            self._clear_builder()

        self._mark_dirty()
        self._render_table()

        if successor is not None:
            self.table.selection_set(successor)
            self.table.focus(successor)
        self.table.focus_set()

        plural = "s" if len(doomed) > 1 else ""
        self._toast(f"Deleted {len(doomed)} mapping{plural} — Ctrl+Z to "
                    f"undo, Save Config to make it permanent.", MUTED)

    # Shorter alias, so either name resolves to the same implementation.
    delete_mapping = delete_selected_mapping

    def undo_delete(self) -> None:
        """Restore the most recent deletion to its original position."""
        if not self._undo_stack:
            self._toast("Nothing to undo.", AMBER)
            return

        restored = self._undo_stack.pop()
        for index, rule in sorted(restored, key=lambda item: item[0]):
            self.rules.insert(min(index, len(self.rules)), rule)

        if not self._undo_stack:
            self.undo_button.set_enabled(False)

        self._mark_dirty()
        self._render_table()

        ids = [rule["id"] for _index, rule in restored]
        self.table.selection_set(*ids)
        self.table.focus_set()

        plural = "s" if len(restored) > 1 else ""
        self._toast(f"Restored {len(restored)} mapping{plural}.", ACCENT)

    # ── test runner ─────────────────────────────────────────────────────

    def _test_selected(self) -> None:
        rule = self._selected()
        if rule is None:
            return
        if self._test_job is not None:
            return

        action = rule["action"]
        chord = rule.get("keys") or ACTION_MACROS.get(action, "")
        if action in (DRAG_START, DRAG_STOP):
            note = "presses and releases the left button"
        elif chord:
            note = f"sends {chord}"
        else:
            note = f"sends {action}"

        if not messagebox.askokcancel(
                "Test mapping",
                f"This performs the action for real: {note}.\n\n"
                f"You get 3 seconds to focus the window you want it to hit.",
                parent=self):
            return

        self._countdown(3, rule)

    def _countdown(self, remaining: int, rule) -> None:
        if remaining > 0:
            self._toast(f"Testing {rule['action']} in {remaining}…", CYAN)
            self._test_job = self.after(
                1000, lambda: self._countdown(remaining - 1, rule))
            return

        self._test_job = None
        try:
            if self._executor is None:
                self._executor = ActionExecutor()
        except Exception as exc:
            self._toast(f"pynput unavailable: {exc}", DANGER)
            return

        action = rule["action"]
        event = gesture_fsm.ActionEvent(action, rule_id=rule["id"],
                                        rule_name=rule.get("name"),
                                        keys=rule.get("keys", ""),
                                        at=time.time())
        ok = self._executor.dispatch(event)

        # A test that presses the button must release it, or the tool would
        # leave the desktop mid-drag with no gesture available to end it.
        if action == DRAG_START:
            self.after(800, self._executor.release_all)

        self._toast(f"{action} sent." if ok
                    else f"{action} could not be sent.",
                    ACCENT if ok else DANGER)

    # ── persistence ─────────────────────────────────────────────────────

    def _settings_to_widgets(self) -> None:
        """Push the loaded settings into the panel, without dirtying it."""
        if not hasattr(self, "mirror_toggle"):
            return
        prior, self._loading = self._loading, True
        try:
            try:
                sens = float(self.settings.get("cursor_sensitivity", 1.4))
            except (TypeError, ValueError):
                sens = 1.4
            sens = max(SENSITIVITY_MIN, min(SENSITIVITY_MAX, round(sens, 1)))
            self.sensitivity_var.set(sens)
            self.sens_readout.configure(text=f"{sens:.1f}×")
            self.mirror_toggle.set(bool(self.settings.get("is_mirrored", True)))
            self.invert_toggle.set(
                bool(self.settings.get("invert_cursor_x", False)))
            self._on_direction()
        finally:
            self._loading = prior

    def _widgets_to_settings(self) -> None:
        """Fold the panel back into self.settings, which is what gets saved."""
        if not hasattr(self, "mirror_toggle"):
            return
        self.settings["cursor_sensitivity"] = round(
            float(self.sensitivity_var.get()), 1)
        self.settings["is_mirrored"] = bool(self.mirror_toggle.get())
        self.settings["invert_cursor_x"] = bool(self.invert_toggle.get())

    def _rules_to_config(self) -> dict:
        self._widgets_to_settings()
        transitions, holds = [], []
        for rule in self.rules:
            entry = {k: v for k, v in rule.items() if k != "trigger"}
            (transitions if rule["trigger"] == TRANSITION else holds).append(
                entry)
        return {
            "version": 1,
            "settings": self.settings,
            "transitions": transitions,
            "holds": holds,
        }

    def _config_to_rules(self, config: dict) -> None:
        self.settings = dict(gesture_fsm.DEFAULT_SETTINGS)
        self.settings.update(config.get("settings") or {})

        rules = []
        for kind, key in ((TRANSITION, "transitions"), (HOLD, "holds")):
            for entry in (config.get(key) or []):
                if not isinstance(entry, dict):
                    continue
                rule = dict(entry)
                rule["trigger"] = kind
                rule.setdefault("id", f"r{uuid.uuid4().hex[:8]}")
                rule.setdefault("enabled", True)
                rule.setdefault("cooldown_sec", 0.35)
                poses = ([rule.get("from_state"), rule.get("to_state")]
                         if kind == TRANSITION else [rule.get("pose")])
                rule.setdefault("source", rule_source([p for p in poses if p]))
                rules.append(rule)
        self.rules = rules
        self._settings_to_widgets()
        self._forget_undo()

    def _forget_undo(self) -> None:
        """Swapping the whole list out invalidates any pending restore.

        Undo replays a rule at a recorded index; after Load Defaults or a
        reload those indices point into a list that no longer exists, and
        replaying them would inject rules from a config the user has left.
        """
        self._undo_stack.clear()
        if hasattr(self, "undo_button"):
            self.undo_button.set_enabled(False)

    def _load_from_disk(self) -> None:
        if self._dirty and not messagebox.askokcancel(
                "Discard changes?",
                "You have unsaved mappings. Reload from disk anyway?",
                parent=self):
            return
        exists = os.path.exists(CONFIG_PATH)
        config = load_config(CONFIG_PATH, quiet=True)
        self._config_to_rules(config)
        self._render_table()
        self._clear_builder()
        self._clear_dirty()
        self._toast(
            f"Loaded {len(self.rules)} mapping(s) from "
            f"{os.path.basename(CONFIG_PATH)}." if exists
            else "No config yet — showing the safe fallback binding.",
            MUTED if exists else AMBER)

    def _load_defaults(self) -> None:
        if not messagebox.askokcancel(
                "Load defaults?",
                "This replaces every mapping in the list with the default "
                "scheme.\n\nNothing is written to disk until you press "
                "Save Config.", parent=self):
            return
        self._config_to_rules(default_config())
        self._render_table()
        self._clear_builder()
        self._mark_dirty()
        self._toast("Default scheme loaded — not yet saved.", AMBER)

    def _save(self) -> None:
        # Defaults to Cancel, because saving an empty list is the one action
        # here that silently disables the whole tracker: it starts, the
        # cursor moves, and no gesture ever fires.  Easy to reach by
        # clearing the table to start over, and hard to diagnose afterwards.
        if not self.rules and not messagebox.askokcancel(
                "Save an empty config?",
                "There are no mappings.\n\nSaving now leaves the tracker "
                "with no bindings at all — the cursor will still move, but "
                "no gesture will click, drag or fire a macro.\n\n"
                "Press Load Defaults first if you wanted the standard "
                "scheme.",
                parent=self, icon=messagebox.WARNING,
                default=messagebox.CANCEL):
            self._toast("Nothing saved — the config on disk is unchanged.",
                        AMBER)
            return
        try:
            save_config(self._rules_to_config(), CONFIG_PATH)
        except OSError as exc:
            self._toast(f"Could not write the config: {exc}", DANGER)
            return
        self._clear_dirty()
        self._toast(f"Saved {len(self.rules)} mapping(s) to "
                    f"{os.path.basename(CONFIG_PATH)}.", ACCENT)

    def _mark_dirty(self) -> None:
        # The settings widgets fire their callbacks while the window is
        # being built and again on every load, neither of which is a user
        # edit.  Without this guard the config would open already flagged
        # unsaved and the quit prompt would appear for doing nothing.
        if self._booting or self._loading:
            return
        self._dirty = True
        self.save_button.configure(text="Save Config •")

    def _clear_dirty(self) -> None:
        self._dirty = False
        self.save_button.configure(text="Save Config")

    # ── status bar ──────────────────────────────────────────────────────

    def _build_status(self) -> None:
        bar = tk.Frame(self, bg=BG)
        bar.grid(row=3, column=0, columnspan=2, sticky="ew",
                 padx=PAD, pady=(8, 10))
        self.status = tk.Label(bar, text="", bg=BG, fg=MUTED,
                               font=self.f_sub, anchor="w")
        self.status.pack(side="left")
        tk.Label(bar, text=os.path.basename(CONFIG_PATH), bg=BG, fg=FAINT,
                 font=self.f_mono).pack(side="right")

    def _toast(self, message: str, colour: str = MUTED) -> None:
        self.status.configure(text=message, fg=colour)

    # ── shutdown ────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._dirty and not messagebox.askokcancel(
                "Quit without saving?",
                "You have unsaved mappings. Quit anyway?", parent=self):
            return
        if self._executor is not None:
            # Never exit holding a button, whatever the Test button did.
            self._executor.release_all()
        self.destroy()


def main() -> None:
    if not os.path.isdir(GESTURE_DIR):
        print(f"[app] no gestures/ folder at {GESTURE_DIR} — the catalog "
              f"will show the geometric poses only")
    GestureStudio().mainloop()


if __name__ == "__main__":
    main()
