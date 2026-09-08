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

import json
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

# Optional, and each degrades to a dash on the diagnostics tab rather than
# stopping the window from opening: psutil is a third-party package and
# pynvml only exists where an NVIDIA driver does.
try:
    import psutil
except ImportError:                                       # pragma: no cover
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_READY = True
except Exception:                                         # pragma: no cover
    pynvml = None
    _NVML_READY = False

import gesture_fsm
from monitors import (ALL_SCREENS, enumerate_monitors, monitor_labels,
                      monitor_union, resolve_monitor_target)
from gesture_fsm import (HAND_ANY, HAND_BOTH, HAND_LEFT, HAND_RIGHT,
                         hands_can_coincide, normalise_hand)
from gesture_fsm import (ACTION_MACROS, ACTIONS, CONFIG_PATH, DOUBLE_CLICK,
                         DRAG_START, DRAG_STOP, KEYBOARD_MACRO, LEFT_CLICK,
                         MIDDLE_CLICK, MOUSE_ACTIONS, RIGHT_CLICK,
                         AI_CONFIDENCE_DEFAULT, AI_CONFIDENCE_MAX,
                         AI_CONFIDENCE_MIN,
                         SENSITIVITY_MAX, SENSITIVITY_MIN, SENSITIVITY_STEP,
                         ActionExecutor, load_config, save_config)

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

MIB = 1024 ** 2
GIB = 1024 ** 3
CPU_MODE_NOTE = "N/A (CPU)"

# The only settings the GUI owns.  Everything else under "settings" is
# engine tuning, written back only if it was changed from its default.
SELECTABLE_ACTIONS = [a for a in ACTIONS if a != KEYBOARD_MACRO]

CURSOR_SETTING_KEYS = ("cursor_sensitivity", "is_mirrored",
                       "invert_cursor_x", "ai_confidence", "target_screen")

GESTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gestures")

# ─── No gesture vocabulary ──────────────────────────────────────────────────
# There is deliberately no master list of poses here any more.
#
# There used to be two — GEOMETRIC for detect_gesture()'s output, SEMANTIC
# for the YOLO class list — plus an alias table folding the folder's
# "_inverse" filenames onto the model's "_inverted" labels.  All three
# assumed the vocabulary was fixed and known at edit time, which is exactly
# the assumption that blocks a user-supplied gesture: an unknown label was
# flagged as suspect, and a filename that did not match the table could not
# be picked at all.
#
# The contract is now simply: the PNG's filename IS the label the model
# emits.  Drop point.png in gestures/, train the model to say "point", and
# it works — no list to edit, no alias to add, no validation to satisfy.
#
# DEFAULT_SOURCE is the stream a rule listens to when it does not say.
# "any" means the FSM accepts it from whichever stream feeds it; a config
# may still hand-set "geometry" or "semantic" per rule, which is what keeps
# the dual-stream split in gesture_config.json working.
DEFAULT_SOURCE = "any"

# Hand requirement, as the builder offers it.  ANY leads and is the default
# because it is what every rule written before this feature meant, so
# opening an old config does not silently constrain it.
HAND_CHOICES = (
    (HAND_ANY, "Any"),
    (HAND_RIGHT, "Right"),
    (HAND_LEFT, "Left"),
    (HAND_BOTH, "Both"),
)

HAND_WORDS = {
    HAND_ANY: "either hand",
    HAND_RIGHT: "right hand",
    HAND_LEFT: "left hand",
    HAND_BOTH: "both hands",
}

# The reference image in the builder.  Larger than a socket thumbnail (SLOT)
# because its whole job is to be looked at while you copy the pose.
REFERENCE = 148

TRANSITION = "transition"
HOLD = "hold"


def pretty(label: str) -> str:
    """'little_finger' -> 'Little Finger'."""
    return " ".join(part.capitalize() for part in str(label).split("_"))


# ─── Gesture catalog ────────────────────────────────────────────────────────


# Temporary: set MOUSEGESTURE_DEBUG_ADAPTIVE=1 to trace the calibration
# hand-off.  Silent otherwise, so a normal run prints nothing.
_DEBUG_ADAPTIVE = bool(os.environ.get("MOUSEGESTURE_DEBUG_ADAPTIVE"))


# The mapping-lifecycle trace: what the builder committed, what was
# written, and what came back off disk.  OFF by default — it is several
# lines per rule per save, which is a debugging session rather than
# something a user should have to opt out of.
# Enable with MOUSEGESTURE_CONFIG_DEBUG=1.
CONFIG_DEBUG = os.environ.get("MOUSEGESTURE_CONFIG_DEBUG", "") in (
    "1", "true", "True", "TRUE", "yes", "on")


def _dbg(message: str) -> None:
    if _DEBUG_ADAPTIVE:
        print(f"[adaptive] {message}")


# ── Camera state ───────────────────────────────────────────────────────
# One vocabulary, shared by the button, the calibration dialog and
# anything else that needs to know.  The point of naming these is that
# CONNECTED is DERIVED from the engine on every read rather than latched
# when a button was pressed -- a label reading "Disconnect" while the
# camera has actually stopped is the failure this exists to prevent.
CAM_DISCONNECTED = "DISCONNECTED"
CAM_CONNECTING = "CONNECTING"
CAM_CONNECTED = "CONNECTED"
CAM_DISCONNECTING = "DISCONNECTING"
CAM_ERROR = "ERROR"

# How often the button re-reads the engine.  Two attribute lookups, so
# it costs nothing, and it means an engine that stops on its own is
# reflected in the UI without anything having to notice and report it.
CAMERA_SYNC_MS = 300


def timing_label(rule) -> str:
    """How a transition's timing reads in the UI.

    Three distinct states, and they must not be conflated: the engine
    owns it, the user pinned a number, or the user asked for no limit at
    all (which max_time_sec spells as 0).
    """
    if rule.get("adaptive_timing"):
        return "timing: automatic"
    window = rule.get("max_time_sec", 0.0) or 0.0
    return f"within {window:.2f}s" if window > 0 else "no time limit"


class GestureLibrary:
    """Scans gestures/ and decodes thumbnails off the main thread.

    Thirty large PNGs take a noticeable moment to decode, and doing it
    inline would freeze the window before it had finished drawing.  The
    worker hands back PIL images; the Tk side turns them into PhotoImages,
    because those may only be created on the thread running the mainloop.
    """

    # Poses hidden from the UI: two-handed or awkward to hold, so there is
    # no point offering them as bindings.  THIS IS A GUI FILTER ONLY — the
    # model still recognises every one of them and the FSM still executes
    # any rule already bound to one; they simply cannot be picked here any
    # more, so no new mapping can be built on them.
    #
    # Both spellings are listed for three of them on purpose.  The
    # classifier reports "take_picture", "hand_heart" and "hand_heart2", so
    # listing only the informal "take_photo", "heart" and "heart2" would
    # match nothing and leave three of the seven still on screen.
    EXCLUDED_GESTURES = {
        "thumb_index2",
        "holy",
        "timeout",
        "take_photo", "take_picture",
        "xsign", "x_sign",
        "heart", "hand_heart",
        "heart2", "hand_heart2",
        # "open" is the geometry classifier's flat-palm state and "palm" is
        # the network's class for the same shape, and open.png is a byte
        # copy of palm.png — so the catalog showed the same drawing twice
        # with two names and nothing to tell them apart.  "palm" is kept
        # because it is the one the trained model reports.
        #
        # The engine still classifies "open" every frame and any rule
        # already bound to it still fires; it simply cannot be picked here.
        "open",
    }

    def __init__(self, directory: str = GESTURE_DIR) -> None:
        self.directory = directory
        self.labels = []              # canonical, sorted
        self._stem_for = {}           # canonical -> file stem
        self._path_for = {}           # canonical -> full path
        self._results = queue.Queue()
        self._scan()

    @classmethod
    def is_excluded(cls, *names) -> bool:
        """True if any spelling of this pose is blacklisted.

        Every name a pose is known by is checked, because the file stem and
        the canonical label can differ and only one of them may be listed.
        Case, spaces and hyphens are all normalised away first.
        """
        for name in names:
            if not name:
                continue
            key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
            if key in cls.EXCLUDED_GESTURES:
                return True
        return False

    def _scan(self) -> None:
        """The catalog is the directory.  Nothing else decides what exists.

        Whatever image files are in gestures/ become the pose list, minus
        the blacklist.  There is no vocabulary to check against and no
        alias table: the filename stem is taken as the exact label the
        classifier emits, so adding a gesture means adding a PNG.
        """
        found = {}
        if not os.path.isdir(self.directory):
            self._path_for = found
            self.labels = []
            return

        for name in sorted(os.listdir(self.directory)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            label = stem.strip().lower()
            if not label or self.is_excluded(stem, label):
                continue
            found[label] = os.path.join(self.directory, name)
            self._stem_for[label] = stem

        self._path_for = found
        self.labels = sorted(found)

    def path_for(self, label):
        """Full path to a pose's artwork, or None when it has none."""
        return self._path_for.get(str(label or "").strip().lower())

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
    """Segmented control for any number of mutually exclusive options.

    Used for the trigger-type switch (2) and the hand requirement (4).
    """

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
        self._caption.configure(text=pretty(label), fg=ACCENT)


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
        # Reference renders are built on demand rather than in the
        # background pass: only one is on screen at a time, and
        # decoding 29 PNGs at a second size to show one of them would
        # cost startup time for nothing.
        self.reference_large = {}
        self.rules = []               # rule dicts; the source of truth
        self.settings = dict(gesture_fsm.DEFAULT_SETTINGS)
        self.editing_id = None
        self.armed_slot = None
        self._executor = None
        self._test_job = None
        self._dirty = False
        self._loading = False         # suppresses builder auto-defaults
        self._booting = True          # suppresses dirty-marking until shown

        # The recycle bin is the single store behind both the Ctrl+Z undo
        # on Tab 1 and the Restore button on Tab 3.  _undo_stack holds
        # batches of ids that point into it, so the two can never disagree
        # about what was deleted.
        self.deleted = []
        self._undo_stack = []         # lists of binned rule ids, newest last

        self.engine = None
        self._engine_busy = False
        self._cam_disconnecting = False
        self._cam_error = ""
        self._camera_watchers = []   # notified on any change
        self._camera_last_state = None

        # The authoritative selection.  Every other place that needs to
        # know which camera is chosen -- the calibration popup included
        # -- asks selected_camera() rather than reading a widget.
        self.cameras = []
        self.camera_index = None
        self._camera_scan = queue.Queue()
        self._camera_scan_busy = False
        self._engine_result = queue.Queue()
        self._video_job = None
        self._monitor_job = None
        self._proc = None
        self._scrollers = []       # (canvas, inner frame) pairs for the wheel

        self._init_fonts()
        self._init_styles()
        self._build()

        self.library.start_loading()
        self.after(60, self._pump_thumbnails)

        self._load_from_disk()
        self._booting = False
        self._refresh_metrics()       # self-rescheduling, every 1000 ms
        # Re-reads the engine on a timer, so the button follows the
        # camera even when nothing pressed it.
        self._sync_camera_ui()
        # First scan runs off-thread, so the window opens at once
        # and the dropdown fills in a moment later.
        self.after(200, self.refresh_cameras)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── chrome ──────────────────────────────────────────────────────────

    def _init_fonts(self) -> None:
        family = ("Segoe UI" if "Segoe UI" in tkfont.families()
                  else "TkDefaultFont")
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

        # The notebook's own frame is drawn behind the tab strip, so it has
        # to lose its border and take the window colour or a pale seam
        # appears under every tab.
        style.configure("Dark.TNotebook", background=BG, borderwidth=0,
                        tabmargins=(2, 6, 2, 0))
        style.configure("Dark.TNotebook.Tab", background=CARD,
                        foreground=MUTED,
                        bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD,
                        padding=(20, 9), font=(self.f_body[0], 9, "bold"))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", ACCENT), ("active", HOVER)],
                  foreground=[("selected", ON_ACCENT), ("active", TEXT)],
                  lightcolor=[("selected", ACCENT)],
                  bordercolor=[("selected", ACCENT)])

        style.configure("Dark.Treeview", background=FIELD,
                        fieldbackground=FIELD, foreground=TEXT,
                        bordercolor=BORDER, lightcolor=FIELD,
                        darkcolor=FIELD, rowheight=30, font=self.f_body)
        style.configure("Dark.Treeview.Heading", background=CARD,
                        foreground=MUTED, relief="flat", font=self.f_body,
                        padding=6)
        style.map("Dark.Treeview.Heading",
                  background=[("active", HOVER)],
                  foreground=[("active", TEXT)])
        style.map("Dark.Treeview",
                  background=[("selected", "#14532d")],
                  foreground=[("selected", TEXT)])
        style.layout("Dark.Treeview", [
            ("Dark.Treeview.treearea", {"sticky": "nswe"}),
        ])

    def _build(self) -> None:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self, style="Dark.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew",
                           padx=PAD, pady=(PAD, 0))

        # Tab 1 is the original window, moved wholesale into a frame.  Its
        # internals are untouched: every widget below still grids into the
        # same rows and columns it always did, the only difference being
        # that the container is this page rather than the toplevel.
        self._page = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self._page, text="  Mapping Studio  ")

        self._tab_camera = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self._tab_camera,
                          text="  Live Camera & Diagnostics  ")

        self._tab_bin = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self._tab_bin, text="  Recycle Bin  ")

        self._build_page()
        self._build_camera_tab(self._tab_camera)
        self._build_bin_tab(self._tab_bin)

    def _build_page(self) -> None:
        """The original single-window layout, verbatim, inside Tab 1."""
        self._page.rowconfigure(1, weight=1)
        self._page.columnconfigure(1, weight=1)

        self._build_header()

        left = tk.Frame(self._page, bg=BG)
        left.grid(row=1, column=0, sticky="nsew", padx=(PAD, 0), pady=(0, 0))
        # The catalog is now the only thing in this column, so it takes the
        # whole height that Cursor Settings used to share with it.
        self._build_catalog(left)

        right = tk.Frame(self._page, bg=BG)
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
        card.pack(fill="x", pady=(PAD, 0))

        tk.Label(card, text="CURSOR SETTINGS", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(anchor="w", padx=PAD, pady=(PAD, 2))
        tk.Label(card, text="Saved to gesture_config.json; the engine reads "
                            "them when it starts.",
                 bg=CARD, fg=FAINT, font=("Segoe UI", 8), wraplength=232,
                 justify="left").pack(anchor="w", padx=PAD)

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
        self.direction_note.pack(fill="x", padx=PAD, pady=(6, 0))

        # ── target screen ──────────────────────────────────────────────
        # Built from the displays attached RIGHT NOW, so the list is
        # however many there are — one, two or six — and no count is
        # hard-coded anywhere.  Refreshed when the dropdown is opened,
        # because monitors get plugged in while the app is running.
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=PAD, pady=(12, 0))
        tk.Label(row, text="CURSOR AREA", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(side="left")

        self.screen_var = tk.StringVar(value=ALL_SCREENS)
        self.screen_box = ttk.Combobox(
            card, textvariable=self.screen_var, state="readonly",
            font=("Segoe UI", 9))
        self.screen_box.pack(fill="x", padx=PAD, pady=(4, 0))
        self.screen_box.bind("<<ComboboxSelected>>", self._on_screen_target)
        self.screen_box.bind("<Button-1>", lambda _e: self._refresh_screens())

        self.screen_note = tk.Label(
            card, text="", bg=CARD, fg=FAINT, font=("Segoe UI", 8),
            wraplength=232, justify="left", anchor="w")
        self.screen_note.pack(fill="x", padx=PAD, pady=(4, 0))

        self._refresh_screens()

        # ── AI confidence ──────────────────────────────────────────────
        # One floor for both models.  Low finds a hand in poor light and
        # invents poses in clutter; high is certain but drops out when you
        # move.  A custom gesture set is exactly when this needs tuning.
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=PAD, pady=(12, 0))
        tk.Label(row, text="AI CONFIDENCE", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        self.confidence_readout = tk.Label(row, text="0.60", bg=CARD,
                                           fg=CYAN,
                                           font=("Consolas", 10, "bold"))
        self.confidence_readout.pack(side="right")

        self.confidence_var = tk.DoubleVar(value=AI_CONFIDENCE_DEFAULT)
        self.confidence_scale = ttk.Scale(
            card, from_=AI_CONFIDENCE_MIN, to=AI_CONFIDENCE_MAX,
            variable=self.confidence_var, orient="horizontal",
            style="Dark.Horizontal.TScale", command=self._on_confidence)
        self.confidence_scale.pack(fill="x", padx=PAD, pady=(4, 0))

        ends = tk.Frame(card, bg=CARD)
        ends.pack(fill="x", padx=PAD)
        tk.Label(ends, text=f"{AI_CONFIDENCE_MIN:.1f} loose", bg=CARD,
                 fg=FAINT, font=("Segoe UI", 7)).pack(side="left")
        tk.Label(ends, text=f"strict {AI_CONFIDENCE_MAX:.1f}", bg=CARD,
                 fg=FAINT, font=("Segoe UI", 7)).pack(side="right")

        tk.Frame(card, bg=CARD, height=PAD).pack()

        self._on_direction()

    def _refresh_screens(self) -> None:
        """Re-enumerate the displays and rebuild the dropdown.

        Called when the panel is built and again every time the list is
        opened, so plugging a monitor in mid-session is picked up without
        a restart.  The current choice survives if it still exists and
        falls back to Screen 1 if it does not — the same rule the engine
        applies, so the two never disagree about which screen is live.
        """
        if not hasattr(self, "screen_box"):
            return
        try:
            monitors = enumerate_monitors()
        except Exception:
            monitors = []

        self.monitors = monitors
        labels = monitor_labels(monitors)
        self.screen_box.configure(values=labels)

        wanted = self.screen_var.get() or ALL_SCREENS
        if wanted not in labels:
            wanted = "Screen 1" if len(labels) > 1 else ALL_SCREENS
            self.screen_var.set(wanted)

        if not monitors:
            self.screen_note.configure(
                text="No displays enumerated — the cursor uses the whole "
                     "desktop.")
            return

        if wanted == ALL_SCREENS:
            rect = monitor_union(monitors)
            detail = (f"{len(monitors)} display"
                      f"{'s' if len(monitors) != 1 else ''}, "
                      f"{rect[2]}×{rect[3]} combined")
        else:
            rect, _ = resolve_monitor_target(wanted, monitors)
            detail = f"{rect[2]}×{rect[3]} at ({rect[0]}, {rect[1]})"
        self.screen_note.configure(text=detail)

    def _on_screen_target(self, _event=None) -> None:
        """Dropdown changed: note it, push it, and save it."""
        self._refresh_screens()
        if self._loading:
            return
        self._widgets_to_settings()
        self.update_engine_settings()
        self.save_config()

    def update_engine_settings(self) -> None:
        """Push the panel's three values into the running engine.

        Called from the slider and both toggles, so a change reaches the
        tracker on its very next frame rather than waiting for a
        reconnect, and again on connect so the values loaded from
        gesture_config.json are applied to a freshly started engine.

        A no-op when nothing is running, which is what lets the same call
        sit unconditionally at the end of every handler.
        """
        engine = self.engine
        if engine is None:
            return
        try:
            engine.apply_settings(
                cursor_speed=float(self.sensitivity_var.get()),
                is_mirrored=bool(self.mirror_toggle.get()),
                invert_x=bool(self.invert_toggle.get()),
                ai_confidence=float(self.confidence_var.get()),
                target_screen=str(self.screen_var.get() or ALL_SCREENS),
            )
        except (TypeError, ValueError, AttributeError) as exc:
            self._camera_note(f"Could not apply cursor settings: "
                              f"{exc.__class__.__name__}: {exc}", DANGER)

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
        self.update_engine_settings()

    def _on_confidence(self, raw) -> None:
        """Snap to 0.05, show it, and push it to the live models."""
        value = round(float(raw) / 0.05) * 0.05
        value = round(min(AI_CONFIDENCE_MAX, max(AI_CONFIDENCE_MIN, value)), 2)
        if abs(value - self.confidence_var.get()) > 1e-9:
            self.confidence_var.set(value)
        if hasattr(self, "confidence_readout"):
            self.confidence_readout.configure(text=f"{value:.2f}")
            self._mark_dirty()
        self.update_engine_settings()

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
        self.update_engine_settings()

    # ── hotkey footer ───────────────────────────────────────────────────

    def _build_hotkeys(self) -> None:
        """The same hotkeys the camera overlay prints, mirrored here.

        Reference only — these keys are read by cv2.waitKey() in the
        tracker's preview window, so they do nothing while this window has
        focus.  The panel above is how you set the same things from here.
        """
        bar = tk.Frame(self._page, bg=CARD, highlightbackground=BORDER,
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
        bar = tk.Frame(self._page, bg=BG)
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
        self._scrollers.append((self.canvas, self.grid_frame))
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_wheel(self, event) -> None:
        """Route the wheel to whichever scrolling column is under the pointer.

        The binding is global (bind_all), so without the containment walk
        the wheel would scroll the catalog while the pointer sat over the
        mappings table or the diagnostics column.
        """
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            for canvas, frame in self._scrollers:
                if widget is canvas or widget is frame:
                    canvas.yview_scroll(-int(event.delta / 120), "units")
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
        labels = [name for name in self.library.labels
                  if not needle or needle in name.replace("_", " ")
                  or needle in name]

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

        widgets = (tile, image, name)
        for widget in widgets:
            widget.bind("<Button-1>",
                        lambda _e, name=label: self._pick(name))
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
        self._show_reference(label)
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

        # Reference image ----------------------------------------------
        # The sockets show the pose at thumbnail size, which is enough to
        # confirm a pick but not to copy a hand shape from.  This shows the
        # pose you most recently placed, large enough to actually read.
        # It is DISPLAY ONLY — nothing downstream reads this widget, and
        # recognition never consults the artwork.
        reference = tk.Frame(body, bg=CARD)
        reference.pack(side="left", padx=(0, PAD))

        tk.Label(reference, text="REFERENCE", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")

        self._ref_box = tk.Frame(reference, bg=FIELD,
                                 width=REFERENCE, height=REFERENCE,
                                 highlightbackground=BORDER,
                                 highlightthickness=1)
        self._ref_box.pack_propagate(False)
        self._ref_box.pack(pady=(4, 0))

        self.reference_image = tk.Label(
            self._ref_box, bg=FIELD, fg=FAINT,
            text="pick a pose", font=("Segoe UI", 9), wraplength=REFERENCE - 16)
        self.reference_image.pack(expand=True)

        self.reference_caption = tk.Label(
            reference, text="", bg=CARD, fg=FAINT, font=("Segoe UI", 8),
            wraplength=REFERENCE + 12, justify="left")
        self.reference_caption.pack(anchor="w", pady=(4, 0))

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
            fields, textvariable=self.action_var, values=SELECTABLE_ACTIONS,
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
        self.timing_box = ttk.Spinbox(fields, from_=0.0, to=10.0,
                                      increment=0.05,
                                      textvariable=self.timing_var, width=8,
                                      style="Dark.TSpinbox", font=self.f_body)
        self.timing_box.grid(row=3, column=0, sticky="w")

        self.cooldown_var = tk.StringVar(value="0.35")
        self.cooldown_var.trace_add("write",
                                    lambda *_: self._refresh_preview())
        self.cooldown_caption = tk.Label(
            fields, text="COOLDOWN (SEC)", bg=CARD, fg=FAINT,
            font=("Segoe UI", 8, "bold"))
        self.cooldown_box = ttk.Spinbox(
            fields, from_=0.0, to=30.0, increment=0.05,
            textvariable=self.cooldown_var, width=8,
            style="Dark.TSpinbox", font=self.f_body)
        self.cooldown_caption.grid(row=2, column=1, sticky="w", pady=(0, 2))
        self.cooldown_box.grid(row=3, column=1, sticky="w")

        # No key-combination field.  Every action offered in the dropdown
        # carries its own chord (ACTION_MACROS), so the box sat reading
        # "(n/a)" almost all of the time.  keys_var survives as invisible
        # state only, so a hand-written KEYBOARD_MACRO rule keeps its chord
        # when it is opened in the editor and saved again.
        self.keys_var = tk.StringVar(value="")

        # Hand requirement ---------------------------------------------
        # A separate field rather than four more poses in the catalog: the
        # shape and the hand that makes it are independent, and folding one
        # into the other would double the pose list.
        hand_row = tk.Frame(fields, bg=CARD)
        hand_row.grid(row=4, column=0, columnspan=4, sticky="w", pady=(12, 0))

        tk.Label(hand_row, text="HAND", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(side="left",
                                                    padx=(0, 10))
        self.hand = Segmented(hand_row, list(HAND_CHOICES),
                              lambda _v: self._on_hand_change())
        self.hand.pack(side="left")

        self.hand_note = tk.Label(hand_row, text="", bg=CARD, fg=FAINT,
                                  font=("Segoe UI", 8))
        self.hand_note.pack(side="left", padx=(12, 0))

        toggles = tk.Frame(fields, bg=CARD)
        toggles.grid(row=5, column=0, columnspan=4, sticky="w", pady=(12, 0))
        self.promote_toggle = Toggle(
            toggles,
            "Repeat inside the double-click window fires DOUBLE_CLICK",
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

    def _on_hand_change(self) -> None:
        """Explain what the choice will require, then re-render the preview."""
        if not hasattr(self, "hand_note"):
            return
        value = self.hand.value
        if value == HAND_BOTH:
            note = "requires two hands in frame"
        elif value == HAND_ANY:
            note = "fires on whichever hand performs it"
        else:
            note = f"only the {value} hand"
        self.hand_note.configure(text=note)
        self._refresh_preview()

    def _show_reference(self, label) -> None:
        """Put a pose in the reference panel.  Display only.

        Falls back to the pose name when the catalog has no artwork for it,
        so a geometric pose — which has no PNG — still reads as picked
        rather than looking like a failure.
        """
        if not hasattr(self, "reference_image"):
            return
        if not label:
            self.reference_image.configure(image="", text="pick a pose",
                                           fg=FAINT)
            self.reference_image.image = None
            self.reference_caption.configure(text="")
            return

        photo = self.reference_large.get(label)
        if photo is None:
            path = self.library.path_for(label)
            if path:
                try:
                    image = GestureLibrary._decode(path, REFERENCE)
                    photo = ImageTk.PhotoImage(image)
                    self.reference_large[label] = photo
                except Exception:
                    photo = None
        if photo is not None:
            self.reference_image.configure(image=photo, text="")
            self.reference_image.image = photo
        else:
            self.reference_image.configure(image="", text=pretty(label),
                                           fg=TEXT)
            self.reference_image.image = None
        self.reference_caption.configure(text=pretty(label), fg=ACCENT)

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
            # A transition has no timing field any more: the engine keeps
            # its own per-pose-pair estimate (adaptive_timing.py), so the
            # user configures FROM, TO, ACTION and HAND and nothing else.
            # Cooldown goes with it -- it keeps its default and stays out
            # of the way rather than becoming another number to manage.
            self.timing_caption.grid_remove()
            self.timing_box.grid_remove()
            self.cooldown_caption.grid_remove()
            self.cooldown_box.grid_remove()
        else:
            self.slot_from.pack_forget()
            self.slot_arrow.pack_forget()
            self.slot_to.pack_forget()
            self.slot_pose.pack(side="left")
            # Static Hold: only hold duration is a user parameter. Recognize
            # the pose and fire the action. Cooldown is engine-managed and
            # stays hidden.
            self.timing_caption.configure(text="HOLD FOR (SEC)")
            self.timing_caption.grid()
            self.timing_box.grid()
            self.cooldown_caption.grid_remove()
            self.cooldown_box.grid_remove()

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
            body = (f"{rule['from_state']} → {rule['to_state']} "
                    + timing_label(rule))
        else:
            delay = rule["hold_sec"]
            body = (f"hold {rule['pose']} for {delay:.2f}s" if delay > 0
                    else f"enter {rule['pose']}")

        note = ("   • YOLO path (~4.7 Hz)"
                if rule.get("source") == "semantic" else "")

        requirement = normalise_hand(rule.get("hand"))
        if requirement != HAND_ANY:
            note = f"   • {HAND_WORDS[requirement]}" + note

        self.preview.configure(
            text=f"{body}  ⇒  {action}{suffix}"
                 f"   • cooldown {rule['cooldown_sec']:.2f}s{note}",
            fg=MUTED)

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
            rule.update({"from_state": src, "to_state": dst})

            # MIGRATION.  A rule that already carries an explicit
            # max_time_sec keeps it, untouched, even when it is edited
            # here -- converting it silently would change behaviour the
            # user never asked to change, and there is no longer a field
            # in which to see or undo that.  Only rules created from now
            # on hand their timing to the engine.
            prior = self._rule_by_id(self.editing_id)
            if prior is not None and not prior.get("adaptive_timing") \
                    and prior.get("max_time_sec") is not None:
                rule["max_time_sec"] = prior["max_time_sec"]
            else:
                rule["adaptive_timing"] = True
                rule.pop("max_time_sec", None)
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
            rule.update({"pose": pose,
                         "hold_sec": number(self.timing_var, 0.4)})
            if self.repeat_toggle.get():
                rule["repeat"] = True
                rule["repeat_sec"] = 0.25
            poses = [pose]

        if action == KEYBOARD_MACRO:
            # Only reachable by editing a rule written straight into the
            # JSON, since the dropdown no longer offers this action.  The
            # chord is carried through untouched rather than re-validated:
            # the editor cannot change it, so it is not the editor's to
            # reject.
            chord = self.keys_var.get().strip()
            if not chord:
                if validate:
                    self._toast("That mapping needs a key combination; "
                                "set \"keys\" in gesture_config.json.",
                                DANGER)
                return None
            rule["keys"] = chord

        name = self.name_var.get().strip()
        if name:
            rule["name"] = name

        rule["source"] = DEFAULT_SOURCE
        rule["hand"] = self.hand.value
        return rule

    # ── [config-debug] lifecycle tracing ────────────────────────────
    # Instrumentation for the "a new mapping never reaches the runtime"
    # investigation.  Prints only; nothing here creates, routes or drops
    # a rule.  Off unless MOUSEGESTURE_CONFIG_DEBUG=1.

    def _log_rule_add(self, rule, where: str) -> None:
        """One block per rule committed to self.rules."""
        if not CONFIG_DEBUG:
            return
        print("[config-debug] GUI ADD (%s)" % where)
        for key in ("id", "trigger", "from_state", "to_state", "pose",
                    "action", "hand", "enabled", "adaptive_timing",
                    "max_time_sec", "cooldown_sec", "source"):
            if key in rule:
                print("  %s=%r" % (key, rule.get(key)))
        self._log_active_rules()

    def _log_active_rules(self) -> None:
        """Every ACTIVE rule.  The bin is counted, never listed as active."""
        if not CONFIG_DEBUG:
            return
        print("[config-debug] GUI ACTIVE RULES  count=%d  (bin=%d, not "
              "active)" % (len(self.rules), len(self.deleted)))
        for index, rule in enumerate(self.rules, 1):
            print("  [%d] id=%s  %s -> %s = %s  hand=%s source=%s "
                  "enabled=%s adaptive=%s"
                  % (index, rule.get("id"),
                     rule.get("from_state") or rule.get("pose"),
                     rule.get("to_state") or "-", rule.get("action"),
                     rule.get("hand"), rule.get("source"),
                     rule.get("enabled"), rule.get("adaptive_timing")))

    def _commit(self) -> None:
        rule = self._read_builder()
        if rule is None:
            return

        clash = self._conflict(rule)
        if clash is not None:
            # Naming the clashing rule's hand matters now that per-hand
            # variants are legal: "already bound" alone would look wrong
            # to someone who just set Left and can see a Right mapping
            # sitting in the table.
            whose = HAND_WORDS.get(normalise_hand(clash.get("hand")),
                                   "either hand")
            self._toast(f"That trigger is already bound to "
                        f"{clash['action']} for {whose}.", DANGER)
            return

        if self.editing_id:
            for index, existing in enumerate(self.rules):
                if existing["id"] == self.editing_id:
                    rule["enabled"] = existing.get("enabled", True)
                    self.rules[index] = rule
                    break
            self._toast("Mapping updated.", ACCENT)
        else:
            # A NEW adaptive mapping is not added here.  It asks for its
            # twenty examples first, and only exists if they arrive --
            # so cancelling, closing the window or losing the camera
            # leaves nothing half-created behind.  Editing an existing
            # rule never calibrates: it already has its timing.
            _dbg(f"create requested: {rule.get('from_state')} -> "
                 f"{rule.get('to_state')} adaptive="
                 f"{rule.get('adaptive_timing')!r} "
                 f"max_time_sec={rule.get('max_time_sec')!r} "
                 f"editing_id={self.editing_id!r}")
            if rule.get("adaptive_timing"):
                _dbg("rule detected as adaptive")
                if self._begin_calibration(rule):
                    _dbg("calibration took over; mapping NOT added yet")
                    return
                _dbg("calibration declined; falling through to direct add")
            else:
                _dbg("rule is NOT adaptive; no calibration")
            self.rules.append(rule)
            self._log_rule_add(rule, "direct, no calibration")
            _dbg("mapping added directly")
            self._toast("Mapping added.", ACCENT)

        self._mark_dirty()
        self._clear_builder()
        self._render_table()

    def _begin_calibration(self, rule) -> bool:
        """Ask for the initial examples.  True if the dialog took over.

        The dialog opens whether or not a camera is running.  With one,
        it starts counting immediately.  Without, it says so and offers
        to start one -- and because it follows the camera state rather
        than sampling it once, connecting from inside the dialog carries
        straight on into collection instead of making the user close it
        and rebuild the mapping.

        Nothing is created either way until twenty valid samples land.

        Returning True means "do not add this mapping here", which is as
        true of a session waiting for a camera as of one collecting.
        """
        engine = getattr(self, "engine", None)
        _dbg(f"starting calibration (camera={self.camera_state()})")
        try:
            import calibration_ui
        except Exception as exc:
            self._toast(f"Calibration unavailable "
                        f"({exc.__class__.__name__}), using defaults.",
                        FAINT)
            return False

        def finish(durations):
            for name in ("semantic_learner", "geometry_learner"):
                learner = getattr(engine, name, None)
                if learner is not None:
                    learner.seed(rule["from_state"], rule["to_state"],
                                 durations)
            _dbg(f"calibration complete: {len(durations)} samples")
            self.rules.append(rule)
            self._log_rule_add(rule, "after calibration")
            self._mark_dirty()
            self._clear_builder()
            self._render_table()
            _dbg("mapping added")
            self._toast("Mapping added and timing initialised.", ACCENT)

        try:
            panel = calibration_ui.CalibrationDialog(self, rule, finish)

            # Drop the handle when the window goes, so a cancelled
            # session does not leave a destroyed widget behind for
            # _forget_learning to trip over.  Filtered to the dialog
            # itself: <Destroy> fires for every child on the way down.
            def _forget_panel(event, _p=panel):
                if event.widget is _p and self._calibration is _p:
                    self._calibration = None

            panel.bind("<Destroy>", _forget_panel, "+")
            self._calibration = panel
            _dbg("dialog opened")
        except Exception as exc:
            _dbg(f"dialog FAILED to open: {exc.__class__.__name__}: {exc}")
            self._toast(f"Calibration unavailable "
                        f"({exc.__class__.__name__}), using defaults.",
                        FAINT)
            return False
        return True

    def _conflict(self, rule):
        """Two rules that could both fire on one frame.  Refuse the second.

        Sharing a trigger is not enough to be a duplicate.  Two mappings on
        the same gesture are a genuine conflict only if some frame satisfies
        BOTH hand requirements — otherwise they are alternatives, which is
        the entire point of per-hand mappings:

            point → grip [right] = LEFT_CLICK
            point → grip [left]  = RIGHT_CLICK

        A hand is left or right, never both, so those two can never fire
        together and are allowed to coexist.  Pairing either of them with
        an "Any" mapping on the same gesture still IS a conflict, because
        Any matches the very frames the specific one does.

        hands_can_coincide() lives in gesture_fsm so this check and the
        engine's own guard cannot drift apart.
        """
        for existing in self.rules:
            if existing["id"] == rule["id"]:
                continue
            if existing["trigger"] != rule["trigger"]:
                continue
            if rule["trigger"] == TRANSITION:
                same_trigger = (
                    existing.get("from_state") == rule.get("from_state")
                    and existing.get("to_state") == rule.get("to_state"))
            else:
                same_trigger = existing.get("pose") == rule.get("pose")
            if not same_trigger:
                continue
            if hands_can_coincide(existing.get("hand"), rule.get("hand")):
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
        self.keys_var.set("")
        self.promote_toggle.set(False)
        self.repeat_toggle.set(False)
        self.hand.select(HAND_ANY, notify=False)
        self._on_hand_change()
        self._show_reference(None)
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
                timing = timing_label(rule)
            else:
                gestures = f"hold {pretty(rule['pose'])}"
                delay = rule.get("hold_sec", 0.0)
                timing = f"{delay:.2f}s" if delay > 0 else "instant"
                if rule.get("repeat"):
                    timing += " ↻"

            # Shown on the gesture, not in a column of its own: it is a
            # qualifier on the pose, and an extra column would be blank on
            # most rows.
            requirement = normalise_hand(rule.get("hand"))
            if requirement != HAND_ANY:
                gestures = f"{gestures}   [{requirement}]"

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

    def _rule_by_id(self, rule_id):
        """The saved rule behind an id, or None for a brand new one."""
        if not rule_id:
            return None
        for rule in self.rules:
            if rule.get("id") == rule_id:
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
            self._show_reference(rule["to_state"])
        else:
            self.slot_pose.set_pose(rule["pose"],
                                    self.thumbs.get(rule["pose"]))
            self.timing_var.set(f"{rule.get('hold_sec', 0.4):.2f}")
            self.repeat_toggle.set(bool(rule.get("repeat")))
            self._show_reference(rule["pose"])

        self.action_var.set(rule["action"])
        self.cooldown_var.set(f"{rule.get('cooldown_sec', 0.35):.2f}")
        self.keys_var.set(rule.get("keys", ""))
        self.name_var.set(rule.get("name", ""))
        # An absent key means HAND_ANY, so a rule written before this
        # field existed opens unconstrained rather than defaulting to a
        # hand its author never chose.
        self.hand.select(normalise_hand(rule.get("hand")), notify=False)
        self._on_hand_change()

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

    def _forget_learning(self, rules) -> int:
        """Retire the adaptive state behind deleted mappings.

        A deleted mapping must leave nothing behind: no history, no
        learned tolerance, and nothing still queued for the learner.
        Both streams are asked, because a pose pair is fed by whichever
        classifier reports it and the GUI does not track which.

        Only ever called with pose pairs that are genuinely going away,
        and it removes exactly those keys -- another mapping on a
        different pair is untouched by construction.

        Any calibration still running for one of these pairs is
        cancelled too: it would otherwise finish and recreate the
        mapping the user just deleted.
        """
        engine = getattr(self, "engine", None)
        pairs = {(r.get("from_state"), r.get("to_state"))
                 for r in rules
                 if r.get("trigger") == TRANSITION
                 and r.get("from_state") and r.get("to_state")}
        if not pairs:
            return 0

        panel = getattr(self, "_calibration", None)
        if panel is not None:
            try:
                if panel.winfo_exists() and (
                        panel.session.from_state,
                        panel.session.to_state) in pairs:
                    panel._cancel()
                    self._calibration = None
            except Exception:
                self._calibration = None

        if engine is None:
            return 0
        forgotten = 0
        for name in ("semantic_learner", "geometry_learner"):
            learner = getattr(engine, name, None)
            if learner is None:
                continue
            for src, dst in pairs:
                if learner.forget(src, dst):
                    forgotten += 1
        return forgotten

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

        # SOFT DELETE.  The rule is popped out of the active list and
        # appended to the bin, carrying the index it came from so a restore
        # rebuilds the original order rather than appending to the bottom.
        #
        # Popped highest-index-first so the remaining indices stay valid
        # mid-loop, then reversed so the bin keeps the table's order.
        positions = {rule["id"]: index
                     for index, rule in enumerate(self.rules)}
        moved = []
        targets = sorted((positions[r["id"]] for r in doomed), reverse=True)
        for index in targets:
            try:
                entry = self.rules.pop(index)
            except IndexError:                        # pragma: no cover
                continue
            entry["_origin_index"] = index
            entry["_deleted_at"] = time.time()
            moved.append(entry)
        moved.reverse()

        if not moved:
            self._toast("Nothing was deleted.", AMBER)
            return

        # The mapping is inactive from here, so its learning must be
        # too -- an estimator that kept accumulating for a binned rule
        # would still be shaping timing nobody can see.
        self._forget_learning(moved)

        self.deleted.extend(moved)
        self._undo_stack.append([entry["id"] for entry in moved])
        self.undo_button.set_enabled(True)

        if self.editing_id in doomed_ids:
            self._clear_builder()

        self._render_table()
        self._render_bin()

        if successor is not None and successor in self.table.get_children():
            self.table.selection_set(successor)
            self.table.focus(successor)
        self.table.focus_set()

        plural = "s" if len(moved) > 1 else ""
        saved = self.save_config()
        self._toast(
            f"Moved {len(moved)} mapping{plural} to the Recycle Bin"
            + (" and saved." if saved else " — but the config could NOT be "
                                           "written.")
            + "  Ctrl+Z to undo, or restore it from the Recycle Bin tab.",
            MUTED if saved else DANGER)

    # Shorter alias, so either name resolves to the same implementation.
    delete_mapping = delete_selected_mapping

    def undo_delete(self) -> None:
        """Pull the most recent deletion back out of the recycle bin."""
        if not self._undo_stack:
            self._toast("Nothing to undo.", AMBER)
            return

        ids = self._undo_stack.pop()
        count, disabled = self._restore_ids(ids)

        if not self._undo_stack:
            self.undo_button.set_enabled(False)

        self._render_table()
        self._render_bin()

        present = [i for i in ids if i in self.table.get_children()]
        if present:
            self.table.selection_set(*present)
        self.table.focus_set()

        saved = self.save_config()
        plural = "s" if count != 1 else ""
        note = f"Restored {count} mapping{plural}"
        if disabled:
            note += f" — {disabled} came back disabled (trigger already bound)"
        note += "." if saved else " but the config could NOT be written."
        self._toast(note, AMBER if disabled or not saved else ACCENT)

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
            try:
                conf = float(self.settings.get("ai_confidence",
                                               AI_CONFIDENCE_DEFAULT))
            except (TypeError, ValueError):
                conf = AI_CONFIDENCE_DEFAULT
            conf = round(min(AI_CONFIDENCE_MAX,
                             max(AI_CONFIDENCE_MIN, conf)), 2)
            self.confidence_var.set(conf)
            self.confidence_readout.configure(text=f"{conf:.2f}")

            self.mirror_toggle.set(
                bool(self.settings.get("is_mirrored", True)))
            self.invert_toggle.set(
                bool(self.settings.get("invert_cursor_x", False)))
            self._on_direction()

            # Set the raw value first, then refresh: _refresh_screens is
            # what validates it against the displays actually attached and
            # rewrites it to Screen 1 if the saved one is gone.
            self.screen_var.set(str(self.settings.get("target_screen")
                                    or ALL_SCREENS))
            self._refresh_screens()
        finally:
            self._loading = prior
        self.update_engine_settings()

    def _widgets_to_settings(self) -> None:
        """Fold the panel back into self.settings, which is what gets saved."""
        if not hasattr(self, "mirror_toggle"):
            return
        self.settings["cursor_sensitivity"] = round(
            float(self.sensitivity_var.get()), 1)
        self.settings["is_mirrored"] = bool(self.mirror_toggle.get())
        self.settings["invert_cursor_x"] = bool(self.invert_toggle.get())
        self.settings["ai_confidence"] = round(
            float(self.confidence_var.get()), 2)
        self.settings["target_screen"] = str(self.screen_var.get()
                                             or ALL_SCREENS)

    def _rules_to_config(self) -> dict:
        """Serialise the model, split by which classifier feeds each rule.

        Every entry keeps its `trigger` key, which is what lets the engine
        route a list that mixes transitions and holds back into the right
        rule class.  `deleted_bindings` is never read by the engine, so a
        binned rule cannot be mistaken for an active one.
        """
        self._widgets_to_settings()

        geometry, semantic = [], []
        for rule in self.rules:
            entry = dict(rule)
            target = (semantic if entry.get("source") == "semantic"
                      else geometry)
            target.append(entry)

        # Only the three cursor values the panel owns, plus any engine knob
        # actually retuned away from its default.  Writing the whole
        # DEFAULT_SETTINGS block instead would bury those three in eleven
        # internals and make a fresh save stop matching the documented
        # default shape — the file would grow keys nobody set.
        settings = {key: self.settings[key] for key in CURSOR_SETTING_KEYS
                    if key in self.settings}
        for key, value in self.settings.items():
            if key in settings:
                continue
            if value != gesture_fsm.DEFAULT_SETTINGS.get(key):
                settings[key] = value

        return {
            "settings": settings,
            "geometry_bindings": geometry,
            "yolo_bindings": semantic,
            "deleted_bindings": [dict(entry) for entry in self.deleted],
        }

    def save_config(self, message: str = "") -> bool:
        """Write the whole model to gesture_config.json.  Never raises.

        Called after every bin operation, so the file always matches what
        the two tables show: a delete or a restore cannot be lost by a
        crash, or by quitting without pressing Save.
        """
        try:
            payload = self._rules_to_config()
        except Exception as exc:                      # pragma: no cover
            self._toast(f"Could not assemble the config: "
                        f"{exc.__class__.__name__}: {exc}", DANGER)
            return False

        if CONFIG_DEBUG:
            print("[config-debug] SAVE ABOUT TO WRITE  path=%s"
                  % CONFIG_PATH)
            for key in ("geometry_bindings", "yolo_bindings"):
                rows = payload.get(key) or []
                print("  %s=%d" % (key, len(rows)))
                for row in rows:
                    print("      id=%s  %s -> %s = %s  source=%s"
                          % (row.get("id"), row.get("from_state"),
                             row.get("to_state"), row.get("action"),
                             row.get("source")))
            print("  deleted_bindings=%d (never compiled)"
                  % len(payload.get("deleted_bindings") or []))

        try:
            gesture_fsm.save_config(payload, CONFIG_PATH)
        except (OSError, TypeError, ValueError) as exc:
            self._toast(f"Could not write "
                        f"{os.path.basename(CONFIG_PATH)}: {exc}", DANGER)
            return False

        # Read back from DISK, not from the payload: the question this
        # answers is whether the file on disk carries the new mapping in
        # an ACTIVE list, and only re-reading it can answer that.
        if CONFIG_DEBUG:
            try:
                with open(CONFIG_PATH, encoding="utf-8") as handle:
                    on_disk = json.load(handle)
            except Exception as exc:
                print("[config-debug] DISK AFTER SAVE  unreadable: %s: %s"
                      % (exc.__class__.__name__, exc))
            else:
                print("[config-debug] DISK AFTER SAVE  path=%s mtime=%.3f"
                      % (CONFIG_PATH, os.path.getmtime(CONFIG_PATH)))
                for key in ("geometry_bindings", "yolo_bindings",
                            "bindings", "transitions", "mappings", "holds",
                            "deleted_bindings"):
                    rows = on_disk.get(key)
                    if rows is None:
                        continue
                    note = ("  <- NEVER COMPILED"
                            if key == "deleted_bindings" else "")
                    print("  %s=%d%s" % (key, len(rows), note))
                    for row in rows:
                        print("      id=%s  %s -> %s = %s"
                              % (row.get("id"), row.get("from_state"),
                                 row.get("to_state"), row.get("action")))

        self._clear_dirty()
        self._resync_engine()
        if message:
            self._toast(message, ACCENT)
        return True

    def _resync_engine(self) -> None:
        """Push the file just written into the running tracker.

        Without this the engine keeps whatever it compiled when it was
        constructed: HandTrackerEngine._apply_config() runs once in
        __init__, reload_config() had no caller anywhere in the project,
        and apply_settings() carries only the five cursor values — so a
        mapping created here reached the JSON and stopped there.  The
        user saw their new mapping listed as active and the OLD rule
        firing, with nothing short of restarting the application to
        clear it.

        A no-op when no engine exists, which is the case while the
        camera has never been connected; the mapping is on disk and the
        engine will read it when it starts.

        Never fatal.  Saving succeeded whatever happens here, and a
        tracker that could not be resynchronised is worth a note in the
        camera tab rather than an exception on top of a good save.
        """
        engine = self.engine
        if engine is None:
            return
        reload_config = getattr(engine, "reload_config", None)
        if reload_config is None:
            return
        try:
            reload_config()
        except Exception as exc:
            self._camera_note(
                f"Saved, but the running tracker kept its old mappings "
                f"({exc.__class__.__name__}: {exc}). Reconnect the camera "
                f"to pick them up.", DANGER)

    @staticmethod
    def _normalise_rule(entry, trigger=None):
        """One config entry → one in-memory rule, or None if unusable.

        Tolerant on purpose: this parses a file a human may have edited,
        so a missing id, cooldown or source is filled in rather than
        rejected.  Only an entry with no action at all is dropped, because
        there is nothing to bind it to.
        """
        if not isinstance(entry, dict):
            return None

        rule = dict(entry)
        kind = trigger or rule.get("trigger")
        if kind not in (TRANSITION, HOLD):
            kind = HOLD if rule.get("pose") else TRANSITION
        rule["trigger"] = kind

        if not str(rule.get("action", "")).strip():
            return None

        rule.setdefault("id", f"r{uuid.uuid4().hex[:8]}")
        rule.setdefault("enabled", True)
        rule.setdefault("cooldown_sec", 0.35)

        poses = ([rule.get("from_state"), rule.get("to_state")]
                 if kind == TRANSITION else [rule.get("pose")])
        rule.setdefault("source", DEFAULT_SOURCE)
        # Normalised rather than defaulted: "Right", "right_hand" and a
        # typo all resolve here, so the table and the engine agree on one
        # spelling whatever the file said.
        rule["hand"] = normalise_hand(rule.get("hand"))
        return rule

    def _config_to_rules(self, config: dict) -> None:
        """Load the model, accepting both the current and legacy shapes."""
        self.settings = dict(gesture_fsm.DEFAULT_SETTINGS)
        self.settings.update(config.get("settings") or {})

        rules, seen = [], set()

        def take(entries, trigger=None):
            for entry in (entries or []):
                rule = self._normalise_rule(entry, trigger)
                # Ids are how every table row, undo batch and bin entry
                # refers to a rule, so a duplicate would make two rows
                # indistinguishable.  First one wins.
                if rule is None or rule["id"] in seen:
                    continue
                seen.add(rule["id"])
                rules.append(rule)

        take(config.get("geometry_bindings"))
        take(config.get("yolo_bindings"))
        take(config.get("transitions"), TRANSITION)      # legacy
        take(config.get("holds"), HOLD)                  # legacy
        take(config.get("mappings"))                     # legacy
        self.rules = rules

        binned = []
        for entry in (config.get("deleted_bindings") or []):
            item = self._normalise_rule(entry)
            if item is None or item["id"] in seen:
                continue
            seen.add(item["id"])
            binned.append(item)
        self.deleted = binned

        self._settings_to_widgets()
        self._forget_undo()
        self._render_bin()

    def _forget_undo(self) -> None:
        """Swapping the whole list out invalidates any pending restore.

        Only the undo *stack* is cleared, not the bin itself: the bin is
        part of the config being loaded, whereas the stack's ids point at
        positions in a list that no longer exists.
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
        if not exists:
            self._toast("No config yet — starting with a blank slate. "
                        "Pick two poses to build your first mapping.",
                        AMBER)
        elif not self.rules:
            self._toast(f"{os.path.basename(CONFIG_PATH)} has no mappings — "
                        f"nothing is bound.", AMBER)
        else:
            self._toast(f"Loaded {len(self.rules)} mapping(s) from "
                        f"{os.path.basename(CONFIG_PATH)}.", MUTED)

    def _save(self) -> None:
        # No confirmation on an empty list any more.  Nothing regenerates
        # bindings behind the user's back now, so an empty scheme is a
        # choice the tool has to be able to express — warning about it on
        # every save would fight the blank slate rather than protect it.
        # The status line still says plainly that nothing is bound.
        binned = f", {len(self.deleted)} in the bin" if self.deleted else ""
        if not self.rules:
            if self.save_config():
                self._toast(f"Saved an empty scheme{binned} — no gesture is "
                            f"bound, so the cursor will move but not act.",
                            AMBER)
            return

        self.save_config(f"Saved {len(self.rules)} mapping(s){binned} to "
                         f"{os.path.basename(CONFIG_PATH)}.")

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

    # ══ TAB 2 — live camera and diagnostics ═════════════════════════════

    def _build_camera_tab(self, master) -> None:
        master.rowconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)

        # Three fixed-height cards stack to more than a laptop screen once
        # Cursor Settings joins them, and the last one packed would simply
        # be clipped.  A scrolling column keeps all three reachable at any
        # window size instead of picking one to lose.
        column = tk.Frame(master, bg=BG)
        column.grid(row=0, column=0, sticky="ns", pady=(PAD, PAD))

        side_canvas = tk.Canvas(column, bg=BG, highlightthickness=0,
                                width=262, bd=0)
        side_scroll = ttk.Scrollbar(column, orient="vertical",
                                    style="Dark.Vertical.TScrollbar",
                                    command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_canvas.pack(side="left", fill="both", expand=True)
        side_scroll.pack(side="right", fill="y")

        side = tk.Frame(side_canvas, bg=BG)
        window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>",
                  lambda _e: side_canvas.configure(
                      scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>",
                         lambda e: side_canvas.itemconfigure(window,
                                                             width=e.width))
        self._scrollers.append((side_canvas, side))

        self._build_camera_controls(side)
        # Sited between the camera and the meters because that is the order
        # you use them in: connect, tune the feel, watch the cost.  The
        # widgets and their config bindings are unchanged by the move.
        self._build_settings(side)
        self._build_diagnostics(side)

        stage = Card(master)
        stage.grid(row=0, column=1, sticky="nsew", padx=(PAD, 0),
                   pady=(PAD, PAD))
        stage.rowconfigure(1, weight=1)
        stage.columnconfigure(0, weight=1)

        head = tk.Frame(stage, bg=CARD)
        head.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(PAD, 6))
        tk.Label(head, text="LIVE PREVIEW", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(side="left")
        self.preview_meta = tk.Label(head, text="not connected", bg=CARD,
                                     fg=FAINT, font=self.f_sub)
        self.preview_meta.pack(side="left", padx=8)

        # The frame is rendered into a Label rather than a Canvas: there is
        # exactly one image, it fills the widget, and nothing is drawn on
        # top of it — the overlays are already burned in by the engine.
        self.video_label = tk.Label(
            stage, bg=FIELD, fg=FAINT, font=self.f_body,
            text="Choose a camera index and press Connect.\n\n"
                 "The tracker runs on its own thread; this tab only "
                 "displays what it produces.")
        self.video_label.grid(row=1, column=0, sticky="nsew",
                              padx=PAD, pady=(0, PAD))
        self._video_photo = None

    def _build_camera_controls(self, master) -> None:
        card = Card(master)
        card.pack(fill="x")

        tk.Label(card, text="CAMERA", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(anchor="w", padx=PAD, pady=(PAD, 2))
        tk.Label(card, text="Starts the tracking engine in the background.",
                 bg=CARD, fg=FAINT, font=("Segoe UI", 8)).pack(
            anchor="w", padx=PAD)

        tk.Label(card, text="CAMERA", bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).pack(
            anchor="w", padx=PAD, pady=(10, 2))

        # A dropdown of what is actually attached, not a spinner over
        # indices that may not exist.  Index 0 is not reliably the
        # camera anyone wants.
        self.camera_var = tk.StringVar(value="")
        self.camera_box = ttk.Combobox(
            card, textvariable=self.camera_var, state="readonly",
            values=[], style="Dark.TCombobox", font=self.f_body)
        self.camera_box.pack(fill="x", padx=PAD)
        self.camera_box.bind("<<ComboboxSelected>>",
                             lambda _e: self._on_camera_pick())

        pick_row = tk.Frame(card, bg=CARD)
        pick_row.pack(fill="x", padx=PAD, pady=(6, 0))
        self.refresh_button = FlatButton(pick_row, "Refresh Cameras",
                                         self.refresh_cameras)
        self.refresh_button.pack(side="left")
        self.change_button = FlatButton(pick_row, "Change Camera",
                                        self._change_camera)
        self.change_button.pack(side="left", padx=(8, 0))

        buttons = tk.Frame(card, bg=CARD)
        buttons.pack(fill="x", padx=PAD, pady=(12, 0))
        # ONE control, not two.  Two buttons meant two widgets that
        # could disagree with each other and with the engine; a single
        # toggle whose label is recomputed from the live state cannot.
        self.camera_button = FlatButton(buttons, "Connect Camera",
                                        self._toggle_camera, kind="accent")
        self.camera_button.pack(side="left")
        self.camera_state_label = tk.Label(buttons, text="", bg=CARD,
                                           fg=FAINT,
                                           font=("Segoe UI", 8, "bold"))
        self.camera_state_label.pack(side="left", padx=(10, 0))

        # Diagnostics, not configuration.  It opens its own window and
        # reads the rule list without touching it, which is why it sits
        # here beside the camera rather than in the Mapping Studio.
        test_row = tk.Frame(card, bg=CARD)
        test_row.pack(fill="x", padx=PAD, pady=(8, 0))
        self.test_button = FlatButton(test_row, "Test Gesture",
                                      self._open_gesture_test)
        self.test_button.pack(side="left")

        self.camera_status = tk.Label(card, text="Idle.", bg=CARD, fg=FAINT,
                                      font=("Segoe UI", 8), wraplength=232,
                                      justify="left", anchor="w")
        self.camera_status.pack(fill="x", padx=PAD, pady=(10, PAD))

    def _build_diagnostics(self, master) -> None:
        card = Card(master)
        card.pack(fill="x", pady=(PAD, 0))

        tk.Label(card, text="RESOURCES", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(anchor="w", padx=PAD, pady=(PAD, 2))
        note = ("psutil + NVML, refreshed every second."
                if psutil is not None else
                "psutil is not installed — pip install psutil")
        tk.Label(card, text=note, bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8), wraplength=232, justify="left").pack(
            anchor="w", padx=PAD)

        self.metrics = {}

        def section(title, rows):
            tk.Label(card, text=title, bg=CARD, fg=FAINT,
                     font=("Segoe UI", 8, "bold")).pack(
                anchor="w", padx=PAD, pady=(10, 2))
            for key, label in rows:
                line = tk.Frame(card, bg=CARD)
                line.pack(fill="x", padx=PAD)
                tk.Label(line, text=label, bg=CARD, fg=MUTED,
                         font=("Segoe UI", 8)).pack(side="left")
                value = tk.Label(line, text="—", bg=CARD, fg=TEXT,
                                 font=("Consolas", 9))
                value.pack(side="right")
                self.metrics[key] = value

        section("SYSTEM TOTALS", (("cpu", "CPU"), ("ram", "RAM"),
                                  ("gpu", "GPU"), ("vram", "VRAM")))
        section("THIS APP & MODEL", (("app_cpu", "App CPU"),
                                     ("app_ram", "App RAM"),
                                     ("model_vram", "Model VRAM"),
                                     ("fps", "Tracker FPS")))

        tk.Frame(card, bg=CARD, height=PAD).pack()

    # ── engine glue ─────────────────────────────────────────────────────

    # ── camera selection ───────────────────────────────────────────────
    def selected_camera(self):
        """The chosen index, or None.  The one authoritative answer."""
        return self.camera_index

    def selected_camera_label(self) -> str:
        for cam in self.cameras:
            if cam.index == self.camera_index:
                return cam.label
        if self.camera_index is None:
            return "no camera selected"
        return f"Camera {self.camera_index}"

    def refresh_cameras(self, select=None) -> None:
        """Re-enumerate, off the Tk thread.

        Probing a camera costs the best part of two seconds, so this
        cannot happen inline without freezing the window.  The result
        comes back through a queue, the same hand-off the engine start
        already uses.
        """
        if self._camera_scan_busy:
            return
        self._camera_scan_busy = True
        self._camera_note("Looking for cameras…", CYAN)
        try:
            self.refresh_button.set_enabled(False)
        except Exception:
            pass

        # The engine holds its camera open, and probing an open device
        # fails -- so the live one is carried through as present rather
        # than vanishing from its own list mid-session.
        busy = []
        if self.camera_is_live() and self.camera_index is not None:
            busy.append(self.camera_index)

        def work():
            try:
                import cameras
                found = cameras.enumerate_cameras(skip=busy)
                error = ""
            except Exception as exc:
                found, error = [], f"{exc.__class__.__name__}: {exc}"
            self._camera_scan.put((found, error, select))

        threading.Thread(target=work, daemon=True).start()
        self.after(120, self._poll_camera_scan)

    def _poll_camera_scan(self) -> None:
        try:
            found, error, select = self._camera_scan.get_nowait()
        except queue.Empty:
            if self._camera_scan_busy:
                self.after(120, self._poll_camera_scan)
            return
        self._camera_scan_busy = False
        try:
            self.refresh_button.set_enabled(True)
        except Exception:
            pass
        self.cameras = list(found)
        self._render_camera_list(error, select)

    def _render_camera_list(self, error="", select=None) -> None:
        labels = [cam.label for cam in self.cameras]
        try:
            self.camera_box.configure(values=labels)
        except Exception:
            return

        if not self.cameras:
            self.camera_index = None
            self.camera_var.set("")
            self._camera_note(
                error or "No camera detected. Connect a camera and "
                         "refresh the list.", DANGER)
            self._notify_camera_watchers()
            return

        wanted = select if select is not None else self.camera_index
        chosen = next((c for c in self.cameras if c.index == wanted),
                      self.cameras[0])
        self.camera_index = chosen.index
        self.camera_var.set(chosen.label)
        self._camera_note(
            f"{len(self.cameras)} camera(s) found. "
            f"Selected {chosen.label}.", FAINT)
        self._notify_camera_watchers()

    def _on_camera_pick(self) -> None:
        label = self.camera_var.get()
        for cam in self.cameras:
            if cam.label == label:
                self.camera_index = cam.index
                break
        self._notify_camera_watchers()

    def _change_camera(self) -> None:
        """Release the current device, then let the user choose another.

        Disconnect first and deliberately: the device has to be free
        before it can be re-enumerated, and calibration has to be told
        before the frames stop.  Mappings are untouched throughout.
        """
        if self.camera_is_live():
            self._disconnect()
        self.refresh_cameras()
        try:
            self.camera_box.focus_set()
        except Exception:
            pass

    def _notify_camera_watchers(self) -> None:
        for watcher in list(self._camera_watchers):
            try:
                watcher(self.camera_state())
            except Exception:
                self._camera_watchers.remove(watcher)

    # ── camera state ───────────────────────────────────────────────────
    def camera_state(self) -> str:
        """The one answer everything else asks.

        CONNECTED is read from the engine every time rather than
        remembered from a button press, so a camera that stops on its own
        cannot leave the UI claiming otherwise.
        """
        if self._engine_busy:
            return CAM_CONNECTING
        if self._cam_disconnecting:
            return CAM_DISCONNECTING
        engine = getattr(self, "engine", None)
        if engine is not None and getattr(engine, "running", False):
            return CAM_CONNECTED
        if self._cam_error:
            return CAM_ERROR
        return CAM_DISCONNECTED

    def camera_is_live(self) -> bool:
        return self.camera_state() == CAM_CONNECTED

    def watch_camera(self, callback) -> None:
        """Be told when the state changes.  Used by the calibration
        dialog so it can start collecting the moment a camera appears."""
        if callback not in self._camera_watchers:
            self._camera_watchers.append(callback)

    def unwatch_camera(self, callback) -> None:
        if callback in self._camera_watchers:
            self._camera_watchers.remove(callback)

    def _sync_camera_ui(self, repeat: bool = True) -> None:
        """Make the button agree with reality, then say so if it moved."""
        if repeat:
            self.after(CAMERA_SYNC_MS, self._sync_camera_ui)
        try:
            state = self.camera_state()
        except Exception:                       # pragma: no cover
            return

        label, enabled, tint = {
            CAM_CONNECTED: ("Disconnect Camera", True, ACCENT),
            CAM_CONNECTING: ("Connecting…", False, CYAN),
            CAM_DISCONNECTING: ("Disconnecting…", False, MUTED),
            CAM_ERROR: ("Connect Camera", True, DANGER),
            CAM_DISCONNECTED: ("Connect Camera", True, FAINT),
        }[state]

        try:
            self.camera_button.configure(text=label)
            self.camera_button.set_enabled(enabled)
            self.camera_state_label.configure(text=state, fg=tint)
        except tk.TclError:                     # pragma: no cover
            return

        if state != self._camera_last_state:
            self._camera_last_state = state
            for watcher in list(self._camera_watchers):
                try:
                    watcher(state)
                except Exception:
                    self._camera_watchers.remove(watcher)

    def _toggle_camera(self) -> None:
        """One button, two directions, decided by the real state."""
        state = self.camera_state()
        if state == CAM_CONNECTED:
            self._disconnect()
        elif state in (CAM_DISCONNECTED, CAM_ERROR):
            self._connect()
        # CONNECTING / DISCONNECTING: the button is disabled anyway, and
        # ignoring a stray click is better than queuing a contradiction.

    def _connect(self) -> None:
        """Start the tracker, importing it off the GUI thread.

        `import hand_cursor_2` pulls in mediapipe and costs seconds, and
        opening a camera costs more.  Doing either inline would freeze the
        window mid-click, so both happen on a worker and the result comes
        back through `after`.
        """
        if self._engine_busy:
            return
        index = self.selected_camera()
        if index is None:
            self._camera_note("No camera selected. Refresh the list and "
                              "choose one.", DANGER)
            return

        self._engine_busy = True
        self._cam_error = ""
        self._sync_camera_ui(repeat=False)
        self._camera_note(f"Starting the engine on index {index}…", CYAN)

        def work():
            try:
                if self.engine is None:
                    import hand_cursor_2
                    self.engine = hand_cursor_2.HandTrackerEngine(
                        enable_monitor=False, verbose=True)
                ok = self.engine.start(index)
                error = self.engine.status.get("error")
            except Exception as exc:
                ok, error = False, f"{exc.__class__.__name__}: {exc}"
            # Tk is not thread-safe, and `after` is no exception — calling
            # it from here raises "main thread is not in main loop".  The
            # result goes on a queue instead, and the main thread collects
            # it from the poller below.
            self._engine_result.put((ok, index, error))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll_engine_result)

    def _poll_engine_result(self) -> None:
        """Main-thread half of the connect hand-off."""
        try:
            ok, index, error = self._engine_result.get_nowait()
        except queue.Empty:
            if self._engine_busy:
                self.after(100, self._poll_engine_result)
            return
        self._connected(ok, index, error)

    def _connected(self, ok: bool, index: int, error) -> None:
        self._engine_busy = False
        self._cam_error = "" if ok else (error or "camera unavailable")
        if ok:
            # The engine adopted whatever gesture_config.json held when it
            # was constructed; the panel may have been changed since, and
            # on a reconnect it certainly has.  Pushing here makes the
            # panel authoritative from the first frame.
            self.update_engine_settings()
            self._camera_note(f"Connected to camera {index}.", ACCENT)
            self._toast(f"Tracker running on camera {index}.", ACCENT)
            self._schedule_video()
        else:
            self._camera_note(f"Could not open camera {index}. {error or ''}",
                              DANGER)
        self._sync_camera_ui(repeat=False)

    def _disconnect(self) -> None:
        """Stop the camera cleanly, taking everything that rides on it.

        Order matters: the calibration session is ended and the probe
        detached BEFORE the engine stops, so nothing is left recording
        into a buffer that will never be drained.  Mappings are not
        touched -- disconnecting a camera is not a reason to lose
        configuration.
        """
        if self.engine is None:
            return
        self._cam_disconnecting = True
        self._sync_camera_ui(repeat=False)

        panel = getattr(self, "_calibration", None)
        if panel is not None:
            try:
                if panel.winfo_exists():
                    panel.camera_lost()
            except tk.TclError:
                pass
        try:
            self.engine.probe = None
        except Exception:
            pass

        # Stopping the loop ends the cursor freeze with it: the freeze is
        # loop-local state, not a flag that outlives the thread.
        self.engine.stop()
        self._cam_disconnecting = False
        self._video_photo = None
        self.video_label.configure(
            image="", text="Disconnected. Press Connect to resume.")
        self.video_label.image = None
        self.preview_meta.configure(text="not connected")
        self._camera_note("Camera released.", MUTED)
        self._sync_camera_ui(repeat=False)

    def _open_gesture_test(self) -> None:
        """Open the gesture test bench, or raise the one already open.

        Imported here rather than at module scope so a broken or missing
        gesture_test.py costs this button and nothing else — the studio
        still starts and every mapping still works.
        """
        existing = getattr(self, "_test_panel", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.deiconify()
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass
        try:
            import gesture_test
            self._test_panel = gesture_test.GestureTestPanel(self)
        except Exception as exc:
            self._toast(f"Gesture testing unavailable "
                        f"({exc.__class__.__name__}: {exc})", DANGER)

    def _camera_note(self, message: str, colour: str = FAINT) -> None:
        self.camera_status.configure(text=message, fg=colour)

    def _schedule_video(self) -> None:
        if self._video_job is None:
            self._video_job = self.after(30, self.update_video_feed)

    def update_video_feed(self) -> None:
        """Draw the newest engine frame, at ~33 Hz."""
        self._video_job = None
        engine = self.engine

        if engine is None or not engine.running:
            return

        frame = engine.get_latest_frame()
        if frame is not None:
            width = max(160, self.video_label.winfo_width())
            height = max(120, self.video_label.winfo_height())
            # [:, :, ::-1] is the BGR→RGB swap; done with a numpy view so
            # the GUI needs no OpenCV import of its own.
            image = Image.fromarray(frame[:, :, ::-1])
            image.thumbnail((width, height), Image.BILINEAR)
            self._video_photo = ImageTk.PhotoImage(image)
            self.video_label.configure(image=self._video_photo, text="")
            self.video_label.image = self._video_photo
            self.preview_meta.configure(
                text=f"{frame.shape[1]}×{frame.shape[0]}  "
                     f"{engine.status.get('fps', 0.0):.0f} fps")

        self._schedule_video()

    def _refresh_metrics(self) -> None:
        """One pass over psutil and NVML.  Never raises, never blocks."""
        self._monitor_job = self.after(1000, self._refresh_metrics)

        def put(key, text):
            widget = self.metrics.get(key)
            if widget is not None:
                widget.configure(text=text)

        if psutil is not None:
            try:
                put("cpu", f"{psutil.cpu_percent(interval=None):5.1f} %")
                memory = psutil.virtual_memory()
                put("ram", f"{memory.used / GIB:.1f} / "
                           f"{memory.total / GIB:.1f} GiB")
                if self._proc is None:
                    self._proc = psutil.Process(os.getpid())
                # cpu_percent is relative to the last call on this object,
                # so the first reading is always 0.0 and the rest are real.
                share = self._proc.cpu_percent(interval=None) / max(
                    1, psutil.cpu_count())
                put("app_cpu", f"{share:5.1f} %")
                put("app_ram", f"{self._proc.memory_info().rss / MIB:.0f} MiB")
            except Exception:
                for key in ("cpu", "ram", "app_cpu", "app_ram"):
                    put(key, "n/a")
        else:
            for key in ("cpu", "ram", "app_cpu", "app_ram"):
                put(key, "psutil?")

        if _NVML_READY:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                put("gpu", f"{util.gpu:5d} %")
                put("vram", f"{info.used / MIB:.0f} / "
                            f"{info.total / MIB:.0f} MiB")

                # Our own slice of VRAM, which is what the YOLO weights
                # actually occupy — the total above includes every other
                # process on the card.
                mine = 0
                running = pynvml.nvmlDeviceGetComputeRunningProcesses(
                    handle)
                for proc in running:
                    if proc.pid == os.getpid() and proc.usedGpuMemory:
                        mine = proc.usedGpuMemory
                put("model_vram", f"{mine / MIB:.0f} MiB" if mine
                    else CPU_MODE_NOTE)
            except Exception:
                for key in ("gpu", "vram", "model_vram"):
                    put(key, "n/a")
        else:
            for key in ("gpu", "vram", "model_vram"):
                put(key, CPU_MODE_NOTE)

        engine = self.engine
        put("fps", f"{engine.status.get('fps', 0.0):5.1f}"
            if engine is not None and engine.running else "—")

    # ══ TAB 3 — recycle bin ═════════════════════════════════════════════

    def _build_bin_tab(self, master) -> None:
        master.rowconfigure(1, weight=1)
        master.columnconfigure(0, weight=1)

        head = tk.Frame(master, bg=BG)
        head.grid(row=0, column=0, sticky="ew", pady=(PAD, 8))
        tk.Label(head, text="Recycle Bin", bg=BG, fg=TEXT,
                 font=self.f_title).pack(anchor="w")
        tk.Label(head,
                 text="Deleted mappings are kept here and saved to "
                      "gesture_config.json, so nothing is lost until you "
                      "delete it permanently.",
                 bg=BG, fg=MUTED, font=self.f_sub).pack(anchor="w")

        body = tk.Frame(master, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", pady=(0, PAD))
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        # ── list ───────────────────────────────────────────────────────
        left = Card(body)
        left.grid(row=0, column=0, sticky="nsew")

        bar = tk.Frame(left, bg=CARD)
        bar.pack(fill="x", padx=PAD, pady=(PAD, 8))
        tk.Label(bar, text="DELETED MAPPINGS", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(side="left")
        self.bin_count = tk.Label(bar, text="", bg=CARD, fg=FAINT,
                                  font=self.f_sub)
        self.bin_count.pack(side="left", padx=8)

        # Only the list-wide action stays here; the per-mapping pair lives
        # at the bottom of the details card, next to what they act on.
        buttons = tk.Frame(left, bg=CARD)
        buttons.pack(side="bottom", fill="x", padx=PAD, pady=PAD)
        FlatButton(buttons, "Empty Bin", self.empty_bin,
                   kind="danger").pack(side="left")
        tk.Label(buttons, text="Select a row to see it on the right.",
                 bg=CARD, fg=FAINT, font=("Segoe UI", 8)).pack(side="left",
                                                               padx=10)

        holder = tk.Frame(left, bg=CARD)
        holder.pack(side="top", fill="both", expand=True, padx=PAD)

        columns = ("trigger", "gestures", "action", "when")
        self.bin_table = ttk.Treeview(holder, columns=columns,
                                      show="headings", style="Dark.Treeview",
                                      selectmode="extended", height=6)
        for key, title, width in (("trigger", "Trigger", 90),
                                  ("gestures", "Gestures", 200),
                                  ("action", "Action", 170),
                                  ("when", "Deleted", 90)):
            self.bin_table.heading(key, text=title)
            self.bin_table.column(key, width=width,
                                  stretch=key == "gestures")
        scroll = ttk.Scrollbar(holder, orient="vertical",
                               style="Dark.Vertical.TScrollbar",
                               command=self.bin_table.yview)
        self.bin_table.configure(yscrollcommand=scroll.set)
        self.bin_table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.bin_table.tag_configure("odd", background="#232327")
        self.bin_table.bind("<<TreeviewSelect>>",
                            lambda _e: self._show_bin_details())
        self.bin_table.bind("<Double-1>", lambda _e: self.restore_selected())

        # ── details ────────────────────────────────────────────────────
        right = Card(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(PAD, 0))
        self._build_bin_details(right)

        self._render_bin()

    def _build_bin_details(self, card) -> None:
        """The details card: poses as pictures, then the numbers, then act.

        Built once and repopulated on selection.  The two per-mapping
        buttons sit at the bottom of this card rather than under the list,
        so the thing being restored or destroyed is on screen beside them.
        """
        tk.Label(card, text="DETAILS", bg=CARD, fg=TEXT,
                 font=self.f_head).pack(anchor="w", padx=PAD, pady=(PAD, 2))
        self.bin_subtitle = tk.Label(card, text="", bg=CARD, fg=FAINT,
                                     font=("Segoe UI", 8))
        self.bin_subtitle.pack(anchor="w", padx=PAD)

        # Buttons first, against the bottom: the panels above expand, and
        # whatever is packed after an expanding widget loses its space.
        actions = tk.Frame(card, bg=CARD)
        actions.pack(side="bottom", fill="x", padx=PAD, pady=PAD)
        self.bin_restore_button = FlatButton(
            actions, "Restore Mapping", self.restore_selected, kind="accent")
        self.bin_restore_button.pack(side="left")
        self.bin_purge_button = FlatButton(
            actions, "Permanently Delete", self.purge_selected,
            kind="danger-solid")
        self.bin_purge_button.pack(side="right")

        # The pose strip: one tile for a hold, two and an arrow for a
        # transition.  Rebuilt per selection because the shape changes.
        self.bin_poses = tk.Frame(card, bg=CARD)
        self.bin_poses.pack(fill="x", padx=PAD, pady=(12, 0))

        self.bin_fields = tk.Frame(card, bg=CARD)
        self.bin_fields.pack(fill="both", expand=True, padx=PAD, pady=(14, 0))
        self.bin_fields.columnconfigure(1, weight=1)

        self.bin_empty = tk.Label(
            card, text="Select a deleted mapping to see it here.",
            bg=CARD, fg=FAINT, font=self.f_body)
        self.bin_empty.pack(fill="both", expand=True, padx=PAD, pady=PAD)

    def _pose_tile(self, parent, label, caption) -> tk.Frame:
        """One framed gesture image with its name underneath.

        Reuses the catalog's already-decoded PhotoImages: they are the
        gestures/ PNGs at 76 px, re-inked for the dark theme, with a
        lettered placeholder standing in for the poses that ship no art.
        Decoding them a second time here would only add a delay.
        """
        tile = tk.Frame(parent, bg=CARD)

        box = tk.Frame(tile, bg=FIELD, width=SLOT, height=SLOT,
                       highlightbackground=BORDER, highlightthickness=1)
        box.pack_propagate(False)
        box.pack()

        photo = self.thumbs.get(label)
        image = tk.Label(box, bg=FIELD, fg=TEXT, font=("Segoe UI", 14, "bold"))
        if photo is not None:
            image.configure(image=photo)
            image.image = photo
        else:
            image.configure(text=pretty(label or "?")[:2].upper())
        image.pack(expand=True)

        tk.Label(tile, text=caption, bg=CARD, fg=FAINT,
                 font=("Segoe UI", 7, "bold")).pack(pady=(5, 0))
        tk.Label(tile, text=pretty(label or "—"), bg=CARD, fg=ACCENT,
                 font=("Segoe UI", 8, "bold"),
                 wraplength=SLOT + 24).pack()
        return tile

    def _bin_field(self, label: str, value: str,
                   colour: str = TEXT) -> None:
        """Append one label/value row.  Owns the counter so callers cannot
        drift out of step with it."""
        row = self._bin_row
        self._bin_row += 1
        tk.Label(self.bin_fields, text=label, bg=CARD, fg=FAINT,
                 font=("Segoe UI", 8, "bold")).grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 12))
        tk.Label(self.bin_fields, text=value, bg=CARD, fg=colour,
                 font=("Consolas", 9), anchor="w").grid(
            row=row, column=1, sticky="ew", pady=2)

    def _render_bin(self) -> None:
        if not hasattr(self, "bin_table"):
            return
        self.bin_table.delete(*self.bin_table.get_children())
        for index, rule in enumerate(self.deleted):
            if rule.get("trigger") == TRANSITION:
                gestures = (f"{pretty(rule.get('from_state', '?'))}  →  "
                            f"{pretty(rule.get('to_state', '?'))}")
            else:
                gestures = f"hold {pretty(rule.get('pose', '?'))}"
            action = rule.get("action", "?")
            if action == KEYBOARD_MACRO:
                action = f"{action}  {rule.get('keys', '')}"
            stamp = rule.get("_deleted_at")
            when = (time.strftime("%H:%M:%S", time.localtime(stamp))
                    if stamp else "—")
            self.bin_table.insert(
                "", "end", iid=rule["id"],
                values=(str(rule.get("trigger", "?")).capitalize(),
                        gestures, action, when),
                tags=("odd",) if index % 2 else ())
        self.bin_count.configure(
            text=f"{len(self.deleted)} item(s)"
            + ("" if self.deleted else " — nothing deleted yet"))
        self._show_bin_details()

    def _show_bin_details(self) -> None:
        """Repaint the details card for whatever the bin list has selected."""
        selection = self.bin_table.selection()
        rule = None
        for entry in self.deleted:
            if selection and entry["id"] == selection[0]:
                rule = entry
                break

        for child in self.bin_poses.winfo_children():
            child.destroy()
        for child in self.bin_fields.winfo_children():
            child.destroy()

        if rule is None:
            self.bin_poses.pack_forget()
            self.bin_fields.pack_forget()
            self.bin_empty.pack(fill="both", expand=True, padx=PAD, pady=PAD)
            self.bin_subtitle.configure(text="")
            self.bin_restore_button.set_enabled(False)
            self.bin_purge_button.set_enabled(False)
            return

        self.bin_empty.pack_forget()
        self.bin_poses.pack(fill="x", padx=PAD, pady=(12, 0))
        self.bin_fields.pack(fill="both", expand=True, padx=PAD, pady=(14, 0))
        self.bin_restore_button.set_enabled(True)
        self.bin_purge_button.set_enabled(True)

        name = rule.get("name") or rule.get("action", "mapping")
        self.bin_subtitle.configure(text=name)

        strip = tk.Frame(self.bin_poses, bg=CARD)
        strip.pack()

        if rule.get("trigger") == TRANSITION:
            self._pose_tile(strip, rule.get("from_state"),
                            "FROM POSE").pack(side="left")
            tk.Label(strip, text="→", bg=CARD, fg=ACCENT,
                     font=("Segoe UI", 18, "bold")).pack(side="left", padx=12,
                                                         pady=(0, 24))
            self._pose_tile(strip, rule.get("to_state"),
                            "TO POSE").pack(side="left")
        else:
            self._pose_tile(strip, rule.get("pose"), "POSE").pack(side="left")

        self._bin_row = 0
        self._bin_field("TRIGGER",
                        str(rule.get("trigger", "—")).capitalize())

        if rule.get("trigger") == TRANSITION:
            self._bin_field("MAX TIME", timing_label(rule))
        else:
            delay = rule.get("hold_sec", 0.0)
            self._bin_field("HOLD FOR",
                            f"{delay:.2f} s" if delay > 0 else "instant")
            self._bin_field("REPEAT",
                            "yes" if rule.get("repeat") else "no")

        action = rule.get("action", "—")
        self._bin_field("ACTION", action, ACCENT)

        chord = rule.get("keys") or ACTION_MACROS.get(action, "")
        self._bin_field("KEYS", chord or "—")
        self._bin_field("COOLDOWN",
                        f"{rule.get('cooldown_sec', 0.0):.2f} s")

        if rule.get("trigger") == TRANSITION:
            self._bin_field("PROMOTE 2×",
                            "yes" if rule.get("promote_double") else "no")

        path = {"geometry": "30 Hz geometric", "semantic": "YOLO semantic"}
        self._bin_field("PATH",
                        path.get(rule.get("source", "any"), "either path"),
                        CYAN if rule.get("source") == "semantic" else TEXT)
        self._bin_field("ENABLED",
                        "yes" if rule.get("enabled", True) else "no")

        stamp = rule.get("_deleted_at")
        self._bin_field("DELETED",
                        time.strftime("%H:%M:%S", time.localtime(stamp))
                        if stamp else "—")
        self._bin_field("RULE ID", str(rule.get("id", "—")), FAINT)

    def _selected_bin_ids(self) -> list:
        chosen = set(self.bin_table.selection())
        return [r["id"] for r in self.deleted if r["id"] in chosen]

    def restore_selected(self) -> None:
        """Move the selected mappings back into the active list."""
        ids = self._selected_bin_ids()
        if not ids:
            self._toast("Select something in the bin to restore.", AMBER)
            return

        restored, disabled = self._restore_ids(ids)
        if not restored:
            self._toast("Nothing was restored.", AMBER)
            return

        self._render_table()
        self._render_bin()

        saved = self.save_config()
        note = f"Restored {restored} mapping(s)"
        if disabled:
            note += (f" — {disabled} came back DISABLED because its trigger "
                     f"is already bound")
        note += "." if saved else " but the config could NOT be written."
        self._toast(note, AMBER if disabled or not saved else ACCENT)

    def _restore_ids(self, ids):
        """Put binned rules back where they came from.

        Returns (restored, disabled).  Two things make this more than a
        list move:

        Ascending origin order — each insert shifts everything after it, so
        restoring high-index rules first would land the rest one slot short
        of where they started.

        Trigger collisions — the same gesture pair may have been re-bound
        while the rule sat in the bin.  Two rules on one trigger both fire,
        which is a broken control scheme rather than a conflict the user
        can see, so the returning rule comes back disabled and says so.

        ADAPTIVE STATE IS NOT RESURRECTED.  Deleting retires the pose
        pair's learning outright, so a restored mapping comes back to a
        fresh context: baseline tolerance, zero samples, learning again
        from what it sees.  The alternative — holding the old history
        aside in case of an undo — would mean a "deleted" mapping still
        had state in the engine, which is exactly the influence deletion
        is supposed to remove.  Restoring stale state cannot be done
        atomically with the rule move either, and a half-restored
        estimator is worse than an honest fresh start.
        """
        wanted_ids = set(ids)
        wanted = [entry for entry in self.deleted
                  if entry.get("id") in wanted_ids]
        wanted.sort(key=lambda entry: self._origin_of(entry))

        restored = disabled = 0
        for entry in wanted:
            try:
                self.deleted.remove(entry)
            except ValueError:                        # pragma: no cover
                continue

            rule = {key: value for key, value in entry.items()
                    if key not in ("_origin_index", "_deleted_at")}
            rule.setdefault("enabled", True)

            if rule.get("enabled", True) and self._conflict(rule) is not None:
                rule["enabled"] = False
                disabled += 1

            index = self._origin_of(entry)
            self.rules.insert(min(max(0, index), len(self.rules)), rule)
            restored += 1

        self._prune_undo(wanted_ids)
        return restored, disabled

    def _origin_of(self, entry) -> int:
        """The index a binned rule should return to, coerced to a sane int."""
        try:
            return max(0, int(entry.get("_origin_index", len(self.rules))))
        except (TypeError, ValueError):
            return len(self.rules)

    def _prune_undo(self, gone) -> None:
        """Drop ids that have left the bin from every pending undo batch."""
        self._undo_stack = [[rule_id for rule_id in batch
                             if rule_id not in gone]
                            for batch in self._undo_stack]
        self._undo_stack = [batch for batch in self._undo_stack if batch]
        if not self._undo_stack and hasattr(self, "undo_button"):
            self.undo_button.set_enabled(False)

    def purge_selected(self) -> None:
        """Hard-delete the selected mappings.  This one is irreversible."""
        ids = set(self._selected_bin_ids())
        if not ids:
            self._toast("Select something in the bin to delete.", AMBER)
            return
        if not messagebox.askokcancel(
                "Delete permanently?",
                f"Permanently delete {len(ids)} mapping(s)?\n\n"
                f"This cannot be undone.", parent=self,
                icon=messagebox.WARNING, default=messagebox.CANCEL):
            return
        # HARD DELETE.  Removed by identity from the bin and from every
        # pending undo batch, so nothing can point at it afterwards.
        removed = 0
        for entry in [e for e in self.deleted if e.get("id") in ids]:
            try:
                self.deleted.remove(entry)
                removed += 1
            except ValueError:                        # pragma: no cover
                continue

        self._forget_learning(
            [e for e in self.deleted if e.get("id") in ids])
        self._prune_undo(ids)
        self._render_bin()

        saved = self.save_config()
        self._toast(
            f"Permanently deleted {removed} mapping(s)"
            + ("." if saved else " but the config could NOT be written."),
            DANGER)

    def empty_bin(self) -> None:
        if not self.deleted:
            self._toast("The bin is already empty.", AMBER)
            return
        if not messagebox.askokcancel(
                "Empty the bin?",
                f"Permanently delete all {len(self.deleted)} mapping(s) in "
                f"the bin?\n\nThis cannot be undone.", parent=self,
                icon=messagebox.WARNING, default=messagebox.CANCEL):
            return
        count = len(self.deleted)
        self.deleted.clear()
        self._undo_stack.clear()
        if hasattr(self, "undo_button"):
            self.undo_button.set_enabled(False)
        self._render_bin()

        saved = self.save_config()
        self._toast(
            f"Bin emptied — {count} mapping(s) gone"
            + ("." if saved else " but the config could NOT be written."),
            DANGER)

    # ── status bar ──────────────────────────────────────────────────────

    def _build_status(self) -> None:
        bar = tk.Frame(self._page, bg=BG)
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

        for job in (self._video_job, self._monitor_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self._video_job = self._monitor_job = None

        # The engine owns a camera, two worker threads and possibly a held
        # mouse button.  Closing it before destroy() is what stops the
        # device staying locked and the button staying down after we exit.
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass
            self.engine = None

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
