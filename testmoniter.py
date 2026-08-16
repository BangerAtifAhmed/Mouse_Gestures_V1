"""
Live system + per-process resource monitor for the hand-gesture cursor.

Launched by test.py when the camera opens, terminated when it shuts down.
Takes the PID to watch as argv[1]:

    python testmoniter.py <pid>

Its own process on purpose: NVML queries and Tk each want a thread of their
own, and neither belongs near the 30 Hz cursor loop.

ON "MODEL VRAM", because the honest answer has four cases and only three of
them are what you would guess:

  * no pynvml, or no NVIDIA GPU        -> "N/A (CPU Mode)"
  * the PID is not on the GPU at all   -> "N/A (CPU Mode)"
  * NVML reports its memory            -> the real figure
  * NVML lists the PID but reports no
    memory for it                      -> device-delta estimate, "est."

That last case is the normal one on Windows and it is NOT CPU mode.
Measured here — RTX 3050, driver 610.88, driver model WDDM — a child that
had provably allocated 256 MB of CUDA memory came back with
usedGpuMemory = None, as did all 27 other compute processes, from all three
API versions.  Under WDDM the display driver owns allocation and does not
tell NVML the per-process split; TCC mode reports it, but TCC is not
available on a GeForce part that is also driving a monitor.

PRESENCE in the compute list is still trustworthy, and that is what
separates real CPU mode from unreadable GPU mode: measured, a CUDA child
appears in the list and a busy CPU-only child does not.  So "N/A (CPU Mode)"
means it, and the estimate is only ever shown for a process genuinely on the
GPU.  The estimate is the rise in device VRAM since this window opened —
fair here because the monitor starts when the camera opens and the model
loads about eleven seconds later, but it moves if anything else on the GPU
does, which is why it is labelled rather than presented as measurement.
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk

import psutil

try:
    import pynvml
    _NVML_IMPORTED = True
except Exception:                       # pragma: no cover - optional
    pynvml = None
    _NVML_IMPORTED = False

REFRESH_MS = 1000
MIB = 1024 ** 2
GIB = 1024 ** 3

CPU_MODE = "N/A (CPU Mode)"

BG = "#161616"
CARD = "#1e1e1e"
FG = "#e6e6e6"
DIM = "#8a8a8a"
SYS_ACCENT = "#00c8ff"
APP_ACCENT = "#c792ea"

WARN_AT = 75.0
CRIT_AT = 90.0
OK_COLOUR = "#3ddc84"
WARN_COLOUR = "#ffb300"
CRIT_COLOUR = "#ff5252"


def human_bytes(value: float) -> str:
    """MB below a gibibyte, GB above it — the units people expect."""
    if value >= GIB:
        return f"{value / GIB:.2f} GB"
    return f"{value / MIB:.0f} MB"


class Meter(ttk.Frame):
    """One row: name, progress bar, right-aligned readout, detail line."""

    def __init__(self, parent, name: str, accent: str = SYS_ACCENT):
        super().__init__(parent, style="Card.TFrame")
        self.columnconfigure(1, weight=1)
        self._accent = accent

        tk.Label(self, text=name, bg=CARD, fg=FG, width=11, anchor="w",
                 font=("Segoe UI", 9, "bold")).grid(
                     row=0, column=0, sticky="w", padx=(12, 6), pady=(6, 0))

        self._bar = ttk.Progressbar(self, maximum=100.0, length=150,
                                    style="OK.Horizontal.TProgressbar")
        self._bar.grid(row=0, column=1, sticky="ew", pady=(6, 0))

        self._value = tk.Label(self, text="--", bg=CARD, fg=accent, width=13,
                               anchor="e", font=("Consolas", 10))
        self._value.grid(row=0, column=2, sticky="e", padx=(6, 12),
                         pady=(6, 0))

        self._detail = tk.Label(self, text="", bg=CARD, fg=DIM, anchor="w",
                                font=("Segoe UI", 8))
        self._detail.grid(row=1, column=0, columnspan=3, sticky="w",
                          padx=12, pady=(0, 6))

    def set(self, percent, value_text: str = "", detail: str = "") -> None:
        if percent is None:
            self._value.config(text=value_text or "n/a", fg=DIM)
            self._bar.config(value=0.0,
                             style="OK.Horizontal.TProgressbar")
        else:
            percent = max(0.0, min(100.0, float(percent)))
            self._value.config(text=value_text or f"{percent:.1f}%",
                               fg=self._accent)
            self._bar.config(value=percent)
            if percent >= CRIT_AT:
                style = "CRIT.Horizontal.TProgressbar"
            elif percent >= WARN_AT:
                style = "WARN.Horizontal.TProgressbar"
            else:
                style = "OK.Horizontal.TProgressbar"
            self._bar.config(style=style)
        self._detail.config(text=detail)


class Gpu:
    """Every NVML call lives behind this and returns None rather than raising.

    Nothing here is allowed to propagate an exception: a driver that goes
    away mid-session, a laptop switching to integrated graphics, or a
    machine with no NVIDIA hardware at all must all degrade to a label.
    """

    def __init__(self):
        self.ok = False
        self.reason = CPU_MODE
        self.name = "no NVIDIA GPU"
        self.per_process_supported = False
        self._handle = None
        self._baseline = None

        if not _NVML_IMPORTED:
            self.name = "pynvml not installed"
            return
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() < 1:
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetName(self._handle)
            self.name = raw.decode() if isinstance(raw, bytes) else raw
            self.ok = True
            self.reason = ""
        except Exception as exc:
            self.ok = False
            self.name = f"NVML unavailable ({exc.__class__.__name__})"

    def shutdown(self) -> None:
        if self.ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def totals(self):
        """(util_percent, used_bytes, total_bytes).  All None when unread."""
        if not self.ok:
            return None, None, None
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        except Exception:
            return None, None, None
        if self._baseline is None:
            self._baseline = mem.used
        return float(util.gpu), float(mem.used), float(mem.total)

    def _compute_entry(self, pid: int):
        """(found, used_bytes_or_None).  Tries every API version there is."""
        for fn_name in ("nvmlDeviceGetComputeRunningProcesses_v3",
                        "nvmlDeviceGetComputeRunningProcesses_v2",
                        "nvmlDeviceGetComputeRunningProcesses"):
            fn = getattr(pynvml, fn_name, None)
            if fn is None:
                continue
            try:
                procs = fn(self._handle)
            except Exception:
                continue
            for proc in procs:
                if proc.pid == pid:
                    return True, getattr(proc, "usedGpuMemory", None)
            return False, None
        return False, None

    def process_vram(self, pid: int):
        """(bytes_or_None, state) where state is 'measured', 'estimate' or
        'cpu'.  'cpu' is the only one that becomes N/A (CPU Mode)."""
        if not self.ok:
            return None, "cpu"

        try:
            found, used = self._compute_entry(pid)
        except Exception:
            return None, "cpu"

        if not found:
            # Genuinely not on the GPU — verified separately that a CUDA
            # process appears in this list and a CPU-only one does not.
            return None, "cpu"

        if used:
            self.per_process_supported = True
            return float(used), "measured"

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        except Exception:
            return None, "cpu"
        if self._baseline is None:
            self._baseline = mem.used
        return max(0.0, float(mem.used) - float(self._baseline)), "estimate"


class SystemMonitor(tk.Tk):

    def __init__(self, pid: int):
        super().__init__()
        self.title("Resource Monitor")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.wm_attributes("-topmost", True)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._pid = pid
        self._cores = psutil.cpu_count(logical=True) or 1
        self._gpu = Gpu()
        self._running = True

        try:
            self._proc = psutil.Process(pid)
            self._proc.cpu_percent(interval=None)       # prime
        except Exception:
            self._proc = None

        self._styles()

        self._section("SYSTEM TOTALS")
        self._sys_cpu = self._row("CPU")
        self._sys_ram = self._row("RAM")
        self._sys_gpu = self._row("GPU")
        self._sys_vram = self._row("VRAM")

        self._section("APP & MODEL USAGE", top=8)
        self._app_cpu = self._row("App CPU", APP_ACCENT)
        self._app_ram = self._row("App RAM", APP_ACCENT)
        self._app_vram = self._row("Model VRAM", APP_ACCENT)

        self._footer = tk.Label(self, bg=BG, fg=DIM, font=("Segoe UI", 8),
                                anchor="w", justify="left")
        self._footer.pack(anchor="w", padx=12, pady=(6, 8))

        psutil.cpu_percent(interval=None)               # prime
        self._refresh()

    # ── layout ─────────────────────────────────────────────────────────
    def _section(self, text: str, top: int = 10) -> None:
        tk.Label(self, text=text, bg=BG, fg=DIM,
                 font=("Segoe UI", 8, "bold")).pack(
                     anchor="w", padx=12, pady=(top, 2))

    def _row(self, name: str, accent: str = SYS_ACCENT) -> Meter:
        meter = Meter(self, name, accent)
        meter.pack(fill="x", padx=6, pady=1)
        return meter

    def _styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")     # the only built-in that recolours bars
        except tk.TclError:             # pragma: no cover
            pass
        style.configure("Card.TFrame", background=CARD)
        for tag, colour in (("OK", OK_COLOUR), ("WARN", WARN_COLOUR),
                            ("CRIT", CRIT_COLOUR)):
            style.configure(f"{tag}.Horizontal.TProgressbar",
                            troughcolor="#2b2b2b", bordercolor=CARD,
                            background=colour, lightcolor=colour,
                            darkcolor=colour, thickness=9)

    # ── sampling ───────────────────────────────────────────────────────
    def _refresh(self) -> None:
        """One tick.  Reschedules itself no matter what goes wrong inside.

        The outer guard is the difference between a monitor that reports a
        bad reading and a monitor that silently stops updating: a single
        unhandled exception in a Tk callback kills the chain, and the window
        would sit there showing stale numbers forever.
        """
        if not self._running:
            return
        try:
            self._sample_system()
            self._sample_app()
        except Exception as exc:        # last resort, never reached in tests
            self._footer.config(text=f"sampling error: "
                                     f"{exc.__class__.__name__}")
        finally:
            if self._running:
                self.after(REFRESH_MS, self._refresh)

    def _sample_system(self) -> None:
        self._sys_cpu.set(psutil.cpu_percent(interval=None),
                          detail=f"{self._cores} logical cores")

        mem = psutil.virtual_memory()
        self._sys_ram.set(mem.percent,
                          detail=f"{human_bytes(mem.used)} / "
                                 f"{human_bytes(mem.total)}")

        util, used, total = self._gpu.totals()
        self._sys_gpu.set(util, detail=self._gpu.name)
        if total:
            self._sys_vram.set(used / total * 100.0,
                               value_text=human_bytes(used),
                               detail=f"{human_bytes(used)} / "
                                      f"{human_bytes(total)} device-wide")
        else:
            self._sys_vram.set(None, value_text=CPU_MODE,
                               detail=self._gpu.name)

    def _sample_app(self) -> None:
        if self._proc is None or not self._proc.is_running():
            for row in (self._app_cpu, self._app_ram, self._app_vram):
                row.set(None, detail=f"pid {self._pid} is not running")
            self._footer.config(text="watched process ended — closing")
            # An always-on-top window outliving the app it describes is
            # worse than closing unasked.
            self.after(1500, self._close)
            return

        try:
            raw_cpu = self._proc.cpu_percent(interval=None)
            rss = self._proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            for row in (self._app_cpu, self._app_ram, self._app_vram):
                row.set(None, detail=exc.__class__.__name__)
            return

        # psutil returns percent of ONE core, so it can exceed 100 on a
        # multi-core box.  Dividing by the core count puts it on the same
        # scale as the System CPU row, which is what makes them comparable.
        share = raw_cpu / self._cores
        self._app_cpu.set(share, value_text=f"{share:.1f}%",
                          detail=f"{raw_cpu:.0f}% of one core  ·  "
                                 f"pid {self._pid}")

        total_ram = psutil.virtual_memory().total
        self._app_ram.set(rss / total_ram * 100.0,
                          value_text=human_bytes(rss),
                          detail="resident set size")

        _, _, vram_total = self._gpu.totals()
        vram, state = self._gpu.process_vram(self._pid)

        if state == "cpu":
            self._app_vram.set(None, value_text=CPU_MODE,
                               detail=self._gpu.name if self._gpu.ok
                               else "no NVIDIA GPU available")
            self._footer.config(text=f"refreshing every {REFRESH_MS} ms")
            return

        pct = (vram / vram_total * 100.0) if vram_total else None
        if state == "measured":
            self._app_vram.set(pct, value_text=human_bytes(vram),
                               detail="reported by NVML")
            self._footer.config(text=f"refreshing every {REFRESH_MS} ms")
        else:
            self._app_vram.set(pct, value_text=human_bytes(vram),
                               detail="est. — rise in device VRAM since start")
            self._footer.config(
                text="Model VRAM is an estimate: this driver (WDDM) does\n"
                     "not report per-process GPU memory.")

    def _close(self) -> None:
        self._running = False
        self._gpu.shutdown()
        self.destroy()


def main() -> None:
    pid = os.getppid()
    if len(sys.argv) > 1:
        try:
            pid = int(sys.argv[1])
        except ValueError:
            print(f"[monitor] bad pid {sys.argv[1]!r}, watching parent {pid}")

    try:
        app = SystemMonitor(pid)
    except tk.TclError as exc:          # no display, or Tk missing
        print(f"[monitor] cannot open a window: {exc}")
        return
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

