"""
A frame progress bar for the headless tools.

Uses tqdm when it is installed and falls back to a small built-in bar
otherwise — the fallback exists because this project ships as a Nuitka
build, and making a progress bar a hard dependency would mean bundling
tqdm for something purely cosmetic.

    from src.GUI.pipeline_editor.progress import frame_progress

    with frame_progress(total, "my_pipeline.json") as bar:
        for index, total, frame in pipeline.iter_sequence():
            bar.update(1)

The bar writes to stderr, so piping stdout to a file keeps the log clean
and still shows progress on the terminal. It disables itself when stderr
is not a terminal, so a redirected run does not accumulate thousands of
bar redraws in a file.
"""
import sys
import time
from contextlib import contextmanager

from src.GUI.pipeline_editor import run_log

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


class _SimpleBar:
    """Minimal stand-in for tqdm: one line, rewritten in place."""

    WIDTH = 28

    def __init__(self, total: int, label: str, stream):
        self.total = max(1, int(total))
        self.label = label
        self.stream = stream
        self.n = 0
        self.start = time.perf_counter()
        self._last_draw = 0.0
        self._draw()

    def update(self, count: int = 1):
        self.n += count
        # Redraw at most ~20x a second: a fast pipeline would otherwise
        # spend real time formatting the bar.
        now = time.perf_counter()
        if now - self._last_draw < 0.05 and self.n < self.total:
            return
        self._last_draw = now
        self._draw()

    def _draw(self):
        frac = min(1.0, self.n / self.total)
        filled = int(self.WIDTH * frac)
        bar = "#" * filled + "-" * (self.WIDTH - filled)
        elapsed = time.perf_counter() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        self.stream.write(
            f"\r  {self.label[:28]:28s} [{bar}] {self.n}/{self.total} "
            f"{rate:5.1f} f/s eta {eta:5.1f}s")
        self.stream.flush()

    def close(self):
        self._draw()
        self.stream.write("\n")
        self.stream.flush()


class _NullBar:
    def update(self, count: int = 1):
        pass

    def close(self):
        pass


@contextmanager
def frame_progress(total: int, label: str = "", stream=None):
    """Progress over `total` frames. Yields an object with .update(n).

    Silent when stderr is not a terminal, or when a per-frame --log
    channel is active — a bar redrawing between log lines makes both
    unreadable.
    """
    stream = stream or sys.stderr
    quiet = (not getattr(stream, "isatty", lambda: False)()
             or run_log.per_frame_active())
    if quiet:
        bar = _NullBar()
    elif _tqdm is not None:
        bar = _tqdm(total=total, desc=label[:28], unit="frame",
                    file=stream, leave=True,
                    bar_format="  {desc:28s} {bar} {n_fmt}/{total_fmt} "
                               "{rate_fmt} eta {remaining}")
    else:
        bar = _SimpleBar(total, label, stream)
    try:
        yield bar
    finally:
        bar.close()
