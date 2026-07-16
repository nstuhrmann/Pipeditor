"""
LivePreviewWindow: a per-node floating preview window with zoom/pan
canvas, live visible-region histogram, pixel readout, and optional
view-lock syncing across windows. Moved out of main.py.
"""
import numpy as np

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStatusBar
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, Signal, QTimer

from src.GUI.pipeline_editor.image_canvas import ZoomPanCanvas, HistogramWidget


class LivePreviewWindow(QMainWindow):
    """
    A regular (non-fullscreen) window showing a single node's output,
    opened by double-clicking that node's thumbnail. Multiple can be
    open at once — one per node. Stays open across pipeline re-runs
    (manual Run or Live Update) and refreshes automatically.

    Shows the image on a zoomable/pannable canvas (see image_canvas.py
    for the mouse controls) with a live RGB histogram of whatever
    region is currently visible, underneath.

    View > Lock View syncs zoom/pan across every *other* locked preview
    window — pan or zoom in one, and all locked windows follow, which is
    the point when comparing several nodes' output side by side.
    """
    closed      = Signal(str)         # node_id
    viewChanged = Signal(str)         # node_id — forwarded from the canvas,
                                       # only meaningful to act on when locked
    lockToggled = Signal(str, bool)   # node_id, locked

    def __init__(self, node_id: str, title: str, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self._base_title = title
        self.setWindowTitle(title)
        self._locked = False

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas = ZoomPanCanvas()
        self._histogram = HistogramWidget()
        self._canvas.viewChanged.connect(self._refresh_histogram)
        self._canvas.viewChanged.connect(
            lambda: self.viewChanged.emit(self.node_id))
        self._canvas.pixelHovered.connect(self._on_pixel_hovered)

        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._histogram)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

        # No menu bar — Lock View lives in the canvas's right-click
        # context menu, alongside zoom/fit/save, so there's one place
        # for all view actions.
        self._act_lock = QAction("Lock View (sync zoom/pan)", self)
        self._act_lock.setCheckable(True)
        self._act_lock.toggled.connect(self._on_lock_toggled)
        self._canvas.context_actions.append(self._act_lock)

        self._sized_once = False

    def _on_lock_toggled(self, checked: bool):
        self._locked = checked
        self.setWindowTitle(
            f"{self._base_title}  [Locked]" if checked else self._base_title)
        self.lockToggled.emit(self.node_id, checked)

    def is_locked(self) -> bool:
        return self._locked

    def view_state(self):
        return self._canvas.view_state()

    def apply_view_state(self, scale: float, offset):
        self._canvas.set_view_state(scale, offset, emit=False)

    def show_image(self, arr: np.ndarray):
        self._canvas.set_image(arr)
        self._refresh_histogram()
        # Only auto-size the window the first time an image arrives, so
        # later re-runs don't keep resetting a window you've since resized.
        first = not self._sized_once
        if first:
            self._fit_to_image(arr.shape[1], arr.shape[0])
            self._sized_once = True
        self.show()
        self.raise_()
        self.activateWindow()
        if first:
            # The canvas gets its real size only after the window is
            # shown and laid out — fit *then*, or the initial view is
            # computed against a stale/default widget size.
            QTimer.singleShot(0, self._canvas.fit_to_window)

    def _refresh_histogram(self):
        self._histogram.set_data(self._canvas.visible_array())

    def _on_pixel_hovered(self, x: int, y: int, value):
        self._histogram.set_hover_value(value)
        if value is None:
            self.statusBar().clearMessage()
            return
        if isinstance(value, tuple):
            text = f"x={x}, y={y}    " + "  ".join(
                f"{ch}: {self._fmt_value(v)}" for ch, v in zip("RGBA", value))
        else:
            text = f"x={x}, y={y}    Value: {self._fmt_value(value)}"
        self.statusBar().showMessage(text)

    @staticmethod
    def _fmt_value(v):
        return f"{v}" if isinstance(v, int) else f"{v:.3f}"

    def _fit_to_image(self, img_w, img_h):
        """Size the window so the canvas area matches the frame's aspect
        ratio exactly — including when the image is bigger than the
        screen: one uniform scale factor is applied to both dimensions
        (instead of capping each axis independently, which distorted the
        window's shape for large frames)."""
        margin_w = 40
        margin_h = 60 + self._histogram.height()
        avail_w, avail_h = 1600, 1000
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            avail_w = avail.width() - 80
            avail_h = avail.height() - 80
        scale = min((avail_w - margin_w) / img_w,
                    (avail_h - margin_h) / img_h,
                    1.0)
        w = int(img_w * scale) + margin_w
        h = int(img_h * scale) + margin_h
        self.resize(max(w, 300), max(h, 300))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed.emit(self.node_id)
        super().closeEvent(event)

