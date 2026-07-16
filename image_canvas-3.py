"""
Zoomable / pannable image canvas, plus a live RGB histogram of whatever
region is currently visible. Used by LivePreviewWindow.

Controls on ZoomPanCanvas:
    Mouse wheel             zoom in/out, centered on the cursor
    Left click  (no drag)   zoom in one step, centered on the click
    Left drag                pan
    Shift + Left drag        rubber-band box zoom
    Double-click              reset to fit-the-window
    Right click              context menu (zoom in/out, fit, save image)

Left-drag is used for both panning and (with Shift) box-zoom, since a
plain click and a drag are already distinguished by movement distance —
adding a modifier for the box-zoom case avoids overloading the same
gesture for two different things. Right click used to be an instant
"zoom out one step" — it's now a context menu instead (which needs the
click for itself), with "Zoom Out" as one of its items so that gesture
still works, just via the menu rather than immediately on release.
"""
import numpy as np
from PySide6.QtWidgets import QWidget, QSizePolicy, QMenu, QFileDialog, QMessageBox
from PySide6.QtGui import (
    QPainter, QImage, QColor, QPen, QBrush, QPainterPath,
)
from PySide6.QtCore import Qt, QPointF, QRectF, QRect, Signal

from src.GUI.pipeline_editor.image_utils import arr_to_uint8 as _arr_to_uint8
from src.GUI.pipeline_editor.image_utils import numpy_to_qimage as _numpy_to_qimage


class ZoomPanCanvas(QWidget):
    """
    Displays a numpy image with zoom/pan. Emits `viewChanged` whenever the
    visible region changes (zoom, pan, resize, or a new image arrives) so
    a histogram (or any other overlay) can stay in sync.
    """
    viewChanged   = Signal()
    pixelHovered  = Signal(int, int, object)   # x, y, value (None if off-image)

    MIN_SCALE_FACTOR = 0.05   # relative to fit-to-window scale
    MAX_SCALE        = 64.0  # screen pixels per image pixel, upper bound
    ZOOM_STEP        = 1.5   # multiplier per click / wheel notch
    DRAG_THRESHOLD   = 4     # pixels — distinguishes a click from a drag

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background:#1a1a1a;")

        self._image: np.ndarray | None = None
        self._qimage: QImage | None = None
        self._fit_scale = 1.0
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)   # image coords shown at widget (0,0)

        self._press_pos = None
        self._dragging = False
        self._pan_last = None
        self._box_start = None
        self._box_rect = None   # widget-space QRectF while dragging a box

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def set_image(self, arr: np.ndarray):
        first_image = self._image is None
        same_size = (self._image is not None
                    and arr.shape[:2] == self._image.shape[:2])
        self._image = arr
        self._qimage = _numpy_to_qimage(arr)
        if first_image or not same_size:
            self.fit_to_window()
        else:
            # Same size as before (typical live-mode re-run) — keep the
            # user's current zoom/pan instead of resetting it every frame.
            self.update()
            self.viewChanged.emit()

    # ------------------------------------------------------------------
    # View transform
    # ------------------------------------------------------------------

    def fit_to_window(self):
        if self._qimage is None:
            return
        W, H = self._qimage.width(), self._qimage.height()
        w, h = max(1, self.width()), max(1, self.height())
        self._fit_scale = min(w / W, h / H) if W and H else 1.0
        self._scale = self._fit_scale
        self._offset = QPointF(W / 2 - w / (2 * self._scale),
                               H / 2 - h / (2 * self._scale))
        self.update()
        self.viewChanged.emit()

    def _clamp_scale(self, scale: float) -> float:
        min_scale = self._fit_scale * self.MIN_SCALE_FACTOR
        return max(min_scale, min(scale, self.MAX_SCALE))

    def view_state(self):
        """Current (scale, offset) — offset is image coords at widget (0,0)."""
        return self._scale, QPointF(self._offset)

    def set_view_state(self, scale: float, offset: QPointF, emit: bool = True):
        """Apply an externally-supplied view (e.g. from another locked
        preview window). `emit=False` avoids re-triggering viewChanged,
        which matters for the caller doing the syncing to not loop back."""
        if self._qimage is None:
            return
        self._scale = self._clamp_scale(scale)
        self._offset = QPointF(offset)
        self.update()
        if emit:
            self.viewChanged.emit()

    def _zoom_to(self, new_scale: float, center_widget: QPointF):
        if self._qimage is None:
            return
        new_scale = self._clamp_scale(new_scale)
        # Keep the image point currently under `center_widget` fixed on
        # screen while the scale changes.
        ix = self._offset.x() + center_widget.x() / self._scale
        iy = self._offset.y() + center_widget.y() / self._scale
        self._scale = new_scale
        self._offset = QPointF(ix - center_widget.x() / new_scale,
                               iy - center_widget.y() / new_scale)
        self.update()
        self.viewChanged.emit()

    def _visible_image_rect(self) -> QRect:
        """Currently visible region, in image pixel coords, clipped to bounds."""
        if self._qimage is None:
            return QRect()
        W, H = self._qimage.width(), self._qimage.height()
        x0, y0 = self._offset.x(), self._offset.y()
        x1 = x0 + self.width() / self._scale
        y1 = y0 + self.height() / self._scale
        ix0 = max(0, int(np.floor(x0)))
        iy0 = max(0, int(np.floor(y0)))
        ix1 = min(W, int(np.ceil(x1)))
        iy1 = min(H, int(np.ceil(y1)))
        return QRect(ix0, iy0, max(0, ix1 - ix0), max(0, iy1 - iy0))

    def visible_array(self):
        """Sub-array of the original image currently on screen — this is
        what drives the histogram, so it reflects the current zoom."""
        if self._image is None:
            return None
        r = self._visible_image_rect()
        if r.width() <= 0 or r.height() <= 0:
            return None
        return self._image[r.y():r.y() + r.height(),
                           r.x():r.x() + r.width()]

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        if self._qimage is not None:
            painter.save()
            painter.translate(-self._offset.x() * self._scale,
                              -self._offset.y() * self._scale)
            painter.scale(self._scale, self._scale)
            painter.setRenderHint(QPainter.SmoothPixmapTransform,
                                  self._scale < 4.0)
            painter.drawImage(0, 0, self._qimage)
            painter.restore()
        if self._box_rect is not None:
            painter.setPen(QPen(QColor("#ffcc00"), 1, Qt.DashLine))
            painter.setBrush(QBrush(QColor(255, 204, 0, 40)))
            painter.drawRect(self._box_rect)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image is not None:
            self.viewChanged.emit()

    # ------------------------------------------------------------------
    # Mouse: zoom / pan / box-zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        if self._qimage is None:
            return
        notches = event.angleDelta().y() / 120.0
        self._zoom_to(self._scale * (self.ZOOM_STEP ** notches),
                      event.position())

    def mousePressEvent(self, event):
        self._press_pos = event.position()
        self._dragging = False
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ShiftModifier:
                self._box_start = event.position()
                self._box_rect = QRectF(self._box_start, self._box_start)
            else:
                self._pan_last = event.position()

    def mouseMoveEvent(self, event):
        self._emit_hover(event.position())

        if self._press_pos is None:
            return
        if (event.position() - self._press_pos).manhattanLength() > self.DRAG_THRESHOLD:
            self._dragging = True

        if self._box_start is not None:
            self._box_rect = QRectF(self._box_start, event.position()).normalized()
            self.update()
        elif self._pan_last is not None and self._dragging:
            delta = event.position() - self._pan_last
            self._offset = QPointF(self._offset.x() - delta.x() / self._scale,
                                   self._offset.y() - delta.y() / self._scale)
            self._pan_last = event.position()
            self.update()
            self.viewChanged.emit()

    def leaveEvent(self, event):
        self.pixelHovered.emit(-1, -1, None)
        super().leaveEvent(event)

    def _emit_hover(self, widget_pos: QPointF):
        if self._image is None:
            self.pixelHovered.emit(-1, -1, None)
            return
        ix = int(np.floor(self._offset.x() + widget_pos.x() / self._scale))
        iy = int(np.floor(self._offset.y() + widget_pos.y() / self._scale))
        H, W = self._image.shape[:2]
        if 0 <= ix < W and 0 <= iy < H:
            raw = self._image[iy, ix]
            value = tuple(raw.tolist()) if np.ndim(raw) else raw.item()
            self.pixelHovered.emit(ix, iy, value)
        else:
            self.pixelHovered.emit(-1, -1, None)

    def mouseReleaseEvent(self, event):
        if self._box_start is not None and self._box_rect is not None:
            self._zoom_to_box(self._box_rect)
            self._box_start = None
            self._box_rect = None
            self.update()
        elif (not self._dragging and self._qimage is not None
              and event.button() == Qt.LeftButton):
            self._zoom_to(self._scale * self.ZOOM_STEP, event.position())

        self._press_pos = None
        self._dragging = False
        self._pan_last = None

    def mouseDoubleClickEvent(self, event):
        self.fit_to_window()

    def contextMenuEvent(self, event):
        if self._qimage is None:
            return
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                color: #eeeeee;
                border: 1px solid #444444;
            }
            QMenu::item {
                padding: 4px 24px 4px 12px;
            }
            QMenu::item:selected {
                background-color: #3a6ea5;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #444444;
                margin: 4px 6px;
            }
        """)
        act_zoom_in  = menu.addAction("Zoom In")
        act_zoom_out = menu.addAction("Zoom Out")
        act_fit      = menu.addAction("Fit to Window")
        menu.addSeparator()
        act_save_full    = menu.addAction("Save Image As…")
        act_save_visible = menu.addAction("Save Visible Region As…")

        chosen = menu.exec(event.globalPos())
        center = QPointF(event.pos())
        if chosen == act_zoom_in:
            self._zoom_to(self._scale * self.ZOOM_STEP, center)
        elif chosen == act_zoom_out:
            self._zoom_to(self._scale / self.ZOOM_STEP, center)
        elif chosen == act_fit:
            self.fit_to_window()
        elif chosen == act_save_full:
            self._save_array_as(self._image)
        elif chosen == act_save_visible:
            self._save_array_as(self.visible_array())

    def _save_array_as(self, arr):
        if arr is None or arr.size == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As", "image.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)")
        if not path:
            return
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(_arr_to_uint8(arr)).save(path)
        except ImportError:
            _numpy_to_qimage(arr).save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _zoom_to_box(self, box: QRectF):
        if box.width() < 4 or box.height() < 4 or self._qimage is None:
            return
        ix0 = self._offset.x() + box.left() / self._scale
        iy0 = self._offset.y() + box.top() / self._scale
        iw = box.width() / self._scale
        ih = box.height() / self._scale
        if iw <= 0 or ih <= 0:
            return
        new_scale = self._clamp_scale(
            min(self.width() / iw, self.height() / ih))
        self._scale = new_scale
        self._offset = QPointF(
            ix0 - (self.width() / new_scale - iw) / 2,
            iy0 - (self.height() / new_scale - ih) / 2)
        self.update()
        self.viewChanged.emit()


class HistogramWidget(QWidget):
    """
    Live per-channel (R/G/B, or single-channel for grayscale) histogram
    of whatever numpy array it's given via set_data(). Intended to be fed
    the *visible* crop from a ZoomPanCanvas so it reflects the current
    zoomed-in window rather than the whole image.
    """
    AXIS_LABEL_H = 14   # px reserved at the bottom for the lo/hi scale labels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setStyleSheet("background:#111;")
        self._hists = None   # list[(QColor, np.ndarray[256])] or None
        self._range = None   # (lo, hi) the histogram bins currently span
        self._display_range = None   # (lo, hi) shown as axis labels
        self._hover_value = None   # None | scalar | tuple, from ZoomPanCanvas

    def set_data(self, arr):
        if arr is None or arr.size == 0:
            self._hists = None
            self._range = None
            self._display_range = None
            self.update()
            return

        if arr.dtype == np.uint8:
            lo, hi = 0, 256
            self._display_range = (0, 255)
        else:
            lo = float(arr.min())
            hi = float(arr.max())
            if hi <= lo:
                hi = lo + 1.0
            self._display_range = (lo, hi)
        self._range = (lo, hi)

        if arr.ndim == 2 or arr.shape[2] == 1:
            channels = [(QColor("#dddddd"),
                        arr if arr.ndim == 2 else arr[..., 0])]
        else:
            names = [QColor("#ff5555"), QColor("#55ff55"), QColor("#5599ff")]
            channels = [(names[c], arr[..., c])
                       for c in range(min(3, arr.shape[2]))]

        self._hists = []
        for color, data in channels:
            hist, _ = np.histogram(data, bins=256, range=(lo, hi))
            self._hists.append((color, hist))
        self.update()

    def set_hover_value(self, value):
        """value: None (mouse off-image), a scalar (grayscale), or a
        tuple of 2+ numbers (RGB[A]) — as emitted by
        ZoomPanCanvas.pixelHovered."""
        self._hover_value = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if not self._hists:
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data")
            return

        w, h_total = self.width(), self.height()
        h = h_total - self.AXIS_LABEL_H   # plot area height, labels live below it
        max_count = max(hist.max() for _, hist in self._hists) or 1

        painter.setRenderHint(QPainter.Antialiasing)
        for color, hist in self._hists:
            path = QPainterPath()
            path.moveTo(0, h)
            n = len(hist)
            for i, count in enumerate(hist):
                x = i / (n - 1) * w
                y = h - (count / max_count) * (h - 4)
                path.lineTo(x, y)
            path.lineTo(w, h)
            path.closeSubpath()
            fill = QColor(color)
            fill.setAlpha(90)
            painter.setPen(QPen(color, 1))
            painter.setBrush(QBrush(fill))
            painter.drawPath(path)

        self._paint_hover_marker(painter, w, h)
        self._paint_axis_labels(painter, w, h, h_total)

    def _paint_axis_labels(self, painter, w, h, h_total):
        lo, hi = self._display_range
        label_rect_h = self.AXIS_LABEL_H
        painter.setPen(QColor("#999999"))
        painter.drawText(
            QRectF(2, h, w / 2 - 4, label_rect_h),
            Qt.AlignLeft | Qt.AlignVCenter, self._fmt(lo))
        painter.drawText(
            QRectF(w / 2 + 4, h, w / 2 - 6, label_rect_h),
            Qt.AlignRight | Qt.AlignVCenter, self._fmt(hi))

    def _paint_hover_marker(self, painter, w, h):
        if self._hover_value is None or self._range is None:
            return
        lo, hi = self._range
        span = (hi - lo) or 1.0

        def to_x(v):
            v = max(lo, min(hi, v))
            return (v - lo) / span * w

        is_color = isinstance(self._hover_value, tuple)
        if is_color:
            colors = [QColor("#ff5555"), QColor("#55ff55"), QColor("#5599ff")]
            values = self._hover_value[:3]
            for color, v in zip(colors, values):
                x = to_x(v)
                painter.setPen(QPen(color, 1, Qt.DashLine))
                painter.drawLine(int(x), 0, int(x), h)
            text = "  ".join(f"{ch}:{self._fmt(v)}"
                             for ch, v in zip("RGB", values))
        else:
            x = to_x(self._hover_value)
            painter.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine))
            painter.drawLine(int(x), 0, int(x), h)
            text = f"Val: {self._fmt(self._hover_value)}"

        painter.setPen(QColor("#eeeeee"))
        painter.drawText(self.rect().adjusted(4, 2, -4, -2),
                         Qt.AlignTop | Qt.AlignRight, text)

    @staticmethod
    def _fmt(v):
        return f"{v}" if isinstance(v, int) else f"{v:.3g}"
