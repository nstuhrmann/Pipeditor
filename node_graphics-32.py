"""
Node editor graphics.

NodeItem      box for a ProcessingStep
PortItem      input (blue) or output (green) port circle
EdgeItem      bezier connection with optional image thumbnail
PipelineScene manages the graph visuals and syncs with pipeline data
"""
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsObject, QGraphicsScene,
    QGraphicsPathItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QGraphicsPixmapItem, QMenu, QGraphicsView,
)
from PySide6.QtGui import (
    QBrush, QColor, QPen, QPainterPath, QFont, QPixmap,
    QImage, QPainterPathStroker, QFontMetrics,
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
import numpy as np

from src.GUI.pipeline_editor.image_utils import numpy_to_thumbnail

NODE_WIDTH        = 170
NODE_HEADER_H     = 28
NODE_ROW_H        = 14
PORT_RADIUS       = 6
THUMB_W, THUMB_H  = 96, 72

# Header colours per node kind
_HEADER = {
    "source": QColor("#1a6b3a"),   # green
    "sink":   QColor("#7a2020"),   # dark red
    "metric": QColor("#4a2070"),   # purple
    "normal": QColor("#555555"),   # grey
}

# Port colours. Input port labels reuse the input-port blue so they
# read as belonging to the port — deliberately NOT the edge grey
# (#cccccc), since edges run straight through the label area and a
# matching tone made the labels unreadable where they crossed.
# Bus messages are control flow, not image data — deliberately a
# different colour family from the ports and edges.
BUS_MSG_COLOR   = "#ffb74d"
from src.GUI.pipeline_editor.pipeline import EDGE_CONTROL, EDGE_DATA

CONTROL_EDGE_COLOR = "#ffb74d"     # matches the side-channel text
PORT_IN_COLOR   = "#2196f3"
PORT_OUT_COLOR  = "#4caf50"
PORT_LABEL_COLOR = PORT_IN_COLOR

# Shared dark stylesheet for right-click context menus in this module —
# without it, QMenu can end up with mismatched (e.g. black-on-black) text
# depending on the OS theme, since it otherwise inherits system defaults.
_DARK_MENU_STYLE = """
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
"""


def _numpy_to_pixmap(arr: np.ndarray) -> QPixmap:
    return numpy_to_thumbnail(arr, THUMB_W, THUMB_H)


# ---------------------------------------------------------------------------
# PortItem
# ---------------------------------------------------------------------------

def _fmt_scalar(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (int, bool, str)):
        return str(v)
    return type(v).__name__


def format_bus_value(value) -> str:
    """Compact one-line rendering of a posted message, for the canvas."""
    if isinstance(value, dict):
        items = list(value.items())
        shown = ", ".join(f"{k}={_fmt_scalar(v)}" for k, v in items[:3])
        return "{" + shown + ("…" if len(items) > 3 else "") + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_scalar(v) for v in value[:3]) + \
               ("…" if len(value) > 3 else "") + "]"
    return _fmt_scalar(value)


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node_item: "NodeItem", is_output: bool,
                 port_index: int = 0, label: str = "", tooltip: str = "",
                 is_control: bool = False):
        super().__init__(-PORT_RADIUS, -PORT_RADIUS,
                         PORT_RADIUS * 2, PORT_RADIUS * 2)
        self.node_item = node_item
        self.is_output = is_output
        self.port_index = port_index
        self.is_control = is_control
        self.setBrush(QBrush(QColor(
            CONTROL_EDGE_COLOR if is_control
            else (PORT_OUT_COLOR if is_output else PORT_IN_COLOR))))
        self.setPen(QPen(QColor("#222"), 1))
        self.setParentItem(node_item)
        self.setAcceptHoverEvents(True)
        self.edges: list["EdgeItem"] = []
        if tooltip:
            self.setToolTip(tooltip)
        # NOTE: the port's text label is NOT created here. It sits to the
        # LEFT of the port, i.e. outside this item's 12px bounding rect —
        # Qt culls and misrepaints children drawn outside their parent's
        # bounds, which is why labels were invisible. NodeItem draws them
        # instead and widens its own boundingRect to cover them.

    def scene_pos(self) -> QPointF:
        return self.mapToScene(0, 0)


# ---------------------------------------------------------------------------
# EdgeItem
# ---------------------------------------------------------------------------

class EdgeItem(QGraphicsPathItem):
    """An edge. kind="data" carries images; kind="control" carries values
    backwards to the target's NEXT frame and is drawn dashed, since it
    deliberately does not participate in execution order."""

    def __init__(self, source_port: PortItem,
                 target_port: "PortItem | None" = None,
                 show_preview: bool = False, kind: str = EDGE_DATA):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        self.kind = kind
        self._is_permanent = target_port is not None
        self.setZValue(-1)
        self.temp_end: QPointF | None = None

        # The thumbnail is intentionally NOT a child of this item. Qt groups
        # a child's stacking order with its parent's, so a thumbnail parented
        # to its edge could still end up rendered behind an unrelated edge
        # added later. Keeping it as an independent top-level scene item
        # with a high Z value guarantees previews always sit above every
        # connection line, regardless of add/draw order.
        self._thumb = QGraphicsPixmapItem()
        self._thumb.setZValue(10)
        self._thumb.setVisible(False)
        # Back-reference so PipelineScene can identify which edge (and
        # therefore which node's output) a double-clicked thumbnail
        # belongs to, without the thumbnail being a child item.
        self._thumb._owner_edge = self
        self._show_preview = False
        self._image_data: np.ndarray | None = None

        self._update_pen()
        if self._is_permanent:
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setAcceptHoverEvents(True)
        if show_preview:
            self.set_preview_visible(True)

    def _pen_for_kind(self) -> QPen:
        """Control edges are dashed amber: they carry values, not images,
        and are excluded from execution order, so they must not read as
        an ordinary connection."""
        if self.kind == EDGE_CONTROL:
            pen = QPen(QColor(CONTROL_EDGE_COLOR), 2, Qt.DashLine)
            pen.setDashPattern([5, 4])
            return pen
        return QPen(QColor("#cccccc"), 2)

    def set_kind(self, kind: str):
        self.kind = kind
        if kind == EDGE_CONTROL:
            self.set_preview_visible(False)   # no image on a control edge
        self._update_pen()
        self.update()

    def _update_pen(self):
        if not self._is_permanent:
            self.setPen(QPen(QColor("#888"), 1.5, Qt.DashLine))
        elif self.isSelected():
            self.setPen(QPen(QColor("#ff5555"), 2.5))
        else:
            self.setPen(self._pen_for_kind())

    def shape(self):
        ps = QPainterPathStroker()
        ps.setWidth(12)
        return ps.createStroke(self.path())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self._update_pen()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        if not self.isSelected():
            self.setPen(QPen(QColor("#ffffff"), 2.5))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._update_pen()
        super().hoverLeaveEvent(event)

    def update_path(self):
        start = self.source_port.scene_pos()
        end = (self.target_port.scene_pos() if self.target_port
               else (self.temp_end or start))
        path = QPainterPath(start)
        dx = max(abs(end.x() - start.x()) * 0.5, 40)
        path.cubicTo(QPointF(start.x() + dx, start.y()),
                     QPointF(end.x() - dx, end.y()), end)
        self.setPath(path)
        self._update_pen()
        self._reposition_thumb()

    def _reposition_thumb(self):
        if not self._show_preview or not self.path():
            return
        mid = self.path().pointAtPercent(0.5)
        px = self._thumb.pixmap()
        self._thumb.setPos(mid.x() - px.width() / 2,
                           mid.y() - px.height() / 2)

    def set_image(self, arr: np.ndarray):
        self._image_data = arr
        if self._show_preview:
            self._refresh_thumb()

    def set_preview_visible(self, visible: bool):
        self._show_preview = visible
        if visible and self._image_data is not None:
            self._refresh_thumb()
        self._thumb.setVisible(visible and self._image_data is not None)

    def toggle_preview(self):
        self.set_preview_visible(not self._show_preview)

    def _refresh_thumb(self):
        if self._image_data is None:
            return
        px = _numpy_to_pixmap(self._image_data)
        self._thumb.setPixmap(px)
        self._thumb.setVisible(True)
        self._reposition_thumb()

    def add_to_scene(self, scene):
        """Add this edge and its (independent) thumbnail to the scene."""
        scene.addItem(self)
        scene.addItem(self._thumb)

    def remove_from_scene(self, scene):
        """Remove this edge and its thumbnail from the scene."""
        if self._thumb.scene() is not None:
            scene.removeItem(self._thumb)
        if self.scene() is not None:
            scene.removeItem(self)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(_DARK_MENU_STYLE)
        is_control = self.kind == EDGE_CONTROL
        act_prev = None
        if not is_control:
            act_prev = menu.addAction(
                "Hide Preview" if self._show_preview else "Show Preview")
        act_data = menu.addAction("Data Edge")
        act_data.setCheckable(True)
        act_data.setChecked(not is_control)
        act_ctrl = menu.addAction("Control Edge")
        act_ctrl.setCheckable(True)
        act_ctrl.setChecked(is_control)
        menu.addSeparator()
        act_rem = menu.addAction("Remove Connection")

        chosen = menu.exec(event.screenPos())
        scene = self.scene()
        if act_prev is not None and chosen == act_prev:
            self.toggle_preview()
        elif chosen == act_data and is_control:
            self.set_kind(EDGE_DATA)
            if isinstance(scene, PipelineScene):
                scene.edgeKindChanged.emit(self, EDGE_DATA)
        elif chosen == act_ctrl and not is_control:
            self.set_kind(EDGE_CONTROL)
            if isinstance(scene, PipelineScene):
                scene.edgeKindChanged.emit(self, EDGE_CONTROL)
        elif chosen == act_rem:
            if isinstance(scene, PipelineScene):
                scene.remove_edge_item_by_ref(self)


# ---------------------------------------------------------------------------
# NodeItem
# ---------------------------------------------------------------------------

class NodeItem(QGraphicsObject):
    doubleClicked      = Signal(object)
    bypassToggled      = Signal(object)   # pipeline_node
    deleteRequested    = Signal(object)   # pipeline_node
    duplicateRequested = Signal(object)   # pipeline_node

    BYPASSED_OPACITY = 0.45
    # Space reserved LEFT of the box for input port labels. It is part of
    # boundingRect() (so Qt paints the labels) but not of node_rect()
    # (so it isn't painted or clickable).
    LABEL_MARGIN = 74
    MSG_FONT_PT = 7
    MSG_ROW_H = 12          # one WRAPPED line, under the node

    def __init__(self, pipeline_node, label: str):
        super().__init__()
        step = pipeline_node.step
        self.pipeline_node = pipeline_node

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        is_source = getattr(step, "IS_SOURCE", False)
        is_metric = getattr(step, "IS_METRIC", False)
        n_inputs  = getattr(step, "N_INPUTS", 1)
        n_params  = len(step.PARAMS)

        body_rows = max(1, n_params)
        if is_metric:
            body_rows += 1   # extra row for the value display
        self.height = NODE_HEADER_H + body_rows * NODE_ROW_H + 10

        # ── Input ports ────────────────────────────────────────────────
        # Input 0 sits at the SAME height as the output port (header
        # center), so two adjacent modules connect with a straight
        # horizontal line instead of a slanted one. Additional inputs
        # (metric A/B etc.) are spaced below it.
        PORT_SPACING = 22
        self.input_ports: list[PortItem] = []
        # (text, y) for each input port label — painted directly in
        # paint() rather than as child QGraphicsTextItems. Child items
        # positioned outside the parent's bounds get culled by Qt, which
        # is why these labels never showed up.
        self._input_labels: list[tuple[str, float]] = []
        # Bus messages this node emitted on the last executed frame,
        # already formatted. Shown under the box so a control loop is
        # visible in the graph instead of only in the code.
        self._bus_messages: list[str] = []
        self._bus_lines: list[str] = []      # after wrapping to box width
        if not is_source:
            declared = tuple(getattr(step, "INPUT_LABELS", ()) or ())
            fallback = ["A", "B", "C", "D", "E", "F", "G", "H"]
            if n_inputs > 1:
                # make sure the box is tall enough for the lowest port
                need = int(NODE_HEADER_H / 2
                           + (n_inputs - 1) * PORT_SPACING
                           + PORT_RADIUS + 8)
                self.height = max(self.height, need)
            for i in range(n_inputs):
                name = declared[i] if i < len(declared) else ""
                # Show a label whenever the step named the port (even for
                # a single input) or whenever there's more than one port
                # to tell apart.
                shown = name or (fallback[i] if n_inputs > 1
                                 and i < len(fallback) else "")
                slot = (fallback[i] if i < len(fallback) else str(i))
                tip = (f"Input {slot}: {name}" if name
                       else f"Input {slot}" if n_inputs > 1 else "Input")
                y = NODE_HEADER_H / 2 + i * PORT_SPACING
                p = PortItem(self, is_output=False, port_index=i,
                             tooltip=tip)
                p.setPos(0, y)
                self.input_ports.append(p)
                if shown:
                    self._input_labels.append((shown, y))

        # ── Output port (hidden for metric nodes) ──────────────────────
        # ── Control ports ─────────────────────────────────────────────
        # Amber, on the BOTTOM edge, so a control link is visually a
        # different plane from the left-to-right data flow. A source has
        # no data inputs, so this is the only way to wire a feedback loop
        # back into one.
        step = pipeline_node.step
        self.control_in = None
        self.control_out = None
        if getattr(step, "ACCEPTS_CONTROL", False):
            self.control_in = PortItem(self, is_output=False, port_index=0,
                                       tooltip="Control in (values arrive "
                                               "on the next frame)",
                                       is_control=True)
            self.control_in.setPos(NODE_WIDTH * 0.28, self.height)
        if getattr(step, "EMITS_CONTROL", False):
            self.control_out = PortItem(self, is_output=True, port_index=0,
                                        tooltip="Control out (sent to the "
                                                "target's next frame)",
                                        is_control=True)
            self.control_out.setPos(NODE_WIDTH * 0.72, self.height)

        # ── Output port ───────────────────────────────────────────────
        # Metric nodes have one too now: their graph output is the first
        # input with the metric value overlaid, so it can be previewed
        # or fed into a writer like any other image.
        self.output_port: PortItem | None = None
        self.output_port = PortItem(self, is_output=True)
        self.output_port.setPos(NODE_WIDTH, NODE_HEADER_H / 2)

        # ── Labels ─────────────────────────────────────────────────────
        f = QFont(); f.setBold(True)
        fm = QFontMetrics(f)
        # Elide titles that would overflow the node box; the full name
        # stays available as a tooltip on hover.
        elided = fm.elidedText(label, Qt.ElideRight, NODE_WIDTH - 16)
        self.title_item = QGraphicsTextItem(elided, self)
        self.title_item.setDefaultTextColor(QColor("white"))
        self.title_item.setFont(f)
        self.title_item.setPos(8, 4)
        if elided != label:
            self.setToolTip(label)

        self.params_text = QGraphicsTextItem(self._params_preview(), self)
        self.params_text.setDefaultTextColor(QColor("#dddddd"))
        pf = QFont(); pf.setPointSize(7)
        self.params_text.setFont(pf)
        self.params_text.setPos(8, NODE_HEADER_H + 2)

        # metric value display
        self._metric_text: QGraphicsTextItem | None = None
        if is_metric:
            self._metric_text = QGraphicsTextItem("—", self)
            self._metric_text.setDefaultTextColor(QColor("#ffdd44"))
            f2 = QFont(); f2.setBold(True); f2.setPointSize(11)
            self._metric_text.setFont(f2)
            # Place it just below the params text's REAL extent (row
            # arithmetic broke when the param font/row height shrank),
            # and grow the box if the value line wouldn't fit.
            y_val = (self.params_text.pos().y()
                     + self.params_text.boundingRect().height() + 2)
            self._metric_text.setPos(8, y_val)
            needed = int(y_val
                         + self._metric_text.boundingRect().height() + 6)
            self.height = max(self.height, needed)

        x, y = pipeline_node.pos
        self.setPos(x, y)
        if pipeline_node.bypassed:
            self.setOpacity(self.BYPASSED_OPACITY)

    # ── bypass ─────────────────────────────────────────────────────────
    def set_bypassed(self, value: bool):
        self.pipeline_node.bypassed = value
        self.setOpacity(self.BYPASSED_OPACITY if value else 1.0)
        self.bypassToggled.emit(self.pipeline_node)

    # ── node kind ──────────────────────────────────────────────────────
    def _kind(self) -> str:
        step = self.pipeline_node.step
        if getattr(step, "IS_SOURCE", False): return "source"
        if getattr(step, "IS_SINK",   False): return "sink"
        if getattr(step, "IS_METRIC", False): return "metric"
        return "normal"

    # ── labels ─────────────────────────────────────────────────────────
    def _params_preview(self) -> str:
        import os
        vals = self.pipeline_node.step.get_param_values()
        if not vals:
            return "(no parameters)"
        lines = []
        for k, v in vals.items():
            if isinstance(v, str) and len(v) > 22:
                # Strip the directory, then cap the basename too — a long
                # filename alone can still overflow the node box.
                v = os.path.basename(v)
                if len(v) > 20:
                    v = "…" + v[-19:]
                else:
                    v = "…" + v
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def refresh_params_preview(self):
        self.params_text.setPlainText(self._params_preview())

    def set_metric_value(self, value):
        if self._metric_text is not None:
            if isinstance(value, float):
                text = f"{value:.4g}"
            else:
                text = str(value)
            self._metric_text.setPlainText(text)

    # ── geometry ────────────────────────────────────────────────────────
    def node_rect(self) -> QRectF:
        """The visible box. Painting and hit-testing use THIS, not
        boundingRect() — which is deliberately wider (see below)."""
        return QRectF(0, 0, NODE_WIDTH, self.height)

    def boundingRect(self) -> QRectF:
        # Extends to the left of the box so the input port labels, which
        # are drawn outside it, are inside this item's declared area.
        # Without that Qt culls / fails to repaint them and they simply
        # never appear. The bottom is extended the same way for bus
        # messages.
        extra = (len(self._bus_lines) * self.MSG_ROW_H + 4
                 if self._bus_lines else 0)
        return QRectF(-self.LABEL_MARGIN, 0,
                      NODE_WIDTH + self.LABEL_MARGIN, self.height + extra)

    @staticmethod
    def _wrap(text: str, fm: QFontMetrics, max_width: int) -> list:
        """Greedy word wrap to max_width. A single word too long to fit
        is elided rather than overflowing — message values can be one
        unbroken token."""
        words = text.split(" ")
        lines, current = [], ""
        for word in words:
            trial = word if not current else current + " " + word
            if fm.horizontalAdvance(trial) <= max_width:
                current = trial
                continue
            if current:
                lines.append(current)
            if fm.horizontalAdvance(word) <= max_width:
                current = word
            else:
                lines.append(fm.elidedText(word, Qt.ElideRight, max_width))
                current = ""
        if current:
            lines.append(current)
        return lines or [""]

    def set_bus_messages(self, messages: list):
        """Replace the message lines shown under this node. Text is
        wrapped to the width of the box, so the line COUNT (and thus
        boundingRect) depends on the content — Qt must be told before it
        re-reads the geometry, hence prepareGeometryChange()."""
        messages = list(messages)
        if messages == self._bus_messages:
            return
        f = QFont(); f.setPointSize(self.MSG_FONT_PT)
        fm = QFontMetrics(f)
        lines = []
        for text in messages:
            lines.extend(self._wrap(text, fm, NODE_WIDTH - 6))
        self.prepareGeometryChange()
        self._bus_messages = messages
        self._bus_lines = lines
        self.update()

    def _paint_bus_messages(self, painter):
        if not self._bus_lines:
            return
        f = QFont(); f.setPointSize(self.MSG_FONT_PT)
        y = self.height + self.MSG_ROW_H - 2
        for shown in self._bus_lines:
            path = QPainterPath()
            path.addText(3, y, f, shown)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#141414"), 3))
            painter.drawPath(path)                       # halo
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(BUS_MSG_COLOR)))
            painter.drawPath(path)
            y += self.MSG_ROW_H

    def shape(self):
        # Hit-testing must NOT include the label margin, or clicking in
        # the empty space left of a node would select and drag it.
        path = QPainterPath()
        path.addRoundedRect(self.node_rect(), 6, 6)
        return path

    def _paint_port_labels(self, painter):
        """Input port labels, drawn left of the box. Done with the
        painter (not child items) so they can't be culled, and elided to
        the reserved LABEL_MARGIN so they never run into the canvas.

        Drawn as an outlined glyph path rather than plain text: incoming
        edges are light grey and pass straight through this area, so flat
        text in a similar tone is unreadable where they cross. The dark
        halo keeps the label legible over an edge, the canvas, or a
        thumbnail; the fill matches the input-port blue so it reads as
        belonging to the port."""
        if not self._input_labels:
            return
        f = QFont(); f.setPointSize(7); f.setBold(True)
        fm = QFontMetrics(f)
        avail = self.LABEL_MARGIN - PORT_RADIUS - 6
        for text, y in self._input_labels:
            shown = fm.elidedText(text, Qt.ElideRight, avail)
            w = fm.horizontalAdvance(shown)
            path = QPainterPath()
            path.addText(-(PORT_RADIUS + 5 + w), y + 4, f, shown)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#141414"), 3))
            painter.drawPath(path)                      # halo
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(PORT_LABEL_COLOR)))
            painter.drawPath(path)                      # fill

    def paint(self, painter, option, widget=None):
        # 1. Body fill (no border yet)
        painter.setBrush(QBrush(QColor("#3a3a3a")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.node_rect(), 6, 6)

        # 2. Header fill
        painter.setBrush(QBrush(_HEADER[self._kind()]))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(0, 0, NODE_WIDTH, NODE_HEADER_H), 6, 6)
        painter.drawRect(QRectF(0, NODE_HEADER_H - 6, NODE_WIDTH, 6))

        # 3. Outline drawn last, on top of both fills, so it's a uniform
        #    width all the way around instead of being partially covered
        #    by the header fill.
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(
            QColor("#ffcc00") if self.isSelected() else QColor("#222"), 2))
        painter.drawRoundedRect(self.node_rect(), 6, 6)

        # 4. Input port labels, outside the box to the left.
        self._paint_port_labels(painter)

        # 5. Bus messages emitted by this node, under the box.
        self._paint_bus_messages(painter)

        # 5. Execution time, bottom-right corner (mean over the runs
        #    since the last reset — i.e. over the current batch).
        timing = self.pipeline_node.timing
        if timing.count:
            f = QFont(); f.setPointSize(6)
            painter.setFont(f)
            painter.setPen(QColor("#8a8a8a"))
            text = f"{timing.mean_ms:.1f} ms"
            fm = QFontMetrics(f)
            painter.drawText(
                QPointF(NODE_WIDTH - 6 - fm.horizontalAdvance(text),
                        self.height - 4), text)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.pipeline_node.pos = (value.x(), value.y())
        if change in (QGraphicsItem.ItemPositionChange,
                      QGraphicsItem.ItemPositionHasChanged):
            all_ports = list(self.input_ports)
            if self.output_port:
                all_ports.append(self.output_port)
            for port in all_ports:
                for edge in port.edges:
                    edge.update_path()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit(self.pipeline_node)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(_DARK_MENU_STYLE)
        # Bypass isn't meaningful for sources (nothing upstream to pass
        # through) or metrics (no single output to pass through as) —
        # they still get the menu for Delete, just without that entry.
        act_bypass = None
        if self._kind() not in ("source", "metric"):
            act_bypass = menu.addAction("Bypass")
            act_bypass.setCheckable(True)
            act_bypass.setChecked(self.pipeline_node.bypassed)
            menu.addSeparator()
        act_dup = menu.addAction("Duplicate")
        act_delete = menu.addAction("Delete")
        chosen = menu.exec(event.screenPos())
        if act_bypass is not None and chosen == act_bypass:
            self.set_bypassed(not self.pipeline_node.bypassed)
        elif chosen == act_dup:
            self.duplicateRequested.emit(self.pipeline_node)
        elif chosen == act_delete:
            # Deletion is owned by MainWindow (it also closes the node's
            # preview window and updates the model) — just request it.
            self.deleteRequested.emit(self.pipeline_node)


# ---------------------------------------------------------------------------
# PipelineScene
# ---------------------------------------------------------------------------

class PipelineView(QGraphicsView):
    """QGraphicsView for the node editor with middle-mouse-drag panning.
    Left-drag stays rubber-band selection / node dragging, so panning
    gets its own button rather than overloading an existing gesture."""

    def __init__(self, scene=None, parent=None):
        super().__init__(scene, parent)
        self._panning = False
        self._pan_last = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class PipelineScene(QGraphicsScene):
    nodeDoubleClicked      = Signal(object)
    graphChanged           = Signal()
    edgeRequested          = Signal(str, str, int, str)  # from, to, port, kind
    edgeRemoved            = Signal(str, str, int)   # from_id, to_id, to_port
    thumbnailDoubleClicked = Signal(object)          # EdgeItem
    nodeBypassToggled      = Signal(object)          # PipelineNode
    nodeDeleteRequested    = Signal(object)          # PipelineNode
    nodeDuplicateRequested = Signal(object)          # PipelineNode
    edgeKindChanged        = Signal(object, str)     # EdgeItem, kind

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: list[EdgeItem] = []
        self._drag_edge: EdgeItem | None = None
        self._drag_source_port: PortItem | None = None

    # ── nodes ──────────────────────────────────────────────────────────

    def add_node_item(self, pipeline_node) -> NodeItem:
        item = NodeItem(pipeline_node, pipeline_node.display_name)
        item.doubleClicked.connect(self.nodeDoubleClicked.emit)
        item.bypassToggled.connect(self.nodeBypassToggled.emit)
        item.deleteRequested.connect(self.nodeDeleteRequested.emit)
        item.duplicateRequested.connect(self.nodeDuplicateRequested.emit)
        self.addItem(item)
        self.node_items[pipeline_node.id] = item
        return item

    def remove_node_item(self, node_id: str):
        item = self.node_items.pop(node_id, None)
        if not item:
            return
        all_ports = list(item.input_ports)
        if item.output_port:
            all_ports.append(item.output_port)
        for port in all_ports:
            for edge in list(port.edges):
                self._remove_edge_item(edge)
        self.removeItem(item)

    # ── edges ──────────────────────────────────────────────────────────

    def add_edge_item(self, from_id: str, to_id: str,
                      to_port: int = 0, show_preview: bool = False,
                      kind: str = EDGE_DATA):
        src_item = self.node_items[from_id]
        dst_item = self.node_items[to_id]
        if kind == EDGE_CONTROL:
            src_port = src_item.control_out
            dst_port = dst_item.control_in
        else:
            src_port = src_item.output_port
            dst_port = (dst_item.input_ports[to_port]
                        if to_port < len(dst_item.input_ports) else None)
        if src_port is None or dst_port is None:
            return

        # clear old edge to this specific input port
        for old in list(dst_port.edges):
            self._remove_edge_item(old)

        edge = EdgeItem(src_port, dst_port,
                        show_preview=show_preview, kind=kind)
        src_port.edges.append(edge)
        dst_port.edges.append(edge)
        edge.add_to_scene(self)
        self.edge_items.append(edge)
        edge.update_path()
        return edge

    def _remove_edge_item(self, edge: EdgeItem):
        if edge.source_port:
            edge.source_port.edges = [e for e in edge.source_port.edges
                                      if e is not edge]
        if edge.target_port:
            edge.target_port.edges = [e for e in edge.target_port.edges
                                      if e is not edge]
        if edge in self.edge_items:
            self.edge_items.remove(edge)
        edge.remove_from_scene(self)

    def remove_edge_item_by_ref(self, edge: EdgeItem):
        from_id = self._find_node_id_for_port(edge.source_port)
        to_id   = self._find_node_id_for_port(edge.target_port)
        to_port = edge.target_port.port_index if edge.target_port else 0
        self._remove_edge_item(edge)
        if from_id and to_id:
            self.edgeRemoved.emit(from_id, to_id, to_port)

    # ── previews / metric values ───────────────────────────────────────

    def update_side_channels(self, result):
        """Show what each node put on the non-image channels for the
        frame just executed: metadata it emitted (forward, with the
        frame) and control values it sent (backward, next frame)."""
        for nid, item in self.node_items.items():
            node = self.pipeline.nodes.get(nid)
            lines = []
            if node is not None:
                own = getattr(node, "last_emitted", None) or {}
                for key, value in own.items():
                    lines.append(f"↦ {key}: {format_bus_value(value)}")
                sent = getattr(node, "last_control", None) or {}
                if sent:
                    lines.append(f"⤳ control: {format_bus_value(sent)}")
            item.set_bus_messages(lines)

    def update_previews(self, results: dict, metric_values: dict):
        """Update edge thumbnails and metric value labels after a run.
        `results` holds images for every node (metrics included — theirs
        is input A with the value overlaid); the raw metric values come
        separately in `metric_values` (node_id -> float/str)."""
        target_ids = {e[1] for e in self.pipeline.edges}
        source_ids = {e[0] for e in self.pipeline.edges}
        root_ids     = {nid for nid in self.pipeline.nodes
                        if nid not in target_ids}
        terminal_ids = {nid for nid in self.pipeline.nodes
                        if nid not in source_ids}

        # A node whose output fans out to several downstream nodes would
        # otherwise get an identical thumbnail on every outgoing edge.
        # Auto-show at most one preview per SOURCE node — and seed the
        # set with sources that already have a preview visible, so a
        # manually enabled one counts and doesn't get a duplicate added
        # next to it.
        previewed_sources = {
            self._find_node_id_for_port(e.source_port)
            for e in self.edge_items
            if getattr(e, "_show_preview", False) and e.source_port is not None
        }

        for edge in self.edge_items:
            if not edge._is_permanent or edge.source_port is None:
                continue
            from_id = self._find_node_id_for_port(edge.source_port)
            to_id   = self._find_node_id_for_port(edge.target_port)
            if from_id and from_id in results:
                val = results[from_id]
                if isinstance(val, np.ndarray):
                    # Every edge still gets the image, so a preview the
                    # user switched on by hand keeps updating.
                    edge.set_image(val)
                    is_first = from_id in root_ids
                    is_last  = to_id in terminal_ids
                    if ((is_first or is_last)
                            and from_id not in previewed_sources):
                        edge.set_preview_visible(True)
                        previewed_sources.add(from_id)

        # repaint all nodes so the timing corner text refreshes
        for item in self.node_items.values():
            item.update()

        # update metric node labels
        for nid, item in self.node_items.items():
            if getattr(item.pipeline_node.step, "IS_METRIC", False):
                if metric_values and nid in metric_values:
                    item.set_metric_value(metric_values[nid])

    # ── keyboard ────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in list(self.selectedItems()):
                if isinstance(item, EdgeItem) and item._is_permanent:
                    self.remove_edge_item_by_ref(item)
        super().keyPressEvent(event)

    # ── mouse: draw connections ─────────────────────────────────────────

    def mousePressEvent(self, event):
        view = self.views()[0] if self.views() else None
        item = self.itemAt(event.scenePos(), view.transform()) if view else None
        if isinstance(item, PortItem) and item.is_output:
            self._drag_source_port = item
            self._drag_edge = EdgeItem(item)
            self._drag_edge.temp_end = event.scenePos()
            self._drag_edge.add_to_scene(self)
            self._drag_edge.update_path()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        view = self.views()[0] if self.views() else None
        item = self.itemAt(event.scenePos(), view.transform()) if view else None
        # Thumbnails are independent top-level items (see EdgeItem), so a
        # double-click on one won't reach NodeItem/EdgeItem handlers on
        # its own — catch it here via the back-reference we tagged it with.
        owner_edge = getattr(item, "_owner_edge", None)
        if owner_edge is not None:
            self.thumbnailDoubleClicked.emit(owner_edge)
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_edge:
            self._drag_edge.temp_end = event.scenePos()
            self._drag_edge.update_path()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_edge:
            view = self.views()[0] if self.views() else None
            target = self.itemAt(event.scenePos(),
                                 view.transform()) if view else None
            self._drag_edge.remove_from_scene(self)
            self._drag_edge = None
            src_port = self._drag_source_port
            self._drag_source_port = None
            if (isinstance(target, PortItem) and not target.is_output
                    and src_port):
                from_id = self._find_node_id_for_port(src_port)
                to_id   = self._find_node_id_for_port(target)
                # Control and data ports don't interconnect: the two
                # carry different things, and silently coercing one into
                # the other is how you get a loop that mysteriously does
                # nothing.
                if src_port.is_control != target.is_control:
                    return
                kind = EDGE_CONTROL if target.is_control else EDGE_DATA
                if from_id and to_id and (from_id != to_id
                                          or kind == EDGE_CONTROL):
                    self.edgeRequested.emit(from_id, to_id,
                                            target.port_index, kind)
            return
        super().mouseReleaseEvent(event)

    def _find_node_id_for_port(self, port):
        if port is None:
            return None
        for nid, item in self.node_items.items():
            all_ports = list(item.input_ports)
            if item.output_port:
                all_ports.append(item.output_port)
            for extra in (getattr(item, "control_in", None),
                          getattr(item, "control_out", None)):
                if extra is not None:
                    all_ports.append(extra)
            if port in all_ports:
                return nid
        return None
