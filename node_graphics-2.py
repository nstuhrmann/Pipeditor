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
    QGraphicsPixmapItem, QMenu,
)
from PySide6.QtGui import (
    QBrush, QColor, QPen, QPainterPath, QFont, QPixmap,
    QImage, QPainterPathStroker,
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
import numpy as np

NODE_WIDTH        = 170
NODE_HEADER_H     = 28
NODE_ROW_H        = 20
PORT_RADIUS       = 6
THUMB_W, THUMB_H  = 96, 72

# Header colours per node kind
_HEADER = {
    "source": QColor("#1a6b3a"),   # green
    "sink":   QColor("#7a2020"),   # dark red
    "metric": QColor("#4a2070"),   # purple
    "normal": QColor("#555555"),   # grey
}


def _numpy_to_pixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32) - arr.min()
        mx = a.max()
        if mx > 0:
            a = a / mx * 255
        arr = a.astype(np.uint8)
    if arr.ndim == 2:
        h, w = arr.shape
        qi = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, c = arr.shape
        if c == 3:
            qi = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        elif c == 4:
            qi = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
        else:
            qi = QImage(arr[..., 0].copy().data, w, h, w,
                        QImage.Format_Grayscale8)
    px = QPixmap.fromImage(qi.copy())
    return px.scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatio,
                     Qt.SmoothTransformation)


# ---------------------------------------------------------------------------
# PortItem
# ---------------------------------------------------------------------------

class PortItem(QGraphicsEllipseItem):
    def __init__(self, node_item: "NodeItem", is_output: bool,
                 port_index: int = 0, label: str = ""):
        super().__init__(-PORT_RADIUS, -PORT_RADIUS,
                         PORT_RADIUS * 2, PORT_RADIUS * 2)
        self.node_item = node_item
        self.is_output = is_output
        self.port_index = port_index
        self.setBrush(QBrush(QColor("#4caf50" if is_output else "#2196f3")))
        self.setPen(QPen(QColor("#222"), 1))
        self.setParentItem(node_item)
        self.setAcceptHoverEvents(True)
        self.edges: list["EdgeItem"] = []

        if label:
            lbl = QGraphicsTextItem(label, self)
            lbl.setDefaultTextColor(QColor("#aaaaaa"))
            f = QFont(); f.setPointSize(7); lbl.setFont(f)
            lbl.setPos(PORT_RADIUS + 2 if is_output else
                       -(PORT_RADIUS + 2 + lbl.boundingRect().width()), -8)

    def scene_pos(self) -> QPointF:
        return self.mapToScene(0, 0)


# ---------------------------------------------------------------------------
# EdgeItem
# ---------------------------------------------------------------------------

class EdgeItem(QGraphicsPathItem):
    def __init__(self, source_port: PortItem,
                 target_port: "PortItem | None" = None,
                 show_preview: bool = False):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        self._is_permanent = target_port is not None
        self.setZValue(-1)
        self.temp_end: QPointF | None = None

        self._thumb = QGraphicsPixmapItem(self)
        self._thumb.setVisible(False)
        self._show_preview = False
        self._image_data: np.ndarray | None = None

        self._update_pen()
        if self._is_permanent:
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setAcceptHoverEvents(True)
        if show_preview:
            self.set_preview_visible(True)

    def _update_pen(self):
        if not self._is_permanent:
            self.setPen(QPen(QColor("#888"), 1.5, Qt.DashLine))
        elif self.isSelected():
            self.setPen(QPen(QColor("#ff5555"), 2.5))
        else:
            self.setPen(QPen(QColor("#cccccc"), 2))

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

    def contextMenuEvent(self, event):
        menu = QMenu()
        act_prev = menu.addAction(
            "Hide Preview" if self._show_preview else "Show Preview")
        act_rem = menu.addAction("Remove Connection")
        chosen = menu.exec(event.screenPos())
        if chosen == act_prev:
            self.toggle_preview()
        elif chosen == act_rem:
            s = self.scene()
            if isinstance(s, PipelineScene):
                s.remove_edge_item_by_ref(self)


# ---------------------------------------------------------------------------
# NodeItem
# ---------------------------------------------------------------------------

class NodeItem(QGraphicsObject):
    doubleClicked = Signal(object)

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
        self.input_ports: list[PortItem] = []
        if not is_source:
            port_labels = (["A", "B", "C", "D"][:n_inputs]
                           if n_inputs > 1 else [""] * n_inputs)
            body_h = self.height - NODE_HEADER_H
            for i in range(n_inputs):
                y = (NODE_HEADER_H
                     + (i + 0.5) * body_h / n_inputs)
                p = PortItem(self, is_output=False,
                             port_index=i, label=port_labels[i])
                p.setPos(0, y)
                self.input_ports.append(p)

        # ── Output port (hidden for metric nodes) ──────────────────────
        self.output_port: PortItem | None = None
        if not is_metric:
            self.output_port = PortItem(self, is_output=True)
            self.output_port.setPos(NODE_WIDTH, NODE_HEADER_H / 2)

        # ── Labels ─────────────────────────────────────────────────────
        self.title_item = QGraphicsTextItem(label, self)
        self.title_item.setDefaultTextColor(QColor("white"))
        f = QFont(); f.setBold(True); self.title_item.setFont(f)
        self.title_item.setPos(8, 4)

        self.params_text = QGraphicsTextItem(self._params_preview(), self)
        self.params_text.setDefaultTextColor(QColor("#dddddd"))
        self.params_text.setPos(8, NODE_HEADER_H + 2)

        # metric value display
        self._metric_text: QGraphicsTextItem | None = None
        if is_metric:
            self._metric_text = QGraphicsTextItem("—", self)
            self._metric_text.setDefaultTextColor(QColor("#ffdd44"))
            f2 = QFont(); f2.setBold(True); f2.setPointSize(11)
            self._metric_text.setFont(f2)
            self._metric_text.setPos(8,
                NODE_HEADER_H + max(1, n_params) * NODE_ROW_H + 4)

        x, y = pipeline_node.pos
        self.setPos(x, y)

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
                v = "…" + os.path.basename(v)
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
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, NODE_WIDTH, self.height)

    def paint(self, painter, option, widget=None):
        painter.setBrush(QBrush(QColor("#3a3a3a")))
        painter.setPen(QPen(
            QColor("#ffcc00") if self.isSelected() else QColor("#222"), 2))
        painter.drawRoundedRect(self.boundingRect(), 6, 6)

        painter.setBrush(QBrush(_HEADER[self._kind()]))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(0, 0, NODE_WIDTH, NODE_HEADER_H), 6, 6)
        painter.drawRect(QRectF(0, NODE_HEADER_H - 6, NODE_WIDTH, 6))

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


# ---------------------------------------------------------------------------
# PipelineScene
# ---------------------------------------------------------------------------

class PipelineScene(QGraphicsScene):
    nodeDoubleClicked = Signal(object)
    graphChanged      = Signal()
    edgeRequested     = Signal(str, str, int)   # from_id, to_id, to_port
    edgeRemoved       = Signal(str, str, int)   # from_id, to_id, to_port

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
        item = NodeItem(pipeline_node, pipeline_node.step.NAME)
        item.doubleClicked.connect(self.nodeDoubleClicked.emit)
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
                      to_port: int = 0, show_preview: bool = False):
        src_item = self.node_items[from_id]
        dst_item = self.node_items[to_id]
        if src_item.output_port is None:
            return   # metric nodes have no output
        dst_port = (dst_item.input_ports[to_port]
                    if to_port < len(dst_item.input_ports) else None)
        if dst_port is None:
            return

        # clear old edge to this specific input port
        for old in list(dst_port.edges):
            self._remove_edge_item(old)

        edge = EdgeItem(src_item.output_port, dst_port,
                        show_preview=show_preview)
        src_item.output_port.edges.append(edge)
        dst_port.edges.append(edge)
        self.addItem(edge)
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
        self.removeItem(edge)

    def remove_edge_item_by_ref(self, edge: EdgeItem):
        from_id = self._find_node_id_for_port(edge.source_port)
        to_id   = self._find_node_id_for_port(edge.target_port)
        to_port = edge.target_port.port_index if edge.target_port else 0
        self._remove_edge_item(edge)
        if from_id and to_id:
            self.edgeRemoved.emit(from_id, to_id, to_port)

    # ── previews / metric values ───────────────────────────────────────

    def update_previews(self, results: dict):
        """Update edge thumbnails and metric value labels after a run."""
        target_ids = {e[1] for e in self.pipeline.edges}
        source_ids = {e[0] for e in self.pipeline.edges}
        root_ids     = {nid for nid in self.pipeline.nodes
                        if nid not in target_ids}
        terminal_ids = {nid for nid in self.pipeline.nodes
                        if nid not in source_ids}

        for edge in self.edge_items:
            if not edge._is_permanent or edge.source_port is None:
                continue
            from_id = self._find_node_id_for_port(edge.source_port)
            to_id   = self._find_node_id_for_port(edge.target_port)
            if from_id and from_id in results:
                val = results[from_id]
                if isinstance(val, np.ndarray):
                    edge.set_image(val)
                    is_first = from_id in root_ids
                    is_last  = to_id in terminal_ids
                    if is_first or is_last:
                        edge.set_preview_visible(True)

        # update metric node labels
        for nid, item in self.node_items.items():
            if getattr(item.pipeline_node.step, "IS_METRIC", False):
                if nid in results:
                    item.set_metric_value(results[nid])

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
            self.addItem(self._drag_edge)
            self._drag_edge.update_path()
            return
        super().mousePressEvent(event)

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
            self.removeItem(self._drag_edge)
            self._drag_edge = None
            src_port = self._drag_source_port
            self._drag_source_port = None
            if (isinstance(target, PortItem) and not target.is_output
                    and src_port):
                from_id = self._find_node_id_for_port(src_port)
                to_id   = self._find_node_id_for_port(target)
                if from_id and to_id and from_id != to_id:
                    self.edgeRequested.emit(from_id, to_id,
                                            target.port_index)
            return
        super().mouseReleaseEvent(event)

    def _find_node_id_for_port(self, port):
        if port is None:
            return None
        for nid, item in self.node_items.items():
            all_ports = list(item.input_ports)
            if item.output_port:
                all_ports.append(item.output_port)
            if port in all_ports:
                return nid
        return None
