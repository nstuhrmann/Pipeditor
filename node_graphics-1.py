"""
Grafische Darstellung der Pipeline als Node-Diagramm mit PySide6.

NodeItem      = Box für einen ProcessingStep (source nodes haben keinen Input-Port)
PortItem      = Kreis links (Input) / rechts (Output) an einer Box
EdgeItem      = Verbindungslinie mit optionalem Bild-Thumbnail in der Mitte
PipelineScene = synchronisiert Grafik und Pipeline-Datenmodell
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

NODE_WIDTH = 160
NODE_HEADER_HEIGHT = 28
NODE_ROW_HEIGHT = 22
PORT_RADIUS = 6
THUMB_W = 96
THUMB_H = 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_pixmap(arr: np.ndarray, max_w=THUMB_W, max_h=THUMB_H) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        a -= a.min()
        if a.max() > 0:
            a = a / a.max() * 255
        arr = a.astype(np.uint8)
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, c = arr.shape
        if c == 3:
            qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        elif c == 4:
            qimg = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
        else:
            qimg = QImage(arr[..., 0].copy().data, w, h, w, QImage.Format_Grayscale8)
    px = QPixmap.fromImage(qimg.copy())
    return px.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ---------------------------------------------------------------------------
# PortItem
# ---------------------------------------------------------------------------

class PortItem(QGraphicsEllipseItem):
    def __init__(self, node_item: "NodeItem", is_output: bool):
        super().__init__(-PORT_RADIUS, -PORT_RADIUS, PORT_RADIUS * 2, PORT_RADIUS * 2)
        self.node_item = node_item
        self.is_output = is_output
        self.setBrush(QBrush(QColor("#4caf50" if is_output else "#2196f3")))
        self.setPen(QPen(QColor("#222"), 1))
        self.setParentItem(node_item)
        self.setAcceptHoverEvents(True)
        self.edges: list["EdgeItem"] = []

    def scene_pos(self) -> QPointF:
        return self.mapToScene(0, 0)


# ---------------------------------------------------------------------------
# EdgeItem — with optional image thumbnail at the midpoint
# ---------------------------------------------------------------------------

class EdgeItem(QGraphicsPathItem):
    def __init__(self, source_port: PortItem, target_port: "PortItem | None" = None,
                 show_preview: bool = False):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        self._is_permanent = target_port is not None
        self.setZValue(-1)
        self.temp_end: QPointF | None = None

        # thumbnail child item
        self._thumb_item = QGraphicsPixmapItem(self)
        self._thumb_item.setVisible(False)
        self._show_preview = False
        self._image_data: np.ndarray | None = None

        self._update_pen()

        if self._is_permanent:
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setAcceptHoverEvents(True)

        if show_preview:
            self.set_preview_visible(True)

    # ---- appearance -------------------------------------------------------

    def _update_pen(self):
        if not self._is_permanent:
            self.setPen(QPen(QColor("#888888"), 1.5, Qt.DashLine))
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

    # ---- path / position --------------------------------------------------

    def update_path(self):
        start = self.source_port.scene_pos()
        end = (self.target_port.scene_pos() if self.target_port
               else (self.temp_end or start))
        path = QPainterPath(start)
        dx = max(abs(end.x() - start.x()) * 0.5, 40)
        path.cubicTo(
            QPointF(start.x() + dx, start.y()),
            QPointF(end.x() - dx, end.y()),
            end,
        )
        self.setPath(path)
        self._update_pen()
        self._reposition_thumb()

    def _reposition_thumb(self):
        if not self._show_preview:
            return
        mid = self.path().pointAtPercent(0.5)
        px = self._thumb_item.pixmap()
        self._thumb_item.setPos(
            mid.x() - px.width() / 2,
            mid.y() - px.height() / 2,
        )

    # ---- preview ----------------------------------------------------------

    def set_image(self, arr: np.ndarray):
        """Store the image data and refresh the thumbnail if visible."""
        self._image_data = arr
        if self._show_preview:
            self._refresh_thumb()

    def set_preview_visible(self, visible: bool):
        self._show_preview = visible
        if visible and self._image_data is not None:
            self._refresh_thumb()
        self._thumb_item.setVisible(visible and self._image_data is not None)

    def toggle_preview(self):
        self.set_preview_visible(not self._show_preview)

    def _refresh_thumb(self):
        if self._image_data is None:
            return
        px = _numpy_to_pixmap(self._image_data)
        self._thumb_item.setPixmap(px)
        self._thumb_item.setVisible(True)
        self._reposition_thumb()

    # ---- context menu -----------------------------------------------------

    def contextMenuEvent(self, event):
        menu = QMenu()
        preview_label = "Hide Preview" if self._show_preview else "Show Preview"
        act_preview = menu.addAction(preview_label)
        act_remove = menu.addAction("Remove Connection")
        chosen = menu.exec(event.screenPos())
        if chosen == act_preview:
            self.toggle_preview()
        elif chosen == act_remove:
            scene = self.scene()
            if isinstance(scene, PipelineScene):
                scene.remove_edge_item_by_ref(self)


# ---------------------------------------------------------------------------
# NodeItem
# ---------------------------------------------------------------------------

class NodeItem(QGraphicsObject):
    doubleClicked = Signal(object)

    def __init__(self, pipeline_node, label: str):
        super().__init__()
        self.pipeline_node = pipeline_node
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        is_source = getattr(pipeline_node.step, "IS_SOURCE", False)
        n_params = len(pipeline_node.step.PARAMS)
        self.height = NODE_HEADER_HEIGHT + max(1, n_params) * NODE_ROW_HEIGHT + 10

        # Input port — hidden for source nodes
        self.input_port = PortItem(self, is_output=False)
        self.input_port.setPos(0, NODE_HEADER_HEIGHT / 2)
        if is_source:
            self.input_port.setVisible(False)
            self.input_port.setAcceptedMouseButtons(Qt.NoButton)

        self.output_port = PortItem(self, is_output=True)
        self.output_port.setPos(NODE_WIDTH, NODE_HEADER_HEIGHT / 2)

        self.title_item = QGraphicsTextItem(label, self)
        self.title_item.setDefaultTextColor(QColor("white"))
        font = QFont()
        font.setBold(True)
        self.title_item.setFont(font)
        self.title_item.setPos(8, 4)

        self.params_text = QGraphicsTextItem(self._params_preview(), self)
        self.params_text.setDefaultTextColor(QColor("#dddddd"))
        self.params_text.setPos(8, NODE_HEADER_HEIGHT + 4)

        x, y = pipeline_node.pos
        self.setPos(x, y)

    def _params_preview(self) -> str:
        import os
        values = self.pipeline_node.step.get_param_values()
        if not values:
            return "(no parameters)"
        lines = []
        for k, v in values.items():
            # shorten long file paths
            if isinstance(v, str) and len(v) > 22:
                v = "…" + os.path.basename(v)
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def refresh_params_preview(self):
        self.params_text.setPlainText(self._params_preview())

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, NODE_WIDTH, self.height)

    def paint(self, painter, option, widget=None):
        is_source = getattr(self.pipeline_node.step, "IS_SOURCE", False)
        is_sink   = getattr(self.pipeline_node.step, "IS_SINK",   False)
        if is_source:
            header_color = QColor("#1a6b3a")   # green
        elif is_sink:
            header_color = QColor("#7a2020")   # dark red
        else:
            header_color = QColor("#555555")   # default grey

        painter.setBrush(QBrush(QColor("#3a3a3a")))
        painter.setPen(QPen(QColor("#ffcc00") if self.isSelected() else QColor("#222"), 2))
        painter.drawRoundedRect(self.boundingRect(), 6, 6)

        painter.setBrush(QBrush(header_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(0, 0, NODE_WIDTH, NODE_HEADER_HEIGHT), 6, 6)
        painter.drawRect(QRectF(0, NODE_HEADER_HEIGHT - 6, NODE_WIDTH, 6))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.pipeline_node.pos = (value.x(), value.y())
        if change in (QGraphicsItem.ItemPositionChange,
                      QGraphicsItem.ItemPositionHasChanged):
            for port in (self.input_port, self.output_port):
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
    graphChanged = Signal()
    edgeRequested = Signal(str, str)
    edgeRemoved = Signal(str, str)

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: list[EdgeItem] = []
        self._drag_edge: EdgeItem | None = None
        self._drag_source_port: PortItem | None = None

    # ---- nodes ------------------------------------------------------------

    def add_node_item(self, pipeline_node):
        item = NodeItem(pipeline_node, pipeline_node.step.NAME)
        item.doubleClicked.connect(self.nodeDoubleClicked.emit)
        self.addItem(item)
        self.node_items[pipeline_node.id] = item
        return item

    def remove_node_item(self, node_id: str):
        item = self.node_items.pop(node_id, None)
        if item:
            for edge in list(item.input_port.edges) + list(item.output_port.edges):
                self._remove_edge_item(edge)
            self.removeItem(item)

    # ---- edges ------------------------------------------------------------

    def add_edge_item(self, from_id: str, to_id: str, show_preview: bool = False):
        src = self.node_items[from_id].output_port
        dst = self.node_items[to_id].input_port
        for old in list(dst.edges):
            self._remove_edge_item(old)
        edge = EdgeItem(src, dst, show_preview=show_preview)
        src.edges.append(edge)
        dst.edges.append(edge)
        self.addItem(edge)
        self.edge_items.append(edge)
        edge.update_path()
        return edge

    def _remove_edge_item(self, edge: EdgeItem):
        if edge.source_port:
            edge.source_port.edges = [e for e in edge.source_port.edges if e is not edge]
        if edge.target_port:
            edge.target_port.edges = [e for e in edge.target_port.edges if e is not edge]
        if edge in self.edge_items:
            self.edge_items.remove(edge)
        self.removeItem(edge)

    def remove_edge_between(self, from_id: str, to_id: str):
        src = self.node_items[from_id].output_port
        dst = self.node_items[to_id].input_port
        for edge in list(src.edges):
            if edge.target_port is dst:
                self._remove_edge_item(edge)
                self.edgeRemoved.emit(from_id, to_id)

    def remove_edge_item_by_ref(self, edge: EdgeItem):
        from_id = self._find_node_id_for_port(edge.source_port)
        to_id = self._find_node_id_for_port(edge.target_port)
        self._remove_edge_item(edge)
        if from_id and to_id:
            self.edgeRemoved.emit(from_id, to_id)

    # ---- previews ---------------------------------------------------------

    def update_previews(self, results: dict):
        """
        Called after pipeline run. results maps node_id -> output np.ndarray.
        Each edge shows the image produced by its source node.
        By default, only first-edge (out of a root) and last-edge (into a
        terminal) are made visible; all others keep their current visibility.
        """
        # find root node ids (no incoming edges) and terminal node ids (no outgoing)
        targets = {e[1] for e in self.pipeline.edges}
        sources = {e[0] for e in self.pipeline.edges}
        root_ids = {nid for nid in self.pipeline.nodes if nid not in targets}
        terminal_ids = {nid for nid in self.pipeline.nodes if nid not in sources}

        for edge in self.edge_items:
            if not edge._is_permanent:
                continue
            from_id = self._find_node_id_for_port(edge.source_port)
            to_id = self._find_node_id_for_port(edge.target_port)
            if from_id and from_id in results:
                edge.set_image(results[from_id])
                # auto-show for first edge out of root or last edge into terminal
                is_first = from_id in root_ids
                is_last = to_id in terminal_ids
                if is_first or is_last:
                    edge.set_preview_visible(True)

    # ---- keyboard ---------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in list(self.selectedItems()):
                if isinstance(item, EdgeItem) and item._is_permanent:
                    self.remove_edge_item_by_ref(item)
        super().keyPressEvent(event)

    # ---- mouse: draw connections ------------------------------------------

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
            target = self.itemAt(event.scenePos(), view.transform()) if view else None
            self.removeItem(self._drag_edge)
            self._drag_edge = None
            src_port = self._drag_source_port
            self._drag_source_port = None
            if isinstance(target, PortItem) and not target.is_output and src_port:
                from_id = self._find_node_id_for_port(src_port)
                to_id = self._find_node_id_for_port(target)
                if from_id and to_id and from_id != to_id:
                    self.edgeRequested.emit(from_id, to_id)
            return
        super().mouseReleaseEvent(event)

    def _find_node_id_for_port(self, port):
        if port is None:
            return None
        for nid, item in self.node_items.items():
            if item.output_port is port or item.input_port is port:
                return nid
        return None
