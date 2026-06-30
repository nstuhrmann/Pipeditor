"""
Grafische Darstellung der Pipeline als Node-Diagramm mit PySide6
QGraphicsView/QGraphicsScene.

NodeItem      = die Box für einen einzelnen ProcessingStep
PortItem      = kleiner Kreis links (Input) / rechts (Output) an einer Box
EdgeItem      = Verbindungslinie zwischen zwei Ports
PipelineScene = Scene, die das Pipeline-Datenmodell mit der Grafik synchron hält
"""
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsObject, QGraphicsScene,
    QGraphicsPathItem, QGraphicsTextItem, QGraphicsEllipseItem,
)
from PySide6.QtGui import QBrush, QColor, QPen, QPainterPath, QFont
from PySide6.QtCore import Qt, QPointF, QRectF, Signal

NODE_WIDTH = 160
NODE_HEADER_HEIGHT = 28
NODE_ROW_HEIGHT = 22
PORT_RADIUS = 6


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


class EdgeItem(QGraphicsPathItem):
    def __init__(self, source_port: PortItem, target_port: "PortItem | None" = None):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        self.setPen(QPen(QColor("#cccccc"), 2))
        self.setZValue(-1)
        self.temp_end: QPointF | None = None

    def update_path(self):
        start = self.source_port.scene_pos()
        end = self.target_port.scene_pos() if self.target_port else (self.temp_end or start)
        path = QPainterPath(start)
        dx = max(abs(end.x() - start.x()) * 0.5, 40)
        c1 = QPointF(start.x() + dx, start.y())
        c2 = QPointF(end.x() - dx, end.y())
        path.cubicTo(c1, c2, end)
        self.setPath(path)


class NodeItem(QGraphicsObject):
    doubleClicked = Signal(object)   # emits the PipelineNode

    def __init__(self, pipeline_node, label: str):
        super().__init__()
        self.pipeline_node = pipeline_node
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        n_params = len(pipeline_node.step.PARAMS)
        self.height = NODE_HEADER_HEIGHT + max(1, n_params) * NODE_ROW_HEIGHT + 10

        self.label = label
        self.input_port = PortItem(self, is_output=False)
        self.input_port.setPos(0, NODE_HEADER_HEIGHT / 2)
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
        values = self.pipeline_node.step.get_param_values()
        if not values:
            return "(keine Parameter)"
        lines = [f"{k}: {v}" for k, v in values.items()]
        return "\n".join(lines)

    def refresh_params_preview(self):
        self.params_text.setPlainText(self._params_preview())

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, NODE_WIDTH, self.height)

    def paint(self, painter, option, widget=None):
        rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#3a3a3a")))
        painter.setPen(QPen(QColor("#ffcc00") if self.isSelected() else QColor("#222"), 2))
        painter.drawRoundedRect(rect, 6, 6)

        header_rect = QRectF(0, 0, NODE_WIDTH, NODE_HEADER_HEIGHT)
        painter.setBrush(QBrush(QColor("#555555")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(header_rect, 6, 6)
        painter.drawRect(QRectF(0, NODE_HEADER_HEIGHT - 6, NODE_WIDTH, 6))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.pipeline_node.pos = (value.x(), value.y())
        if change in (QGraphicsItem.ItemPositionChange, QGraphicsItem.ItemPositionHasChanged):
            for port in (self.input_port, self.output_port):
                for edge in port.edges:
                    edge.update_path()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit(self.pipeline_node)
        super().mouseDoubleClickEvent(event)


class PipelineScene(QGraphicsScene):
    nodeDoubleClicked = Signal(object)   # emits PipelineNode
    graphChanged = Signal()
    edgeRequested = Signal(str, str)     # from_node_id, to_node_id (after valid drop)

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: list[EdgeItem] = []
        self._drag_edge: EdgeItem | None = None
        self._drag_source_port: PortItem | None = None

    # ---- Nodes ------------------------------------------------------------
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

    # ---- Edges --------------------------------------------------------------
    def add_edge_item(self, from_id: str, to_id: str):
        src = self.node_items[from_id].output_port
        dst = self.node_items[to_id].input_port
        # falls Ziel-Input schon belegt ist: alte Kante visuell entfernen
        for old_edge in list(dst.edges):
            self._remove_edge_item(old_edge)
        edge = EdgeItem(src, dst)
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

    # ---- Maus-Interaktion: Verbindungen ziehen ------------------------------
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
            target_item = self.itemAt(event.scenePos(), view.transform()) if view else None
            self.removeItem(self._drag_edge)
            self._drag_edge = None
            src_port = self._drag_source_port
            self._drag_source_port = None
            if isinstance(target_item, PortItem) and not target_item.is_output and src_port:
                from_id = self._find_node_id_for_port(src_port)
                to_id = self._find_node_id_for_port(target_item)
                if from_id and to_id and from_id != to_id:
                    self.edgeRequested.emit(from_id, to_id)
            return
        super().mouseReleaseEvent(event)

    def _find_node_id_for_port(self, port):
        for nid, item in self.node_items.items():
            if item.output_port is port or item.input_port is port:
                return nid
        return None
