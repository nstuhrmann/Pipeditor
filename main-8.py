"""
ImgPipe main window.

New in this version:
  - Fan-out: one node output → multiple downstream nodes
  - Metric nodes: two inputs (A/B), purple header, value shown on node
  - Sink fullscreen: each IS_SINK node gets its own maximized window
  - Live mode: re-runs the pipeline automatically after any param change
  - Background thread: pipeline always runs off the UI thread
"""
import sys
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout,
    QTreeWidget, QTreeWidgetItem, QGraphicsView,
    QMessageBox, QSplitter, QStatusBar, QFileDialog,
    QLabel,
)
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QKeySequence
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer

from base_step import STEP_REGISTRY
import steps  # noqa: F401
from pipeline import Pipeline
from node_graphics import PipelineScene
from param_dialog import ParamDialog
from image_canvas import ZoomPanCanvas, HistogramWidget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
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
    return QPixmap.fromImage(qi.copy())


def arr_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32) - arr.min()
    mx = a.max()
    if mx > 0:
        a = a / mx * 255
    return a.astype(np.uint8)


# ---------------------------------------------------------------------------
# Background pipeline worker
# ---------------------------------------------------------------------------

class _PipelineWorker(QObject):
    finished = Signal(dict)
    failed   = Signal(str)

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline

    def run(self):
        try:
            dummy = np.zeros((1, 1, 3), dtype=np.uint8)
            results = self.pipeline.run(dummy)
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Floating image window (used for output / sink nodes)
# ---------------------------------------------------------------------------

class LivePreviewWindow(QMainWindow):
    """
    A regular (non-fullscreen) window showing a single node's output,
    opened by double-clicking that node's thumbnail. Multiple can be
    open at once — one per node. Stays open across pipeline re-runs
    (manual Run or Live Update) and refreshes automatically.

    Shows the image on a zoomable/pannable canvas (see image_canvas.py
    for the mouse controls) with a live RGB histogram of whatever
    region is currently visible, underneath.
    """
    closed = Signal(str)   # node_id

    def __init__(self, node_id: str, title: str, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.setWindowTitle(title)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas = ZoomPanCanvas()
        self._histogram = HistogramWidget()
        self._canvas.viewChanged.connect(self._refresh_histogram)
        self._canvas.pixelHovered.connect(self._on_pixel_hovered)

        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._histogram)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

        self._sized_once = False

    def show_image(self, arr: np.ndarray):
        self._canvas.set_image(arr)
        self._refresh_histogram()
        # Only auto-size the window the first time an image arrives, so
        # later re-runs don't keep resetting a window you've since resized.
        if not self._sized_once:
            self._fit_to_image(arr.shape[1], arr.shape[0])
            self._sized_once = True
        self.show()
        self.raise_()
        self.activateWindow()

    def _refresh_histogram(self):
        self._histogram.set_data(self._canvas.visible_array())

    def _on_pixel_hovered(self, x: int, y: int, value):
        self._histogram.set_hover_value(value)
        if value is None:
            self.statusBar().clearMessage()
            return
        if isinstance(value, tuple):
            text = f"x={x}, y={y}    " + "  ".join(
                f"{ch}: {v}" for ch, v in zip("RGBA", value))
        else:
            text = f"x={x}, y={y}    Value: {value}"
        self.statusBar().showMessage(text)

    def _fit_to_image(self, img_w, img_h):
        margin_w, margin_h = 40, 60 + self._histogram.height()
        w = img_w + margin_w
        h = img_h + margin_h
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(w, avail.width() - 80)
            h = min(h, avail.height() - 80)
        self.resize(max(w, 300), max(h, 300))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed.emit(self.node_id)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImgPipe – Pipeline Editor")
        self.resize(1200, 800)

        self.pipeline = Pipeline()
        self.scene    = PipelineScene(self.pipeline)
        self._connect_scene(self.scene)

        self._last_results: dict | None = None
        self._preview_windows: dict[str, LivePreviewWindow] = {}   # node_id → window

        self._live_mode  = False
        self._is_running = False
        self._thread: QThread | None = None
        self._worker: _PipelineWorker | None = None

        # Debounces rapid-fire parameter edits (e.g. dragging a spin box)
        # so we don't kick off a pipeline run on every single keystroke/tick.
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(150)
        self._live_timer.timeout.connect(self.run_pipeline)

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------
    # UI / menu
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.palette = QTreeWidget()
        self.palette.setHeaderLabel("Available Steps")
        self.palette.setMinimumWidth(180)
        self.palette.itemDoubleClicked.connect(self.on_palette_item_double_clicked)
        self._populate_palette()

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)

        splitter = QSplitter()
        splitter.addWidget(self.palette)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())

    def _populate_palette(self):
        self.palette.clear()
        categories: dict[str, QTreeWidgetItem] = {}
        for cls_name, cls in sorted(STEP_REGISTRY.items(),
                                    key=lambda kv: kv[1].NAME):
            cat = cls.CATEGORY
            if cat not in categories:
                cat_item = QTreeWidgetItem([cat])
                self.palette.addTopLevelItem(cat_item)
                categories[cat] = cat_item
            leaf = QTreeWidgetItem([cls.NAME])
            leaf.setData(0, Qt.UserRole, cls_name)
            categories[cat].addChild(leaf)
        self.palette.expandAll()

    def _build_menu(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu("&File")
        self._add_action(fm, "Save Output to File…",
                         self.save_output_image, "Ctrl+S")
        fm.addSeparator()
        self._add_action(fm, "Open Pipeline…",
                         self.load_pipeline, "Ctrl+L")
        self._add_action(fm, "Save Pipeline…",
                         self.save_pipeline, "Ctrl+Shift+S")
        fm.addSeparator()
        self._add_action(fm, "Quit", self.close, "Ctrl+Q")

        # Pipeline
        pm = mb.addMenu("&Pipeline")
        self._act_run = self._add_action(pm, "Run",
                                          self.run_pipeline, "Ctrl+R")

        self._act_live = QAction("Live Update", self)
        self._act_live.setCheckable(True)
        self._act_live.toggled.connect(self._on_live_toggled)
        pm.addAction(self._act_live)

        # Edit
        em = mb.addMenu("&Edit")
        self._add_action(em, "Delete Selected Node",
                         self.delete_selected_node,
                         QKeySequence.Delete)
        self._add_action(em, "Clear Pipeline", self.clear_pipeline)

    def _add_action(self, menu, label, slot, shortcut=None):
        act = QAction(label, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    # ------------------------------------------------------------------
    # Scene wiring (reused after load)
    # ------------------------------------------------------------------

    def _connect_scene(self, scene: PipelineScene):
        scene.nodeDoubleClicked.connect(self.on_node_double_clicked)
        scene.edgeRequested.connect(self.on_edge_requested)
        scene.edgeRemoved.connect(self.on_edge_removed)
        scene.thumbnailDoubleClicked.connect(self.on_thumbnail_double_clicked)

    # ------------------------------------------------------------------
    # Palette
    # ------------------------------------------------------------------

    def _populate_palette(self):
        self.palette.clear()
        categories: dict[str, QTreeWidgetItem] = {}
        for cls_name, cls in sorted(STEP_REGISTRY.items(),
                                    key=lambda kv: kv[1].NAME):
            cat = cls.CATEGORY
            if cat not in categories:
                ci = QTreeWidgetItem([cat])
                self.palette.addTopLevelItem(ci)
                categories[cat] = ci
            leaf = QTreeWidgetItem([cls.NAME])
            leaf.setData(0, Qt.UserRole, cls_name)
            categories[cat].addChild(leaf)
        self.palette.expandAll()

    def on_palette_item_double_clicked(self, item: QTreeWidgetItem, _):
        cls_name = item.data(0, Qt.UserRole)
        if not cls_name:
            return
        step = STEP_REGISTRY[cls_name]()
        node = self.pipeline.add_node(step, pos=(50, 50))
        self.scene.add_node_item(node)
        self.statusBar().showMessage(f"Added '{step.NAME}'", 3000)

    # ------------------------------------------------------------------
    # Parameters + live update
    # ------------------------------------------------------------------

    def on_node_double_clicked(self, pipeline_node):
        node_item = self.scene.node_items[pipeline_node.id]
        original_values = pipeline_node.step.get_param_values()

        dialog = ParamDialog(pipeline_node.step, self)

        def _apply_live():
            # Push the current form values into the step and refresh the
            # node's on-canvas summary immediately; only the (debounced)
            # pipeline re-run is throttled.
            pipeline_node.step.set_param_values(dialog.get_values())
            node_item.refresh_params_preview()
            if self._live_mode:
                self._live_timer.start()

        dialog.valuesChanged.connect(_apply_live)

        accepted = dialog.exec() == ParamDialog.Accepted
        self._live_timer.stop()

        final_values = dialog.get_values() if accepted else original_values
        pipeline_node.step.set_param_values(final_values)
        node_item.refresh_params_preview()
        if self._live_mode:
            self.run_pipeline()

    def _on_live_toggled(self, checked: bool):
        self._live_mode = checked
        self.statusBar().showMessage(
            "Live update ON — pipeline runs after every parameter change."
            if checked else "Live update OFF.", 3000)
        if checked:
            self.run_pipeline()

    # ------------------------------------------------------------------
    # Thumbnail → fullscreen live preview
    # ------------------------------------------------------------------

    def on_thumbnail_double_clicked(self, edge):
        node_id = self.scene._find_node_id_for_port(edge.source_port)
        if not node_id or node_id not in self.pipeline.nodes:
            return
        node = self.pipeline.nodes[node_id]

        win = self._preview_windows.get(node_id)
        if win is None:
            win = LivePreviewWindow(node_id, node.display_name, self)
            win.closed.connect(self._on_preview_window_closed)
            self._preview_windows[node_id] = win

        if self._last_results and node_id in self._last_results:
            val = self._last_results[node_id]
            if isinstance(val, np.ndarray):
                win.show_image(val)
            else:
                win.show()
        else:
            win.show()
        win.raise_()
        win.activateWindow()

    def _on_preview_window_closed(self, node_id: str):
        self._preview_windows.pop(node_id, None)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def on_edge_requested(self, from_id: str, to_id: str, to_port: int):
        try:
            self.pipeline.add_edge(from_id, to_id, to_port)
            self.scene.add_edge_item(from_id, to_id, to_port)
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc))

    def on_edge_removed(self, from_id: str, to_id: str, to_port: int):
        self.pipeline.remove_edge(from_id, to_id, to_port)
        self.statusBar().showMessage("Connection removed", 2000)

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------

    def delete_selected_node(self):
        for item in list(self.scene.selectedItems()):
            pnode = getattr(item, "pipeline_node", None)
            if pnode is not None:
                # close any associated live preview window
                win = self._preview_windows.pop(pnode.id, None)
                if win:
                    win.close()
                self.pipeline.remove_node(pnode.id)
                self.scene.remove_node_item(pnode.id)

    def clear_pipeline(self):
        if QMessageBox.question(
            self, "Clear Pipeline",
            "Remove all nodes and connections?"
        ) != QMessageBox.Yes:
            return
        for nid in list(self.pipeline.nodes.keys()):
            win = self._preview_windows.pop(nid, None)
            if win:
                win.close()
            self.scene.remove_node_item(nid)
            self.pipeline.remove_node(nid)

    # ------------------------------------------------------------------
    # Run (background thread)
    # ------------------------------------------------------------------

    def run_pipeline(self):
        if self._is_running:
            return
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline",
                                    "Add at least one node first.")
            return
        has_source = any(getattr(n.step, "IS_SOURCE", False)
                         for n in self.pipeline.nodes.values())
        if not has_source:
            QMessageBox.information(
                self, "No Source",
                "Add an Image Source node (Input / Output category).")
            return

        self._is_running = True
        self._act_run.setEnabled(False)
        self.statusBar().showMessage("Running pipeline…")

        self._worker = _PipelineWorker(self.pipeline)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _cleanup_thread(self):
        self._is_running = False
        self._act_run.setEnabled(True)

    def _on_run_finished(self, results: dict):
        self._last_results = results
        self.scene.update_previews(results)

        # Refresh any fullscreen preview windows that are currently open,
        # so live mode / manual re-runs keep them up to date.
        for nid, win in list(self._preview_windows.items()):
            if nid in results:
                val = results[nid]
                if isinstance(val, np.ndarray):
                    win.show_image(val)

        self.statusBar().showMessage("Pipeline executed successfully", 3000)

    def _on_run_failed(self, error: str):
        QMessageBox.critical(self, "Pipeline Error", error)
        self.statusBar().showMessage("Pipeline failed.", 5000)

    def _find_terminal_node(self):
        source_ids = {e[0] for e in self.pipeline.edges}
        all_ids = list(self.pipeline.nodes.keys())
        terminal = [nid for nid in all_ids if nid not in source_ids]
        return terminal[-1] if terminal else (all_ids[-1] if all_ids else None)

    # ------------------------------------------------------------------
    # File: save output
    # ------------------------------------------------------------------

    def save_output_image(self):
        if self._last_results is None:
            QMessageBox.information(self, "No Output", "Run the pipeline first.")
            return
        last_id = self._find_terminal_node()
        arr = (self._last_results.get(last_id)
               if last_id else None)
        if arr is None or not isinstance(arr, np.ndarray):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Image", "output.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)")
        if not path:
            return
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(arr_to_uint8(arr)).save(path)
            self.statusBar().showMessage(f"Saved: {path}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    # ------------------------------------------------------------------
    # Pipeline save / load
    # ------------------------------------------------------------------

    def save_pipeline(self):
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "pipeline.json", "JSON (*.json)")
        if not path:
            return
        self.pipeline.save(path)
        self.statusBar().showMessage(f"Pipeline saved: {path}", 3000)

    def load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pipeline", "", "JSON (*.json)")
        if not path:
            return
        try:
            new_pipeline = Pipeline.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        for win in self._preview_windows.values():
            win.close()
        self._preview_windows.clear()

        self.scene.clear()
        self.pipeline = new_pipeline
        self.scene = PipelineScene(self.pipeline)
        self._connect_scene(self.scene)
        self.view.setScene(self.scene)

        for node in self.pipeline.nodes.values():
            self.scene.add_node_item(node)
        for f, t, p in self.pipeline.edges:
            self.scene.add_edge_item(f, t, p)

        self.statusBar().showMessage(f"Pipeline loaded: {path}", 3000)


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
