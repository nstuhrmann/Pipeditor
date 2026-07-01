"""
ImgPipe – main window.

Layout:
  left   : palette of available steps (grouped by category)
  centre : node editor (QGraphicsView / PipelineScene)

All image I/O is handled by pipeline nodes:
  - "Image Source"      (green header)  — loads a file, no input port
  - "Directory Writer"  (red header)    — writes to a folder, passthrough

Run the pipeline via Pipeline → Run  (Ctrl+R).
The output of the terminal node is shown in a floating output window.
Edge thumbnails show the image stream at each connection point.
"""
import sys
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout,
    QTreeWidget, QTreeWidgetItem, QGraphicsView,
    QMessageBox, QSplitter, QStatusBar, QFileDialog, QScrollArea, QLabel,
)
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QKeySequence
from PySide6.QtCore import Qt

from base_step import STEP_REGISTRY
import steps  # noqa: F401  — triggers auto-registration of all steps
from pipeline import Pipeline
from node_graphics import PipelineScene
from param_dialog import ParamDialog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
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
    return QPixmap.fromImage(qimg.copy())


def arr_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    a -= a.min()
    if a.max() > 0:
        a = a / a.max() * 255
    return a.astype(np.uint8)


# ---------------------------------------------------------------------------
# Floating output viewer
# ---------------------------------------------------------------------------

class ImageWindow(QMainWindow):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 520)
        self._label = QLabel(f"— {title} —")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background:#1a1a1a; color:#666;")
        scroll = QScrollArea()
        scroll.setWidget(self._label)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(scroll)

    def show_image(self, arr: np.ndarray):
        px = numpy_to_qpixmap(arr)
        self._label.setPixmap(px)
        self._label.resize(px.size())
        base = self.windowTitle().split("  ")[0]
        self.setWindowTitle(f"{base}  [{arr.shape[1]}×{arr.shape[0]}]")
        self.show()
        self.raise_()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImgPipe – Bildverarbeitungs-Pipeline-Editor")
        self.resize(1200, 800)

        self.pipeline = Pipeline()
        self.scene = PipelineScene(self.pipeline)
        self.scene.nodeDoubleClicked.connect(self.on_node_double_clicked)
        self.scene.edgeRequested.connect(self.on_edge_requested)
        self.scene.edgeRemoved.connect(self.on_edge_removed)

        self._last_results: dict | None = None
        self._output_window = ImageWindow("Output")

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------
    # UI
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
        for cls_name, cls in sorted(STEP_REGISTRY.items(), key=lambda kv: kv[1].NAME):
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
        menubar = self.menuBar()

        # ── File ──────────────────────────────────────────────────────────
        file_menu = menubar.addMenu("&File")

        act_save_out = QAction("Save Output to File…", self)
        act_save_out.setShortcut("Ctrl+S")
        act_save_out.triggered.connect(self.save_output_image)
        file_menu.addAction(act_save_out)

        file_menu.addSeparator()

        act_open_pipe = QAction("Open Pipeline…", self)
        act_open_pipe.setShortcut("Ctrl+L")
        act_open_pipe.triggered.connect(self.load_pipeline)
        file_menu.addAction(act_open_pipe)

        act_save_pipe = QAction("Save Pipeline…", self)
        act_save_pipe.setShortcut("Ctrl+Shift+S")
        act_save_pipe.triggered.connect(self.save_pipeline)
        file_menu.addAction(act_save_pipe)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ── Pipeline ──────────────────────────────────────────────────────
        pipe_menu = menubar.addMenu("&Pipeline")

        act_run = QAction("Run", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self.run_pipeline)
        pipe_menu.addAction(act_run)

        pipe_menu.addSeparator()

        act_show_out = QAction("Show Output Window", self)
        act_show_out.triggered.connect(lambda: self._output_window.show())
        pipe_menu.addAction(act_show_out)

        # ── Edit ──────────────────────────────────────────────────────────
        edit_menu = menubar.addMenu("&Edit")

        act_delete = QAction("Delete Selected Node", self)
        act_delete.setShortcut(QKeySequence.Delete)
        act_delete.triggered.connect(self.delete_selected_node)
        edit_menu.addAction(act_delete)

        act_clear = QAction("Clear Pipeline", self)
        act_clear.triggered.connect(self.clear_pipeline)
        edit_menu.addAction(act_clear)

    # ------------------------------------------------------------------
    # Palette
    # ------------------------------------------------------------------

    def on_palette_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        cls_name = item.data(0, Qt.UserRole)
        if not cls_name:
            return
        step = STEP_REGISTRY[cls_name]()
        node = self.pipeline.add_node(step, pos=(50, 50))
        self.scene.add_node_item(node)
        self.statusBar().showMessage(f"Added '{step.NAME}'", 3000)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def on_node_double_clicked(self, pipeline_node):
        dialog = ParamDialog(pipeline_node.step, self)
        if dialog.exec() == ParamDialog.Accepted:
            pipeline_node.step.set_param_values(dialog.get_values())
            self.scene.node_items[pipeline_node.id].refresh_params_preview()

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def on_edge_requested(self, from_id, to_id):
        try:
            self.pipeline.add_edge(from_id, to_id)
            self.scene.add_edge_item(from_id, to_id)
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc))

    def on_edge_removed(self, from_id, to_id):
        self.pipeline.remove_edge(from_id, to_id)
        self.statusBar().showMessage("Connection removed", 2000)

    # ------------------------------------------------------------------
    # Edit actions
    # ------------------------------------------------------------------

    def delete_selected_node(self):
        for item in list(self.scene.selectedItems()):
            pnode = getattr(item, "pipeline_node", None)
            if pnode is not None:
                self.pipeline.remove_node(pnode.id)
                self.scene.remove_node_item(pnode.id)

    def clear_pipeline(self):
        if QMessageBox.question(
            self, "Clear Pipeline", "Remove all nodes and connections?"
        ) != QMessageBox.Yes:
            return
        for nid in list(self.pipeline.nodes.keys()):
            self.scene.remove_node_item(nid)
            self.pipeline.remove_node(nid)

    # ------------------------------------------------------------------
    # Pipeline: run
    # ------------------------------------------------------------------

    def run_pipeline(self):
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline",
                                    "Add at least one node first.")
            return
        has_source = any(
            getattr(n.step, "IS_SOURCE", False)
            for n in self.pipeline.nodes.values()
        )
        if not has_source:
            QMessageBox.information(
                self, "No Source",
                "Add an Image Source node (Input / Output category) "
                "as the starting point of the pipeline."
            )
            return
        try:
            # Source nodes ignore this dummy image and load their own file.
            dummy = np.zeros((1, 1, 3), dtype=np.uint8)
            self._last_results = self.pipeline.run(dummy)
        except Exception as exc:
            QMessageBox.critical(self, "Pipeline Error", str(exc))
            return

        self.scene.update_previews(self._last_results)

        last_id = self._find_terminal_node()
        if last_id and last_id in self._last_results:
            self._output_window.show_image(self._last_results[last_id])

        self.statusBar().showMessage("Pipeline executed successfully", 3000)

    def _find_terminal_node(self):
        sources = {e[0] for e in self.pipeline.edges}
        all_ids = list(self.pipeline.nodes.keys())
        terminal = [nid for nid in all_ids if nid not in sources]
        return terminal[-1] if terminal else (all_ids[-1] if all_ids else None)

    # ------------------------------------------------------------------
    # File: save output
    # ------------------------------------------------------------------

    def save_output_image(self):
        if self._last_results is None:
            QMessageBox.information(self, "No Output", "Run the pipeline first.")
            return
        last_id = self._find_terminal_node()
        arr = self._last_results.get(last_id) if last_id else None
        if arr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Image", "output.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)"
        )
        if not path:
            return
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(arr_to_uint8(arr)).save(path)
            self.statusBar().showMessage(f"Saved: {path}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    # ------------------------------------------------------------------
    # Pipeline: save / load
    # ------------------------------------------------------------------

    def save_pipeline(self):
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "pipeline.json", "JSON (*.json)"
        )
        if not path:
            return
        self.pipeline.save(path)
        self.statusBar().showMessage(f"Pipeline saved: {path}", 3000)

    def load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pipeline", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            new_pipeline = Pipeline.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self.scene.clear()
        self.pipeline = new_pipeline
        self.scene = PipelineScene(self.pipeline)
        self.scene.nodeDoubleClicked.connect(self.on_node_double_clicked)
        self.scene.edgeRequested.connect(self.on_edge_requested)
        self.scene.edgeRemoved.connect(self.on_edge_removed)
        self.view.setScene(self.scene)

        for node in self.pipeline.nodes.values():
            self.scene.add_node_item(node)
        for from_id, to_id in self.pipeline.edges:
            self.scene.add_edge_item(from_id, to_id)

        self.statusBar().showMessage(f"Pipeline loaded: {path}", 3000)


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
