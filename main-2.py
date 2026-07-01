"""
Hauptfenster der Anwendung:
  - Menüleiste: File, Pipeline, Edit
  - links: Palette aller verfügbaren Steps (gruppiert nach CATEGORY)
  - Mitte: Node-Editor (QGraphicsView über PipelineScene)
  - Bildvorschau: Input und Output als eigene, frei positionierbare Fenster
"""
import sys
import os
import glob
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout,
    QTreeWidget, QTreeWidgetItem, QGraphicsView, QLabel, QFileDialog,
    QMessageBox, QSplitter, QStatusBar, QListWidget, QScrollArea,
)
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QKeySequence
from PySide6.QtCore import Qt, QSize

from base_step import STEP_REGISTRY
import steps  # noqa: F401  (triggers auto-registration of all steps)
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


def load_image_as_array(path: str) -> np.ndarray:
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except ImportError:
        qimg = QImage(path).convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        arr = np.array(ptr).reshape(h, qimg.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
        return arr.copy()


def arr_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    a -= a.min()
    if a.max() > 0:
        a = a / a.max() * 255
    return a.astype(np.uint8)


# ---------------------------------------------------------------------------
# Standalone image viewer window
# ---------------------------------------------------------------------------

class ImageWindow(QMainWindow):
    """A simple resizable window that displays a single numpy image."""

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
        self.setWindowTitle(
            f"{self.windowTitle().split('  ')[0]}  [{arr.shape[1]}×{arr.shape[0]}]"
        )
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

        self.current_image: np.ndarray | None = None
        self.current_stack: list[np.ndarray] | None = None
        self.current_stack_paths: list[str] | None = None
        self._last_results: dict | None = None

        # Separate image viewer windows
        self._input_window = ImageWindow("Input Image")
        self._output_window = ImageWindow("Output Image")

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Palette (left)
        self.palette = QTreeWidget()
        self.palette.setHeaderLabel("Available Steps")
        self.palette.setMinimumWidth(180)
        self.palette.itemDoubleClicked.connect(self.on_palette_item_double_clicked)
        self._populate_palette()

        # Node editor (center)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)

        # Stack list (right)
        stack_widget = QWidget()
        stack_layout = QVBoxLayout(stack_widget)
        stack_layout.addWidget(QLabel("<b>Image Stack</b>"))
        self.stack_list = QListWidget()
        self.stack_list.itemClicked.connect(self.on_stack_item_clicked)
        stack_layout.addWidget(self.stack_list)
        stack_widget.setMinimumWidth(160)

        splitter = QSplitter()
        splitter.addWidget(self.palette)
        splitter.addWidget(self.view)
        splitter.addWidget(stack_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 1)
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

        act_open_img = QAction("Open Image…", self)
        act_open_img.setShortcut(QKeySequence.Open)
        act_open_img.triggered.connect(self.load_image)
        file_menu.addAction(act_open_img)

        act_open_stack = QAction("Open Image Stack…", self)
        act_open_stack.setShortcut("Ctrl+Shift+O")
        act_open_stack.triggered.connect(self.load_stack)
        file_menu.addAction(act_open_stack)

        file_menu.addSeparator()

        act_save_img = QAction("Save Output to File…", self)
        act_save_img.setShortcut("Ctrl+S")
        act_save_img.triggered.connect(self.save_output_image)
        file_menu.addAction(act_save_img)

        act_save_video = QAction("Save Stack Output as Video…", self)
        act_save_video.triggered.connect(self.save_output_video)
        file_menu.addAction(act_save_video)

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

        act_run = QAction("Run on Current Image", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self.run_pipeline_on_current)
        pipe_menu.addAction(act_run)

        act_run_stack = QAction("Run on Entire Stack…", self)
        act_run_stack.setShortcut("Ctrl+Shift+R")
        act_run_stack.triggered.connect(self.run_pipeline_on_stack)
        pipe_menu.addAction(act_run_stack)

        pipe_menu.addSeparator()

        act_show_input = QAction("Show Input Window", self)
        act_show_input.triggered.connect(lambda: self._input_window.show())
        pipe_menu.addAction(act_show_input)

        act_show_output = QAction("Show Output Window", self)
        act_show_output.triggered.connect(lambda: self._output_window.show())
        pipe_menu.addAction(act_show_output)

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
    # Palette → add node
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
    # Parameter editing
    # ------------------------------------------------------------------

    def on_node_double_clicked(self, pipeline_node):
        dialog = ParamDialog(pipeline_node.step, self)
        if dialog.exec() == ParamDialog.Accepted:
            pipeline_node.step.set_param_values(dialog.get_values())
            self.scene.node_items[pipeline_node.id].refresh_params_preview()

    # ------------------------------------------------------------------
    # Edge handling
    # ------------------------------------------------------------------

    def on_edge_requested(self, from_id, to_id):
        try:
            self.pipeline.add_edge(from_id, to_id)
            self.scene.add_edge_item(from_id, to_id)
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc))

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
        if QMessageBox.question(self, "Clear Pipeline",
                                "Remove all nodes and connections?") != QMessageBox.Yes:
            return
        for nid in list(self.pipeline.nodes.keys()):
            self.scene.remove_node_item(nid)
            self.pipeline.remove_node(nid)

    # ------------------------------------------------------------------
    # File → images
    # ------------------------------------------------------------------

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            self.current_image = load_image_as_array(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._input_window.show_image(self.current_image)
        self.statusBar().showMessage(f"Loaded: {path}", 3000)

    def load_stack(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Stack Folder")
        if not folder:
            return
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        paths = sorted(sum((glob.glob(os.path.join(folder, e)) for e in exts), []))
        if not paths:
            QMessageBox.warning(self, "No Images",
                                "No image files found in this folder.")
            return
        self.current_stack_paths = paths
        self.current_stack = [load_image_as_array(p) for p in paths]
        self.stack_list.clear()
        self.stack_list.addItems([os.path.basename(p) for p in paths])
        # show first image in input window
        self.current_image = self.current_stack[0]
        self._input_window.show_image(self.current_image)
        self.statusBar().showMessage(f"{len(paths)} images loaded as stack", 3000)

    def on_stack_item_clicked(self, item):
        idx = self.stack_list.row(item)
        if self.current_stack:
            self.current_image = self.current_stack[idx]
            self._input_window.show_image(self.current_image)

    # ------------------------------------------------------------------
    # File → save output
    # ------------------------------------------------------------------

    def save_output_image(self):
        if self._last_results is None:
            QMessageBox.information(self, "No Output",
                                    "Run the pipeline first.")
            return
        last_id = self._find_terminal_node()
        arr = self._last_results.get(last_id)
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

    def save_output_video(self):
        if not self.current_stack:
            QMessageBox.information(self, "No Stack",
                                    "Load an image stack first.")
            return
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline",
                                    "Add at least one node first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output as Video", "output.mp4",
            "MP4 (*.mp4);;AVI (*.avi)"
        )
        if not path:
            return
        try:
            import cv2
            last_id = self._find_terminal_node()
            first_result = self.pipeline.run(self.current_stack[0])
            first = arr_to_uint8(first_result[last_id])
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, 10, (w, h))
            writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
                         if first.ndim == 3 else cv2.cvtColor(first, cv2.COLOR_GRAY2BGR))
            for img in self.current_stack[1:]:
                res = self.pipeline.run(img)
                frame = arr_to_uint8(res[last_id])
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                             if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            writer.release()
            self.statusBar().showMessage(f"Video saved: {path}", 5000)
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency",
                                 "opencv-python is required for video export.\n"
                                 "Install with:  pip install opencv-python")
        except Exception as exc:
            QMessageBox.critical(self, "Video Error", str(exc))

    # ------------------------------------------------------------------
    # Pipeline: run
    # ------------------------------------------------------------------

    def run_pipeline_on_current(self):
        if self.current_image is None:
            QMessageBox.information(self, "No Image", "Please load an image first.")
            return
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline",
                                    "Add at least one node first.")
            return
        try:
            self._last_results = self.pipeline.run(self.current_image)
        except Exception as exc:
            QMessageBox.critical(self, "Pipeline Error", str(exc))
            return
        last_id = self._find_terminal_node()
        if last_id and last_id in self._last_results:
            self._output_window.show_image(self._last_results[last_id])
        self.statusBar().showMessage("Pipeline executed successfully", 3000)

    def run_pipeline_on_stack(self):
        if not self.current_stack:
            QMessageBox.information(self, "No Stack", "Load an image stack first.")
            return
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline",
                                    "Add at least one node first.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Processed Images"
        )
        if not out_dir:
            return
        last_id = self._find_terminal_node()
        try:
            from PIL import Image as PILImage
            for img, path in zip(self.current_stack, self.current_stack_paths):
                results = self.pipeline.run(img)
                out_arr = results.get(last_id)
                if out_arr is None:
                    continue
                out_name = os.path.splitext(os.path.basename(path))[0] + "_processed.png"
                PILImage.fromarray(arr_to_uint8(out_arr)).save(
                    os.path.join(out_dir, out_name)
                )
        except Exception as exc:
            QMessageBox.critical(self, "Stack Error", str(exc))
            return
        self.statusBar().showMessage(
            f"Stack processed, results saved to {out_dir}", 5000
        )

    def _find_terminal_node(self):
        sources = {e[0] for e in self.pipeline.edges}
        all_ids = list(self.pipeline.nodes.keys())
        terminal = [nid for nid in all_ids if nid not in sources]
        return terminal[-1] if terminal else (all_ids[-1] if all_ids else None)

    # ------------------------------------------------------------------
    # Pipeline: save / load
    # ------------------------------------------------------------------

    def save_pipeline(self):
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline",
                                    "Nothing to save.")
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
