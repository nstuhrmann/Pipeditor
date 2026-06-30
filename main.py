"""
Hauptfenster der Anwendung:
  - links: Palette aller verfügbaren Steps (gruppiert nach CATEGORY)
  - Mitte: Node-Editor (QGraphicsView über PipelineScene)
  - rechts: Bildvorschau (Original / Ergebnis)
  - Toolbar: Bild laden, Stack laden, Pipeline ausführen, speichern/laden
"""
import sys
import os
import glob
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeWidget, QTreeWidgetItem, QGraphicsView, QLabel, QFileDialog,
    QToolBar, QMessageBox, QSplitter, QStatusBar, QListWidget,
)
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter
from PySide6.QtCore import Qt

from base_step import STEP_REGISTRY
import steps  # noqa: F401  (Import löst Auto-Registrierung aller Steps aus)
from pipeline import Pipeline
from node_graphics import PipelineScene
from param_dialog import ParamDialog


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
    """Lädt ein Bild als numpy-Array. Nutzt Pillow, falls vorhanden, sonst
    QImage als Fallback (damit die App auch ohne Pillow läuft)."""
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImgPipe – Bildverarbeitungs-Pipeline-Editor")
        self.resize(1400, 850)

        self.pipeline = Pipeline()
        self.scene = PipelineScene(self.pipeline)
        self.scene.nodeDoubleClicked.connect(self.on_node_double_clicked)
        self.scene.edgeRequested.connect(self.on_edge_requested)

        self.current_image: np.ndarray | None = None
        self.current_stack: list[np.ndarray] | None = None
        self.current_stack_paths: list[str] | None = None

        self._build_ui()

    # ---------------------------------------------------------------------
    def _build_ui(self):
        # --- Palette links --------------------------------------------------
        self.palette = QTreeWidget()
        self.palette.setHeaderLabel("Verfügbare Steps")
        self.palette.itemDoubleClicked.connect(self.on_palette_item_double_clicked)
        self._populate_palette()

        # --- Node-Editor Mitte ----------------------------------------------
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)

        # --- Vorschau rechts --------------------------------------------------
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.addWidget(QLabel("<b>Original</b>"))
        self.original_label = QLabel("Kein Bild geladen")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumHeight(220)
        self.original_label.setStyleSheet("background:#1e1e1e; color:#888;")
        preview_layout.addWidget(self.original_label)

        preview_layout.addWidget(QLabel("<b>Ergebnis</b>"))
        self.result_label = QLabel("Noch nicht ausgeführt")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(220)
        self.result_label.setStyleSheet("background:#1e1e1e; color:#888;")
        preview_layout.addWidget(self.result_label)

        preview_layout.addWidget(QLabel("<b>Bild-Stack</b>"))
        self.stack_list = QListWidget()
        self.stack_list.itemClicked.connect(self.on_stack_item_clicked)
        preview_layout.addWidget(self.stack_list)

        # --- Splitter zusammenbauen --------------------------------------------
        splitter = QSplitter()
        splitter.addWidget(self.palette)
        splitter.addWidget(self.view)
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        self.setCentralWidget(splitter)

        self._build_toolbar()
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

    def _build_toolbar(self):
        tb = QToolBar("Aktionen")
        self.addToolBar(tb)

        act_load_img = QAction("Bild laden", self)
        act_load_img.triggered.connect(self.load_image)
        tb.addAction(act_load_img)

        act_load_stack = QAction("Bild-Stack laden", self)
        act_load_stack.triggered.connect(self.load_stack)
        tb.addAction(act_load_stack)

        tb.addSeparator()

        act_run = QAction("Pipeline ausführen", self)
        act_run.triggered.connect(self.run_pipeline_on_current)
        tb.addAction(act_run)

        act_run_stack = QAction("Pipeline auf Stack ausführen", self)
        act_run_stack.triggered.connect(self.run_pipeline_on_stack)
        tb.addAction(act_run_stack)

        tb.addSeparator()

        act_save = QAction("Pipeline speichern", self)
        act_save.triggered.connect(self.save_pipeline)
        tb.addAction(act_save)

        act_open = QAction("Pipeline laden", self)
        act_open.triggered.connect(self.load_pipeline)
        tb.addAction(act_open)

        tb.addSeparator()

        act_delete = QAction("Box löschen", self)
        act_delete.triggered.connect(self.delete_selected_node)
        tb.addAction(act_delete)

    # ---- Palette -> Node erzeugen ---------------------------------------
    def on_palette_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        cls_name = item.data(0, Qt.UserRole)
        if not cls_name:
            return  # Kategorie-Knoten, kein Step
        step_cls = STEP_REGISTRY[cls_name]
        step = step_cls()
        node = self.pipeline.add_node(step, pos=(50, 50))
        self.scene.add_node_item(node)
        self.statusBar().showMessage(f"Box '{step.NAME}' hinzugefügt", 3000)

    # ---- Parameter bearbeiten ---------------------------------------------
    def on_node_double_clicked(self, pipeline_node):
        dialog = ParamDialog(pipeline_node.step, self)
        if dialog.exec() == ParamDialog.Accepted:
            pipeline_node.step.set_param_values(dialog.get_values())
            item = self.scene.node_items[pipeline_node.id]
            item.refresh_params_preview()

    # ---- Verbindungen -------------------------------------------------------
    def on_edge_requested(self, from_id, to_id):
        try:
            self.pipeline.add_edge(from_id, to_id)
            self.scene.add_edge_item(from_id, to_id)
        except Exception as exc:
            QMessageBox.warning(self, "Fehler", str(exc))

    def delete_selected_node(self):
        for item in list(self.scene.selectedItems()):
            node_id = getattr(item, "pipeline_node", None)
            if node_id is not None:
                self.pipeline.remove_node(item.pipeline_node.id)
                self.scene.remove_node_item(item.pipeline_node.id)

    # ---- Bild laden ----------------------------------------------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Bild laden", "", "Bilder (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            self.current_image = load_image_as_array(path)
        except Exception as exc:
            QMessageBox.critical(self, "Fehler beim Laden", str(exc))
            return
        self.original_label.setPixmap(
            numpy_to_qpixmap(self.current_image).scaled(
                self.original_label.width() or 300, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self.statusBar().showMessage(f"Bild geladen: {path}", 3000)

    def load_stack(self):
        folder = QFileDialog.getExistingDirectory(self, "Ordner mit Bild-Stack wählen")
        if not folder:
            return
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        paths = sorted(sum((glob.glob(os.path.join(folder, e)) for e in exts), []))
        if not paths:
            QMessageBox.warning(self, "Kein Stack", "Keine Bilddateien in diesem Ordner gefunden.")
            return
        self.current_stack_paths = paths
        self.current_stack = [load_image_as_array(p) for p in paths]
        self.stack_list.clear()
        self.stack_list.addItems([os.path.basename(p) for p in paths])
        self.statusBar().showMessage(f"{len(paths)} Bilder als Stack geladen", 3000)

    def on_stack_item_clicked(self, item):
        idx = self.stack_list.row(item)
        if self.current_stack:
            self.current_image = self.current_stack[idx]
            self.original_label.setPixmap(
                numpy_to_qpixmap(self.current_image).scaled(
                    self.original_label.width() or 300, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    # ---- Pipeline ausführen ----------------------------------------------
    def run_pipeline_on_current(self):
        if self.current_image is None:
            QMessageBox.information(self, "Kein Bild", "Bitte zuerst ein Bild laden.")
            return
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Leere Pipeline", "Bitte mindestens eine Box hinzufügen.")
            return
        try:
            results = self.pipeline.run(self.current_image)
        except Exception as exc:
            QMessageBox.critical(self, "Fehler bei Ausführung", str(exc))
            return
        last_node_id = self._find_terminal_node()
        if last_node_id and last_node_id in results:
            self.result_label.setPixmap(
                numpy_to_qpixmap(results[last_node_id]).scaled(
                    self.result_label.width() or 300, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        self.statusBar().showMessage("Pipeline erfolgreich ausgeführt", 3000)

    def run_pipeline_on_stack(self):
        if not self.current_stack:
            QMessageBox.information(self, "Kein Stack", "Bitte zuerst einen Bild-Stack laden.")
            return
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Leere Pipeline", "Bitte mindestens eine Box hinzufügen.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Ausgabeordner für Ergebnisse wählen")
        if not out_dir:
            return
        last_node_id = self._find_terminal_node()
        try:
            from PIL import Image as PILImage
            for i, (img, path) in enumerate(zip(self.current_stack, self.current_stack_paths)):
                results = self.pipeline.run(img)
                out_arr = results.get(last_node_id)
                if out_arr is None:
                    continue
                out_name = os.path.splitext(os.path.basename(path))[0] + "_processed.png"
                arr = out_arr
                if arr.dtype != np.uint8:
                    a = arr.astype(np.float32)
                    a -= a.min()
                    if a.max() > 0:
                        a = a / a.max() * 255
                    arr = a.astype(np.uint8)
                PILImage.fromarray(arr).save(os.path.join(out_dir, out_name))
        except Exception as exc:
            QMessageBox.critical(self, "Fehler bei Stack-Verarbeitung", str(exc))
            return
        self.statusBar().showMessage(f"Stack verarbeitet, gespeichert in {out_dir}", 5000)

    def _find_terminal_node(self):
        """Findet die Box ohne ausgehende Kante (= Ende der Pipeline)."""
        sources = {e[0] for e in self.pipeline.edges}
        all_ids = list(self.pipeline.nodes.keys())
        terminal = [nid for nid in all_ids if nid not in sources]
        return terminal[-1] if terminal else (all_ids[-1] if all_ids else None)

    # ---- Pipeline speichern / laden -------------------------------------
    def save_pipeline(self):
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Leere Pipeline", "Es gibt nichts zu speichern.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Pipeline speichern", "pipeline.json", "JSON (*.json)")
        if not path:
            return
        self.pipeline.save(path)
        self.statusBar().showMessage(f"Pipeline gespeichert: {path}", 3000)

    def load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pipeline laden", "", "JSON (*.json)")
        if not path:
            return
        try:
            new_pipeline = Pipeline.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Fehler beim Laden", str(exc))
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

        self.statusBar().showMessage(f"Pipeline geladen: {path}", 3000)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
