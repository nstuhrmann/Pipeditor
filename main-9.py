"""
ImgPipe main window (MainWindow + entry point only).

The rest of the app lives in focused modules:
  pipeline.py        graph data model + execution (no Qt)
  base_step.py       ProcessingStep base class, ParamSpec, registry
  node_graphics.py   node editor scene/items
  param_dialog.py    auto-generated parameter dialogs
  image_canvas.py    zoom/pan canvas + histogram widget
  preview_window.py  per-node LivePreviewWindow
  workers.py         background-thread pipeline/sequence workers
  image_utils.py     shared numpy <-> Qt image conversion
"""
import sys
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QTreeWidget, QTreeWidgetItem,
    QGraphicsView, QMessageBox, QSplitter, QStatusBar, QFileDialog,
    QLabel, QSpinBox, QProgressDialog,
)
from PySide6.QtGui import QAction, QPainter, QKeySequence
from PySide6.QtCore import Qt, QThread, QTimer

from src.GUI.pipeline_editor.base_step import STEP_REGISTRY
import src.GUI.pipeline_editor.steps  # noqa: F401  (registers all steps)
from src.GUI.pipeline_editor.pipeline import Pipeline
from src.GUI.pipeline_editor.node_graphics import PipelineScene
from src.GUI.pipeline_editor.param_dialog import ParamDialog
from src.GUI.pipeline_editor.preview_window import LivePreviewWindow
from src.GUI.pipeline_editor.workers import PipelineWorker, SequenceWorker
from src.GUI.pipeline_editor.image_utils import arr_to_uint8


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

        # Frame selector — only shown once the graph actually has a
        # video/image-stack source with more than one frame. Lets Run /
        # Live Update preview any single frame of a sequence, not just
        # frame 0, without processing the whole thing.
        self._frame_label = QLabel("Frame:")
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 0)
        self._frame_label.setVisible(False)
        self._frame_spin.setVisible(False)
        self.statusBar().addPermanentWidget(self._frame_label)
        self.statusBar().addPermanentWidget(self._frame_spin)

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
        self._act_process_sequence = self._add_action(
            pm, "Process Full Sequence…",
            self.process_full_sequence, "Ctrl+Shift+R")

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
        scene.nodeBypassToggled.connect(self.on_node_bypass_toggled)

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
            win.viewChanged.connect(self._on_preview_view_changed)
            win.lockToggled.connect(self._on_preview_lock_toggled)
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

    def _on_preview_view_changed(self, node_id: str):
        """Propagate one locked preview window's zoom/pan to every other
        locked preview window. Windows that aren't locked are untouched."""
        src_win = self._preview_windows.get(node_id)
        if src_win is None or not src_win.is_locked():
            return
        scale, offset = src_win.view_state()
        for nid, win in self._preview_windows.items():
            if nid != node_id and win.is_locked():
                win.apply_view_state(scale, offset)

    def _on_preview_lock_toggled(self, node_id: str, locked: bool):
        if not locked:
            return
        # Snap the newly-locked window to match whatever view the other
        # locked windows are already showing, instead of waiting for the
        # next pan/zoom to bring it into sync.
        for nid, win in self._preview_windows.items():
            if nid != node_id and win.is_locked():
                scale, offset = win.view_state()
                self._preview_windows[node_id].apply_view_state(scale, offset)
                break

    def on_node_bypass_toggled(self, pipeline_node):
        state = "bypassed" if pipeline_node.bypassed else "active"
        self.statusBar().showMessage(
            f"{pipeline_node.display_name}: {state}", 3000)
        if self._live_mode:
            self.run_pipeline()

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

        total = self._refresh_frame_selector()
        frame_index = self._frame_spin.value() if total > 1 else 0

        self._is_running = True
        self._act_run.setEnabled(False)
        self._act_process_sequence.setEnabled(False)
        self.statusBar().showMessage("Running pipeline…")

        self._worker = _PipelineWorker(self.pipeline, frame_index, total)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _refresh_frame_selector(self) -> int:
        """Recompute total_frames() and show/hide the Frame: spinbox
        accordingly. Returns total_frames for convenience."""
        total = self.pipeline.total_frames()
        is_sequence = total > 1
        self._frame_label.setVisible(is_sequence)
        self._frame_spin.setVisible(is_sequence)
        if is_sequence:
            self._frame_spin.setMaximum(total - 1)
        return total

    # ------------------------------------------------------------------
    # Process Full Sequence (background thread + progress/cancel)
    # ------------------------------------------------------------------

    def process_full_sequence(self):
        if self._is_running:
            return
        total = self.pipeline.total_frames()
        if total <= 1:
            QMessageBox.information(
                self, "No Sequence Source",
                "This pipeline has no video or image-stack source with "
                "more than one frame — add one (Video File Source / Image "
                "Stack Source), or just use Run for a single image.")
            return

        self._is_running = True
        self._act_run.setEnabled(False)
        self._act_process_sequence.setEnabled(False)

        progress = QProgressDialog(
            "Processing frame 0 / %d…" % total, "Cancel", 0, total, self)
        progress.setWindowTitle("Process Full Sequence")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self._seq_worker = _SequenceWorker(self.pipeline)
        self._seq_thread = QThread()
        self._seq_worker.moveToThread(self._seq_thread)
        self._seq_thread.started.connect(self._seq_worker.run)

        def on_frame_done(done, tot):
            progress.setValue(done)
            progress.setLabelText(f"Processing frame {done} / {tot}…")

        def on_finished(processed):
            progress.setValue(total)
            self._seq_thread.quit()
            if self._seq_worker.last_results is not None:
                self._last_results = self._seq_worker.last_results
                self.scene.update_previews(self._last_results)
                for nid, win in self._preview_windows.items():
                    val = self._last_results.get(nid)
                    if isinstance(val, np.ndarray):
                        win.show_image(val)
            note = (f"Sequence processing complete: {processed}/{total} frames."
                   if processed == total else
                   f"Sequence processing cancelled after {processed}/{total} frames.")
            if self._seq_worker.warnings:
                note += f"  ⚠ {len(self._seq_worker.warnings)} warning(s)."
                self.statusBar().setToolTip(
                    "\n".join(self._seq_worker.warnings))
            self.statusBar().showMessage(note, 6000)

        def on_failed(err):
            progress.close()
            self._seq_thread.quit()
            QMessageBox.critical(self, "Sequence Processing Error", err)

        self._seq_worker.frameDone.connect(on_frame_done)
        self._seq_worker.finished.connect(on_finished)
        self._seq_worker.failed.connect(on_failed)
        progress.canceled.connect(self._seq_worker.cancel)
        self._seq_thread.finished.connect(self._cleanup_sequence_thread)

        self._seq_thread.start()

    def _cleanup_sequence_thread(self):
        self._is_running = False
        self._act_run.setEnabled(True)
        self._act_process_sequence.setEnabled(True)

    def _cleanup_thread(self):
        self._is_running = False
        self._act_run.setEnabled(True)
        self._act_process_sequence.setEnabled(True)

    def _on_run_finished(self, results: dict, warnings: list = None):
        self._last_results = results
        self.scene.update_previews(results)

        # Refresh any fullscreen preview windows that are currently open,
        # so live mode / manual re-runs keep them up to date.
        for nid, win in list(self._preview_windows.items()):
            if nid in results:
                val = results[nid]
                if isinstance(val, np.ndarray):
                    win.show_image(val)

        if warnings:
            # Non-modal on purpose: a half-wired scratch node shouldn't
            # produce a dialog on every live-update tick. First reason in
            # the status bar; full list in its tooltip.
            head = warnings[0]
            more = f"  (+{len(warnings) - 1} more)" if len(warnings) > 1 else ""
            self.statusBar().showMessage(f"⚠ {head}{more}", 8000)
            self.statusBar().setToolTip("\n".join(warnings))
        else:
            self.statusBar().showMessage("Pipeline executed successfully", 3000)
            self.statusBar().setToolTip("")

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
