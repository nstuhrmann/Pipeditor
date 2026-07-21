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
    QLabel, QSpinBox, QProgressDialog, QSlider,
)
from PySide6.QtGui import QAction, QPainter, QKeySequence
from PySide6.QtCore import Qt, QThread, QTimer

from src.GUI.pipeline_editor.base_step import STEP_REGISTRY
import src.GUI.pipeline_editor.steps  # noqa: F401  (registers all steps)
from src.GUI.pipeline_editor.pipeline import Pipeline
from src.GUI.pipeline_editor.node_graphics import PipelineScene, PipelineView
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
        self._pipeline_path: str | None = None   # current file for File→Save
        self._preview_windows: dict[str, LivePreviewWindow] = {}   # node_id → window

        self._live_mode  = False
        self._is_running = False
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None

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

        self.view = PipelineView(self.scene)
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
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setFixedWidth(160)
        self._frame_slider.setRange(0, 0)
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 0)
        # slider and spinbox mirror each other (guarded against loops)
        self._frame_slider.valueChanged.connect(self._frame_spin.setValue)
        self._frame_spin.valueChanged.connect(self._frame_slider.setValue)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        for w in (self._frame_label, self._frame_slider, self._frame_spin):
            w.setVisible(False)
            self.statusBar().addPermanentWidget(w)

    def _on_frame_spin_changed(self, _value: int):
        # In live mode, scrubbing the frame selector should re-run the
        # pipeline for the newly selected frame — debounced through the
        # same timer as parameter edits, so holding the arrow / typing
        # doesn't fire a run per tick.
        if self._live_mode:
            self._live_timer.start()

    def _build_menu(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu("&File")
        self._add_action(fm, "New Pipeline", self.new_pipeline, "Ctrl+N")
        self._add_action(fm, "Open Pipeline…",
                         self.load_pipeline, "Ctrl+O")
        fm.addSeparator()
        self._add_action(fm, "Save", self.save_pipeline, "Ctrl+S")
        self._add_action(fm, "Save As…",
                         self.save_pipeline_as, "Ctrl+Shift+S")
        fm.addSeparator()
        self._add_action(fm, "Save Output to File…",
                         self.save_output_image)
        fm.addSeparator()
        self._add_action(fm, "Quit", self.close, "Ctrl+Q")

        # Pipeline
        pm = mb.addMenu("&Pipeline")
        self._act_run = self._add_action(pm, "Run",
                                          self.run_pipeline, "F8")
        self._act_process_sequence = self._add_action(
            pm, "Process Full Sequence…",
            self.process_full_sequence, "Ctrl+Shift+R")

        pm.addSeparator()
        self._add_action(pm, "Optimize Parameters…",
                         self.optimize_parameters, "Ctrl+Shift+O")

        self._act_live = QAction("Live Update", self)
        self._act_live.setCheckable(True)
        self._act_live.toggled.connect(self._on_live_toggled)
        pm.addAction(self._act_live)

        # Edit
        em = mb.addMenu("&Edit")
        self._add_action(em, "Delete Selected",
                         self.delete_selected,
                         QKeySequence.Delete)

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
        scene.nodeDeleteRequested.connect(self.on_node_delete_requested)

    # ------------------------------------------------------------------
    # Palette
    # ------------------------------------------------------------------

    def _populate_palette(self):
        """Builds the step palette. CATEGORY supports hierarchy via
        '/' (or '\\'), e.g. CATEGORY = "Filter/Denoise" nests Denoise
        under Filter. Plain single-level categories work unchanged."""
        self.palette.clear()
        categories: dict[tuple, QTreeWidgetItem] = {}

        def category_item(path_parts: tuple) -> QTreeWidgetItem:
            item = categories.get(path_parts)
            if item is not None:
                return item
            item = QTreeWidgetItem([path_parts[-1]])
            if len(path_parts) == 1:
                self.palette.addTopLevelItem(item)
            else:
                category_item(path_parts[:-1]).addChild(item)
            categories[path_parts] = item
            return item

        for cls_name, cls in sorted(STEP_REGISTRY.items(),
                                    key=lambda kv: kv[1].NAME):
            parts = tuple(p.strip() for p in
                          cls.CATEGORY.replace("\\", "/").split("/")
                          if p.strip()) or ("General",)
            leaf = QTreeWidgetItem([cls.NAME])
            leaf.setData(0, Qt.UserRole, cls_name)
            category_item(parts).addChild(leaf)
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

    def optimize_parameters(self):
        """Search selected parameter ranges to minimize/maximize a metric."""
        if self._is_running:
            return
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline",
                                    "Add nodes first.")
            return
        from src.GUI.pipeline_editor.optimize_dialog import OptimizeDialog
        total = self._refresh_frame_selector()
        frame = self._frame_spin.value() if total > 1 else 0
        dlg = OptimizeDialog(self.pipeline, frame_index=frame,
                             total_frames=total, parent=self)
        if dlg.exec() == OptimizeDialog.Accepted:
            for nid in dlg.changed_node_ids():
                item = self.scene.node_items.get(nid)
                if item is not None:
                    item.refresh_params_preview()
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

    def _delete_node(self, pnode):
        """Remove one node: its preview window, the model entry, and the
        scene item. Used by Edit→Delete and the node context menu."""
        win = self._preview_windows.pop(pnode.id, None)
        if win:
            win.close()
        self.pipeline.remove_node(pnode.id)
        self.scene.remove_node_item(pnode.id)

    def on_node_delete_requested(self, pnode):
        self._delete_node(pnode)

    def delete_selected(self):
        """Delete whatever is selected — nodes AND edges. The menu
        shortcut (Del) grabs the key before the scene's own
        keyPressEvent ever sees it, which is why edge deletion must be
        handled here too (previously only Backspace reached the scene,
        so Del appeared broken for edges)."""
        from src.GUI.pipeline_editor.node_graphics import EdgeItem
        for item in list(self.scene.selectedItems()):
            pnode = getattr(item, "pipeline_node", None)
            if pnode is not None:
                self._delete_node(pnode)
            elif isinstance(item, EdgeItem) and item._is_permanent:
                self.scene.remove_edge_item_by_ref(item)

    def new_pipeline(self):
        if self.pipeline.nodes and QMessageBox.question(
            self, "New Pipeline",
            "Discard the current pipeline and start a new one?"
        ) != QMessageBox.Yes:
            return
        for nid in list(self.pipeline.nodes.keys()):
            win = self._preview_windows.pop(nid, None)
            if win:
                win.close()
            self.scene.remove_node_item(nid)
            self.pipeline.remove_node(nid)
        self._pipeline_path = None
        self._update_window_title()

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

        self._worker = PipelineWorker(self.pipeline, frame_index, total)
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
        for w in (self._frame_label, self._frame_slider, self._frame_spin):
            w.setVisible(is_sequence)
        if is_sequence:
            self._frame_spin.setMaximum(total - 1)
            self._frame_slider.setMaximum(total - 1)
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

        self._seq_worker = SequenceWorker(self.pipeline)
        self._seq_thread = QThread()
        self._seq_worker.moveToThread(self._seq_thread)
        self._seq_thread.started.connect(self._seq_worker.run)

        # IMPORTANT — these must be bound methods of a QObject (this
        # window), NOT local closures. Qt invokes plain-function
        # receivers in the EMITTING thread; with closures here, all the
        # GUI work (progress dialog, scene/thumbnail updates, QPixmap
        # creation) ran on the worker thread — undefined behavior that
        # showed up as vanishing/misdrawn previews after a batch. Bound
        # methods of a main-thread QObject get queued onto the GUI
        # thread automatically.
        self._seq_progress = progress
        self._seq_total = total
        self._seq_worker.frameDone.connect(self._on_seq_frame_done)
        self._seq_worker.finished.connect(self._on_seq_finished)
        self._seq_worker.failed.connect(self._on_seq_failed)
        # The worker's thread is blocked inside run_sequence() and never
        # spins an event loop, so a queued call to worker.cancel would
        # never be delivered. A lambda receiver runs directly in the
        # GUI thread (the emitter), and setting the flag cross-thread is
        # safe — it's polled between frames.
        progress.canceled.connect(lambda: self._seq_worker.cancel())
        self._seq_thread.finished.connect(self._cleanup_sequence_thread)

        self._seq_thread.start()

    def _on_seq_frame_done(self, done: int, total: int):
        self._seq_progress.setValue(done)
        self._seq_progress.setLabelText(f"Processing frame {done} / {total}…")

    def _on_seq_finished(self, processed: int):
        total = self._seq_total
        self._seq_progress.setValue(total)
        self._seq_thread.quit()
        if self._seq_worker.last_results is not None:
            self._last_results = self._seq_worker.last_results
            self.scene.update_previews(
                self._last_results, self._seq_worker.last_metric_values)
            for edge in self.scene.edge_items:
                edge.update_path()
            self.scene.update()
            for nid, win in self._preview_windows.items():
                val = self._last_results.get(nid)
                if isinstance(val, np.ndarray):
                    win.show_image(val)
        note = (f"Sequence processing complete: {processed}/{total} frames."
               if processed == total else
               f"Sequence processing cancelled after {processed}/{total} frames.")
        if self._seq_worker.warnings:
            note += f"  ⚠ {len(self._seq_worker.warnings)} warning(s)."
            self.statusBar().setToolTip("\n".join(self._seq_worker.warnings))
        self.statusBar().showMessage(note, 6000)

    def _on_seq_failed(self, err: str):
        self._seq_progress.close()
        self._seq_thread.quit()
        QMessageBox.critical(self, "Sequence Processing Error", err)

    def _cleanup_sequence_thread(self):
        self._is_running = False
        self._act_run.setEnabled(True)
        self._act_process_sequence.setEnabled(True)

    def _cleanup_thread(self):
        self._is_running = False
        self._act_run.setEnabled(True)
        self._act_process_sequence.setEnabled(True)

    def _on_run_finished(self, results: dict, warnings: list = None,
                         metric_values: dict = None):
        self._last_results = results
        self.scene.update_previews(results, metric_values)

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
        """File → Save: overwrite the current file; only prompts for a
        filename if this pipeline has never been saved/loaded."""
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline", "Nothing to save.")
            return
        if self._pipeline_path is None:
            self.save_pipeline_as()
            return
        self.pipeline.save(self._pipeline_path)
        self.statusBar().showMessage(
            f"Pipeline saved: {self._pipeline_path}", 3000)

    def save_pipeline_as(self):
        """File → Save As: always prompts, then becomes the current file."""
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline As", self._pipeline_path or "pipeline.json",
            "JSON (*.json)")
        if not path:
            return
        self.pipeline.save(path)
        self._pipeline_path = path
        self._update_window_title()
        self.statusBar().showMessage(f"Pipeline saved: {path}", 3000)

    def _update_window_title(self):
        base = "ImgPipe – Pipeline Editor"
        if self._pipeline_path:
            import os
            self.setWindowTitle(
                f"{base} — {os.path.basename(self._pipeline_path)}")
        else:
            self.setWindowTitle(base)

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

        self._pipeline_path = path
        self._update_window_title()
        self.statusBar().showMessage(f"Pipeline loaded: {path}", 3000)


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
