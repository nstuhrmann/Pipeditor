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
    QLabel, QSpinBox, QProgressDialog, QSlider, QLineEdit,
    QWidget, QVBoxLayout, QMenu, QToolButton, QStyle, QAbstractItemView,
)
from PySide6.QtGui import (
    QAction, QActionGroup, QPainter, QKeySequence, QDrag,
)
from PySide6.QtCore import (
    Qt, QThread, QTimer, QSettings, QMimeData, QByteArray,
)

import src.GUI.pipeline_editor.steps  # noqa: F401  (registers every step)
from src.GUI.pipeline_editor.base_step import (
    STEP_REGISTRY, step_source_location,
)
from src.GUI.pipeline_editor.pipeline import EDGE_CONTROL, EDGE_DATA, Pipeline
from src.GUI.pipeline_editor import run_log, theme
from src.GUI.pipeline_editor.node_graphics import (
    EDGE_STYLES, PALETTE_MIME, PipelineScene, PipelineView,
    clipboard_put_step, clipboard_step,
)
from src.GUI.pipeline_editor.param_dialog import ParamDialog
from src.GUI.pipeline_editor.preview_window import LivePreviewWindow
from src.GUI.pipeline_editor.workers import PipelineWorker, SequenceWorker
from src.GUI.pipeline_editor.image_utils import arr_to_uint8


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

def _step_tooltip(step_cls) -> str:
    """What a palette entry says on hover: the class, where it is
    defined, and the first paragraph of its docstring.

    The docstring is the step author's own description, so this needs no
    parallel help text that could drift out of date."""
    path, line = step_source_location(step_cls)
    parts = [f"{step_cls.__name__}  ({step_cls.KIND})"]
    doc = (step_cls.__doc__ or "").strip()
    if doc:
        # First paragraph only — several steps have long docstrings and a
        # tooltip that fills the screen is worse than none.
        para = doc.split("\n\n")[0]
        parts.append(" ".join(w for w in para.split()))
    if path:
        parts.append(f"{path}:{line}")
    return "\n\n".join(parts)


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

        self._settings = QSettings("ImgPipe", "PipelineEditor")
        # Canvas side-channel display, remembered between sessions.
        self._show_meta = self._settings.value(
            "show_meta", True, type=bool)
        self._show_control = self._settings.value(
            "show_control", True, type=bool)
        self._edge_style = self._settings.value("edge_style", "curved")
        self._full_repaint = self._settings.value(
            "full_repaint", True, type=bool)

        # --- frame-by-frame playback state ---
        # Playback pulls frames from pipeline.iter_sequence() one per
        # timer tick. It is the SAME generator Process Sequence and
        # the optimizer drain, so stateful steps advance, control edges
        # deliver and sinks write exactly as in a batch — only watchable.
        self._playback_active = False       # timer is ticking
        self._playback_in_sequence = False  # generator open
        self._playback_iter = None          # the live iter_sequence()
        self._playback_index = 0
        self._playback_total = 1
        self._playback_timer = QTimer(self)
        self._playback_timer.setSingleShot(True)
        self._playback_timer.timeout.connect(self._playback_tick)

        self._build_ui()
        self._build_menu()

        # Reopen the last pipeline (defer one event-loop turn so the
        # window exists before any load-error dialog could appear).
        # Everything the following depend on now exists: settings read,
        # widgets built, menu actions created.
        self._refresh_side_channels()
        self.scene.set_edge_style(self._edge_style)
        self._pipeline_changed()

        last = self._settings.value("last_pipeline", "")
        if last:
            QTimer.singleShot(0, lambda: self._load_pipeline_path(
                last, silent=True))

    # ------------------------------------------------------------------
    # UI / menu
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.palette = QTreeWidget()
        self.palette.setHeaderLabel("Available Steps")
        self.palette.setMinimumWidth(180)
        self.palette.itemDoubleClicked.connect(self.on_palette_item_double_clicked)
        # Dragging a step onto the canvas places it where you drop it,
        # rather than at a fixed spot you then have to move it from.
        self.palette.setDragEnabled(True)
        self.palette.setDragDropMode(QAbstractItemView.DragOnly)
        self.palette.startDrag = self._palette_start_drag
        self._populate_palette()

        # Search/filter above the palette — with 30+ steps in nested
        # categories, scrolling stopped scaling.
        self._palette_filter = QLineEdit()
        self._palette_filter.setPlaceholderText("Search steps…")
        self._palette_filter.setClearButtonEnabled(True)
        self._palette_filter.textChanged.connect(self._filter_palette)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        left_layout.addWidget(self._palette_filter)
        left_layout.addWidget(self.palette)

        self.view = PipelineView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        # Qt's default MinimalViewportUpdate repaints only the rectangles
        # items declare. Anything painted even slightly outside a
        # boundingRect — an antialiased edge, a text halo — leaves the
        # old pixels behind as streaks. Redrawing the whole viewport
        # costs a little on very large graphs but cannot smear.
        self.view.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())

        # Frame selector — only shown once the graph actually has a
        # video/image-stack source with more than one frame. Lets Run /
        # Auto Preview can show any single frame of a sequence, not just
        # frame 0, without processing the whole thing.
        self._frame_label = QLabel("Frame:")
        self._frame_total = QLabel("")   # "/ N" — absolute context for the slider
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setFixedWidth(160)
        self._frame_slider.setRange(0, 0)
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 0)
        # slider and spinbox mirror each other (guarded against loops)
        self._frame_slider.valueChanged.connect(self._frame_spin.setValue)
        self._frame_spin.valueChanged.connect(self._frame_slider.setValue)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        # Transport controls, right of the frame slider. Same actions as
        # the Pipeline menu entries — deliberately the same methods, so
        # menu and buttons can never diverge in behaviour.
        # Qt's own media icons: nothing to bundle through Nuitka, and
        # they follow the platform style and DPI for free. Text lives in
        # the tooltips and the Pipeline menu.
        st = self.style()
        self._icon_play = st.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self._icon_pause = st.standardIcon(QStyle.StandardPixmap.SP_MediaPause)

        self._btn_play = QToolButton()
        self._btn_play.setIcon(self._icon_play)
        self._btn_play.setToolTip("Play / pause the sequence (F5)")
        self._btn_play.clicked.connect(self.toggle_playback)

        self._btn_step = QToolButton()
        self._btn_step.setIcon(
            st.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self._btn_step.setToolTip("Advance one frame (F6)")
        self._btn_step.clicked.connect(self.step_frame)

        self._btn_stop = QToolButton()
        self._btn_stop.setIcon(
            st.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self._btn_stop.setToolTip("Stop the sequence and release its "
                                  "outputs (Shift+F5)")
        self._btn_stop.clicked.connect(self.stop_sequence)

        for w in (self._frame_label, self._frame_slider, self._frame_spin,
                  self._frame_total, self._btn_play, self._btn_step,
                  self._btn_stop):
            w.setVisible(False)
            self.statusBar().addPermanentWidget(w)
        self._update_playback_buttons()

    def _on_frame_spin_changed(self, _value: int):
        # Scrubbing during playback would race the sequence generator:
        # a preview at a jumped index resets the stateful steps the
        # generator is mid-way through, silently corrupting the rest of
        # the run. You scrubbed because you wanted to look at a frame, so
        # pause — the sequence stays open and Play resumes from here.
        # (Programmatic moves come through _sync_frame_widgets with
        # signals blocked, so this only fires for a real user scrub.)
        if self._playback_active:
            self._playback_active = False
            self._playback_timer.stop()
            self._update_playback_buttons()
            self.statusBar().showMessage(
                "Sequence paused — frame selector moved", 4000)

        # In live mode, scrubbing re-runs the pipeline for the newly
        # selected frame — debounced through the same timer as parameter
        # edits, so holding the arrow / typing doesn't fire a run per tick.
        if self._live_mode:
            self._live_timer.start()

    def _build_menu(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu("&File")
        self._add_action(fm, "New Pipeline", self.new_pipeline, "Ctrl+N")
        self._add_action(fm, "Open Pipeline…",
                         self.load_pipeline, "Ctrl+O")
        self._recent_menu = fm.addMenu("Open Recent")
        self._recent_menu.aboutToShow.connect(self._rebuild_recent_menu)
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
        # Two things you can run — the current frame, or the sequence —
        # and the names say which, plus whether it commits. "Preview"
        # never writes and can be repeated; "Process"/"Play" are the same
        # committed run, differing only in whether you watch it.
        pm = mb.addMenu("&Pipeline")
        self._act_run = self._add_action(
            pm, "Preview Frame", self.run_pipeline, "F8")
        self._act_live = QAction("Auto Preview", self)
        self._act_live.setCheckable(True)
        self._act_live.setToolTip(
            "Re-preview the current frame whenever a parameter changes")
        self._act_live.toggled.connect(self._on_live_toggled)
        pm.addAction(self._act_live)

        pm.addSeparator()
        self._act_process_sequence = self._add_action(
            pm, "Process Sequence…", self.process_full_sequence, "F9")
        self._act_play = self._add_action(
            pm, "Play Sequence", self.toggle_playback, "F5")
        self._act_step = self._add_action(
            pm, "Step One Frame", self.step_frame, "F6")
        self._act_stop = self._add_action(
            pm, "Stop Sequence", self.stop_sequence, "Shift+F5")

        pm.addSeparator()
        self._add_action(pm, "Optimize Parameters…",
                         self.optimize_parameters, "Ctrl+Shift+O")

        # Edit
        vm = mb.addMenu("&View")
        self._act_show_meta = QAction("Show Metadata", self, checkable=True)
        self._act_show_meta.setChecked(self._show_meta)
        self._act_show_meta.setToolTip(
            "Teal lines under each node: metadata it emitted")
        self._act_show_meta.toggled.connect(self._on_show_meta_toggled)
        vm.addAction(self._act_show_meta)

        self._act_show_control = QAction("Show Control Messages", self,
                                         checkable=True)
        self._act_show_control.setChecked(self._show_control)
        self._act_show_control.setToolTip(
            "Amber lines under each node: control values it sent")
        self._act_show_control.toggled.connect(self._on_show_control_toggled)
        vm.addAction(self._act_show_control)

        self._act_full_repaint = QAction("Full Viewport Repaint", self,
                                         checkable=True)
        self._act_full_repaint.setChecked(self._full_repaint)
        self._act_full_repaint.setToolTip(
            "Redraw the whole canvas each frame. Fixes leftover streaks "
            "on some displays; slightly slower on very large graphs.")
        self._act_full_repaint.toggled.connect(self._on_full_repaint_toggled)
        vm.addAction(self._act_full_repaint)
        vm.addSeparator()

        self._add_action(vm, "Zoom In", 
                         lambda: self.view.zoom_by(self.view.ZOOM_STEP),
                         "Ctrl++")
        self._add_action(vm, "Zoom Out",
                         lambda: self.view.zoom_by(1.0 / self.view.ZOOM_STEP),
                         "Ctrl+-")
        self._add_action(vm, "Reset Zoom", self.view.reset_zoom, "Ctrl+0")
        self._add_action(vm, "Fit to Content", self.view.fit_to_content,
                         "Ctrl+Shift+F")
        vm.addSeparator()

        style_menu = vm.addMenu("Edge Style")
        self._edge_style_group = QActionGroup(self)
        self._edge_style_group.setExclusive(True)
        for style in EDGE_STYLES:
            act = QAction(style.replace("_", " ").title(), self,
                          checkable=True)
            act.setChecked(style == self._edge_style)
            act.triggered.connect(
                lambda _checked=False, s=style: self._on_edge_style(s))
            self._edge_style_group.addAction(act)
            style_menu.addAction(act)

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
        scene.nodeDuplicateRequested.connect(self.on_node_duplicate_requested)
        scene.statusMessage.connect(
            lambda text: self.statusBar().showMessage(text, 5000))
        scene.nodeCopyRequested.connect(self.on_node_copy_requested)
        scene.outputPreviewRequested.connect(self.on_output_preview_requested)
        scene.fitRequested.connect(lambda: self.view.fit_to_content())
        scene.pasteRequested.connect(self.on_paste_requested)
        scene.addStepRequested.connect(self.on_add_step_requested)
        scene.edgeStyleRequested.connect(self._on_edge_style)

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
            leaf.setToolTip(0, _step_tooltip(cls))
            category_item(parts).addChild(leaf)
        self.palette.expandAll()

    def _filter_palette(self, text: str):
        """Hide steps not matching the filter; category items stay
        visible only while they have visible children. Matching is
        case-insensitive on the step name."""
        needle = text.strip().lower()

        def apply(item: QTreeWidgetItem) -> bool:
            if item.childCount() == 0:
                visible = (not needle) or (needle in item.text(0).lower())
                item.setHidden(not visible)
                return visible
            any_child = False
            for i in range(item.childCount()):
                if apply(item.child(i)):
                    any_child = True
            item.setHidden(not any_child)
            return any_child

        for i in range(self.palette.topLevelItemCount()):
            apply(self.palette.topLevelItem(i))
        if needle:
            self.palette.expandAll()

    def _palette_start_drag(self, supported_actions):
        item = self.palette.currentItem()
        class_name = item.data(0, Qt.UserRole) if item is not None else None
        if not class_name:
            return
        mime = QMimeData()
        mime.setData(PALETTE_MIME,
                     QByteArray(str(class_name).encode("utf-8")))
        drag = QDrag(self.palette)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)

    def on_palette_item_double_clicked(self, item: QTreeWidgetItem, _):
        cls_name = item.data(0, Qt.UserRole)
        if not cls_name:
            return
        step = STEP_REGISTRY[cls_name]()
        node = self.pipeline.add_node(step, pos=(50, 50))
        self.scene.add_node_item(node)
        self._pipeline_changed()
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
        # A parameter can change the frame count (a source's file path
        # being the obvious case), so the transport controls have to be
        # re-evaluated here too, not just on graph edits.
        self._pipeline_changed()
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

    def on_edge_requested(self, from_id: str, to_id: str, to_port: int,
                          kind: str = EDGE_DATA):
        try:
            self.pipeline.add_edge(from_id, to_id, to_port, kind=kind)
            self.scene.add_edge_item(from_id, to_id, to_port, kind=kind)
            if kind == EDGE_CONTROL:
                self.statusBar().showMessage(
                    "Control edge added — values arrive on the target's "
                    "next frame", 4000)
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
        self._pipeline_changed()

    def on_node_delete_requested(self, pnode):
        self._delete_node(pnode)

    def on_output_preview_requested(self, pnode):
        """Open (or raise) the live frame window for this node.

        Shares the window registry with the thumbnail double-click path,
        so a node has at most one output window however it was opened."""
        win = self._preview_windows.get(pnode.id)
        if win is None:
            win = LivePreviewWindow(pnode.id, pnode.display_name, self)
            win.closed.connect(self._on_preview_window_closed)
            win.viewChanged.connect(self._on_preview_view_changed)
            win.lockToggled.connect(self._on_preview_lock_toggled)
            self._preview_windows[pnode.id] = win
        val = (self._last_results or {}).get(pnode.id)
        if isinstance(val, np.ndarray):
            win.show_image(val)
        win.show()
        win.raise_()
        win.activateWindow()

    def on_node_copy_requested(self, pnode):
        clipboard_put_step(pnode.class_name, pnode.step.get_param_values())
        self.statusBar().showMessage(
            f"Copied '{pnode.display_name}'", 3000)

    def on_paste_requested(self, pos):
        """Paste the clipboard step at the cursor. Parameters are applied
        through set_param_values, which ignores keys the class no longer
        has — so a step pasted from an older build loads with whatever
        still applies rather than failing."""
        data = clipboard_step()
        if not data:
            return
        step_cls = STEP_REGISTRY.get(data.get("class_name", ""))
        if step_cls is None:
            QMessageBox.warning(
                self, "Paste Step",
                f"Unknown step type '{data.get('class_name')}' — is its "
                f"module imported?")
            return
        step = step_cls()
        step.set_param_values(data.get("params", {}))
        node = self.pipeline.add_node(step, pos=(pos.x(), pos.y()))
        item = self.scene.add_node_item(node)
        item.setSelected(True)
        self._pipeline_changed()
        self.statusBar().showMessage(f"Pasted '{node.display_name}'", 3000)

    def on_add_step_requested(self, class_name: str, pos):
        step_cls = STEP_REGISTRY.get(class_name)
        if step_cls is None:
            return
        node = self.pipeline.add_node(step_cls(), pos=(pos.x(), pos.y()))
        self.scene.add_node_item(node)
        self._pipeline_changed()
        self.statusBar().showMessage(f"Added '{node.display_name}'", 3000)

    def on_node_duplicate_requested(self, pnode):
        """Clone a node: same step class, same parameter values, same
        bypass state — fresh id/number, offset so it doesn't hide the
        original exactly. Edges are NOT copied: a duplicate usually gets
        wired differently (that's why you duplicated it)."""
        new_step = type(pnode.step)()
        new_step.set_param_values(pnode.step.get_param_values())
        node = self.pipeline.add_node(
            new_step, pos=(pnode.pos[0] + 40, pnode.pos[1] + 40))
        node.bypassed = pnode.bypassed
        item = self.scene.add_node_item(node)
        item.setSelected(True)
        self.statusBar().showMessage(
            f"Duplicated as '{node.display_name}'", 3000)

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
        self._pipeline_changed()

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
        has_source = any(n.step.KIND == "source"
                         for n in self.pipeline.nodes.values())
        if not has_source:
            QMessageBox.information(
                self, "No Source",
                "Add an Image Source node (Input / Output category).")
            return

        total = self._refresh_frame_selector()
        frame_index = self._frame_spin.value() if total > 1 else 0
        self.statusBar().showMessage("Previewing…")
        self._start_worker(frame_index, total)

    def _start_worker(self, frame_index: int, total: int):
        """One frame on a worker thread. During playback that means the
        next frame of the open sequence; otherwise a preview, which is
        what makes Run and live-update repeatable and side-effect free."""
        self._is_running = True
        self._act_run.setEnabled(False)
        self._act_process_sequence.setEnabled(False)

        self._worker = PipelineWorker(
            self.pipeline, frame_index, total,
            iterator=self._playback_iter if self._playback_in_sequence
            else None)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.exhausted.connect(self._on_sequence_exhausted)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.exhausted.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    # ------------------------------------------------------------------
    # Frame-by-frame playback
    # ------------------------------------------------------------------
    def _on_full_repaint_toggled(self, checked: bool):
        self._full_repaint = checked
        self._settings.setValue("full_repaint", checked)
        mode = (QGraphicsView.ViewportUpdateMode.FullViewportUpdate if checked
                else QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.view.setViewportUpdateMode(mode)
        self.view.viewport().update()

    def _on_edge_style(self, style: str):
        self._edge_style = style
        self._settings.setValue("edge_style", style)
        self.scene.set_edge_style(style)

    def _on_show_meta_toggled(self, checked: bool):
        self._show_meta = checked
        self._settings.setValue("show_meta", checked)
        self._refresh_side_channels()

    def _on_show_control_toggled(self, checked: bool):
        self._show_control = checked
        self._settings.setValue("show_control", checked)
        self._refresh_side_channels()

    def _refresh_side_channels(self):
        """Redraw the text under the nodes. No re-run needed: the values
        live on the nodes from the last execution."""
        self.scene.show_meta = self._show_meta
        self.scene.show_control = self._show_control
        self.scene.update_side_channels()

    def _update_playback_buttons(self):
        """Play doubles as pause; Stop only means anything while a
        sequence generator is still open."""
        self._btn_play.setIcon(self._icon_pause if self._playback_active
                               else self._icon_play)
        self._btn_stop.setEnabled(self._playback_in_sequence)
        self._btn_step.setEnabled(not self._playback_active)

    def _playback_ready(self) -> bool:
        if self._is_running:
            return False
        if not self.pipeline.nodes:
            QMessageBox.information(self, "Empty Pipeline",
                                    "Add at least one node first.")
            return False
        if not any(n.step.KIND == "source"
                   for n in self.pipeline.nodes.values()):
            QMessageBox.information(self, "No Source",
                                    "Add a source node first.")
            return False
        return True

    def _playback_begin(self) -> bool:
        """Enter sequence mode if not already in it."""
        if self._playback_in_sequence:
            return True
        total = self._refresh_frame_selector()
        self._playback_total = max(1, total)
        self._playback_index = 0
        # Opening the generator does nothing until the first next();
        # resources open on that first pull.
        self._playback_iter = self.pipeline.iter_sequence()
        self._playback_in_sequence = True
        self._update_playback_buttons()
        return True

    def toggle_playback(self):
        if self._playback_active:
            self._playback_active = False
            self._playback_timer.stop()
            self.statusBar().showMessage(
                f"Paused at frame {self._playback_index}/"
                f"{self._playback_total - 1}", 5000)
            self._update_playback_buttons()
            return
        if not self._playback_ready() or not self._playback_begin():
            return
        self._playback_active = True
        self._update_playback_buttons()
        self._playback_tick()

    def step_frame(self):
        """Advance exactly one frame and stop."""
        if not self._playback_ready() or not self._playback_begin():
            return
        self._playback_active = False
        self._update_playback_buttons()
        self._playback_timer.stop()
        self._playback_run_current()

    def stop_sequence(self):
        """Leave sequence mode. The generator must be closed even if
        playback was only paused, or writers and captures stay open."""
        self._playback_active = False
        self._playback_timer.stop()
        if self._playback_in_sequence:
            self._playback_in_sequence = False
            it, self._playback_iter = self._playback_iter, None
            try:
                # Closing the generator runs its finally: block, which is
                # what releases captures and flushes writer threads.
                if it is not None:
                    it.close()
            except Exception as exc:
                QMessageBox.critical(self, "Sequence Error", str(exc))
        self._playback_index = 0
        self._update_playback_buttons()
        self.statusBar().showMessage("Sequence stopped", 3000)

    def _playback_tick(self):
        if not self._playback_active:
            return
        self._playback_run_current()

    def _playback_run_current(self):
        if self._is_running:
            # A run is still in flight; retry shortly rather than
            # queueing two runs onto the same stateful pipeline.
            self._playback_timer.start(20)
            return
        idx = min(self._playback_index, self._playback_total - 1)
        self._sync_frame_widgets(idx)
        self.statusBar().showMessage(
            f"Frame {idx}/{self._playback_total - 1}"
            + ("  (playing)" if self._playback_active else "  (stepped)"))
        self._start_worker(idx, self._playback_total)

    def _sync_frame_widgets(self, index: int):
        """Move the slider/spinbox to `index` WITHOUT triggering the
        live-update re-run their valueChanged normally fires."""
        for w in (self._frame_slider, self._frame_spin):
            blocked = w.blockSignals(True)
            w.setValue(index)
            w.blockSignals(blocked)

    def _playback_after_frame(self):
        """Called once a playback frame has finished rendering."""
        # Playback drives run() directly rather than run_sequence(), so
        # it has to emit the progress channel itself — same format, so a
        # played run and a batch produce identical stdout.
        run_log.log("progress",
                    f"[{self._playback_index}/{self._playback_total - 1}] "
                    f"frame done ({self._playback_index + 1}/"
                    f"{self._playback_total})")
        self._playback_index += 1
        self._update_playback_buttons()
        if self._playback_active:
            self._playback_timer.start(0)   # next frame, GUI stays live

    def _pipeline_changed(self):
        """Call after ANY change to the graph or to a step's parameters.

        total_frames() depends on both — adding a video source changes
        it, and so does typing a path into an existing one — so the frame
        widgets have to be re-evaluated on every mutation, not only when
        something is run. Previously they were refreshed inside
        run_pipeline() alone, which is why a freshly loaded sequence
        pipeline showed no transport controls until you pressed Run."""
        self._refresh_frame_selector()

    def _refresh_frame_selector(self) -> int:
        """Recompute total_frames() and show/hide the Frame: spinbox
        accordingly. Returns total_frames for convenience."""
        total = self.pipeline.total_frames()
        is_sequence = total > 1
        for w in (self._frame_label, self._frame_slider, self._frame_spin,
                  self._frame_total, self._btn_play, self._btn_step,
                  self._btn_stop):
            w.setVisible(is_sequence)
        if is_sequence:
            self._frame_spin.setMaximum(total - 1)
            self._frame_slider.setMaximum(total - 1)
            self._frame_total.setText(f"/ {total - 1}")
        return total

    # ------------------------------------------------------------------
    # Process Sequence (background thread + progress/cancel)
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
        progress.setWindowTitle("Process Sequence")
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
        # progress(done, total) matches this handler exactly; frame_done
        # additionally carries the RunResult but the dialog doesn't need it.
        self._seq_worker.progress.connect(self._on_seq_progress)
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

    def _on_seq_progress(self, done: int, total: int):
        self._seq_progress.setValue(done)
        self._seq_progress.setLabelText(f"Processing frame {done} / {total}…")

    def _on_seq_finished(self, batch):
        """batch is the RunResult for the whole sequence."""
        total = self._seq_total
        processed = batch.frames_processed if batch else 0
        self._seq_progress.setValue(total)
        self._seq_thread.quit()

        if batch is not None and batch.images:
            self._last_results = batch.images
            self.scene.update_previews(batch.images, batch.metrics)
            self.scene.update_side_channels()
            for edge in self.scene.edge_items:
                edge.update_path()
            self.scene.update()
            for nid, win in self._preview_windows.items():
                val = batch.images.get(nid)
                if isinstance(val, np.ndarray):
                    win.show_image(val)

        note = (f"Sequence processing complete: {processed}/{total} frames."
                if not batch.cancelled else
                f"Sequence processing cancelled after {processed}/{total} "
                f"frames.")
        self._print_timing(f"batch, {processed} frame(s)", use_mean=True)

        tooltip_parts = []
        warnings = batch.warnings if batch else []
        if warnings:
            note += f"  ⚠ {len(warnings)} warning(s)."
            tooltip_parts.append("Warnings:\n" + "\n".join(warnings))
        self.statusBar().setToolTip("\n\n".join(tooltip_parts))
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

    def _on_run_finished(self, result):
        try:
            self._present_result(result)
        finally:
            # Advancing must not depend on the display path succeeding.
            if self._playback_in_sequence:
                self._playback_after_frame()

    def _present_result(self, result):
        self._last_results = result.images
        self.scene.update_previews(result.images, result.metrics)
        self.scene.update_side_channels()
        warnings = result.warnings

        # Refresh any fullscreen preview windows that are currently open,
        # so live mode / playback keep them up to date. Guarded: a window
        # that fails to draw must not abort this handler, because the
        # playback advance lives at the end of it — a display problem
        # would otherwise silently halt the whole sequence.
        for nid, win in list(self._preview_windows.items()):
            val = result.images.get(nid)
            if isinstance(val, np.ndarray):
                try:
                    win.show_image(val)
                except Exception as exc:
                    self.statusBar().showMessage(
                        f"Preview window '{nid[:8]}' failed to draw: {exc}",
                        5000)

        if warnings:
            # Non-modal on purpose: a half-wired scratch node shouldn't
            # produce a dialog on every live-update tick. First reason in
            # the status bar; full list in its tooltip.
            head = warnings[0]
            more = f"  (+{len(warnings) - 1} more)" if len(warnings) > 1 else ""
            self.statusBar().showMessage(f"⚠ {head}{more}", 8000)
            self.statusBar().setToolTip("\n".join(warnings))
        elif self._playback_in_sequence:
            # During playback the frame position IS the useful message.
            # A generic "executed successfully" here would overwrite it
            # once per frame, leaving no progress feedback at all.
            self.statusBar().showMessage(
                f"Frame {self._playback_index}/{self._playback_total - 1}"
                + ("  (playing)" if self._playback_active else "  (stepped)"))
            self.statusBar().setToolTip("")
        else:
            self.statusBar().showMessage("Preview complete", 3000)
            self.statusBar().setToolTip("")
        if not self._playback_in_sequence:
            self._print_timing("single run", use_mean=False)

    def _print_timing(self, context: str, use_mean: bool):
        run_log.print_timing(self.pipeline.timing_summary(), context,
                             per_frame=use_mean)

    def _on_sequence_exhausted(self):
        """The playback generator ran out — the sequence is complete."""
        self._is_running = False
        self._act_run.setEnabled(True)
        self._act_process_sequence.setEnabled(True)
        was_playing = self._playback_active
        self.stop_sequence()
        if was_playing:
            self.statusBar().showMessage(
                f"Sequence finished: {self._playback_total} frames", 6000)

    def _on_run_failed(self, error: str):
        if self._playback_in_sequence:
            # Stop first: leaving the generator open would keep writer
            # threads and captures alive behind the error dialog.
            self._playback_active = False
            self._playback_timer.stop()
            self.stop_sequence()
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
        self._remember_recent(path)
        self.statusBar().showMessage(f"Pipeline saved: {path}", 3000)

    MAX_RECENT = 8

    def _remember_recent(self, path: str):
        import os
        path = os.path.abspath(path)
        recent = self._settings.value("recent_files", []) or []
        if isinstance(recent, str):        # QSettings collapses 1-elem lists
            recent = [recent]
        recent = [path] + [p for p in recent if p != path]
        self._settings.setValue("recent_files", recent[:self.MAX_RECENT])
        self._settings.setValue("last_pipeline", path)

    def _rebuild_recent_menu(self):
        import os
        self._recent_menu.clear()
        recent = self._settings.value("recent_files", []) or []
        if isinstance(recent, str):
            recent = [recent]
        recent = [p for p in recent if os.path.exists(p)]
        if not recent:
            act = self._recent_menu.addAction("(empty)")
            act.setEnabled(False)
            return
        for p in recent:
            act = self._recent_menu.addAction(os.path.basename(p))
            act.setToolTip(p)
            act.triggered.connect(
                lambda checked=False, path=p: self._load_pipeline_path(path))

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
        self._load_pipeline_path(path)

    def _load_pipeline_path(self, path: str, silent: bool = False):
        """Load from a known path (Open dialog, Recent menu, startup
        reopen). silent=True suppresses the error dialog — a vanished
        last-session file shouldn't greet you with an error box."""
        try:
            new_pipeline = Pipeline.load(path)
        except Exception as exc:
            if not silent:
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
        for f, t, p, kind in self.pipeline.edges:
            self.scene.add_edge_item(f, t, p, kind=kind)

        self._pipeline_path = path
        self._update_window_title()
        self._remember_recent(path)
        self._pipeline_changed()
        self.statusBar().showMessage(f"Pipeline loaded: {path}", 3000)


# ---------------------------------------------------------------------------

def main():
    try:
        argv = run_log.take_from_argv(sys.argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    # Laptop panels are commonly run at a fractional scale (125%, 150%).
    # Qt's default rounds that to a whole number, so logical and physical
    # pixels stop lining up and repainted regions can miss the edge of
    # what they were meant to cover. PassThrough keeps the true factor.
    # Must be set BEFORE the QApplication is constructed.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    # Qt substitutes silently when a family is missing, so say which one
    # was actually resolved rather than leaving it to be noticed later.
    print(theme.font_report(), flush=True)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
