"""
Parameter optimization dialog.

Pick parameters (any numeric ones with a min/max), pick a metric node as
the objective, choose minimize/maximize, and search. The heavy lifting
is in optimizer.py -- this is only the UI and the worker thread.

Evaluations run OFF the UI thread (each one is a full pipeline run, or a
full sequence run in sequence mode). Results are shown before anything
is applied: the optimizer always restores the original parameter values,
so the user explicitly chooses Apply or Discard.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTreeWidget,
    QTreeWidgetItem, QComboBox, QSpinBox, QPushButton, QLabel,
    QProgressBar, QDialogButtonBox, QGroupBox, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, QObject, Signal

from src.GUI.pipeline_editor.optimizer import (
    ParameterOptimizer, optimizable_params, metric_nodes,
)


class _OptimWorker(QObject):
    progress = Signal(int, int, float, float)   # done, total, current, best
    finished = Signal(object)                   # OptimResult
    failed   = Signal(str)

    def __init__(self, optimizer: ParameterOptimizer):
        super().__init__()
        self.optimizer = optimizer
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self.optimizer.run(
                on_progress=lambda n, t, cur, best:
                    self.progress.emit(n, t, cur, best),
                should_cancel=lambda: self._cancel)
            self.finished.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))


class OptimizeDialog(QDialog):
    def __init__(self, pipeline, frame_index: int = 0,
                 total_frames: int = 1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimize Parameters")
        self.resize(560, 560)
        self.pipeline = pipeline
        self.frame_index = frame_index
        self.total_frames = total_frames
        self._result = None
        self._thread = None
        self._worker = None

        layout = QVBoxLayout(self)

        # --- parameters to optimize -----------------------------------
        self._targets = optimizable_params(pipeline)
        box = QGroupBox("Parameters to optimize")
        box_layout = QVBoxLayout(box)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Parameter", "Range"])
        self.tree.setColumnWidth(0, 320)
        by_node = {}
        for t in self._targets:
            node_name = t.label.split(" — ")[0]
            parent_item = by_node.get(node_name)
            if parent_item is None:
                parent_item = QTreeWidgetItem([node_name, ""])
                parent_item.setFlags(parent_item.flags() & ~Qt.ItemIsUserCheckable)
                self.tree.addTopLevelItem(parent_item)
                by_node[node_name] = parent_item
            leaf = QTreeWidgetItem([t.label.split(" — ", 1)[-1],
                                    f"{t.lo:g} … {t.hi:g}"])
            leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
            leaf.setCheckState(0, Qt.Unchecked)
            leaf.setData(0, Qt.UserRole, t)
            parent_item.addChild(leaf)
        self.tree.expandAll()
        box_layout.addWidget(self.tree)
        if not self._targets:
            box_layout.addWidget(QLabel(
                "No optimizable parameters — a parameter needs a numeric "
                "type and a min/max range."))
        layout.addWidget(box)

        # --- objective / settings -------------------------------------
        form = QFormLayout()
        self.metric_combo = QComboBox()
        for nid, name in metric_nodes(pipeline):
            self.metric_combo.addItem(name, nid)
        form.addRow("Metric:", self.metric_combo)

        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["minimize", "maximize"])
        form.addRow("Direction:", self.dir_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(f"Current frame ({frame_index})", "frame")
        if total_frames > 1:
            self.mode_combo.addItem(
                f"Full sequence ({total_frames} frames)", "sequence")
        form.addRow("Input:", self.mode_combo)

        self.agg_combo = QComboBox()
        self.agg_combo.addItems(["mean", "min", "max", "last"])
        form.addRow("Sequence aggregate:", self.agg_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["random+pattern", "random", "pattern"])
        form.addRow("Method:", self.method_combo)

        self.evals_spin = QSpinBox()
        self.evals_spin.setRange(4, 100000)
        self.evals_spin.setValue(60)
        form.addRow("Max evaluations:", self.evals_spin)
        layout.addLayout(form)

        # --- progress / result ----------------------------------------
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # --- buttons ---------------------------------------------------
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self.start_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self.buttons = QDialogButtonBox()
        self.apply_btn = self.buttons.addButton("Apply Result",
                                                QDialogButtonBox.AcceptRole)
        self.buttons.addButton("Close", QDialogButtonBox.RejectRole)
        self.apply_btn.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.mode_combo.currentIndexChanged.connect(self._sync_agg)
        self._sync_agg()

    def _sync_agg(self):
        self.agg_combo.setEnabled(
            self.mode_combo.currentData() == "sequence")

    def _checked_targets(self) -> list:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                leaf = parent.child(j)
                if leaf.checkState(0) == Qt.Checked:
                    out.append(leaf.data(0, Qt.UserRole))
        return out

    # --- run -----------------------------------------------------------
    def _start(self):
        targets = self._checked_targets()
        if not targets:
            QMessageBox.information(self, "Nothing selected",
                                    "Tick at least one parameter.")
            return
        if self.metric_combo.count() == 0:
            QMessageBox.information(
                self, "No metric",
                "Add a metric node to the pipeline — it is the objective "
                "the search optimizes.")
            return

        mode = self.mode_combo.currentData()
        n_evals = self.evals_spin.value()
        if mode == "sequence":
            runs = n_evals * self.total_frames
            if QMessageBox.question(
                self, "Confirm",
                f"Sequence mode runs the whole pipeline "
                f"{n_evals} × {self.total_frames} = {runs} times. Continue?"
            ) != QMessageBox.Yes:
                return

        optimizer = ParameterOptimizer(
            self.pipeline, targets,
            metric_node_id=self.metric_combo.currentData(),
            direction=self.dir_combo.currentText(),
            mode=mode,
            frame_index=self.frame_index,
            total_frames=self.total_frames,
            aggregate=self.agg_combo.currentText(),
            method=self.method_combo.currentText(),
            max_evals=n_evals)

        self.progress.setRange(0, n_evals)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.apply_btn.setEnabled(False)
        self.status.setText("Running…")

        self._worker = _OptimWorker(optimizer)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # Bound methods of this QObject, so Qt queues them onto the GUI
        # thread rather than running them in the worker thread.
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self.status.setText("Cancelling…")

    def _on_progress(self, done, total, current, best):
        self.progress.setValue(done)
        self.status.setText(
            f"Evaluation {done}/{total} — current {current:.5g}, "
            f"best {best:.5g}")

    def _on_finished(self, result):
        self._thread.quit()
        self._result = result
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if not result.best_values:
            self.status.setText(result.message)
            return
        lines = [f"{result.evaluations} evaluations — "
                 f"metric {result.start_metric:.5g} → {result.best_metric:.5g}"
                 + ("  (cancelled)" if result.cancelled else "")]
        for (nid, pname), v in result.best_values.items():
            node = self.pipeline.nodes.get(nid)
            name = node.display_name if node else nid
            old = (node.step.get_param_values().get(pname) if node else None)
            lines.append(f"  {name} — {pname}: {old} → {v}")
        self.status.setText("\n".join(lines))
        self.apply_btn.setEnabled(True)

    def _on_failed(self, err):
        self._thread.quit()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText(f"Failed: {err}")

    # --- result ---------------------------------------------------------
    def accept(self):
        """Apply the best values found, then close."""
        if self._result and self._result.best_values:
            for (nid, pname), v in self._result.best_values.items():
                node = self.pipeline.nodes.get(nid)
                if node is not None:
                    node.step.set_param_values({pname: v})
        super().accept()

    def changed_node_ids(self) -> set:
        if not self._result:
            return set()
        return {nid for (nid, _) in self._result.best_values}
