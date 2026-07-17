"""
Erzeugt automatisch ein Formular für die Parameter eines ProcessingStep,
basierend auf dessen PARAMS-Liste (ParamSpec). Keine manuelle UI-Arbeit
nötig, wenn ein neuer Algorithmus hinzugefügt wird.
"""
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QLineEdit, QVBoxLayout, QLabel,
    QHBoxLayout, QPushButton, QFileDialog, QWidget,
)
from PySide6.QtCore import Signal


class ParamDialog(QDialog):
    # Emitted whenever any parameter widget's value changes, so callers can
    # apply a live preview without waiting for OK to be pressed.
    valuesChanged = Signal()

    def __init__(self, step, parent=None):
        super().__init__(parent)
        self.step = step
        self.setWindowTitle(f"Parameter: {step.NAME}")
        self._widgets: dict[str, object] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{step.NAME}</b>"))

        form = QFormLayout()
        layout.addLayout(form)

        current = step.get_param_values()
        for spec in step.PARAMS:
            value = current.get(spec.name, spec.default)
            widget = self._make_widget(spec, value)
            self._widgets[spec.name] = widget
            self._wire_change_signal(spec.kind, widget)
            form.addRow(spec.label, widget)

        if not step.PARAMS:
            form.addRow(QLabel("Dieser Step hat keine Parameter."))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _wire_change_signal(self, kind, widget):
        """Connect the widget's native change signal to valuesChanged so
        the caller can offer a live preview while the user is still
        editing, not just after OK is pressed."""
        if kind == "int" or kind == "float":
            widget.valueChanged.connect(lambda *_: self.valuesChanged.emit())
        elif kind == "bool":
            widget.stateChanged.connect(lambda *_: self.valuesChanged.emit())
        elif kind == "choice":
            widget.currentIndexChanged.connect(lambda *_: self.valuesChanged.emit())
        elif kind in ("file", "directory"):
            widget.pathChanged.connect(lambda *_: self.valuesChanged.emit())
        else:
            widget.textChanged.connect(lambda *_: self.valuesChanged.emit())

    def _make_widget(self, spec, value):
        if spec.kind == "int":
            w = QSpinBox()
            w.setMinimum(int(spec.min_value) if spec.min_value is not None else -1_000_000)
            w.setMaximum(int(spec.max_value) if spec.max_value is not None else 1_000_000)
            w.setValue(int(value))
            return w
        if spec.kind == "float":
            w = QDoubleSpinBox()
            w.setMinimum(spec.min_value if spec.min_value is not None else -1e6)
            w.setMaximum(spec.max_value if spec.max_value is not None else 1e6)
            w.setSingleStep(spec.step or 0.1)
            w.setDecimals(3)
            w.setValue(float(value))
            return w
        if spec.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            return w
        if spec.kind == "choice":
            w = QComboBox()
            choices = spec.choices or []
            w.addItems([str(c) for c in choices])
            if value in choices:
                w.setCurrentIndex(choices.index(value))
            return w
        if spec.kind == "file":
            return _FileBrowseWidget(
                str(value),
                file_filter=getattr(spec, "types", "") or "All files (*)")
        if spec.kind == "directory":
            return _DirBrowseWidget(str(value))
        # default: str
        w = QLineEdit(str(value))
        return w

    def get_values(self) -> dict:
        result = {}
        for spec in self.step.PARAMS:
            widget = self._widgets[spec.name]
            if spec.kind == "int":
                result[spec.name] = widget.value()
            elif spec.kind == "float":
                result[spec.name] = widget.value()
            elif spec.kind == "bool":
                result[spec.name] = widget.isChecked()
            elif spec.kind == "choice":
                result[spec.name] = widget.currentText()
            elif spec.kind == "file":
                result[spec.name] = widget.get_path()
            elif spec.kind == "directory":
                result[spec.name] = widget.get_path()
            else:
                result[spec.name] = widget.text()
        return result


class _FileBrowseWidget(QWidget):
    """Inline file path + Browse button for kind='file' params. The
    dialog's file filter comes from the ParamSpec's `types` string
    (ready-made Qt syntax, e.g. "NUC files (*.nuc);;All files (*)")."""

    pathChanged = Signal(str)

    def __init__(self, path: str = "", parent=None,
                 file_filter: str = "All files (*)"):
        super().__init__(parent)
        self._file_filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit(path)
        self._edit.setPlaceholderText("No file selected…")
        btn = QPushButton("Browse…")
        btn.setFixedWidth(72)
        btn.clicked.connect(self._browse)
        layout.addWidget(self._edit)
        layout.addWidget(btn)
        self._edit.textChanged.connect(self.pathChanged)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", self._edit.text(), self._file_filter
        )
        if path:
            self._edit.setText(path)

    def get_path(self) -> str:
        return self._edit.text()


class _DirBrowseWidget(QWidget):
    """Inline directory path + Browse button for kind='directory' params."""

    pathChanged = Signal(str)

    def __init__(self, path: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit(path)
        self._edit.setPlaceholderText("No directory selected…")
        btn = QPushButton("Browse…")
        btn.setFixedWidth(72)
        btn.clicked.connect(self._browse)
        layout.addWidget(self._edit)
        layout.addWidget(btn)
        self._edit.textChanged.connect(self.pathChanged)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory", self._edit.text()
        )
        if path:
            self._edit.setText(path)

    def get_path(self) -> str:
        return self._edit.text()
