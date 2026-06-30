"""
Erzeugt automatisch ein Formular für die Parameter eines ProcessingStep,
basierend auf dessen PARAMS-Liste (ParamSpec). Keine manuelle UI-Arbeit
nötig, wenn ein neuer Algorithmus hinzugefügt wird.
"""
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QLineEdit, QVBoxLayout, QLabel,
)


class ParamDialog(QDialog):
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
            form.addRow(spec.label, widget)

        if not step.PARAMS:
            form.addRow(QLabel("Dieser Step hat keine Parameter."))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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
            else:
                result[spec.name] = widget.text()
        return result
