# ImgPipe – Bildverarbeitungs-Pipeline-Editor

## Installation
```bash
pip install PySide6 numpy pillow scipy
```
(`pillow` und `scipy` sind optional, aber empfohlen – ohne sie greifen
einfache Fallback-Implementierungen.)

## Start
```bash
python main.py
```

## Bedienung
- **Links**: Palette aller registrierten Steps, gruppiert nach Kategorie.
  Doppelklick fügt eine Box im Editor hinzu.
- **Mitte**: Node-Editor. Boxen lassen sich verschieben. Von einem grünen
  Output-Port (rechts an der Box) zu einem blauen Input-Port (links an
  einer anderen Box) ziehen, um eine Verbindung zu erstellen.
  Doppelklick auf eine Box öffnet den Parameter-Dialog.
  Box auswählen + "Box löschen" in der Toolbar entfernt sie.
- **Rechts**: Bildvorschau (Original/Ergebnis) sowie Liste eines geladenen
  Bild-Stacks.
- **Toolbar**: Bild laden, Ordner als Bild-Stack laden, Pipeline auf dem
  aktuellen Bild bzw. auf dem ganzen Stack ausführen, Pipeline als JSON
  speichern/laden.

Das "Ende" der Pipeline wird automatisch als die Box ohne ausgehende
Verbindung erkannt – deren Ergebnis wird angezeigt bzw. gespeichert.

## Neue Algorithmen hinzufügen
Einfach eine neue Datei in `steps/` anlegen (oder eine bestehende
erweitern), z.B. `steps/my_steps.py`:

```python
from base_step import ProcessingStep, ParamSpec, register_step
import numpy as np

@register_step
class MyAlgorithm(ProcessingStep):
    NAME = "Mein Algorithmus"
    CATEGORY = "Eigene"
    PARAMS = [
        ParamSpec("strength", "Stärke", "float", default=1.0,
                   min_value=0.0, max_value=10.0, step=0.1),
        ParamSpec("mode", "Modus", "choice", default="A",
                   choices=["A", "B", "C"]),
    ]

    def process(self, image: np.ndarray, strength=1.0, mode="A") -> np.ndarray:
        # ... eigene Logik mit OpenCV, scikit-image, numpy etc.
        return image
```

Die Datei wird beim Start automatisch gefunden (`steps/__init__.py`
importiert alle Module im Ordner) und der Algorithmus erscheint in der
Palette – kein weiterer Eintrag irgendwo nötig.

`ParamSpec.kind` unterstützt: `"int"`, `"float"`, `"bool"`, `"choice"`,
`"str"`.

## Projektstruktur
```
imgpipe/
  base_step.py       Basisklasse ProcessingStep + ParamSpec + Registry
  pipeline.py         Graph-Datenmodell, Ausführung, JSON-Speichern/Laden
  node_graphics.py     Node-Editor-Grafik (Boxen, Ports, Verbindungen)
  param_dialog.py      Automatisch generierter Parameter-Dialog
  main.py               Hauptfenster, Bild-/Stack-I/O, Toolbar
  steps/
    __init__.py        Auto-Import aller Step-Module
    basic_steps.py      Beispiel-Algorithmen (durch echte Liste ersetzen)
```

## Bekannte Einschränkungen / nächste Schritte
- Aktuell kann jede Box nur **einen** Input und beliebig viele Outputs
  haben (lineare/baumartige Pipelines). Für Multi-Input-Knoten (z.B.
  Bild-Überlagerung zweier Zweige) müsste `pipeline.py` (`run()`) und
  `NodeItem` um mehrere Input-Ports erweitert werden.
- Es gibt noch keinen Undo/Redo-Mechanismus.
- Zwischenergebnisse einzelner Boxen werden aktuell nicht einzeln
  visualisiert, nur das Endergebnis – kann bei Bedarf ergänzt werden
  (z.B. Klick auf eine Box zeigt deren Zwischenresultat).
