"""
Basisklasse für alle Bildverarbeitungs-Schritte.

Eigene Algorithmen werden einfach durch Ableiten von ProcessingStep
hinzugefügt. Parameter werden deklarativ über PARAMS definiert,
daraus wird automatisch die UI im Node-Editor erzeugt.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np


@dataclass
class ParamSpec:
    """Beschreibt einen einzelnen, editierbaren Parameter eines Steps."""
    name: str                      # interner Key (wird an process() übergeben)
    label: str                     # Anzeigename im UI
    kind: str                      # "int" | "float" | "bool" | "choice" | "str"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list] = None  # bei kind == "choice"


class ProcessingStep:
    """
    Basisklasse für einen Schritt in der Bildverarbeitungs-Pipeline.

    Unterklassen müssen überschreiben:
        - NAME        : Anzeigename der Box im Editor
        - CATEGORY    : Gruppierung in der Palette (z.B. "Filter", "Schwelle")
        - PARAMS      : Liste von ParamSpec-Objekten
        - process()   : die eigentliche Bildverarbeitung

    process() bekommt ein numpy-Array (H,W) oder (H,W,C) und die aktuellen
    Parameterwerte als **kwargs und muss ein numpy-Array zurückgeben.
    """

    NAME: str = "Unnamed Step"
    CATEGORY: str = "Allgemein"
    PARAMS: list[ParamSpec] = []

    def __init__(self):
        # aktuelle Parameterwerte, initial = Defaults
        self.values: dict[str, Any] = {p.name: p.default for p in self.PARAMS}

    def get_param_values(self) -> dict:
        return dict(self.values)

    def set_param_values(self, values: dict):
        for k, v in values.items():
            if k in self.values:
                self.values[k] = v

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__}.process() muss implementiert werden"
        )

    def run(self, image: np.ndarray) -> np.ndarray:
        """Wird von der Pipeline aufgerufen."""
        return self.process(image, **self.values)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.values}>"


# ---------------------------------------------------------------------------
# Registry: alle bekannten ProcessingStep-Unterklassen, gruppiert nach Klasse
# ---------------------------------------------------------------------------
STEP_REGISTRY: dict[str, type] = {}


def register_step(cls):
    """Decorator: registriert einen Step automatisch, damit er im Editor
    in der Palette erscheint und beim Laden von Pipelines per Namen
    wiedergefunden werden kann."""
    STEP_REGISTRY[cls.__name__] = cls
    return cls
