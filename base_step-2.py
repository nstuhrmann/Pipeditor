"""
Base class for all image processing steps.
"""
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class ParamSpec:
    name: str
    label: str
    kind: str          # "int"|"float"|"bool"|"choice"|"str"|"file"|"directory"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list] = None


class ProcessingStep:
    """
    Base class for every node in the pipeline.

    Class variables to override:
        NAME        display name shown in the palette and on the node box
        CATEGORY    palette group
        PARAMS      list of ParamSpec — UI is auto-generated from this
        N_INPUTS    number of input ports (default 1)
        IS_SOURCE   True → no input port; process() should ignore the dummy
                    image passed in and load its own data instead
        IS_SINK     True → dark-red header; step writes data but also
                    returns the image unchanged (passthrough)
        IS_METRIC   True → two input ports (A/B), no output port; process()
                    returns a float or str that is displayed on the node

    process() signature:
        N_INPUTS == 1  →  process(self, image: np.ndarray, **params)
        N_INPUTS == 2  →  process(self, image_a: np.ndarray,
                                        image_b: np.ndarray, **params)
    """

    NAME: str = "Unnamed Step"
    CATEGORY: str = "General"
    PARAMS: list[ParamSpec] = []
    N_INPUTS: int = 1
    IS_SOURCE: bool = False
    IS_SINK: bool = False
    IS_METRIC: bool = False

    def __init__(self):
        self.values: dict[str, Any] = {p.name: p.default for p in self.PARAMS}

    def get_param_values(self) -> dict:
        return dict(self.values)

    def set_param_values(self, values: dict):
        for k, v in values.items():
            if k in self.values:
                self.values[k] = v

    def process(self, *images: np.ndarray, **params):
        raise NotImplementedError(
            f"{self.__class__.__name__}.process() must be implemented"
        )

    def run(self, inputs: list) -> "np.ndarray | float | str":
        """Called by the pipeline executor. inputs is a list of numpy arrays."""
        return self.process(*inputs, **self.values)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.values}>"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
STEP_REGISTRY: dict[str, type] = {}


def register_step(cls):
    STEP_REGISTRY[cls.__name__] = cls
    return cls
