"""
Build ProcessingStep classes from plain functions.

Why a factory is needed at all: register_step() keys STEP_REGISTRY on
cls.__name__, and that same name is the persistence key (Pipeline.to_dict
stores class_name, from_dict looks it up). Creating classes the obvious
way inside a factory --

    def make_step(func):
        class GeneratedStep(ProcessingStep):   # <-- always this name
            ...
        return register_step(GeneratedStep)

-- gives EVERY generated class __name__ == "GeneratedStep", so they all
overwrite each other in the registry and saved pipelines can't be
restored. type() with an explicit unique name fixes both.

Usage — decorator form:

    @as_step(category="Filter/Denoise")
    def denoise(image, strength=0.5, radius=3):
        return ...

Usage — call form (e.g. wrapping a function you don't own):

    step_from_function(my_denoise, name="Denoise",
                       category="Filter/Denoise")

Parameters are inferred from the signature's keyword defaults (bool /
int / float / str). Pass an explicit `params=[ParamSpec(...)]` when you
need ranges, choices, or file pickers -- inference can't guess those:

    step_from_function(
        nuc, name="NUC", category="Preprocessing",
        params=[ParamSpec("gain", "Gain", "float", default=1.0,
                          min_value=0.0, max_value=4.0, step=0.05),
                ParamSpec("map_path", "NUC Map", "file", default="",
                          types="NUC files (*.nuc);;All files (*)")])
"""
import inspect

from src.GUI.pipeline_editor.base_step import (
    ProcessingStep, ParamSpec, register_step, STEP_REGISTRY,
)


_KIND_BY_TYPE = {
    bool:  "bool",     # bool before int: bool IS a subclass of int
    int:   "int",
    float: "float",
    str:   "str",
}


def params_from_signature(func, n_inputs: int = 1) -> list:
    """Derive ParamSpecs from a function's keyword defaults, skipping
    the first `n_inputs` positional (image) arguments. Arguments without
    a default, *args/**kwargs, and unsupported types are skipped."""
    specs = []
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return specs
    positional = 0
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if positional < n_inputs and p.default is p.empty:
            positional += 1          # this is an image argument
            continue
        if p.default is p.empty:
            continue                 # required, non-image: can't infer a default
        kind = _KIND_BY_TYPE.get(type(p.default))
        if kind is None:
            continue                 # None/list/... -> declare explicitly
        label = pname.replace("_", " ").title()
        specs.append(ParamSpec(pname, label, kind, default=p.default))
    return specs


def step_from_function(func, name: str = None, category: str = "General",
                       params: list = None, n_inputs: int = 1,
                       class_name: str = None, register: bool = True,
                       **flags):
    """
    Create (and by default register) a ProcessingStep subclass that calls
    `func(image[, image_b], **params)`.

    name        display name in the palette / on the node
    category    palette category, '/' for hierarchy
    params      explicit ParamSpec list; inferred from the signature if None
    n_inputs    number of image arguments (2 for metric-style functions)
    class_name  registry/persistence key. Defaults to the function's
                __name__ in CamelCase. MUST be stable across runs -- it is
                what saved pipelines store.
    flags       any extra class attributes: IS_SOURCE, IS_SINK, IS_METRIC,
                IS_SEQUENCE_AWARE, ...
    """
    display = name or func.__name__.replace("_", " ").title()
    key = class_name or "".join(
        part.title() for part in func.__name__.split("_")) or "GeneratedStep"

    if register and key in STEP_REGISTRY:
        raise ValueError(
            f"step_from_function: '{key}' is already in STEP_REGISTRY "
            f"(registered by {STEP_REGISTRY[key]!r}). Registry keys must be "
            f"unique and stable -- pass class_name= to disambiguate.")

    specs = params if params is not None else params_from_signature(
        func, n_inputs)

    def process(self, *images, **kwargs):
        return func(*images, **kwargs)

    namespace = {
        "NAME": display,
        "CATEGORY": category,
        "PARAMS": specs,
        "N_INPUTS": n_inputs,
        "process": process,
        "__doc__": func.__doc__,
        # so tracebacks and repr point at the wrapped function
        "__module__": getattr(func, "__module__", __name__),
        "_wrapped_function": staticmethod(func),
    }
    namespace.update(flags)

    cls = type(key, (ProcessingStep,), namespace)
    return register_step(cls) if register else cls


def as_step(name: str = None, category: str = "General", params: list = None,
            n_inputs: int = 1, class_name: str = None, **flags):
    """Decorator form of step_from_function. Returns the function
    unchanged, so it stays directly callable elsewhere."""
    def decorator(func):
        step_from_function(func, name=name, category=category,
                           params=params, n_inputs=n_inputs,
                           class_name=class_name, **flags)
        return func
    return decorator
