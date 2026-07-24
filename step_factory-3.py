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
    ProcessingStep, ParamSpec, register_step,
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


    specs = params if params is not None else params_from_signature(
        func, n_inputs)
    param_names = [s.name for s in specs]

    def process(self, *images):
        return func(*images, **{n: self.values[n] for n in param_names})

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


# ---------------------------------------------------------------------------
# Stateful classes: built once, called per frame
# ---------------------------------------------------------------------------

def _params_from_callable(func, skip: int) -> list:
    """ParamSpecs from a callable's keyword defaults, skipping the first
    `skip` positional arguments without defaults (self / image args)."""
    return params_from_signature(func, n_inputs=skip)


def _arg_names(func, skip: int) -> set:
    """Names of the keyword arguments a callable accepts, past the first
    `skip` positional (self / image) arguments."""
    names = set()
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return names
    positional = 0
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if positional < skip and p.default is p.empty:
            positional += 1
            continue
        names.add(pname)
    return names


def step_from_class(cls, name: str = None, category: str = "General",
                    params: list = None, n_inputs: int = 1,
                    class_name: str = None, register: bool = True,
                    call_method: str = "__call__", **flags):
    """
    Create (and by default register) a step from a STATEFUL class that is
    constructed once and then called per frame:

        obj = MyDenoiser(history=8, strength=0.5)   # once
        out = obj(frame)                            # every frame

    The generated class subclasses _TemporalStep, so it inherits the
    frame-index bookkeeping that keeps such objects consistent when the
    executor does NOT walk frames in order (live-mode re-runs of the same
    frame, frame-slider jumps, batch restarts). See temporal_steps.py.

    Parameters are collected from BOTH signatures and routed accordingly:
      * __init__ arguments  -> passed at construction. These become
        RESET_PARAMS: changing one rebuilds the object (its state was
        built around the old value), which is the honest behavior.
      * extra call_method arguments (past the image) -> passed per frame,
        so changing them keeps the accumulated history.

    Pass an explicit `params=[ParamSpec(...)]` for ranges/choices/file
    pickers; routing still follows the signatures, so the names must
    match the real arguments.

    call_method  name of the per-frame method ("__call__", "process", ...)
    """
    display = name or cls.__name__
    key = class_name or cls.__name__

    call_fn = getattr(cls, call_method, None)
    if call_fn is None:
        raise ValueError(
            f"step_from_class: {cls.__name__} has no '{call_method}' method.")

    # inspect.signature(cls) gives __init__ WITHOUT self; the unbound
    # call method still has self, hence the +1.
    init_names = _arg_names(cls, skip=0)
    call_names = _arg_names(call_fn, skip=1 + n_inputs)
    call_names -= init_names          # __init__ wins on name clashes

    if params is not None:
        specs = params
    else:
        specs = _params_from_callable(cls, skip=0)
        specs += [s for s in _params_from_callable(call_fn,
                                                   skip=1 + n_inputs)
                  if s.name in call_names]

    # Constructor arguments with no default can't be supplied from a
    # ParamSpec-less signature -- fail loudly rather than at first run.
    try:
        init_sig = inspect.signature(cls)
        required = [p.name for p in init_sig.parameters.values()
                    if p.default is p.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                    and p.name not in {s.name for s in specs}]
    except (TypeError, ValueError):
        required = []
    if required:
        raise ValueError(
            f"step_from_class: {cls.__name__}.__init__ requires "
            f"{required} with no default -- declare them via "
            f"params=[ParamSpec(...)] so the step can supply values.")

    init_param_names = {s.name for s in specs} & init_names
    call_param_names = {s.name for s in specs} & call_names

    def reset(self):
        kwargs = {k: self.values[k] for k in self._init_param_names}
        self._proc = cls(**kwargs)

    def advance(self, image):
        kwargs = {k: self.values[k] for k in self._call_param_names}
        return getattr(self._proc, call_method)(image, **kwargs)

    namespace = {
        "NAME": display,
        "CATEGORY": category,
        "PARAMS": specs,
        "N_INPUTS": n_inputs,
        # constructor args are baked into the object -> rebuild on change
        "_init_param_names": frozenset(init_param_names),
        "_call_param_names": frozenset(call_param_names),
        "reset": reset,
        "advance": advance,
        "__doc__": cls.__doc__,
        "__module__": getattr(cls, "__module__", __name__),
        "_wrapped_class": cls,
    }
    namespace.update(flags)

    # Imported here so this module stays usable even if temporal_steps
    # hasn't been loaded yet by the steps auto-discovery.
    from src.GUI.pipeline_editor.base_step import StatefulStep

    generated = type(key, (StatefulStep,), namespace)
    return register_step(generated) if register else generated


def as_step_class(name: str = None, category: str = "General",
                  params: list = None, n_inputs: int = 1,
                  class_name: str = None, call_method: str = "__call__",
                  **flags):
    """Decorator form of step_from_class. Returns the class unchanged."""
    def decorator(cls):
        step_from_class(cls, name=name, category=category, params=params,
                        n_inputs=n_inputs, class_name=class_name,
                        call_method=call_method, **flags)
        return cls
    return decorator
