"""
Every module in this package is imported here, and importing a step
module is what puts its steps in STEP_REGISTRY. So one

    import src.GUI.pipeline_editor.steps

in an entry point makes every step available, and adding a new module
means adding one line below.

    ***  MERGE, DON'T REPLACE  ***
    If you already have a steps/__init__.py listing your own modules,
    add the imports below to it rather than overwriting — the modules
    named here are only the ones shipped with the framework.

Listed explicitly rather than discovered with pkgutil.walk_packages().
Auto-discovery is tidier to read but invisible to static analysis, and
Nuitka would leave every step module out of the build unless each were
named again in --include-module — which is the same list, in a worse
place. An explicit import also fails at startup with a real traceback
when a module is broken, instead of silently registering fewer steps.
"""

from src.GUI.pipeline_editor.steps import camera_sim        # noqa: F401
from src.GUI.pipeline_editor.steps import noise_metrics     # noqa: F401
from src.GUI.pipeline_editor.steps import sensor_steps      # noqa: F401
from src.GUI.pipeline_editor.steps import sequence_steps    # noqa: F401
from src.GUI.pipeline_editor.steps import temporal_steps    # noqa: F401
from src.GUI.pipeline_editor.steps import visual_steps      # noqa: F401

# --- your own step modules go here as well -------------------------------
# from src.GUI.pipeline_editor.steps import extra_steps      # noqa: F401
