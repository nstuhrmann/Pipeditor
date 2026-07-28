"""
Generates 'ae_loop_demo.json' — a ready-to-open pipeline demonstrating
the auto-exposure control loop.

    python build_ae_pipeline.py        # writes ae_loop_demo.json

Then File > Open Pipeline in the editor, or run it headlessly with
run_pipeline_headless.py.

Graph:

    [Camera Simulator] -> [Auto Exposure] -+-> [AE Error]   (error_ev)
                                           +-> [AE Gain]    (gain)
                                           +-> [Image Noise](spatial sigma)

Why the monitors hang off the AE rather than off the camera: they read
the loop's state from the message bus, and the bus is only written once
the AE has run. Wiring them downstream is what puts them later in the
executor's topological order. Reading a bus topic from a node that runs
BEFORE its producer would show the previous frame's value instead.

The three metrics are set to dump CSV, so a "Process Full Sequence" run
leaves three files next to the working directory: the convergence
trace, the gain the AE chose, and the noise that resulted. Plotting
gain against noise is the point of the exercise — pushing gain to hit
the exposure target costs SNR, and this pipeline measures exactly that
trade-off.
"""
import src.GUI.pipeline_editor.camera_sim      # noqa: F401  (registers steps)
import src.GUI.pipeline_editor.noise_metrics   # noqa: F401
import src.GUI.pipeline_editor.sequence_steps  # noqa: F401

from src.GUI.pipeline_editor.base_step import STEP_REGISTRY
from src.GUI.pipeline_editor.pipeline import EDGE_CONTROL, Pipeline

OUTPUT = "ae_loop_demo.json"


def build() -> Pipeline:
    pipeline = Pipeline()

    camera = pipeline.add_node(STEP_REGISTRY["CameraSimulator"](),
                               pos=(-360, 40))
    camera.step.set_param_values({
        "resolution": "640x480",
        "num_frames": 60,
        # +3 EV at 1/3, -1 EV at 2/3 — two step changes for the loop to
        # recover from, which is what makes the trace interesting.
        "illumination": "step",
        # Dim enough that the 33 ms exposure ceiling forces the AE to
        # reach for gain in the darker segments — without that, gain
        # stays pinned at 1x and the noise trade-off never shows.
        "illum_level": 0.1,
        "exposure_ms": 5.0,      # deliberately mis-exposed starting point
        "gain": 1.0,
        "accept_control": True,  # obey the bus; uncheck for manual mode
        "read_noise_e": 4.0,
        "bit_depth": "12",
        "encoding": "sRGB",
        "seed": 0,
    })

    ae = pipeline.add_node(STEP_REGISTRY["AutoExposure"](), pos=(-60, 40))
    ae.step.set_param_values({
        "target": 0.45,
        "metering": "average",
        "damping": 0.6,          # lower = calmer, slower; raise to ring
        "max_step_ev": 1.5,
        "min_exposure_ms": 0.05,
        "max_exposure_ms": 33.0,  # ~1/30 s: a real frame-rate ceiling,
        "max_gain": 16.0,         # so the loop must resort to gain
        "linearize": True,
    })

    # Only ONE metric node is needed now. The camera already emits
    # exposure_ms / gain / illumination as metadata and the AE emits
    # ae_error_ev, so the CSV writer picks all of that up without a
    # monitor node per quantity.
    noise = pipeline.add_node(STEP_REGISTRY["SpatialNoise"](), pos=(240, 40))

    csv_out = pipeline.add_node(STEP_REGISTRY["MetadataCSVWriter"](),
                                pos=(540, 40))
    csv_out.step.set_param_values({"path": "ae_loop.csv"})

    pipeline.add_edge(camera.id, ae.id, 0)
    pipeline.add_edge(ae.id, noise.id, 0)
    # The writer records what its INPUT carries, so it goes last.
    pipeline.add_edge(noise.id, csv_out.id, 0)
    pipeline.add_edge(ae.id, camera.id, 0, kind=EDGE_CONTROL)
    return pipeline


if __name__ == "__main__":
    build().save(OUTPUT)
    print(f"wrote {OUTPUT}")
