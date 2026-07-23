"""
Auto-exposure demo: builds the camera -> AE -> monitor loop in code and
runs it headlessly, printing convergence per frame.

    python ae_demo.py

Shows the bus closing a control loop: the AE never touches the camera
object, only posts messages the camera consumes on the next frame.
"""
import src.GUI.pipeline_editor.steps  # noqa: F401  (populates STEP_REGISTRY)

from src.GUI.pipeline_editor.base_step import STEP_REGISTRY
from src.GUI.pipeline_editor.camera_sim import TOPIC_AE, TOPIC_STATE
from src.GUI.pipeline_editor.pipeline import Pipeline


def build(frames: int = 45) -> tuple:
    pipeline = Pipeline()

    camera = pipeline.add_node(STEP_REGISTRY["CameraSimulator"]())
    camera.step.set_param_values({
        "resolution": "640x480",
        "num_frames": frames,
        "illumination": "step",     # +3 EV then -1 EV, for the AE to track
        "illum_level": 0.5,
        "exposure_ms": 5.0,         # deliberately wrong starting point
        "gain": 1.0,
        "accept_control": True,     # obey the bus
    })

    ae = pipeline.add_node(STEP_REGISTRY["AutoExposure"]())
    ae.step.set_param_values({
        "target": 0.45,
        "metering": "average",
        "damping": 0.6,
    })

    monitor = pipeline.add_node(STEP_REGISTRY["AEMonitor"]())
    monitor.step.set_param_values({"quantity": "error_ev"})

    pipeline.add_edge(camera.id, ae.id, 0)
    pipeline.add_edge(ae.id, monitor.id, 0)
    return pipeline, camera, monitor


def main():
    pipeline, camera, monitor = build()

    print(f"{'frame':>5}  {'illum':>7}  {'exp ms':>7}  {'gain':>5}  "
          f"{'error EV':>9}  {'sat':>6}")

    def on_frame(index, total, results):
        cam = pipeline.bus.get(TOPIC_STATE) or {}
        ae = pipeline.bus.get(TOPIC_AE) or {}
        print(f"{index:5d}  {cam.get('illumination', 0):7.3f}  "
              f"{cam.get('exposure_ms', 0):7.2f}  {cam.get('gain', 0):5.2f}  "
              f"{ae.get('error_ev', 0):+9.3f}  "
              f"{cam.get('saturated_fraction', 0):6.3f}")

    warnings = []
    pipeline.run_sequence(on_frame_done=on_frame, warnings_out=warnings)
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
