# ImgPipe – Image Processing Pipeline Editor

A node-based pipeline editor for image processing, built with PySide6.
Each node is a self-contained processing step. Connect them to form a
pipeline, configure parameters by double-clicking, then hit **Pipeline → Run**.

---

## Installation

```bash
pip install PySide6 numpy pillow scipy
```

`pillow` and `scipy` are optional but recommended. Without them, simple
fallback implementations are used.

## Start

```bash
python main.py
```

---

## Workflow

1. **Add nodes** — double-click a step in the left palette.
2. **Connect nodes** — drag from a green output port (right side of a node)
   to a blue input port (left side of another node).
3. **Configure parameters** — double-click any node to edit its settings.
4. **Remove a connection** — right-click it → *Remove Connection*,
   or click to select it (turns red) then press **Delete**.
5. **Run** — `Pipeline → Run` (Ctrl+R).  
   Thumbnails appear on the first and last connections automatically.
   Right-click any other connection to toggle its preview.
6. **Save / load pipelines** — `File → Save Pipeline` / `Open Pipeline`
   stores the graph as JSON including all parameter values.

### Node types

| Header colour | Meaning |
|---|---|
| Green | **Source** — no input port, produces its own image (e.g. *Image Source*) |
| Grey  | **Processing** — transforms the image it receives |
| Dark red | **Sink** — writes data somewhere, passes the image through unchanged |

---

## Project structure

```
imgpipe/
  main.py              Main window, menus, pipeline run
  base_step.py         ProcessingStep base class, ParamSpec, registry
  pipeline.py          Graph model: nodes, edges, execution, JSON I/O
  node_graphics.py     QGraphicsScene node editor (boxes, ports, edges, thumbnails)
  param_dialog.py      Auto-generated parameter form from PARAMS list
  steps/
    __init__.py        Auto-imports every .py file in this folder
    basic_steps.py     Example processing steps
    io_steps.py        Image Source and Directory Writer nodes
```

Every `.py` file placed in `steps/` is imported automatically on startup
— no manual registration needed anywhere.

---

## Adding new steps

### 1 — Processing step (grey node)

```python
# steps/my_steps.py
import numpy as np
from base_step import ProcessingStep, ParamSpec, register_step

@register_step
class MyFilter(ProcessingStep):
    NAME     = "My Filter"
    CATEGORY = "My Category"
    PARAMS   = [
        ParamSpec("strength", "Strength", "float",
                  default=1.0, min_value=0.0, max_value=10.0, step=0.1),
        ParamSpec("mode", "Mode", "choice",
                  default="A", choices=["A", "B", "C"]),
        ParamSpec("invert", "Invert", "bool", default=False),
    ]

    def process(self, image: np.ndarray,
                strength=1.0, mode="A", invert=False) -> np.ndarray:
        # image is a numpy array, shape (H, W) or (H, W, C), dtype uint8
        # use OpenCV, scikit-image, scipy, or pure numpy — your choice
        result = image * strength
        return result.clip(0, 255).astype(image.dtype)
```

**ParamSpec kinds:** `"int"`, `"float"`, `"bool"`, `"choice"`, `"str"`,
`"file"` (file browser), `"directory"` (folder browser).

---

### 2 — Source step (green node, no input port)

Source nodes load or generate their own image instead of transforming one.
Set `IS_SOURCE = True`; the `image` argument in `process()` is a dummy
and should be ignored.

```python
@register_step
class CameraInputStep(ProcessingStep):
    NAME      = "Camera Source"
    CATEGORY  = "Input / Output"
    IS_SOURCE = True
    PARAMS    = [
        ParamSpec("device_id", "Device ID", "int", default=0,
                  min_value=0, max_value=10, step=1),
    ]

    def process(self, image: np.ndarray, device_id=0, **kwargs) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(device_id)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read from camera {device_id}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

---

### 3 — Sink step (dark-red node, passthrough)

Sink nodes write data somewhere and then return the image unchanged so
they can sit anywhere in the pipeline, not just at the end.
Set `IS_SINK = True`.

```python
@register_step
class VideoWriterStep(ProcessingStep):
    """
    Writes frames to a video file using OpenCV.
    Each call to process() appends one frame.
    Call pipeline.run() in a loop (e.g. over a list of source images)
    to build up the video.  The writer is opened on the first call and
    must be released by calling step.release() when done, or by deleting
    the node.
    """
    NAME     = "Video Writer"
    CATEGORY = "Input / Output"
    IS_SINK  = True
    PARAMS   = [
        ParamSpec("output_path", "Output File", "file",   default="output.mp4"),
        ParamSpec("fps",         "FPS",         "float",  default=25.0,
                  min_value=1.0, max_value=120.0, step=1.0),
    ]

    def __init__(self):
        super().__init__()
        self._writer = None

    def process(self, image: np.ndarray,
                output_path="output.mp4", fps=25.0, **kwargs) -> np.ndarray:
        import cv2
        import numpy as np

        # normalise to uint8 BGR
        arr = image if image.dtype == np.uint8 else (
            (image.astype(np.float32) - image.min()) /
            max(image.max() - image.min(), 1) * 255
        ).astype(np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else \
              cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        if self._writer is None:
            h, w = bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        self._writer.write(bgr)
        return image   # passthrough

    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __del__(self):
        self.release()
```

Drop this class into a file in `steps/` and it appears in the palette
immediately. To finalize the video, call `step.release()` after the last
frame, or delete the node.

---

## Tips

- **Multiple source nodes** are supported — each branch of the graph
  gets its own input image.
- **Sink nodes are passthrough**, so you can chain:
  `Source → Blur → DirectoryWriter → Threshold → DirectoryWriter`
  and save intermediate results at multiple points.
- **Thumbnail previews** appear on the first and last connections after
  running. Right-click any other connection to show its preview.
- **Saving to video in batch:** run the pipeline inside a Python script
  (no GUI needed) using `Pipeline.load()` and `pipeline.run()` in a loop,
  then call `step.release()` on the VideoWriterStep instance when done.
