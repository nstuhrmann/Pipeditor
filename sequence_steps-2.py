"""
Video / image-stack sources and sinks. These are the steps that actually
implement the IS_SEQUENCE_AWARE protocol described in base_step.py — see
that docstring first if you're adding another one.

Common shape all four follow:
  - A source distinguishes "inside an active batch" (self._cap/_reader set
    up in begin_sequence) from "a one-off preview call" (no persistent
    resource open — open, seek, read, close, every single process() call).
  - A sink distinguishes the same via a plain self._in_sequence flag: it
    only ever writes to disk while that's True. A preview call is always
    a pure passthrough, so tuning parameters live never spams frames to
    disk or advances a video file.
"""
import os
import glob
import numpy as np
import cv2

from src.GUI.pipeline_editor.base_step import ProcessingStep, ParamSpec, register_step


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@register_step
class VideoFileSource(ProcessingStep):
    NAME     = "Video File Source"
    CATEGORY = "Input / Output"
    IS_SOURCE = True
    IS_SEQUENCE_AWARE = True
    PARAMS = [
        ParamSpec("path", "Video File", "file", default="",
                  types="Videos (*.mp4 *.avi *.mkv *.mov);;All files (*)"),
    ]

    def __init__(self):
        super().__init__()
        self._cap = None            # persistent capture, only while batching
        self._cap_path = None
        self._last_read_index = -1  # lets sequential reads skip re-seeking
        self._pending_index = 0

    def frame_count(self) -> int:
        path = self.values.get("path", "")
        if not path:
            return 1
        cap = cv2.VideoCapture(path)
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return max(1, n)

    def begin_sequence(self, total_frames: int):
        path = self.values.get("path", "")
        self._cap = cv2.VideoCapture(path) if path else None
        self._cap_path = path
        self._last_read_index = -1

    def end_sequence(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._cap_path = None
        self._last_read_index = -1

    def set_frame_index(self, index: int, total_frames: int):
        self._pending_index = index

    def process(self, image: np.ndarray, path: str = "", **kwargs) -> np.ndarray:
        if not path:
            raise ValueError("Video File Source: no file selected.")
        index = self._pending_index

        if self._cap is not None and self._cap_path == path:
            # Inside an active batch — reuse the open capture, and only
            # seek if we're not just reading the next frame in order
            # (seeking every frame is much slower than sequential reads).
            if index != self._last_read_index + 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._cap.read()
            self._last_read_index = index
        else:
            # Single-frame preview — no persistent state to reuse.
            cap = cv2.VideoCapture(path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
            finally:
                cap.release()

        if not ok or frame is None:
            raise ValueError(f"Video File Source: could not read frame {index}.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@register_step
class ImageStackSource(ProcessingStep):
    """A directory of numbered frames (PNG/TIFF/etc.), sorted by filename.
    Reads with IMREAD_UNCHANGED so 16-bit stacks stay 16-bit."""
    NAME     = "Image Stack Source"
    CATEGORY = "Input / Output"
    IS_SOURCE = True
    IS_SEQUENCE_AWARE = True
    PARAMS = [
        ParamSpec("directory", "Directory", "directory", default=""),
        ParamSpec("pattern", "Filename Pattern", "str", default="*.png"),
    ]

    def __init__(self):
        super().__init__()
        self._files: list[str] = []       # only populated during a batch
        self._files_key = None            # (directory, pattern) it was built for
        self._pending_index = 0

    def _list_files(self, directory: str, pattern: str) -> list[str]:
        if not directory:
            return []
        return sorted(glob.glob(os.path.join(directory, pattern)))

    def frame_count(self) -> int:
        directory = self.values.get("directory", "")
        pattern = self.values.get("pattern", "*.png")
        return max(1, len(self._list_files(directory, pattern)))

    def begin_sequence(self, total_frames: int):
        directory = self.values.get("directory", "")
        pattern = self.values.get("pattern", "*.png")
        self._files = self._list_files(directory, pattern)
        self._files_key = (directory, pattern)

    def end_sequence(self):
        self._files = []
        self._files_key = None

    def set_frame_index(self, index: int, total_frames: int):
        self._pending_index = index

    def process(self, image: np.ndarray, directory: str = "",
                pattern: str = "*.png", **kwargs) -> np.ndarray:
        key = (directory, pattern)
        files = self._files if self._files_key == key else self._list_files(directory, pattern)
        if not files:
            raise ValueError(f"Image Stack Source: no files matching "
                            f"'{pattern}' in '{directory}'.")
        index = min(self._pending_index, len(files) - 1)
        arr = cv2.imread(files[index], cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Image Stack Source: failed to read {files[index]}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr


@register_step
class WebcamSource(ProcessingStep):
    """Live camera as a sequence source. Unlike a video file, a webcam
    has no inherent length — num_frames declares how many frames a
    'Process Full Sequence' run should capture. Frames are grabbed
    sequentially (frame_index can't seek a live device). During a batch
    the capture stays open across all frames; a single-frame preview
    opens the device, grabs one frame, and releases it."""
    NAME     = "Webcam Source"
    CATEGORY = "Input / Output"
    IS_SOURCE = True
    IS_SEQUENCE_AWARE = True
    PARAMS = [
        ParamSpec("device", "Device Index", "int", default=0,
                  min_value=0, max_value=16, step=1),
        ParamSpec("num_frames", "Frames to Capture", "int", default=100,
                  min_value=1, max_value=100000, step=1),
    ]

    def __init__(self):
        super().__init__()
        self._cap = None          # persistent capture, only during a batch
        self._cap_device = None

    def frame_count(self) -> int:
        return max(1, int(self.values.get("num_frames", 1)))

    def begin_sequence(self, total_frames: int):
        device = int(self.values.get("device", 0))
        self._cap = cv2.VideoCapture(device)
        self._cap_device = device

    def end_sequence(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._cap_device = None

    def process(self, image: np.ndarray, device: int = 0,
                **kwargs) -> np.ndarray:
        if self._cap is not None and self._cap_device == device:
            ok, frame = self._cap.read()          # sequential live grab
        else:
            cap = cv2.VideoCapture(int(device))   # one-off preview
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
        if not ok or frame is None:
            raise ValueError(
                f"Webcam Source: could not read from device {device}.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

@register_step
class ImageSequenceWriter(ProcessingStep):
    """Writes each frame of a batch run as a numbered file in a directory,
    at either 8-bit or 16-bit depth. Only writes while inside an active
    batch (Pipeline.run_sequence) — a plain preview call is a passthrough,
    so live-mode parameter tuning never spams files to disk."""
    NAME     = "Image Sequence Writer"
    CATEGORY = "Input / Output"
    IS_SINK  = True
    IS_SEQUENCE_AWARE = True
    PARAMS = [
        ParamSpec("output_dir", "Output Directory", "directory", default=""),
        ParamSpec("prefix", "Filename Prefix", "str", default="frame"),
        ParamSpec("format", "Format", "choice", default="PNG",
                  choices=["PNG", "TIFF"]),
        ParamSpec("bit_depth", "Bit Depth", "choice", default="auto",
                  choices=["auto", "8", "16"]),
    ]

    def __init__(self):
        super().__init__()
        self._in_sequence = False
        self._index = 0
        self._pad = 5

    def begin_sequence(self, total_frames: int):
        self._in_sequence = True
        self._pad = max(4, len(str(max(0, total_frames - 1))))

    def end_sequence(self):
        self._in_sequence = False

    def set_frame_index(self, index: int, total_frames: int):
        self._index = index

    def process(self, image: np.ndarray, output_dir: str = "",
                prefix: str = "frame", format: str = "PNG",
                bit_depth: str = "auto", **kwargs) -> np.ndarray:
        if self._in_sequence and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            ext = ".png" if format == "PNG" else ".tiff"
            out_path = os.path.join(
                output_dir, f"{prefix}_{self._index:0{self._pad}d}{ext}")
            depth = self._resolve_depth(image, bit_depth)
            arr = self._to_depth(image, depth)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
            cv2.imwrite(out_path, bgr)
        return image

    @staticmethod
    def _resolve_depth(image: np.ndarray, bit_depth: str) -> str:
        if bit_depth != "auto":
            return bit_depth
        # auto: uint16 stays 16-bit; uint8 and float (and anything else)
        # default to 8-bit — float data in this codebase is generally an
        # intermediate/HDR representation, not meant to be read as-is at
        # 16-bit integer precision, so 8-bit is the safer default.
        if image.dtype == np.uint16:
            return "16"
        return "8"

    @staticmethod
    def _to_depth(image: np.ndarray, bit_depth: str) -> np.ndarray:
        """Inputs arrive as float32 [0,1] (the pipeline contract); scale
        deterministically to the target depth. Int inputs (possible if
        this step is called outside the executor) are handled too."""
        if bit_depth == "16":
            if image.dtype == np.uint16:
                return image
            if image.dtype == np.uint8:
                return (image.astype(np.uint16) * 257)   # exact 8→16 bit
            a = np.clip(image.astype(np.float32), 0.0, 1.0)
            return (a * 65535.0 + 0.5).astype(np.uint16)
        # 8-bit
        if image.dtype == np.uint8:
            return image
        if image.dtype == np.uint16:
            return (image // 257).astype(np.uint8)
        a = np.clip(image.astype(np.float32), 0.0, 1.0)
        return (a * 255.0 + 0.5).astype(np.uint8)


@register_step
class VideoFileWriter(ProcessingStep):
    """Writes each frame of a batch run to a video file. Like
    ImageSequenceWriter, only actually opens/writes while inside an
    active batch — a preview call is a pure passthrough, since there's
    no sensible way to "write one frame of a video" in isolation."""
    NAME     = "Video File Writer"
    CATEGORY = "Input / Output"
    IS_SINK  = True
    IS_SEQUENCE_AWARE = True
    PARAMS = [
        ParamSpec("output_path", "Output Video File", "file",
                  default="output.mp4",
                  types="Videos (*.mp4 *.avi);;All files (*)"),
        ParamSpec("fourcc", "Codec (FOURCC)", "choice", default="mp4v",
                  choices=["mp4v", "XVID", "MJPG", "avc1"]),
        ParamSpec("fps", "FPS", "float", default=30.0,
                  min_value=1.0, max_value=240.0, step=1.0),
    ]

    def __init__(self):
        super().__init__()
        self._in_sequence = False
        self._writer = None

    def begin_sequence(self, total_frames: int):
        self._in_sequence = True
        self._writer = None   # opened lazily on first frame, once we know its size

    def end_sequence(self):
        if self._writer is not None:
            self._writer.release()
        self._writer = None
        self._in_sequence = False

    def process(self, image: np.ndarray, output_path: str = "output.mp4",
                fourcc: str = "mp4v", fps: float = 30.0, **kwargs) -> np.ndarray:
        if self._in_sequence and output_path:
            frame = self._to_bgr_uint8(image)
            if self._writer is None:
                h, w = frame.shape[:2]
                fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                self._writer = cv2.VideoWriter(output_path, fourcc_code, fps, (w, h))
                if not self._writer.isOpened():
                    raise ValueError(
                        f"Video File Writer: could not open '{output_path}' "
                        f"for writing (codec '{fourcc}').")
            self._writer.write(frame)
        return image

    @staticmethod
    def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            # Pipeline contract: float32 [0,1] → scale directly.
            a = np.clip(image.astype(np.float32), 0.0, 1.0)
            image = (a * 255.0 + 0.5).astype(np.uint8)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
