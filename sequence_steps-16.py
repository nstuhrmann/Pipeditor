"""
Video / image-stack sources and sinks, built on the threaded I/O bases
(threaded_io.py): during a batch, sources decode AHEAD of the pipeline
on a prefetch thread and sinks encode BEHIND it on a writer thread, so
disk/codec latency overlaps with compute instead of adding to it.

What the bases guarantee (see threaded_io.py for the details):
  - read_frame()/write_frame() are never called from two threads at
    once, so the persistent cv2 capture/writer objects need no locking;
  - writes happen in frame order, and ONLY during a batch — a preview
    call is a pure passthrough, so live tuning never spams the disk;
  - a single-frame preview reads synchronously (no thread involved);
  - worker-thread exceptions re-surface as normal step errors.

PARAMS / NAME / CATEGORY are identical to the pre-threaded versions, so
existing saved pipelines load unchanged.
"""
import csv
import glob
import os
import numpy as np
import cv2

from src.GUI.pipeline_editor.array_utils import (
    METRIC_META_PREFIX, to_uint8, to_uint16,
)
from src.GUI.pipeline_editor.base_step import (
    ParamSpec, ProcessingStep, register_step,
)
from src.GUI.pipeline_editor.threaded_io import ThreadedSource, ThreadedSink


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@register_step
class VideoFileSource(ThreadedSource):
    NAME     = "Video File Source"
    CATEGORY = "IO/Input"
    PARAMS = [
        ParamSpec("path", "Video File", "file", default="",
                  types="Videos (*.mp4 *.avi *.mkv *.mov);;All files (*)"),
    ]

    def __init__(self):
        super().__init__()
        self._cap = None            # persistent capture (single-threaded use)
        self._cap_path = None
        self._last_read_index = -1  # sequential reads skip the slow re-seek

    def frame_count(self) -> int:
        path = self.p.path
        if not path:
            return 1
        cap = cv2.VideoCapture(path)
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return max(1, n)

    def read_frame(self, index: int):
        path = self.p.path
        if not path:
            raise ValueError("Video File Source: no file selected.")
        if self._cap is None or self._cap_path != path:
            self._release_cap()
            self._cap = cv2.VideoCapture(path)
            self._cap_path = path
            self._last_read_index = -1
        if index != self._last_read_index + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        self._last_read_index = index
        if not ok or frame is None:
            raise ValueError(
                f"Video File Source: could not read frame {index}.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _release_cap(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._cap_path = None
        self._last_read_index = -1

    def close_resources(self):
        # Stop the prefetch thread FIRST (it may be mid-read on the
        # capture), only then release the capture.
        super().close_resources()
        self._release_cap()


@register_step
class ImageStackSource(ThreadedSource):
    """A directory of numbered frames (PNG/TIFF/etc.), sorted by filename.
    Reads with IMREAD_UNCHANGED so 16-bit stacks stay 16-bit."""
    NAME     = "Image Stack Source"
    CATEGORY = "IO/Input"
    PARAMS = [
        ParamSpec("directory", "Directory", "directory", default=""),
        ParamSpec("pattern", "Filename Pattern", "str", default="*.png"),
    ]

    def __init__(self):
        super().__init__()
        self._files: list[str] = []
        self._files_key = None      # (directory, pattern) the list was built for

    @staticmethod
    def _list_files(directory: str, pattern: str) -> list[str]:
        if not directory:
            return []
        return sorted(glob.glob(os.path.join(directory, pattern)))

    def frame_count(self) -> int:
        directory = self.p.directory
        pattern = self.p.pattern
        return max(1, len(self._list_files(directory, pattern)))

    def read_frame(self, index: int):
        directory = self.p.directory
        pattern = self.p.pattern
        key = (directory, pattern)
        if self._files_key != key:
            self._files = self._list_files(directory, pattern)
            self._files_key = key
        if not self._files:
            raise ValueError(f"Image Stack Source: no files matching "
                             f"'{pattern}' in '{directory}'.")
        index = min(index, len(self._files) - 1)
        arr = cv2.imread(self._files[index], cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(
                f"Image Stack Source: failed to read {self._files[index]}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr


@register_step
class WebcamSource(ThreadedSource):
    """Live camera as a sequence source. Unlike a video file, a webcam
    has no inherent length — num_frames declares how many frames a
    'Process Full Sequence' run should capture. The frame index is
    ignored (a live device can't seek); frames are grabbed in call
    order, which the prefetch thread preserves. Prefetching is actively
    useful here: grabbing ahead of the pipeline reduces dropped frames.
    During a batch the capture stays open; a single-frame preview opens
    the device, grabs one frame, and releases it."""
    NAME     = "Webcam Source"
    CATEGORY = "IO/Input"
    PARAMS = [
        ParamSpec("device", "Device Index", "int", default=0,
                  min_value=0, max_value=16, step=1),
        ParamSpec("num_frames", "Frames to Capture", "int", default=100,
                  min_value=1, max_value=100000, step=1),
    ]

    def __init__(self):
        super().__init__()
        self._cap = None
        self._cap_device = None

    def frame_count(self) -> int:
        return max(1, int(self.p.num_frames))

    def read_frame(self, index: int):
        device = int(self.p.device)
        if self._running:
            # Batch: persistent capture, sequential grabs.
            if self._cap is None or self._cap_device != device:
                if self._cap is not None:
                    self._cap.release()
                self._cap = cv2.VideoCapture(device)
                self._cap_device = device
            ok, frame = self._cap.read()
        else:
            # One-off preview: don't hold the device open.
            cap = cv2.VideoCapture(device)
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
        if not ok or frame is None:
            raise ValueError(
                f"Webcam Source: could not read from device {device}.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close_resources(self):
        super().close_resources()   # stop prefetch first, then release
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._cap_device = None


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

@register_step
class ImageSequenceWriter(ThreadedSink):
    """Writes each frame of a batch run as a numbered file in a
    directory, at either 8-bit or 16-bit depth. Encoding happens on the
    writer thread, in frame order, so numbering by write order equals
    numbering by frame index."""
    NAME     = "Image Sequence Writer"
    CATEGORY = "IO/Output"
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
        self._write_index = 0
        self._pad = 5

    def open_resources(self, total_frames: int):
        self._write_index = 0
        self._pad = max(4, len(str(max(0, total_frames - 1))))
        super().open_resources(total_frames)

    def process(self, image):
        # No output dir configured: skip the enqueue (and its frame
        # copy) entirely instead of no-opping on the writer thread.
        if not self.p.output_dir:
            return image
        return super().process(image)

    def write_frame(self, frame: np.ndarray):
        output_dir = self.p.output_dir
        prefix = self.p.prefix
        fmt = self.p.format
        bit_depth = self.p.bit_depth

        os.makedirs(output_dir, exist_ok=True)
        ext = ".png" if fmt == "PNG" else ".tiff"
        out_path = os.path.join(
            output_dir, f"{prefix}_{self._write_index:0{self._pad}d}{ext}")
        self._write_index += 1

        depth = self._resolve_depth(frame, bit_depth)
        arr = self._to_depth(frame, depth)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
        cv2.imwrite(out_path, bgr)

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
        return (to_uint16(image) if bit_depth == "16" else to_uint8(image))


@register_step
class VideoFileWriter(ThreadedSink):
    """Writes each frame of a batch run to a video file. The
    cv2.VideoWriter lives entirely on the writer thread: opened lazily
    on the first frame (once the size is known), fed in order, released
    in finalize() — including after cancellation, so partial videos are
    still closed properly."""
    NAME     = "Video File Writer"
    CATEGORY = "IO/Output"
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
        self._writer = None

    def process(self, image):
        if not self.p.output_path:
            return image
        return super().process(image)

    def write_frame(self, frame: np.ndarray):
        output_path = self.p.output_path
        bgr = self._to_bgr_uint8(frame)
        if self._writer is None:
            h, w = bgr.shape[:2]
            fourcc = self.p.fourcc
            fps = float(self.p.fps)
            self._writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if not self._writer.isOpened():
                self._writer = None
                raise ValueError(
                    f"Video File Writer: could not open '{output_path}' "
                    f"for writing (codec '{fourcc}').")
        self._writer.write(bgr)

    def finalize(self):
        if self._writer is not None:
            self._writer.release()
        self._writer = None

    @staticmethod
    def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
        image = to_uint8(image)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


@register_step
class MetadataCSVWriter(ProcessingStep):
    """Write the metadata carried by each frame to one CSV — one row per
    frame, one column per key.

    Because metric values travel as metadata, this captures every metric
    in the graph at once, in a single file with a column each, instead of
    one file per metric. It also captures metadata that is NOT a metric —
    a camera's exposure_ms, gain and illumination, say — which otherwise
    needs a monitor node per quantity just to make it dumpable.

    It records what its INPUT carries, so put it at the END of the chain:
    metadata accumulates downstream, and a writer placed early simply
    hasn't been told about anything yet.

    Rows are buffered and written on close, so the column set is the
    union over the whole run — a key that only appears from frame 10
    still gets a column, blank before then. Only scalars are written;
    an array-valued metadata entry is skipped rather than mangled.
    """
    NAME = "Metadata CSV Writer"
    CATEGORY = "IO/Output"
    KIND = "sink"
    PARAMS = [
        ParamSpec("path", "CSV File", "file", default="metadata.csv",
                  types="CSV files (*.csv);;All files (*)"),
        ParamSpec("include", "Include", "choice", default="all",
                  choices=["all", "metrics only", "custom"]),
        ParamSpec("columns", "Custom Keys (comma separated)", "str",
                  default=""),
        ParamSpec("strip_prefix", "Strip 'metric:' From Column Names",
                  "bool", default=True),
    ]

    def __init__(self):
        super().__init__()
        self._rows: list = []
        self._columns: list = []      # first-appearance order

    def open_resources(self, total_frames: int):
        self._rows = []
        self._columns = []

    def _wanted(self, key: str) -> bool:
        mode = self.p.include
        if mode == "metrics only":
            return key.startswith(METRIC_META_PREFIX)
        if mode == "custom":
            wanted = {k.strip() for k in str(self.p.columns).split(",")
                      if k.strip()}
            return key in wanted or key[len(METRIC_META_PREFIX):] in wanted
        return True

    def process(self, image):
        # Only a committed run records anything — scrubbing the frame
        # slider must not append rows.
        if not self.ctx.committed:
            return image
        row = {"frame": self.ctx.index}
        for key, value in (getattr(image, "meta", {}) or {}).items():
            if not self._wanted(key):
                continue
            if not isinstance(value, (int, float, bool, str)):
                continue          # arrays and objects don't belong in a cell
            name = key
            if self.p.strip_prefix and key.startswith(METRIC_META_PREFIX):
                name = key[len(METRIC_META_PREFIX):]
            if name not in self._columns:
                self._columns.append(name)
            row[name] = value
        self._rows.append(row)
        return image

    def close_resources(self):
        rows, self._rows = self._rows, []
        if not rows:
            return
        path = (self.p.path or "").strip() or "metadata.csv"
        columns = ["frame"] + self._columns
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
