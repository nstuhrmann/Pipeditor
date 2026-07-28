"""
Worker objects that run the pipeline off the GUI thread.

Both emit a RunResult, so adding an output channel never changes a
signal signature again.
"""
from PySide6.QtCore import QObject, Signal

from src.GUI.pipeline_editor.pipeline import Pipeline


class PipelineWorker(QObject):
    """One frame, off the UI thread.

    Two modes, one signal contract:
      * no iterator -> a PREVIEW of `frame_index` (no writes, stateful
        steps rewind so the same index can be re-run with new params);
      * an open iter_sequence() -> pull the NEXT committed frame from it.
        Playback uses this, so a played run and a batch drain the very
        same generator.
    """
    finished = Signal(object)      # RunResult
    exhausted = Signal()           # iterator mode: sequence ended
    failed = Signal(str)

    def __init__(self, pipeline: Pipeline, frame_index: int = 0,
                 total_frames: int = 1, iterator=None):
        super().__init__()
        self.pipeline = pipeline
        self.frame_index = frame_index
        self.total_frames = total_frames
        self.iterator = iterator

    def run(self):
        try:
            if self.iterator is not None:
                try:
                    _index, _total, result = next(self.iterator)
                except StopIteration:
                    self.exhausted.emit()
                    return
            else:
                result = self.pipeline.preview(
                    frame_index=self.frame_index,
                    total_frames=self.total_frames)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class SequenceWorker(QObject):
    """A full batch, off the UI thread.

    Cancellation is checked between frames, so it is responsive but never
    preempts a frame mid-flight (steps are stateful; interrupting one
    would corrupt it).
    """
    progress = Signal(int, int)    # frames done, total
    frame_done = Signal(int, int, object)   # index, total, RunResult
    finished = Signal(object)      # RunResult for the whole batch
    failed = Signal(str)

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.result = None
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.result = self.pipeline.run_sequence(
                on_frame_done=lambda i, t, r: self.frame_done.emit(i, t, r),
                on_progress=lambda d, t: self.progress.emit(d, t),
                should_cancel=lambda: self._cancel)
            self.finished.emit(self.result)
        except Exception as exc:
            self.failed.emit(str(exc))
