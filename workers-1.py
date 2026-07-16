"""
Background-thread workers for pipeline execution. Both are moved to a
QThread by MainWindow and communicate purely via signals — no direct UI
access from the worker side.
"""
from PySide6.QtCore import Signal, QObject

from src.GUI.pipeline_editor.pipeline import Pipeline


class PipelineWorker(QObject):
    """One single-frame pipeline run (the normal Run / Live Update path)."""
    finished = Signal(dict, list)   # results, warnings (skipped nodes etc.)
    failed   = Signal(str)

    def __init__(self, pipeline: Pipeline, frame_index: int = 0,
                 total_frames: int = 1):
        super().__init__()
        self.pipeline = pipeline
        self.frame_index = frame_index
        self.total_frames = total_frames

    def run(self):
        try:
            warnings: list[str] = []
            results = self.pipeline.run(
                frame_index=self.frame_index,
                total_frames=self.total_frames,
                warnings_out=warnings)
            self.finished.emit(results, warnings)
        except Exception as exc:
            self.failed.emit(str(exc))


class SequenceWorker(QObject):
    """Runs Pipeline.run_sequence() off the UI thread for a full
    video/image-stack batch. frameDone reports progress; cancel() is
    safe to call from the UI thread (should_cancel is polled between
    frames, not preempted mid-frame). last_results holds the final
    processed frame's results dict once finished, so the caller can
    refresh thumbnails/preview windows without streaming every frame's
    data across threads."""
    frameDone = Signal(int, int)   # frames_done, total
    finished  = Signal(int)        # frames actually processed
    failed    = Signal(str)

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._cancel = False
        self.last_results = None
        self.warnings: list[str] = []   # deduped across the batch

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            def _capture(frame_index, total, results):
                self.last_results = results   # cheap: just a reference swap

            processed = self.pipeline.run_sequence(
                on_frame_done=_capture,
                on_progress=lambda done, total: self.frameDone.emit(done, total),
                should_cancel=lambda: self._cancel,
                warnings_out=self.warnings)
            self.finished.emit(processed)
        except Exception as exc:
            self.failed.emit(str(exc))
