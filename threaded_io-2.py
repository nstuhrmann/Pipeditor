"""
Threaded I/O base classes: overlap decode/encode with compute.

During a batch, a plain source decodes frame i while everything else
waits, then the pipeline computes while the decoder idles. These bases
hide that latency:

  ThreadedSource  a prefetch thread decodes ahead of the pipeline into a
                  small ordered buffer. Subclasses implement ONE method:

                      def read_frame(self, index) -> np.ndarray | None

                  (plus frame_count()). The base provides the sequence
                  protocol, the thread, jump handling, and a direct
                  synchronous fallback for single-frame previews.

  ThreadedSink    process() enqueues; a writer thread calls

                      def write_frame(self, frame): ...

                  in the background, with finalize() invoked after the
                  last frame is written. The queue is bounded, so a slow
                  writer applies backpressure instead of buffering the
                  whole sequence in RAM. Writing still happens ONLY
                  during a batch — previews never touch disk, matching
                  the plain writer steps.

Threading model: exactly one extra thread per threaded node, touching
only that node's own decoder/encoder. Steps in between stay untouched,
execution order is unchanged, stateful steps are unaffected — this
composes with everything, unlike graph-level parallelism.

Both bases re-raise worker-thread exceptions on the caller's thread (on
the next process()/end_sequence()), so a failing decode/encode surfaces
as a normal step error attributed to the right node.
"""
import threading
import queue

import numpy as np

from src.GUI.pipeline_editor.base_step import ProcessingStep


class ThreadedSource(ProcessingStep):
    """Sequence source with background prefetch.

    Subclass contract:
        read_frame(index) -> np.ndarray | None
            Called from the PREFETCH THREAD during batches (and directly
            on the caller's thread for previews / after thread death) —
            but never from two threads at once, so a stateful decoder
            (cv2.VideoCapture) is fine without locking.
        frame_count() -> int
            As usual.

    PREFETCH_DEPTH frames are decoded ahead (default 4): enough to hide
    decode latency, small enough to cap memory at DEPTH x frame size.
    """
    IS_SOURCE = True
    PREFETCH_DEPTH = 4

    def __init__(self):
        super().__init__()
        self._buf: dict[int, np.ndarray] = {}
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False
        self._want = 0            # next index the prefetcher should decode
        self._total = 0
        self._exc: Exception | None = None

    # --- subclass API --------------------------------------------------
    def read_frame(self, index: int):
        raise NotImplementedError

    # --- sequence protocol ---------------------------------------------
    def begin_sequence(self, total_frames: int):
        self._total = total_frames
        self._start_thread(start_at=0)

    def end_sequence(self):
        self._stop_thread()

    # --- prefetch machinery ---------------------------------------------
    def _start_thread(self, start_at: int):
        self._stop_thread()
        self._exc = None
        self._buf.clear()
        self._want = start_at
        self._running = True
        self._thread = threading.Thread(target=self._prefetch_loop,
                                        daemon=True)
        self._thread.start()

    def _stop_thread(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._buf.clear()

    def _prefetch_loop(self):
        try:
            while True:
                with self._cond:
                    while (self._running
                           and (len(self._buf) >= self.PREFETCH_DEPTH
                                or self._want >= self._total)):
                        self._cond.wait(timeout=0.25)
                    if not self._running:
                        return
                    idx = self._want
                    self._want += 1
                frame = self.read_frame(idx)     # outside the lock: slow
                with self._cond:
                    if not self._running:
                        return
                    if frame is not None:
                        self._buf[idx] = frame
                    self._cond.notify_all()
        except Exception as exc:
            with self._cond:
                self._exc = exc
                self._running = False
                self._cond.notify_all()

    def process(self):
        idx = self.ctx.index

        if self._thread is None or not self._running:
            # Preview / no batch running: plain synchronous read. Also
            # the fallback after a prefetch-thread failure, so the
            # original exception context isn't lost.
            if self._exc is not None:
                exc, self._exc = self._exc, None
                raise exc
            frame = self.read_frame(idx)
            if frame is None:
                raise ValueError(f"could not read frame {idx}")
            return frame

        with self._cond:
            if idx in self._buf:
                frame = self._buf.pop(idx)
                # drop anything older — it will not be asked for again
                for k in [k for k in self._buf if k < idx]:
                    del self._buf[k]
                self._cond.notify_all()
                return frame

            # Not buffered: either we're ahead of the prefetcher (wait)
            # or the executor jumped (restart the prefetcher there).
            if not (self._want - self.PREFETCH_DEPTH <= idx <= self._want):
                pass   # fall through to restart below
            else:
                while (self._running and idx not in self._buf
                       and self._exc is None):
                    self._cond.wait(timeout=0.25)
                if self._exc is not None:
                    exc, self._exc = self._exc, None
                    raise exc
                if idx in self._buf:
                    frame = self._buf.pop(idx)
                    self._cond.notify_all()
                    return frame

        # Jump (or thread died without an exception): restart at idx.
        self._start_thread(start_at=idx)
        with self._cond:
            while self._running and idx not in self._buf and self._exc is None:
                self._cond.wait(timeout=0.25)
            if self._exc is not None:
                exc, self._exc = self._exc, None
                raise exc
            if idx in self._buf:
                frame = self._buf.pop(idx)
                self._cond.notify_all()
                return frame
        raise ValueError(f"could not read frame {idx}")


class ThreadedSink(ProcessingStep):
    """Sink that writes on a background thread during batches.

    Subclass contract (both called ONLY from the writer thread, in frame
    order, never concurrently):
        write_frame(frame)
            Encode/write one frame. Open outputs lazily on first call.
        finalize()
            Close/release outputs. Called after the last queued frame,
            including on cancellation — same guarantee end_sequence()
            gives the plain writers.

    QUEUE_DEPTH bounds RAM; a slower writer than pipeline blocks
    process() briefly (backpressure) rather than buffering everything.
    Previews pass the image through without writing, exactly like the
    non-threaded writer steps.
    """
    IS_SINK = True
    QUEUE_DEPTH = 8

    _SENTINEL = object()

    def __init__(self):
        super().__init__()
        self._q: queue.Queue = queue.Queue(maxsize=self.QUEUE_DEPTH)
        self._thread: threading.Thread | None = None
        self._in_sequence = False
        self._exc: Exception | None = None

    # --- subclass API --------------------------------------------------
    def write_frame(self, frame: np.ndarray):
        raise NotImplementedError

    def finalize(self):
        pass

    # --- sequence protocol ---------------------------------------------
    def begin_sequence(self, total_frames: int):
        self._in_sequence = True
        self._exc = None
        self._q = queue.Queue(maxsize=self.QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._writer_loop,
                                        daemon=True)
        self._thread.start()

    def end_sequence(self):
        self._in_sequence = False
        if self._thread is not None:
            self._q.put(self._SENTINEL)      # drain, then finalize
            self._thread.join()
            self._thread = None
        if self._exc is not None:
            exc, self._exc = self._exc, None
            raise exc

    def _writer_loop(self):
        try:
            while True:
                item = self._q.get()
                if item is self._SENTINEL:
                    break
                self.write_frame(item)
        except Exception as exc:
            self._exc = exc
            # keep draining so producers never deadlock on a full queue
            while True:
                item = self._q.get()
                if item is self._SENTINEL:
                    break
        finally:
            try:
                self.finalize()
            except Exception as exc:
                if self._exc is None:
                    self._exc = exc

    def process(self, image):
        if self._in_sequence and self._thread is not None:
            if self._exc is not None:
                exc, self._exc = self._exc, None
                self._in_sequence = False
                raise exc
            # copy: downstream/preview code may hold the same array
            self._q.put(np.array(image, copy=True))
        return image
