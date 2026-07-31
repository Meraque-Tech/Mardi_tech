"""RAM-only frame buffering decisions for GNSS movement states."""

from collections import deque


MOVEMENT_STATES = frozenset(
    {
        "initializing",
        "forward",
        "reverse_suspected",
        "reversing",
        "forward_suspected",
        "stationary",
    }
)


class SegmentFrameBuffer:
    """Hold frames until GNSS classifies their movement segment."""

    def __init__(self, max_frames):
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        self.max_frames = max_frames
        self._frames = deque()

    def __len__(self):
        return len(self._frames)

    def add(self, frame):
        """Add one frame and return a dropped oldest frame, if capped."""
        dropped = None
        if len(self._frames) >= self.max_frames:
            dropped = self._frames.popleft()
        self._frames.append(frame)
        return dropped

    def clear(self):
        """Discard and return the number of buffered frames."""
        discarded = len(self._frames)
        self._frames.clear()
        return discarded

    def handle_state(self, movement_state):
        """Return frames safe to persist and the number discarded."""
        if movement_state not in MOVEMENT_STATES:
            raise ValueError("unknown movement state: %s" % movement_state)

        if movement_state == "forward":
            committed = list(self._frames)
            self._frames.clear()
            return committed, 0

        if movement_state == "forward_suspected":
            return [], 0

        return [], self.clear()
