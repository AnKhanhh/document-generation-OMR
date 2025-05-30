import time
from threading import Lock


class IDGenerator:
    def __init__(self):
        self._counter = 0
        self._lock = Lock()
        self._last_timestamp = 0

    def generate(self):
        with self._lock:
            timestamp = int(time.time())
            if timestamp == self._last_timestamp:
                self._counter += 1
            else:
                self._counter = 0
                self._last_timestamp = timestamp

            # Format: {seconds since epoch}-{count} (14 chars total)
            return f"{timestamp:010d}{self._counter:03d}"
