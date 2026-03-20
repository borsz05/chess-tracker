import time
from collections import defaultdict


class PipelineProfiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self._active = {}

    def start(self, name: str):
        self._active[name] = time.perf_counter()

    def stop(self, name: str):
        if name not in self._active:
            return
        dt = time.perf_counter() - self._active[name]
        self.times[name] += dt
        self.counts[name] += 1
        del self._active[name]

    def report(self):
        print("\n---- Pipeline timing ----")
        for k in sorted(self.times.keys()):
            avg = self.times[k] / max(1, self.counts[k])
            print(f"{k:20s}  avg={avg*1000:7.2f} ms   calls={self.counts[k]}")
        print("-------------------------\n")