import sys
import time

if sys.version >= "3":
    from contextlib import ContextDecorator
else:
    from contextdecorator import ContextDecorator


QUIET = "QUIET" in sys.argv


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def DispWarning(text, color=bcolors.WARNING):
    print(color, text, bcolors.ENDC)


class DispDebug(ContextDecorator):
    def __init__(self, name, disp=None, stream=sys.stdout):
        self.name = name
        self.disp = disp if disp is not None else not QUIET
        self.stream = stream

    def __enter__(self):
        if self.disp:
            self.stream.write(f"{self.name}...")
            self.stream.flush()
            self.ts = time.time()
        return self

    def __exit__(self, *exc):
        if self.disp:
            self.stream.write(f" done in {time.time() - self.ts:.3f}s")
            self.stream.flush()
        return False
