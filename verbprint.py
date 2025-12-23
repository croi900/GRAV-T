import time

VERBOSE = 0

DELTA_TIME = 0


def vprint(*args, **kwargs):
    if VERBOSE:
        global DELTA_TIME
        old = DELTA_TIME
        DELTA_TIME = time.time()
        print(f"[{(DELTA_TIME - old):.6f}] ", end="")
        print(*args, **kwargs)
