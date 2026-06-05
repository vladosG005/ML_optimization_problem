from time import perf_counter_ns
import sys
import gc
import resource
from sklearn.metrics import accuracy_score
def get_time(X, model):
    stamp = perf_counter_ns()
    model.predict(X)
    return (perf_counter_ns() - stamp) // 1_000_000
def get_memory(X, model):
    model.predict(X)

    gc.collect()

    if sys.platform == "win32":
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return None
    else:
        import resource
        mem_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        mem_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 if sys.platform.startswith("linux") else 1024 * 1024
        return (mem_children + mem_self) / divisor

def get_accuracy(X, Y, model):
    return round(100 * accuracy_score(Y, model.predict(X)), 2)
