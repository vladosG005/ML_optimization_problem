from time import time_ns
import resource
from sklearn.metrics import accuracy_score
def get_time(X, model):
    stamp = time_ns()
    model.predict(X)
    return (time_ns() - stamp) // 1000000
def get_memory(X, model):
    model.predict(X)
    memory_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    memory_self = resource.getrusage(resource.RUSAGE_SELF)
    return (memory_children.ru_maxrss + memory_self.ru_maxrss) / 1024
def get_accuracy(X, Y, model):
    return round(100 * accuracy_score(Y, model.predict(X)), 2)
