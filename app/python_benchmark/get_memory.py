from sys import argv
from load_model import load_model
import resource
X, Y, model = load_model(argv[1], argv[2])
model.predict(X)
memory_children = resource.getrusage(resource.RUSAGE_CHILDREN)
memory_self = resource.getrusage(resource.RUSAGE_SELF)
print((memory_children.ru_maxrss + memory_self.ru_maxrss) / 1024)
