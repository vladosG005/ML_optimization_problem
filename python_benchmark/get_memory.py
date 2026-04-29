from sys import argv
from load_model import load_model
import resource
X, Y, model = load_model(argv[1], argv[2])
model.predict(X)
print(resource.getrusage(resource.RUSAGE_CHILDREN + resource.RUSAGE_SELF).ru_maxrss / 1024)
