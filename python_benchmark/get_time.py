from sys import argv
from load_model import load_model
from time import time_ns
X, Y, model = load_model(argv[1], argv[2])
stamp = time_ns()
model.predict(X)
print((time_ns() - stamp) // 1000000)