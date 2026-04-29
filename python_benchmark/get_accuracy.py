from sys import argv
from load_model import load_model
from sklearn.metrics import accuracy_score
X, Y, model = load_model(argv[1], argv[2])
print(accuracy_score(Y, model.predict(X)))
