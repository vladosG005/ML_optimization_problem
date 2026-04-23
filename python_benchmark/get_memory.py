from sys import argv
import pickle
filename = argv[1]
with open(filename, 'rb') as file:
    model = pickle.load(file)
    print(typeof(model))
