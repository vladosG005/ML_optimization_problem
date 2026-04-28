from sys import argv
import pickle
import pandas as pd
model_file = argv[1]
dataset_file = argv[2]
with open(model_file, 'rb') as f:
    model = pickle.load(f)
    pd.read_csv(dataset_file)
