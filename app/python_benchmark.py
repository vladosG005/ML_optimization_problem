import joblib
import pandas as pd
import numpy as np
from time import time_ns
import resource
from sklearn.metrics import accuracy_score

def load_model(model_file, dataset_file):
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
        df = pd.read_csv(dataset_file)
        features = model.get_booster().feature_names
        X = []
        Y = []
        for col in df.columns:
            if col in features:
                X.append(df[col].values)
            elif not Y:
                Y.append(df[col].values)
        X = np.transpose(np.array(X))
        Y = np.transpose(np.array(Y))
        return (X, Y, model)

def get_time(model_file, dataset_file):
    X, Y, model = load_model(model_file, dataset_file)
    stamp = time_ns()
    model.predict(X)
    return (time_ns() - stamp) // 1000000

def get_memory(model_file, dataset_file):
    X, Y, model = load_model(model_file, dataset_file)
    model.predict(X)
    memory_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    memory_self = resource.getrusage(resource.RUSAGE_SELF)
    return (memory_children.ru_maxrss + memory_self.ru_maxrss) / 1024

def get_accuracy(model_file, dataset_file):
    X, Y, model = load_model(model_file, dataset_file)
    return round(100 * accuracy_score(Y, model.predict(X)), 2)
