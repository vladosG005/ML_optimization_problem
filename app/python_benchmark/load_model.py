import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pickle
import pandas as pd
import numpy as np
def load_model(model_file, dataset_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
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