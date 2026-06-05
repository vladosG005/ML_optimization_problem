import joblib
import pandas as pd
import sklearn

def load_model(model_file, dataset_file):
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
        df = pd.read_csv(dataset_file)
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
            X = df[features]
            for col in df.columns:
                if col not in features:
                    Y = df[col]
                    if sklearn.metrics.accuracy_score(Y, model.predict(X)) > 0.5:
                        break
        else:
            if hasattr(model, 'n_features_in_'):
                feature_count = model.n_features_in_
            elif hasattr(model, 'n_features_'):
                feature_count = model.n_features_
            X = pd.DataFrame()
            Y = pd.DataFrame()
            target_names = ('target', 'label', 'class', 'y', 'outcome')
            cardinality_threshold = 20
            for col in df.columns:
                if col.lower() in target_names:
                    Y = df[col]
                elif X.columns.shape[0] < feature_count:
                    X = df.drop(columns=[col])
                    break
            if Y.empty:
                for col in df.columns:
                    if df[col].dtype in ('object', 'int64', 'bool'):
                        unique_count = df[col].nunique()
                        if unique_count <= cardinality_threshold and unique_count > 1:
                            Y = df[col]
                            cardinality_threshold = unique_count
                if Y.empty:
                    X = df.iloc[:, :-1]
                    Y = df.iloc[:, -1]
                elif Y.name in X.columns:
                    X = pd.DataFrame()
                    for col in df.columns:
                        if col != Y.name:
                            X[col] = df[col]
                        if X.columns.shape[0] == feature_count:
                            break
        return (X.values, Y, model)