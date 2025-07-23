import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures


def load_data(dataset_path):

    df = pd.read_csv(dataset_path)
    df= df.fillna(df.mean())
    data = df.to_numpy()

    x = data[:, :-1]
    t = data[:, -1].reshape(-1, 1)

    x_train = x[100,:]
    x_val = x[,100:]


    t_train = t[100,:]
    t_val = t[100,:]

    return x, t, x_train, x_val, t_train, t_val


def get_preprocessor(preprocessor_option):

    if preprocessor_option == 0:
        return None

    if preprocessor_option == 1:
        return MinMaxScaler()

    return StandardScaler()

def fit_transform(data, preprocessor_option):
    preprocessor = get_preprocessor(preprocessor_option)

    if preprocessor is not None:
        data = preprocessor.fit_transform(data)


    return data, preprocessor

def monomials_poly_features(X, degree):

    assert degree > 0

    if degree == 1:
        return X

    examples = []

    for example in X:
        example_features = []
        for feature in example:
            cur = 1
            feats = []
            for deg in range(degree):
                cur *= feature
                feats.append(cur)
            example_features.extend(feats)
        examples.append(np.array(example_features))

    return np.vstack(examples)

def preprocess_data(X_train, X_val=None, poly_degree=1, use_cross_features=True, preprocess_option=1):
    if poly_degree > 1:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)  # cross-features

        if use_cross_features:
            X_train = poly.fit_transform(X_train)
        else:
            X_train = monomials_poly_features(X_train, poly_degree)  # no cross features

        if X_val is not None:
            if use_cross_features:
                X_val = poly.fit_transform(X_val)
            else:
                X_val = monomials_poly_features(X_val, poly_degree)

    X_train, processor = fit_transform(X_train, preprocess_option)

    if X_val is not None and processor is not None:
        X_val = processor.transform(X_val)

    return X_train, X_val
