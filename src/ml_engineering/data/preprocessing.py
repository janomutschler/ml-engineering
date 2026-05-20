import numpy as np


def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    """
    Split features and targets into randomized training and test sets.
    """
    rng = np.random.default_rng(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    """
    Standardize features using training set mean and standard deviation.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled
