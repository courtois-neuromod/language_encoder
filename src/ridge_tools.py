from __future__ import division

import time

import numpy as np
from numpy.linalg import inv, svd
from scipy.stats import zscore
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold


def corr(X, Y, axis=0):
    """Compute correlation coefficient."""
    return np.mean(zscore(X) * zscore(Y), axis)


def R2(Pred, Real):
    """Compute coefficient of determination (R^2)."""
    SSres = np.mean((Real - Pred) ** 2, 0)
    SStot = np.var(Real, 0)
    return np.nan_to_num(1 - SSres / SStot)


def fit_predict(data, features, method="plain", n_folds=10):
    """
    Fit and predict using cross-validated Ridge regression.

    Args:
        data (numpy.ndarray): The data array.
        features (numpy.ndarray): The features array.
        method (str): The Ridge regression method. Defaults to 'plain'.
        n_folds (int): The number of folds for cross-validation. Defaults to 10.

    Returns:
        tuple: Tuple containing the correlation and R^2 values.
    """
    n, v = data.shape
    p = features.shape[1]
    corrs = np.zeros((n_folds, v))
    R2s = np.zeros((n_folds, v))
    ind = CV_ind(n, n_folds)
    preds_all = np.zeros_like(data)

    for i in range(n_folds):
        train_data = np.nan_to_num(zscore(data[ind != i]))
        train_features = np.nan_to_num(zscore(features[ind != i]))
        test_data = np.nan_to_num(zscore(data[ind == i]))
        test_features = np.nan_to_num(zscore(features[ind == i]))
        weights, __ = cross_val_ridge(train_features, train_data, method=method)
        preds = np.dot(test_features, weights)
        preds_all[ind == i] = preds

    corrs = corr(preds_all, data)
    R2s = R2(preds_all, data)

    return corrs, R2s


def CV_ind(n, n_folds):
    """Generate cross-validation indices."""
    ind = np.zeros((n))
    n_items = int(np.floor(n / n_folds))

    for i in range(0, n_folds - 1):
        ind[i * n_items : (i + 1) * n_items] = i

    ind[(n_folds - 1) * n_items :] = n_folds - 1

    return ind


def R2r(Pred, Real):
    """Compute square root of R^2."""
    R2rs = R2(Pred, Real)
    ind_neg = R2rs < 0
    R2rs = np.abs(R2rs)
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= -1

    return R2rs


def ridge(X, Y, lmbda):
    """Compute ridge regression weights."""
    return np.dot(inv(X.T.dot(X) + lmbda * np.eye(X.shape[1])), X.T.dot(Y))


def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    """Compute validation errors for ridge regression with different lambda values."""
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = ridge(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def ridge_sk(X, Y, lmbda):
    """Compute ridge regression weights using scikit-learn."""
    rd = Ridge(alpha=lmbda)
    rd.fit(X, Y)
    return rd.coef_.T, rd.fit(X, Y)


def ridgeCV_sk(X, Y, lmbdas):
    """Compute ridge regression weights using scikit-learn with cross-validation."""
    rd = RidgeCV(alphas=lmbdas, solver="svd")
    rd.fit(X, Y)
    return rd.coef_.T, rd.fit(X, Y)


def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    """Compute validation errors for ridge regression with different lambda values using scikit-learn."""
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights, _ = ridge_sk(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def ridge_svd(X, Y, lmbda):
    """
    Ridge regression using singular value decomposition (SVD).
    """
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s**2 + lmbda)
    return np.dot(Vt, np.diag(d).dot(U.T.dot(Y)))


def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    """
    Calculate the validation error of ridge regression using SVD for different lambdas.
    """
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    U, s, Vt = svd(X, full_matrices=False)
    for idx, lmbda in enumerate(lambdas):
        d = s / (s**2 + lmbda)
        weights = np.dot(Vt, np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def cross_val_ridge(
    train_features: np.array,
    train_data: np.array,
    groups: list,
    data_config,
    lambdas=np.array([10**i for i in range(-1, 8)]),
    method="ridge_sk",
    do_plot=False,
    do_zscore=False,
):
    """
    Cross validation for ridge regression.

    Args:
        train_features (array): Array of training features.
        train_data (array): Array of training data.
        lambdas (array): Array of lambda values for Ridge regression.
                          Default is [10^i for i in range(-6, 10)].

    Returns:
        weightMatrix (array): Array of weights for the Ridge regression.
        r (array): Array of regularization parameters.

    """

    ridge_1 = {
        "plain": ridge_by_lambda,
        "svd": ridge_by_lambda_svd,
        "ridge_sk": ridge_by_lambda_sk,
    }[
        method
    ]  # loss of the regressor

    ridge_2 = {
        "plain": ridge,
        "svd": ridge_svd,
        "ridge_sk": ridge_sk,
    }[
        method
    ]  # solver for the weights

    nL = lambdas.shape[0]  # get number of hyperparameter (lambdas) from setting
    r_cv = np.zeros((nL, train_data.shape[1]))  # loss matrix

    group_kfold = GroupKFold(n_splits=data_config.n_splits)

    for icv, (trn, val) in enumerate(
        group_kfold.split(train_features, train_data, groups)
    ):

        if do_zscore:
            cost = ridge_1(
                zscore(train_features[trn]),
                zscore(train_data[trn]),
                zscore(train_features[val]),
                zscore(train_data[val]),
                lambdas=lambdas,
            )  # loss of regressor 1
        else:
            cost = ridge_1(
                train_features[trn],
                train_data[trn],
                train_features[val],
                train_data[val],
                lambdas=lambdas,
            )  # loss of regressor 1

        if do_plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.imshow(cost, aspect="auto")
        print("add cost")
        r_cv += cost
        # print(f"r_cv: {r_cv}")


    if do_plot:  # show loss
        plt.figure()
        plt.imshow(r_cv, aspect="auto", cmap="RdBu_r")

    print("estimating argmin_lambda")
    argmin_lambda = np.argmin(r_cv, axis=0)  # pick the best lambda
    print(f"argmin_lambda: {argmin_lambda}")

    weights = np.zeros(
        (train_features.shape[1], train_data.shape[1])
    )  # initialize the weight
    print(f"lambdas: {lambdas}")
    picked_lambdas = []

    for idx_lambda in range(
        lambdas.shape[0]
    ):  # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        print(f"idx_vox: {idx_vox.shape}")
        print(f"idx_vox: {idx_vox}")

        if np.any(idx_vox):
            print("Some values are True in idx_vox")
            # Continue with the rest of the operations
            print(f"train_data[:, idx_vox]: {train_data[:, idx_vox]}")

            weights[:, idx_vox], model = ridge_2(
                train_features, train_data[:, idx_vox], lambdas[idx_lambda]
            )

            print(f"weights: {weights}")
        picked_lambdas.append(lambdas[idx_lambda])


    return weights, picked_lambdas, model
