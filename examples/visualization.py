import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

from giwerm.covariate_shift_adaptation import CovariateShiftAdaptation
from giwerm.model_selection import train_test_split


def grid_search(x, y, Lambda, Alpha):
    np.random.seed(512)
    cvs = CovariateShiftAdaptation()
    train_x, train_y, test_x, test_y = train_test_split(x, y)
    print(train_x.shape, test_x.shape)

    D = np.zeros((len(Alpha), len(Lambda)))
    for i, alpha in enumerate(Alpha):
        for j, lmd in enumerate(Lambda):
            p, q = get_prob(test_x, train_x)
            weights = cvs.generalized_importance_weight(p, q, alpha=alpha, lmd=lmd) / p

            clf = SVC(kernel='rbf')
            clf.fit(train_x, train_y, sample_weight=weights)
            error = np.sum(clf.predict(test_x) != test_y) / len(test_x)
            D[i][j] = error
    return D

def get_prob(X, Z):
    X = pd.DataFrame(X)
    Z = pd.DataFrame(Z)
    X['is_z'] = 0 # 0 means test set
    Z['is_z'] = 1 # 1 means training set
    XZ = pd.concat([X, Z], ignore_index=True, axis=0 )

    labels = XZ['is_z'].values
    XZ = XZ.drop('is_z', axis=1).values
    X, Z = X.values, Z.values

    clf = RandomForestClassifier(max_depth=2)

    predictions = np.zeros(labels.shape)
    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=1234)
    for fold, (train_idx, test_idx) in enumerate(skf.split(XZ, labels)):
        X_train, X_test = XZ[train_idx], XZ[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs

    predictions_Z = predictions[len(X):]
    q = 1 - predictions_Z

    return predictions_Z, q

def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore


def main():
    data = load_iris()
    x, y = data.data, data.target
    x = zscore(x)
    Lambda = np.arange(0, 1.01, 0.1)
    Alpha = np.arange(1, 4.1, 0.3)
    D = grid_search(x, y, Lambda, Alpha)

    fig = plt.figure(figsize=(12,8))
    ax = Axes3D(fig)
    ax.view_init(30, 40)
    ax.plot_surface(X, Y, gaussian_filter(D, sigma=2), cmap="viridis", alpha=0.6)
    ax.set_xlabel(r"$\lambda$", fontsize=24)
    ax.set_ylabel(r"$\alpha$", fontsize=24)
    ax.set_zlabel("error", fontsize=24)
    plt.savefig("digits_surface.pdf")
    plt.show()


if __name__ == "__main__":
    main()