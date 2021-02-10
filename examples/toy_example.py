import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from giwerm.covariate_shift_adaptation import CovariateShiftAdaptation


def main():
    np.random.seed(100)
    n_train = 400
    n_test = 200
    x = 11 * np.random.random(n_train) - 6.0
    y = x**2 + 10*np.random.random(n_train) - 5
    trainset = np.c_[x, y]

    x = 2*np.random.random(n_test) - 6.0
    y = x**2 + 10*np.random.random(n_test) - 5
    testset = np.c_[x, y]

    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(12, 8))
    assert fig is not None
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.scatter(trainset[:, 0], trainset[:, 1], marker='o', s=40, c='b', label='train data', alpha=0.5)
    plt.scatter(testset[:, 0], testset[:, 1], marker='x', s=40, c='r', label='test data', alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlim([-6, 5])
    plt.xlabel(r"$x$", size=24)
    plt.ylabel(r"$y$", size=24)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.show()

    train_df = pd.DataFrame(trainset)
    test_df = pd.DataFrame(testset)
    test_df['is_train'] = 0  # 0 means test set
    train_df['is_train'] = 1  # 1 means training set
    data = pd.concat([test_df, train_df], ignore_index=True, axis=0)

    labels = data['is_train'].values
    data = data.drop('is_train', axis=1).values
    u, v = test_df.values, train_df.values

    cvs = CovariateShiftAdaptation()

    clf = RandomForestClassifier(max_depth=2)
    p, q = cvs.predict_densities(clf, data, labels,
                                 np.arange(len(test_df), len(data)))
    iw = q / p
    gw = cvs.generalized_importance_weight(p, q, lmd=0.98, alpha=4)

    X_train, y_train = trainset[:,0], trainset[:,1]
    X_test, y_test = testset[:,0], testset[:,1]

    clf_unweighted = LinearRegression()
    clf_unweighted.fit(X_train.reshape(-1,1), y_train)

    clf_iwerm = LinearRegression()
    clf_iwerm.fit(X_train.reshape(-1,1), y_train, sample_weight=iw)

    clf_igiwerm = LinearRegression()
    clf_igiwerm.fit(X_train.reshape(-1,1), y_train, sample_weight=gw)

    plt.rcParams['text.usetex'] = True

    fig = plt.figure(figsize=(12, 8))
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    a = np.arange(-6, 4, 0.5)
    plt.scatter(trainset[:,0], trainset[:,1], marker='o', s=40, c='b', label='train', alpha=0.5)
    plt.scatter(testset[:,0], testset[:,1], marker='x', s=40, c='r', label='test', alpha=0.5)
    plt.plot(a, clf_unweighted.predict(a.reshape(-1,1)), c='black', label="ERM", linestyle="dashed", lw=3, alpha=0.7)
    plt.plot(a,clf_iwerm.predict(a.reshape(-1,1)), c='black', label="IWERM", linestyle="dotted", lw=3, alpha=0.7)
    plt.plot(a,clf_igiwerm.predict(a.reshape(-1,1)), c='black', label="IGIWERM", lw=3, alpha=0.7)
    plt.xlim([-6,-3])
    plt.legend(fontsize=21)
    plt.xlabel(r"$x$", size=24)
    plt.ylabel(r"$y$", size=24)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.show()

    mse_unweighted = mean_squared_error(clf_unweighted.predict(X_test.reshape(-1,1)), y_test)
    mse_iwerm = mean_squared_error(clf_iwerm.predict(X_test.reshape(-1,1)), y_test)
    mse_igiwerm = mean_squared_error(clf_igiwerm.predict(X_test.reshape(-1,1)), y_test)

    print(f"MSE (ERM): {mse_unweighted}")
    print(f"MSE (IWERM): {mse_iwerm}")
    print(f"MSE (IGIWERM): {mse_igiwerm}")

if __name__ == "__main__":
    main()
