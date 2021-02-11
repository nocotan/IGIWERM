import GPyOpt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from giwerm.geometric_functions import alpha_geodesic


class CovariateShiftAdaptation(object):
    def __init__(self, bounds=None):
        self.bounds = bounds

    def predict_densities(self, clf, data, labels, target_idx):
        predictions = np.zeros(labels.shape)
        skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=1234)

        for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
            X_train, X_test = data[train_idx], data[test_idx]
            y_train, _ = labels[train_idx], labels[test_idx]

            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            predictions[test_idx] = probs

        p = predictions[target_idx]
        q = 1 - p

        return (p, q)

    def generalized_importance_weight(self, p, q, lmd, alpha):
        return alpha_geodesic(p, q, lmd=lmd, alpha=alpha) / p

    def optimize(self, p, q, clf, metric, train_x, train_y, val_x, val_y):
        def objective(params):
            lmd = params[:, 0][0]
            alpha = params[:, 1][0]
            w = self.generalized_importance_weight(p, q, lmd, alpha)
            if np.isnan(w).any() or np.isinf(w).any():
                return 1e9
            try:
                clf.fit(train_x, train_y, sample_weight=w)
            except Exception:
                return 1e9

            error = metric(clf.predict(val_x), val_y)
            return error

        bopt = GPyOpt.methods.BayesianOptimization(f=objective, domain=bounds, initial_design_numdata=0)
        e_00 = objective(np.array([[0.0, 0.0]]))
        e_11 = objective(np.array([[1.0, 1.0]]))
        bopt.X = np.array([[1.0, 1.0], [0.0, 0.0]])
        bopt.Y = np.array([[e_11], [e_00]])
        bopt.run_optimization(max_iter=100)
        lmd, alpha = bopt.x_opt

        return self.generalized_importance_weight(p, q, lmd, alpha)
