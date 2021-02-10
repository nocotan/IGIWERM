import numpy as np
from sklearn.model_selection import StratifiedKFold


class CovariateShiftAdaptation(object):
    def __init__(self):
        pass

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
