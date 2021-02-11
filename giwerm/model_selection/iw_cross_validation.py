from sklearn.model_selection import StratifiedKFold


def iw_cross_validation(clf, metric, inputs, targets, sample_weight, n_splits=20, random_state=100):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    losses = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(inputs, targets)):
        X_train, X_test = inputs[train_idx], inputs[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        clf.fit(X_train, y_train)
        loss = metric(clf.predict(X_test) * sample_weight, y_test * sample_weight)
        losses.append(loss)
