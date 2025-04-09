from sklearn import metrics


def get_scores(y_true, y_pred):
    return {
        "accuracy": round(metrics.accuracy_score(y_true, y_pred), 4),
        "precision": round(metrics.precision_score(y_true, y_pred, average="macro"), 4),
        "recall": round(metrics.recall_score(y_true, y_pred, average="macro"), 4),
        "f1": round(metrics.f1_score(y_true, y_pred, average="macro"), 4),
    }
