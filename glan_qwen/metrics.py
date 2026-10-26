"""Metric compatibility with csd/main_12_qwen_1.py.

Preserve its observed-class macro average and its zero per-class fallback when
fewer than three classes occur in the union of predictions and labels.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def scores(labels, preds, grouped=False):
    macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    per_class = f1_score(y_true=labels, y_pred=preds, average=None)
    if len(per_class) >= 3:
        favor, against, neutral = per_class[:3]
    else:
        favor = against = neutral = 0.0
    avg = (favor + neutral + against) / 3.0 if grouped else (favor + against) / 2.0
    return dict(f1_macro=float(macro), f1_favor=float(favor),
                f1_against=float(against), f1_neutral=float(neutral),
                f1_avg=float(avg), accuracy=float(accuracy_score(labels, preds)))


def metrics_by_scope(y_true, y_pred, target_types, prefix):
    y_true, y_pred, target_types = map(np.asarray, (y_true, y_pred, target_types))
    scopes = dict(mixed=np.ones_like(target_types, dtype=bool),
                  c=target_types == 'c', n=target_types == 'n')
    result = {}
    for name, mask in scopes.items():
        values = (scores(y_true[mask], y_pred[mask], grouped=True) if mask.sum() else
                  dict.fromkeys(('f1_macro', 'f1_favor', 'f1_against',
                                 'f1_neutral', 'f1_avg', 'accuracy'), 0.0))
        result.update({f'{prefix}_{name}_{key}': value for key, value in values.items()})
    return result
