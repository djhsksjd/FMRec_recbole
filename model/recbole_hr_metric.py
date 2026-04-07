"""
Expose Hit Rate (HR@K) under the name `hr` for RecBole configs.

RecBole's built-in class is named `Hit`; its docstring defines the same formula as HR@K.
This subclass only changes output keys to `hr@k` (lowercase, consistent with `hit@k` / `ndcg@k`).
"""

from __future__ import annotations

from recbole.evaluator import metrics as rec_metrics
from recbole.evaluator import register as rec_register


class HR(rec_metrics.Hit):
    """Same computation as recbole.evaluator.metrics.Hit; result keys are hr@{k}."""

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = {}
        avg_result = result.mean(axis=0)
        for k in self.topk:
            key = "hr@{}".format(k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict


def register_hr_metric() -> None:
    """Patch RecBole's metric registry so `metrics: [hr, ...]` works."""
    name = "hr"
    rec_register.metrics_dict[name] = HR
    rec_register.metric_information[name] = HR.metric_need
    rec_register.metric_types[name] = HR.metric_type
