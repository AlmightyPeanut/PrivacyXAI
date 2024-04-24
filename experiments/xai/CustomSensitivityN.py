from typing import Any

import numpy as np
from quantus import SensitivityN


class CustomSensitivityN(SensitivityN):
    def custom_postprocess(
            self,
            x_batch: np.ndarray,
            **kwargs,
    ) -> None:
        max_features = int(
            self.n_max_percentage * np.prod(x_batch.shape[1:]) // self.features_in_step
        )

        # Get pred_deltas and att_sums from result list.
        sub_results_pred_deltas: list[Any] = [
            r["pred_deltas"] for r in self.evaluation_scores
        ]
        sub_results_att_sums: list[Any] = [
            r["att_sums"] for r in self.evaluation_scores
        ]

        # Re-arrange sub-lists so that they are sorted by n.
        sub_results_pred_deltas_l: dict[int, Any] = {k: [] for k in range(max_features)}
        sub_results_att_sums_l: dict[int, Any] = {k: [] for k in range(max_features)}

        for k in range(max_features):
            for pred_deltas_instance in sub_results_pred_deltas:
                sub_results_pred_deltas_l[k].append(pred_deltas_instance[k])
            for att_sums_instance in sub_results_att_sums:
                sub_results_att_sums_l[k].append(att_sums_instance[k])

        # Compute the similarity for each n.
        self.evaluation_scores = [
            self.similarity_func(
                a=sub_results_att_sums_l[k], b=sub_results_pred_deltas_l[k]
            )
            for k in range(max_features)
        ]
