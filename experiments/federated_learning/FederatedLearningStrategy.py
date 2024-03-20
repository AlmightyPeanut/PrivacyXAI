import os
from typing import Union, Tuple

import flwr as fl
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from .FederatedLearningClient import scalar
from ..model.ModelManager import ModelManager


class FederatedLearningStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parameters_of_models = None

    def aggregate_fit(
        self,
        server_round: int,
        results: list[(ClientProxy, FitRes)],
        failures: list[Union[Tuple[ClientProxy, FitRes] | BaseException]],
    ) -> (Parameters | None, dict[str, scalar]):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Updating round {server_round} aggregated parameters...")
            self._parameters_of_models = aggregated_parameters

        return aggregated_parameters, aggregated_metrics

    def get_parameters_of_models(self):
        return parameters_to_ndarrays(self._parameters_of_models)

    def save_models(self, path: os.PathLike) -> None:
        if self._parameters_of_models is None:
            raise Exception("No models trained yet!")

        model_manager = ModelManager()
        model_manager.set_parameters_of_models(parameters_to_ndarrays(self._parameters_of_models))
        model_manager.save_models(path)


