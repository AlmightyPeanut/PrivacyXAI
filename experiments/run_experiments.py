from experiments.Experiment import Experiment

if __name__ == '__main__':
    experiment = Experiment()
    experiment.run_model_training()
    experiment.run_xai_evaluation()
    experiment.run_mia()
    experiment.run_model_training(federated_learning=False)
    experiment.run_xai_evaluation(use_federated_model=False)
    experiment.run_mia(use_federated_model=True)
