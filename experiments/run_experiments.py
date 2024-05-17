from experiments.Experiment import Experiment

PARAMETERS_TO_EXPLORE = {
    'number_of_clients': [5],
    'epsilons': [0.5, 3]
}

if __name__ == '__main__':
    experiment = Experiment(**PARAMETERS_TO_EXPLORE)
    experiment.run_model_training()
    # experiment.run_xai_evaluation()
    # experiment.run_mia()
