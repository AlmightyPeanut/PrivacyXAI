from experiments.Experiment import Experiment

PARAMETERS_TO_EXPLORE = {
    'number_of_clients': [5, 10, 20, 50],
    'epsilons': [.5, 1., 3.]
}

if __name__ == '__main__':
    experiment = Experiment(**PARAMETERS_TO_EXPLORE)
    experiment.run_model_training()
    experiment.run_xai_evaluation()
    # experiment.run_mia()
