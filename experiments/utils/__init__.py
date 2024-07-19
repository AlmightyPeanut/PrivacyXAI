from pathlib import Path

import tqdm
from torch.multiprocessing.pool import Pool

MODEL_CHECKPOINTS_PATH = Path(__file__).parent.parent.parent / 'model_checkpoints'
DATASET_PATH = Path(__file__).parent.parent.parent / 'data'
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results'

PRINT_WIDTH = 40


def remove_model_prefix(state_dict, prefix: str):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # remove prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict
