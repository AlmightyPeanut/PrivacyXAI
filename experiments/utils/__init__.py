import marshal
import pickle
import types
from functools import partial
from pathlib import Path

import tqdm
from torch.multiprocessing.pool import Pool

MODEL_CHECKPOINTS_PATH = Path(__file__).parent.parent / 'model_checkpoints'
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


def run_task(*args, **kwargs):
    marshaled = kwargs.pop('marshaled_func')
    func = marshal.loads(marshaled)
    pickled_closure = kwargs.pop('pickled_closure')
    pickled_closure = pickle.loads(pickled_closure)

    restored_function = types.FunctionType(func, globals(), closure=create_closure(func, pickled_closure))
    return restored_function(*args, **kwargs)


def create_closure(func, original_closure):
    indent = " " * 4
    closure_vars_def = f"\n{indent}".join(f"{name}=None" for name in func.co_freevars)
    closure_vars_ref = ",".join(func.co_freevars)
    dynamic_closure = "create_dynamic_closure"
    s = (f"""
def {dynamic_closure}():
    {closure_vars_def}
    def internal():
        {closure_vars_ref}
    return internal.__closure__
    """)
    exec(s)
    created_closure = locals()[dynamic_closure]()
    for closure_var, value in zip(created_closure, original_closure):
        closure_var.cell_contents = value
    return created_closure


def wrap_task(pool: Pool, task, generator):
    closure = task.__closure__
    pickled_closure = pickle.dumps(tuple(x.cell_contents for x in closure))
    marshaled_func = marshal.dumps(task.__code__)
    with tqdm.tqdm(total=len(generator)) as pbar:
        for _ in pool.imap_unordered(partial(run_task, marshaled_func=marshaled_func, pickled_closure=pickled_closure),
                                     generator):
            pbar.update()
