import os
import numpy as np
import pandas as pd
import gym
import copy
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, get_original_cwd
from saic5g.utils import hydra_utils
from tensorflow.python.framework.ops import enable_eager_execution


def evaluate_solvers(evaluator, solvers, out_dir):
    '''
    Iterate over the specified solvers and use the evaluator to
    evaluate each one. The performance of each solver will be saved
    to eval_dir/evaluator_name/performance.csv
    '''
    os.makedirs(out_dir, exist_ok=True)
    performance = defaultdict(lambda: [])
    for name, solv in solvers:
        solv.name = name  # Needed by the evaluator
        solv_eval_dir = os.path.join(out_dir, name)
        performance['solver'].append(name)
        analyzed_results = evaluator.evaluate(solv, solv_eval_dir)
        if np.isscalar(analyzed_results):
            performance['performance'].append(analyzed_results)
        elif isinstance(analyzed_results, dict):
            assert 'performance' in analyzed_results, 'Dictionary returned by the evaluator should contain "performance"'
            for k, v in analyzed_results.items():
                performance[k].append(v)
        else:
            raise ValueError('Format of evaluation result not understood')
    performance_df = pd.DataFrame(performance)
    print(performance_df)
    performance_df.to_csv(os.path.join(out_dir, 'performance.csv'), index=False)


@hydra.main(config_path="../../configs/hydra/", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig):
    if cfg['eager']:
        enable_eager_execution()

    _, env_config = hydra_utils.get_one(cfg['env'], max_selection=1)

    def envgen():
        return instantiate(copy.deepcopy(env_config))

    _, evaluator_config = hydra_utils.get_one(cfg['evaluator'], max_selection=1)
    evaluator = instantiate(evaluator_config, envgen)

    base_out_dir = cfg['eval_dir']
    eval_out_dir = os.path.join(base_out_dir, evaluator.name)

    solvers = []
    for name, solver_config in cfg['solver'].items():
        solver_obj = instantiate(solver_config, envgen)
        solvers.append((name, solver_obj))

    evaluate_solvers(evaluator, solvers, eval_out_dir)


if __name__ == "__main__":
    evaluate()
