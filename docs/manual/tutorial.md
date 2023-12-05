(tutorial)=
# Tutorial

% This tutorial focuses on using MERLIN as a framework. If you are just interested in the simulation environment, you can skip to {ref}`environments`.

## Configuration

MERLIN uses [Hydra](https://hydra.cc/docs/intro/) to manage the configurations for environments and solvers. It is not uncommon to have over a hundred parameters in an experiment given the complexity of simulation environments and learning algorithms. Hydra allows us to decompose complex configuarations into a collection of yaml files, and gives us the flexibility to override any parameter on the command line if needed.

MERLIN's hydra configuration files are stored under `configs/hydra`. They can be used with the {ref}`entry scripts <entry_scripts>` described below.

(entry_scripts)=
## Entry scripts

Entry scripts are stored under `bin/py`. Here we discuss the training and evaluation scripts, which are general scripts that can be used for all gym environments and learning algorithms. Both scripts are structured as hydra applications, and have an accompanying hydra config file. We recommend taking a look at [hydra's documentation](https://hydra.cc/docs/intro/) if you are not familiar with hydra.

### Train script
The train script is located at `bin/py/train.py`, and its corresponding configuration is `configs/hydra/train.yaml`. This script expects the user to overide the `env` (environment) and `solver` parameters on the command line. The following is an example command, which uses hydra's parameter syntax.

```
python bin/py/train.py env=classic_mlb solver=sb3_ppo train_dir=logs/train_logs/my_exp
```

- `classic_mlb` refers to the configuration file `configs/hydra/env/classic_mlb.yaml`, which specifies the parameters for a mobility load balancing simulator.  The action space provided by the `classic_mlb` environment will allow a learning algorithm to control the load balancing parameters, and the environment's reward is based on how evenly the traffic load is distributed among the available frequency bands. See {ref}`environments` for more information on the simulator.
- `sb3_ppo` refers to the configuration file `configs/hydra/solver/sb3_ppo`, which specfies the parameters for [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/)'s implementation of Proximal Policy Optimization (PPO), a popular reinforcement learning algorithm.
- `train_dir` specifies the path where the trained model will be saved. If `train_dir` is not specified, a directory will automatically be created under `logs/train_logs`. 

Please refer to hydra's documentation for information on how multiple configuration files are composed together.

### Evaluation script

The evaluation script is located at `bin/py/evaluate.py`, and its corresponding configuration is `configs/hydra/evaluate.yaml`. This script expects the user to overide the `env` (environment), `solver`, and `evaluator` parameters on the command line. The following is an example command.

```
python bin/py/evaluate.py env=classic_mlb solver=qsim_default evaluator=classic_mlb eval_dir=logs/evals/test
```

- `qsim_default` is a solver that does not perform any load balancing, and can be viewed as a baseline method. 
- `evaluator=classic_mlb` evaluator refers to `configs/hydra/evaluator/classic_mlb`, which will evaluate the solver and output key performance indicators (i.e. standard deviation of the load among frequency bands). `eval_dir` specifies the path where the evaluation results will be saved. If `eval_dir` is not specified, a directory will automatically be created under `logs/evals`.

The following example shows how to evaluate both the baseline method and a trained PPO model. Note that we override the solver's `params_from` parameter to specify the location where our trained model is stored.

```
python bin/py/evaluate.py env=classic_mlb solver=[qsim_default,sb3_ppo] solver.sb3_ppo.params_from=logs/train_logs/my_exp evaluator=classic_mlb eval_dir=logs/evals/test
```

## Extending MERLIN
MERLIN is designed to be easily extendable, allowing you to add custom environments, solvers, and evaluators. You will need to be familiar with the [OpenAI gym interface](https://www.gymlibrary.dev/) and [Hydra](https://hydra.cc/docs/intro/).

Adding an environment involves implementing a class that inherits from `gym.Env`. Place the Python module in `src/py/saic5g/envs`, and the corresponding hydra configuration in `configs/hydra/env`. 

Adding a solver involves implementing a class that inherits from `saic5g.solvers.solvers.SolverBase`, which is located in `src/py/saic5g/solvers/solvers.py`. Refer to the docstrings or the API reference for more details regarding SolverBase. Place your Python module under `src/py/saic5g/solvers`, and the corresponding hydra configuration in `configs/hydra/solver`.

Adding an evaluator involves implementing a class that inherits from `saic5g.evaluators.evaluators.EvaluatorBase`, which is located in `src/py/saic5g/evaluators/evaluators.py`. Refer to the docstrings or the API reference for more details regarding EvaluatorBase. Place your Python module under `src/py/saic5g/evaluators`, and the corresponding hydra configuration in `configs/hydra/evaluator`.

%Use MERLIN **like this**!

%```{admonition} Here's my title
%:class: warning
%
%Here's my admonition content
%```

%(section-two)=
%## Here's another section

%And some more content.

% This comment won't make it into the outputs!
%And here's {ref}`a reference to this section <section-two>`.
%I can also reference the section {ref}`section-two` without specifying my title.
