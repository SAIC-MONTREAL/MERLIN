# MERLIN

The Montreal Environment for Reinforcement Learning and Intelligent Networks (MERLIN) is a Python framework that aims to make it easy for AI scientists to apply machine learning to telecommunication problems, with a focus on cellular networks. MERLIN serves three key roles:

1. MERLIN comes with network simulators for cellular networks that are implemented in Python. Some of the simulators offer OpenAI gym interfaces, which offers a standard way for reinforcement learning algorithms to interact with the simulator.
2. MERLIN offers a standard interface for solvers, which can be used to wrap algorithms in a uniform interface. This makes it easy to directly compare solvers on the same problem, and evaluate using the same set of key performance indicators (KPIs).
3. MERLIN offers integration of experimental management tools to facilitate tasks such as configuration management, hyperparameter tuning and expeiriment monitoring.

# At a glance

MERLIN is a Python framework that facilitates the development of AI-based methods for cellular network management. MERLIN integrates simulators, algorithms, and evaluation benchmarks in a single package.

The following command uses Stable Baselines 3's implementation of Proximal Policy Optimization (PPO), a reinforcement learning (RL) algorithm, to perform mobility load balancing in a simulated cellular network. More specifically, we adjust the cell site parameters to balance the user equipments (UEs) (i.e. cell phones) among the available frequencies.

```bash
python bin/py/train.py env=classic_mlb solver=sb3_ppo solver.sb3_ppo.total_timesteps=200000 train_dir=logs/train_logs/my_experiment
```

MERLIN uses [Hydra](https://hydra.cc/docs/intro/) to manage configuration. The command line syntax you see here is Hydra's syntax.

The following command evaluates the trained model, along with a baseline (qsim_default) that does not perform any load balancing.

```bash
python python bin/py/evaluate.py env=classic_mlb solver=[qsim_default,sb3_ppo] solver.sb3_ppo.params_from=logs/train_logs/my_experiment evaluator=classic_mlb evaluator.mlb_evaluator.n_trials=10 eval_dir=logs/evals/test_my_experiment
```

You should get output like the following. Here, we see that the RL-basd method achieves higher performance than the baseline. In this case, performance is measured in average throughput among cells.

```text
         solver  performance  load_deviation   dropped
0  qsim_default     9.485811        0.444831  6.764646
1       sb3_ppo    13.716233        0.350571  3.052525
```
