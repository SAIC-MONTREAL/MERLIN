(environments)=
# Simulators and Environments

## Introduction
Network simulators play a key role in the development of AI algorithms for network management. We use the term **simulator** to refer to the Python modules that simulate cellular networks, including the simulation of cell sites, user equipment (UEs), and radio signal propagation. An **environment** refers to a Python class that wraps a simulator and conforms to the [OpenAI gym interface](https://www.gymlibrary.dev/index.html). 

Gym environments are designed to facilitate the application of reinforcement learning, and is a key learning paradigm supported by MERLIN. MERLIN's main scripts (```bin/py/train.py, bin/py/evaluate.py```), as well as much of the existing infrasture cater to reinforcement learning.

If you are not interested in using reinforcement learning, you can directly interface with the simulator classes. 

## Simulators
MERLIN currently implements a number simulators. These simulators are geared towards For more details regarding the simulators, please refer to the docstrings in the source code.

- QSim: Qsim stands for quick simulator, which trades off fidelty for speed. See `src/py/saic5g/simulators/qsim.py`.
- DiffQsim: A differentiable version of QSim that is implemented using PyTorch. See `src/py/saic5g/simulators/diffqsim.py`.
- R5GSim: A more realistic simulator than QSim that implements 5G features. See `bin/py/r5gsim.py` and `src/py/saic5g/simulators/R5GSim`.
  - Hierarchical control level (Core network level, basestation level, cell level, band level)
  - Supports multiple layouts based on the map dimensions and the coverage area per BS, and different number of sectors per cell.
  - Modularity: can change any block in the code and improve it without changing the other blocks.
  - Can support up to 1ms simulation resolution, but there is a tradeoff with the simulation time
  - Poisson arrivals for the traffic and random waypoint movement model.
  - Proportional fairness scheduler (More can be added easily with time-frequency allocation)
  - Assumes beamforming in the cell, and accurate interference calculation from adjacent cells.
  - Accurate data rate calculation according the latest 5G standards.
  - Supports mmWave bands and MIMO from a high level.

## Environments
Environments wrap simulators in the Open AI gym interface, allowing them to be easily used used for reinforcement learning. MERLIN currently offers a number of environments based on QSim. Environments based on R5GSim are not currently available, but can be added easily.

The following environments are available. Please see the API reference or docstrings for these classes for additional details:

- PerUeHandover (saic5g.envs.per_ue_base.PerUeHandover): Environment which supports fine-grained control of UE-cell assignments. Since UE-level control often leads to very large action spaces which are difficult to tackle with most RL methods, this class is intended to be used as a base for others with more manageable action spaces.
- A3BasedMLB (saic5g.envs.classic_mlb.A3BasedMLB): Environment for mobility load balancing (MLB) based on A3 event. The implementation is based on the paper [Load Balancing in Cellular Networks: A Reinforcement Learning Approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9045699&tag=1).
- DiffQsimEnv (saic5g.envs.diff_qsim_env.DiffQSimEnv): Environment designed to be used with differentiable QSim.