# Introduction

The Montreal Environment for Reinforcement Learning and Intelligent Networks (MERLIN) is a Python framework that aims to make it easy for AI scientists to apply machine learning to telecommunication problems, with a focus on cellular networks. MERLIN serves three key roles:

1. MERLIN comes with network simulators for cellular networks that are implemented in Python. Some of the simulators offer OpenAI gym interfaces, which offers a standard way for reinforcement learning algorithms to interact with the simulator.
2. MERLIN offers a standard interface for solvers, which can be used to wrap algorithms in a uniform interface. This makes it easy to directly compare solvers on the same problem, and evaluate using the same set of key performance indicators (KPIs).
3. MERLIN offers integration of experimental management tools to facilitate tasks such as configuration management, hyperparameter tuning and expeiriment monitoring.