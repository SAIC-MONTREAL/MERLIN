'''
Solvers based on Stable Baselines3 (Sb3)

Documentation: https://stable-baselines3.readthedocs.io/en/master/

At the time of implementation, Stable Baselines3 is still in beta, and so there
may be breaking changes incoming.
'''
from copy import deepcopy
import os
import glob
from pathlib import Path

import numpy as np
import json
from ray import tune
import stable_baselines3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from saic5g.envs.variant import BaseEnvVariantManager
from .solvers import SolverBase
from saic5g.utils.format import rupdate
import torch
from saic5g.solvers.sb3_policy_networks.lstm_feature_extractor import CustomLSTM

class TuneReportCallback(EvalCallback):
    """
    Callback used for reporting to Tune.

    Based on:
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/8d225c50447d8f0ba6de20f6fe74fd1fbe15d82f/utils/callbacks.py#L11
    """
    def _on_step(self):
        super()._on_step()
        tune.report(num_timesteps=self.num_timesteps, reward=self.last_mean_reward)
        return True

class Sb3Solver(SolverBase):
    """A wrapper for Stable Baselines 3 solvers.

    Args:
        total_timesteps (int): Number of timesteps to train for.
        checkpoint_freq (int): The timestep interval for saving model checkpoints.
        eval_freq (int): The timestep interval for evaluating and updating the learning curve.
        n_eval_episodes (int): The number of episodes to run to evaluate the the current model.
        num_envs (int): The number of parallel environments to use for training.
        verbose (int): 0 no output, 1 info, 2 debug.
        policy_cls (class): The policy class that implements the policy network. Should be a subclass of BasePolicy.
        agent_cls (class): The agent class that implements the reinforcement learning algorithm. Should be a subclass of BaseAlgorithm.
        agent_kwargs (dict): The keyword arguments that will be passed into agent_cls when instantiating it.
            In addition to the argument supported by the agent class, we offer two additional parameters: n_layers and neurons_per_layer, both of which are integers.
            These two arguments offer a convenient way to specify the policy network architecture. They should only be specified
            if policy_kwargs is not in agent_kwargs.
    """
    def __init__(self, *args,
                       total_timesteps=100000,
                       checkpoint_freq=500,
                       eval_freq=500,
                       n_eval_episodes=5,
                       num_envs=1,
                       verbose=1,
                       policy_cls=None,
                       agent_cls=None,
                       agent_kwargs={},
                        **kwargs
                 ):
        self._model_path = None
        self._verbose = verbose
        self._policy_cls = policy_cls
        self._agent_cls = agent_cls
        self._agent_kwargs = agent_kwargs
        self._agent_kwargs.update({
            'tensorboard_log': None,
            'verbose': self._verbose
        })
        self._total_timesteps = total_timesteps
        self._restore_path = None
        self._checkpoint_freq = checkpoint_freq
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._num_envs = num_envs
        self._agent_cls, self._policy_cls = self.get_agent_cls_and_policy()
        self._eval_agent = None
        super().__init__(*args, **kwargs)

    def get_agent_cls_and_policy(self):
        """
        Return the agent class and policy class.

        Returns:
            (BaseAlgorithm, BasePolicy) pair that will be used by the solver.
            These are typically specified when instantiating this solver.
        """
        assert self._agent_cls is not None
        assert self._policy_cls is not None
        return self._agent_cls, self._policy_cls

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        if log_dir is not None:
            self._agent_kwargs['tensorboard_log'] = os.path.join(log_dir, 'tensorboard_log')
            self._model_path = os.path.join(log_dir, 'model')
            Path(self._model_path).mkdir(parents=True, exist_ok=True)

    def load_params(self, path, checkpoint=None):
        """
        This function lazily loads a trained model. This function simply sets
        self._restore_path. The actual loading will be done when _get_agent is called.
        
        Args:
            path (str): Either a directory containing model files (checkpoints, best_model.zip, sb3model.zip),
                or the direct path to a model file.
            checkpoint (int): This should only be specified if path is a directory. If specified, 
                this function will try to load the specified checkpoint.
        """
        if os.path.isdir(path):
            if checkpoint is None:
                best_model_path = os.path.join(path, 'model', 'best_model.zip')
                solution_model_path = os.path.join(path, 'model', 'sb3model.zip')
                if os.path.exists(best_model_path):
                    self._restore_path = best_model_path
                elif os.path.exists(solution_model_path):
                    self._restore_path = solution_model_path
                else:
                    raise ValueError('Did not find %s or %s' % (best_model_path, solution_model_path))
            else:
                globpath = os.path.join(path, 'model', '*_%d_*' % int(checkpoint))
                found = glob.glob(globpath)
                assert len(found) == 1, 'Did not find exactly one matching checkpoint for %d in %s' % (checkpoint, globpath)
                self._restore_path = found[0]
        elif os.path.isfile(path):
            assert checkpoint is None, 'Cannot use the checkpoint keyword argument when directly specifying a model file.'
            self._restore_path = path
        else:
            raise ValueError('Given path %s is not a file or a directory' % path)
        self._eval_agent = None

    def get_env(self, multi_proc=False):
        """Get an environment

        If multi_proc is True, and if num_envs is greater than 1, then
        this function will return a stable_baselines3.common.vec_env.SubprocVecEnv,
        which will allow the agent to interact with multiple environments simultaneously.
        Otherwise, it will return a gym.Env.

        Note that some agents do not support the use of multiple environments (i.e. SAC). For
        these agents, it is important to set num_envs=1 when instantiating this solver.

        Args:
            multi_proc (bool): Whether to instantiate a SubprocVecEnv. This is only
                applicable if num_envs is greater than 1. If num_envs
                is 1, then we just return a gym.Env.

        Returns:
            A gym.Env or a SubprocVecEnv
        """
        def make_env():
            env = self.env_gen()
            return Monitor(env)
        if multi_proc and self._num_envs > 1:
            vec_envs = SubprocVecEnv([make_env for i in range(self._num_envs)])
            return vec_envs
        else:
            return make_env()

    def _instantiate_new_agent(self, env, agent_kwargs):
        """Instantiate an agent

        This is a helper function that instantiates an agent given an
        environment and a set of keyword arguments. Subclasses can override
        this function to perform action just before or after instantiation.

        Args:
            env (gym.Env): A gym.Env or SubprocVecEnv.
            agent_kwargs (dict): Keyword arguments to be passed to the agent class
                when instantiating it.

        Returns:
            A learning agent, typically a subclass of BaseAlgorithm.
        """
        agent = self._agent_cls(self._policy_cls, env, **agent_kwargs)
        return agent

    def get_agent(self, multi_proc=False):
        """Get a learning agent

        This function first instantiates an environment, and then instantiates a
        learning agent, passing in the environment. If load_params has been called,
        then the agent will be loaded with the parameters of the saved model.
        Otherwise, a new agent will be created.

        Args:
            multi_proc (bool): This parameter is passed to get_env when instantiating
                the environment. See get_env for mor details.

        Returns:
            A learning agent, typically a subclass of BaseAlgorithm.
        """
        env = self.get_env(multi_proc=multi_proc)
        agent_kwargs = self._update_architecture(self._agent_kwargs)
        if self._verbose > 0:
            print('---------------------------------')
            print('Initializing agent with desired kwargs')
            print(agent_kwargs)
            print('---------------------------------')
        if self._restore_path is not None:
            if self._verbose > 0:
                print('Loading agent from', self._restore_path)
            load_kwargs = {}
            if hasattr(env, 'num_envs'):
                load_kwargs['n_envs'] = env.num_envs
            agent = self._agent_cls.load(self._restore_path, env, **load_kwargs)
            agent.tensorboard_log = agent_kwargs['tensorboard_log']
            if self._verbose > 0:
                print('The following loaded args should match desired args.')
                for k in agent_kwargs:
                    print(' -', k, ':', getattr(agent, k))
        else:
            if self._verbose > 0:
                print('Creating new agent')
            agent = self._instantiate_new_agent(env, agent_kwargs)
        return agent

    def get_callback_env(self):
        """
        Get the environment used for the callbacks
        (used in get_callback)
        """
        return self.get_env()

    def get_callback(self, tune=False):
        """Get the callbacks to be used during training

        The CallbackList returned by this function is used to instantiate the
        learning agent. Typically the CallbackList will include
        stable_baselines3.common.callbacks.EvalCallback and
        stable_baselines3.common.callbacks.CheckpointCallback.

        Args:
            tune (bool): Whether to use TuneReportCallback instead of EvalCallback.
                This should typically be set to True when using tune to perform
                hyperparameter tuning.

        Returns:
            A CallbackList containing a CheckpointCallback and one of
            EvalCallback/TuneReportCallback.
        """
        checkpoint_callback = CheckpointCallback(save_freq=self._checkpoint_freq,
                                                 save_path=self._model_path)
        # Separate evaluation env
        callback_env = self.get_callback_env()
        if tune:
            eval_callback_class = TuneReportCallback
            log_path = None
            best_model_save_path = None
        else:
            eval_callback_class = EvalCallback
            log_path = os.path.join(self.log_dir, 'eval_callback_result')
            best_model_save_path = self._model_path
        eval_callback = eval_callback_class(callback_env,
                                     best_model_save_path=best_model_save_path,
                                     log_path=log_path,
                                     eval_freq=self._eval_freq,
                                     n_eval_episodes=self._n_eval_episodes,
                                     deterministic=True,
                                     render=False)
        # Create the callback list
        if tune:
            callback = eval_callback
        else:
            callback = CallbackList([checkpoint_callback, eval_callback])
        return callback

    def get_action(self, obs):
        if self._eval_agent is None:
            assert self._restore_path is not None
            self._eval_agent = self.get_agent()
        # Even though SB3 doesn't support multiagent training, we can still
        # do multiagent evaluation by evaluating the policy on the observation
        # of each agent in obs.
        if isinstance(obs, dict):
            return {agent_id: self._eval_agent.predict(agent_obs, deterministic=True)[0]
                    for agent_id, agent_obs in obs.items()}
        return self._eval_agent.predict(obs, deterministic=True)[0]

    def train(self):
        """
        The environment and training agent will be instantiated, callbacks will
        be assigned to the agent, and the training process will be launched.
        """
        if self._verbose > 0:
            print('=== Sb3 starting training for %d timesteps' % self._total_timesteps)
            print('Number of envs', self._num_envs)
        agent = self.get_agent(multi_proc=True)
        callback = self.get_callback()
        agent.learn(total_timesteps=self._total_timesteps, callback=callback)

    def _post_process_architecture(self, param_dict):
        """Translate network architecture schema to be compatible with the chosen algorithm

        Different agents expect slightly different formats for the network
        architecture specification. _update_architecture populates policy_kwargs
        in the format expected by PPO. This function translates the format
        as needed for other algorithms. The given param_dict is modified in place.

        - PPO, A2C use List[Union[int, Dict[str, List[int]]]] and keys pi, vf
        - SAC, TD3, DDPG use Union[List[int], Dict[str, List[int]]] and keys pi, qf
          See: https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/torch_layers.py#L278

        Args:
            param_dict (dict): The keyword arguments that will be passed into the agent.
                This function will affect on param_dict['policy_kwargs'].

        Returns:
            The modified param_dict.
        """
        if self._agent_cls in [stable_baselines3.SAC, stable_baselines3.TD3, stable_baselines3.DDPG]:
            if 'policy_kwargs' in param_dict and 'net_arch' in param_dict['policy_kwargs'] and isinstance(param_dict['policy_kwargs']['net_arch'], list):
                param_dict['policy_kwargs']['net_arch'] = param_dict['policy_kwargs']['net_arch'][0]
                if 'vf' in param_dict['policy_kwargs']['net_arch']:
                    param_dict['policy_kwargs']['net_arch']['qf'] = param_dict['policy_kwargs']['net_arch'].pop('vf')
        else:
            pass
            # raise ValueError('agent_cls %s is not specified or is not supported' % str(self._agent_cls))
        return param_dict

    def _update_architecture(self, param_dict_orig):
        """Update network architecture based on n_layers and neurons_per_layer

        This function is designed to reformat self._agent_kwargs.
        We want to enable network depth and neurons per layer as hyperparams
        but the API accepts these in the form of a net architecture dictionary.
        Thus, this little bit of convenience code. The defaults are based on
        https://github.com/DLR-RM/stable-baselines3/blob/723b341c61d168e1460399592d5cebd4c6ef3cc8/stable_baselines3/common/policies.py#L404
        This has only been tested for ActorCriticPolicy.

        This function does not modify the given dictionary in place, but instead
        returns a copy of it with some elements modified. It reads and removes
        n_layers and neurons_per_layer, and adds policy_kwargs into the dictionary,
        which can then be passed into the agent.

        Note that if policy_kwargs is already present in the given dictionary,
        then specifying n_layers or neurons_per_layer will raise an exception.

        Args:
            param_dict_orig (dict): The keyword arguments for the agent.
                This function will make a copy of param_dict_orig and modify it as needed.

        Returns:
            A modified copy of param_dict_orig.
        """
        param_dict = deepcopy(param_dict_orig)
        
        if 'policy_kwargs' in param_dict:
            assert 'n_layers' not in param_dict and 'neurons_per_layer' not in param_dict, 'agent_kwargs should not contain n_layers and neurons_per_layer if policy_kwargs is directly specified'
            
            if 'feature_extractor' in param_dict['policy_kwargs']:
                feature_extractor = param_dict['policy_network']['feature_extractor']

                if feature_extractor is CustomLSTM:
                    return param_dict
            
            return self._post_process_architecture(param_dict)
            
        policy_kwargs = {}
        policy_network = None

        set_arch = False
        n_layers = 2
        neurons_per_layer = 64

        if 'n_layers' in param_dict:
            n_layers = param_dict['n_layers']
            del param_dict['n_layers']
            set_arch = True
        if 'neurons_per_layer' in param_dict:
            neurons_per_layer = param_dict['neurons_per_layer']
            del param_dict['neurons_per_layer']
            set_arch = True
        if 'activation_fn' in param_dict:
            activation_fn = param_dict['activation_fn']
            del param_dict['activation_fn']
            set_arch = True

        if set_arch:
            policy_kwargs = {
                'activation_fn': activation_fn, 
                'net_arch': [dict(pi=[neurons_per_layer]*n_layers, vf=[neurons_per_layer]*n_layers)]
            }
            param_dict['policy_kwargs'] = policy_kwargs
        
        return self._post_process_architecture(param_dict)

    def get_trainable(self):
        def trainable(config):
            env = self.get_env(multi_proc=True)
            agent_kwargs = deepcopy(self._agent_kwargs)
            agent_kwargs = rupdate(agent_kwargs, config)
            # update the path based on the config to avoid collisions.
            def _config2str(config):
                return '-'.join([str(key) + '=' + str(value) for key, value in config.items()])
            v = _config2str(config)
            agent_kwargs['tensorboard_log'] += '-'+ _config2str(config)
            agent_kwargs = self._update_architecture(agent_kwargs)
            callbacks = self.get_callback(tune=True)
            agent = self._instantiate_new_agent(env, agent_kwargs)
            agent.learn(total_timesteps=self._total_timesteps, callback=callbacks)
        return trainable

class Sb3MultiTaskSolver(Sb3Solver):
    def get_env(self, multi_proc=False, eval_callback=False):
        """Get environments containing multiple tasks

        This function behaves like that of the super class, except that
        it expects the environment generator (self.env_gen) to be an instance of
        BaseEnvVariantManager, and it generates a SubprocVecEnv containing the variants specified by the
        variant manager. num_envs must not be less than the number of available
        variants specified by the variant manager. If num_envs is greater than
        the number of variants, then the variants will be spread evenly across
        the available environments, leading to multiple instances of each variant.

        When using this function to get the environment for the callbacks,
        set eval_callback=True.
        This will return an environment for each variant and 
        evaluate the trained model on each variant. 
        """
        def make_env(i):
            def env_generator():
                assert isinstance(self.env_gen, BaseEnvVariantManager), 'Environment variant manager needs to be specified (i.e. +env_variant_mgr=sls_scenarios)'
                env = self.env_gen(index=i)
                return env
            return env_generator

        colsize = len(self.env_gen)
        if eval_callback:
            envs = SubprocVecEnv([make_env(i) for i in range(colsize)])
            return envs

        if multi_proc and self._num_envs > 1:
            generators = []
            for i in range(self._num_envs):
                # Distribute the number of workers (self._num_envs) evenly among
                # all the environments in the collection
                generators.append(make_env(i%colsize))
            vec_envs = SubprocVecEnv(generators)
            return vec_envs
        else:
            return make_env(0)()

    def get_callback_env(self):
        return self.get_env(eval_callback=True)


class Sb3DDPGSolver(Sb3Solver):
    def _instantiate_new_agent(self, env, agent_kwargs):
        from stable_baselines3.common.noise import NormalActionNoise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        agent = self._agent_cls(self._policy_cls, env, action_noise=action_noise, **agent_kwargs)
        return agent