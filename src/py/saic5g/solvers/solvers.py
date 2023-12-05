"""
Solver and SolverRegistry classes.
"""

import git
import os

class SolverBase:
    """Base class for solvers

    SolverBase is intended unify all solvers and expose them over a unified
    interface. Solvers can be learning algorithms (i.e. reinforcement learning
    algorithms) or analytical algorithms (rule-based methods). However, we
    assume that all solvers will interact with a gym environment.
    To integrate a custom solver or a third party solver, you should
    extend this class and implement the member functions.

    Args:
        env_gen (function): Environment generator that returns a gym Env
        log_dir (str): Path of directory in which logging data should be saved
        params_from (str): Path to the directory where the learned model and checkpoints are stored.
            If not None, parameters will be loaded upon instantiation.
        checkpoint (int): The checkpoint to load

    Attributes:
        env_gen (function): A gym environment generator
    """

    def __init__(self, env_gen, log_dir=None, params_from=None, checkpoint=None):
        self.env_gen = env_gen
        self.set_log_dir(log_dir)
        if params_from is not None:
            self.load_params(params_from, checkpoint)

    def load_params(self, path, checkpoint=None):
        """Load solver parameters from a path.

        If this function is called prior to a rollout,
        the rollout should use the learned parameters.
        If it is called prior to a training session, the
        training session should be initialized from these params.
        If the solver is not learning-based, you can still use this function
        to load necessary parameters, or just override with a pass statement
        to prevent the NotImplementedError.

        Args:
            path (str): Path to the directory where the learned model and checkpoints are stored
            checkpoint (int): The checkpoint to load
        """
        raise NotImplementedError

    def param_out_path(self):
        """Return the path that we can pass to load_params.

        This function is typically used for learning-based methods. The train function
        should write the learned parameters to the path returned by this function.
        The default behavior here is to return return os.path.join(self.log_dir, 'model').

        Returns:
            Path where solver parameters are saved.
        """
        if self.log_dir is not None:
            return os.path.join(self.log_dir, 'model')
        raise ValueError('log_dir should be set before calling param_out_path')

    def set_log_dir(self, path):
        """Set the logging directory used by the solver.

        In many cases, the logging directory can simply be specified when
        instantiating the solver. This function offers the ability to update
        the log directory after instantiation. Subclasses can use this function
        to set up the logging directory (i.e. create necessary folders).

        Args:
            path (str): Path of directory in which logging data should be saved
        """
        self.log_dir = path

    def pre_eval(self, env):
        """Perform any necessary operation before this solver is evaluated on
        the given env.

        This function should be called after load_params, and
        before the first call to get_action. The given env should be created
        using self.env_gen, and is the same environment instance that will
        be used for the upcoming evaluation.

        One use case is to ensure the compatility between the env and this
        solver's loaded params. For example, we can ensure that the dimensions
        of the action and observation spaces are compatible with the learned model.
        For fixed parameters or rule-based methods, we can use this method to
        verify the semantics of the observation/action dimensions. For example,
        we can verify that the IP throughput of cells is present in the
        observation vector, if the env supports such functionality.
        This function does not return anything. It can raise an exception if
        a compatibility check fails.

        Rule-based methods that do not require training can also use this
        function to inspect the environment's settings and perform the
        necessary initialization prior to rollout.

        Args:
            env (gym.Env): This should be the same environment instance
                that will be used for the subsequent evaluation.
        """
        pass

    def get_action(self, obs):
        """Return a control action given an observation.

        Exploration should be disabled, and the best action should be returned
        deterministically

        Args:
            obs (np.ndarray): The observation vector.

        Returns:
            Action vector as np.ndarray.
        """
        raise NotImplementedError

    def train(self):
        """Train the solver.

        The trained model parameters should be saved in the path
        returned by self.param_out_path().
        """
        raise NotImplementedError

    def get_trainable(self):
        """
        Get a trainable to use with Tune for hyperparam tuning.
        https://docs.ray.io/en/latest/tune/key-concepts.html

        The simplest version of a trainable is something that looks like this:

        def trainable(config):
            # config (dict): A dict of hyperparameters.

            for x in range(20):
                score = objective(x, config["a"], config["b"])
                tune.report(score=score)  # This sends the score to Tune.
        """
        raise NotImplementedError
