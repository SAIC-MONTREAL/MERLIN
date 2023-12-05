class BaseEnvVariantManager:
    """Base class for environment variant manager.

    A variant manager's job is to generate different variants of an environment
    by permuting its config.
    One use case for the variant manager is multitask learning, where we
    typically want an agent to learn on multiple related tasks.
    It is often easier to permute the arguments
    passed into the environment generator rather than maintaining numerous
    separate environments, especially if the number of possible task
    parameter combinations is infinite.
    """

    def set_default_env_cfg(self, env_cfg):
        """Set the default hydra configuration

        Args:
            env_cfg (dict): The hydra environment config which we will permute to generate
                the environment variants. Due to the way hydra's intantiate function
                works, env_cfg cannot be directly passed into the __init__
                function when instantiating the object.
        """
        self._env_cfg = env_cfg

    def __len__(self):
        """Return the total number of environment variants.

        This may not be the total number of environments used during training,
        since the learner may instantiate multiple instances of each variant.

        Returns:
            Integer representing the number of variants
        """
        raise NotImplementedError

    def __call__(self, index=None):
        """Generate an environment with the variant index

        Typically, this function would make some modifications to a copy
        of self._env_cfg, and then use the modified config to instantiate
        a gym environment.

        Args:
            index (int): The variant index.

        Returns:
            A gym.Env object
        """
        raise NotImplementedError
