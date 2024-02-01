import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List
import copy
import re
import gym
import hydra
import saic5g.envs  # registers our gym envs
import tensorflow as tf
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sacred import SETTINGS, Experiment
from sacred.observers import QueuedMongoObserver
from sacred.run import Run
from saic5g.utils.git_utils import get_git_email
from saic5g.utils.tensorboard import SummaryReader
from saic5g.utils import hydra_utils

# To enable config handling
# see https://github.com/IDSIA/sacred/issues/492
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment("MERLIN:Train")


class MetricThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super(MetricThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self.curr_row = 0

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def filter_columns(columns: List[str], regex: List[str]):
        matched = set()
        for col in columns:
            for pattern in regex:
                if re.search(pattern, col):
                    matched.add(col)
        return list(matched)

    def read_and_plot_metrics(self, _run, metric_regex):
        for i, path in enumerate(_run.info["tfevent_paths"]):
            sr = SummaryReader(path)
            try:
                df = sr.read()
            except tf.errors.NotFoundError:
                print("[WARNING] Tensorflow was unable to open the events file.")
                continue

            metrics = df[self.curr_row:]
            columns = metrics.columns
            if metric_regex is not None:
                columns = MetricThread.filter_columns(columns, metric_regex)
            for _, row in metrics.iterrows():
                for column in columns:
                    if column == "step":
                        continue
                    if not math.isnan(row[column]):
                        _run.log_scalar("{} {}".format(i, column), row[column], row["step"])
            self.curr_row = len(df)

    @ex.capture
    def run(self, _run, cfg):
        """
        Parse tfevents file to retrieve metrics and report them to sacred.
        This function is run in its own thread. It parses tfevent files every
        METRIC_POLL_INTERVAL seconds and forwards new events to Sacred.
        """

        # Note the output directory of Hydra may not the same as the original
        # working directory
        # (see https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory)
        log_dir = cfg["train_dir"]

        if not tf.executing_eagerly():
            # If eager mode is not enabled, we will not report live metrics
            # since we won't be able to parse the tfevents file.
            return

        metric_poll_interval = cfg["logger"]["sacred"]["metric_poll_interval"]
        metric_regex = cfg["logger"]["sacred"]["metric_regex"]

        while True:
            if self._stop_event.is_set():
                # Make sure we are not missing any data point in the tensorboard
                self.read_and_plot_metrics(_run, metric_regex)
                return

            if "tfevent_paths" not in _run.info:
                readers = SummaryReader.get_readers(Path(log_dir))
                # Find and save the paths to tfevents files.
                # get_readers uses glob to find tfevents files. We don't
                # want to run glob every time, especially since with data collection
                # enabled, there will be a lot of files.
                if len(readers) == 0:
                    time.sleep(metric_poll_interval)
                    continue

                _run.info["tfevent_paths"] = [str(reader.path) for reader in readers]
                print("Bridging the following tfevents to sacred")
                print(_run.info["tfevent_paths"])
            self.read_and_plot_metrics(_run, metric_regex)
            time.sleep(metric_poll_interval)


def extract_env_solver_cfg(cfg):
    """
    Extract the part of hydra config pertaining to to the env and solver.
    Since we are trainig, we expect cfg['env'] and cfg['solver'] to both be
    of length 1. In other words, there should be exactly 1 env and one solver.
    Note that during evaluation, cfg['solver'] can have length
    greater than 1, which allows us to simultaneously evaluate multiple methods.
    """
    _, env_config = hydra_utils.get_one(cfg['env'], max_selection=1)
    _, solver_config = hydra_utils.get_one(cfg['solver'], max_selection=1)
    return env_config, solver_config


def train_gen(cfg: Dict, metric_thread=None):
    """
    Generate a train function
    """
    env_variant_mgr = cfg.get("env_variant_mgr", None)
    if not 'env' in cfg:
        raise ValueError("Env configuration is NOT provided")
    if not 'solver' in cfg:
        raise ValueError("Solver configuration is NOT provided")

    env_config, solver_config = extract_env_solver_cfg(cfg)

    def train_fn():
        # Create environment generator that can be passed into the solver.
        if env_variant_mgr is not None:
            # If a variant manager is specified, we will use it to obtain
            # the generator. In this case the generator will accept a single
            # positional argument that specifies the variant id
            envgen = instantiate(env_variant_mgr)
            envgen.set_default_env_cfg(copy.deepcopy(env_config))
        else:
            # If no variant manager is specified, then our job is fairly simple
            # We just create a generator that creates the gym environment
            def envgen():
                return instantiate(copy.deepcopy(env_config))

        # Pass in the environment generator when creating the solver object.
        # The solver can use isinstance(BaseEnvVariantManager) to check if
        # a variant id should be given to the generator
        solver_obj = instantiate(solver_config, envgen)

        if metric_thread is not None:
            metric_thread.start()

        solver_obj.train()

        if metric_thread is not None:
            metric_thread.stop()
            metric_thread.join()

    return train_fn


@hydra.main(config_path="../../configs/hydra/", config_name="train", version_base=None)
def main(cfg: DictConfig):
    logdir = cfg["train_dir"]
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        print(f"Output will be saved to {os.getcwd()}/{logdir}")

    # Setup logging
    logger = cfg.get("logger", None)
    if logger is None:
        print("No logger is specified!")
        train_fn = train_gen(cfg)
        train_fn()
    elif "sacred" in cfg["logger"]:
        # Sacred config
        sacred_conf = cfg["logger"]["sacred"]
        print("Sacred logging enabled: Data will be visible on omniboard.")
        ex.observers.append(QueuedMongoObserver(url=sacred_conf["url"], db_name=sacred_conf["db_name"]))
        # Resolve all the interpolations to avoid errors in the main function
        OmegaConf.resolve(cfg)
        ex.add_config({"cfg": OmegaConf.to_container(cfg)})

        env_config, solver_config = extract_env_solver_cfg(cfg)
        ex.add_config({'env': env_config['gym_env'], 'solver': solver_config['solver_name']})

        # Add user information to Sacred config
        user = get_git_email()
        ex.add_config({"user": user})

        metric_thread = MetricThread()
        train_fn = train_gen(cfg, metric_thread=metric_thread)
        captured = ex.command(train_fn)
        ex.default_command = captured.__name__

        try:
            ex.run()
        except Exception as e:
            print("Encountered an exception {}".format(e))
            # Safely exit.
            sys.exit(1)
    else:
        raise ValueError("Logger {} not implemented".format(logger))


if __name__ == "__main__":
    main()
