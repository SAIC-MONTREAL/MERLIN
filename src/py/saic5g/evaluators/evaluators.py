"""
Base class for evaluators and generic evaluator.
"""
import os
import time
import uuid
from saic5g.utils.interactions import JsonInteractions

def rollout(env, solver, show_timer=False):
    tic = time.perf_counter()
    obs = env.reset()
    done = False
    rewards= []
    obss = [obs]
    actions = []
    infos = []
    while not done:
        action = solver.get_action(obs)
        obs, reward, done, info = env.step(action)
        if isinstance(done, dict):
            done = done['__all__']
        rewards.append(reward)
        actions.append(action)
        obss.append(obs)
        infos.append(info)
    toc = time.perf_counter()
    if show_timer:
        print(f"Rollout took {toc - tic:0.4f} seconds")
    return rewards, obss, actions, infos

class EvaluatorBase:

    def __init__(self, env_generator, incr_seed=0, start_seed=0, n_trials=1,
        max_workers=10, skip_vis=True, skip_rollouts=False, log_rollouts=False):
        self.env_gen = env_generator
        self.incr_seed = incr_seed
        self.start_seed = start_seed
        self.n_trials = n_trials
        self.max_workers = max_workers
        self.skip_vis = skip_vis
        self.skip_rollouts = skip_rollouts
        self.log_rollouts = log_rollouts

    def get_env(self):
        env = self.env_gen()
        return env
    
    def save_rollout(self, save_dir, rewards, observations, actions, extras={}):
        '''
        Helper function that makes it easy for sublasses to save interaction experiences.
        Subclasses should check self.log_rollouts to see if interaction experineces should 
        be saved, and can call this function to perform the saving.

        Args:
            save_dir (str): A directory where the interaction experience should be saved.
                This function will create an interactions folder under save_dir, and
                generate a random name for the saved file. The data will be saved in
                json format using JsonInteractions. See JsonInteractions for more details
                on the save format.
            rewards (list): List of scalar rewards
            observations (list): List of observation vectors. This list should have one
                more element than actions and rewards, with the first element being the 
                observation from reset.
            actions (list): List of action vectors.
            extras (dict): A dictionary with extra data that will be saved in the json file.
        '''
        save_dir = os.path.join(save_dir, 'interactions')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uuid.uuid4().hex) + '.json'
        interaction_mgr = JsonInteractions()
        data = interaction_mgr.format_arrays(observations, actions, rewards)
        data.update(extras)
        interaction_mgr.save(save_path, data)

    def evaluate(self, solver, out_dir):
        """
        This method can do pretty much anything, as long as it outputs
        to the out_dir. The return value should be either a single number
        that quantifies the quality of the solver, or a dictionary mapping
        performance metrics to values. If a dictionary is given, the key
        "performance" must exist, which maps to a number that indicates the overall
        quality of the solver.
        """
        raise NotImplementedError

    def analyze_saved_results(self, solver, results_dir, out_dir):
        """
        This method analyses previously generated rollouts by the evaluator.
        """
        raise NotImplementedError

    def get_seed_iterable(self):
        if self.incr_seed == 0:
            return [self.start_seed] * self.n_trials
        return range(self.start_seed, self.start_seed + self.incr_seed * self.n_trials, self.incr_seed)
