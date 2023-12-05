import os
import pickle as pkl
import numpy as np
import pandas as pd

from saic5g.evaluators.evaluators import EvaluatorBase, rollout
from saic5g.utils.format import reformat_evaluation_data
from saic5g.vis.timeseries import plot_data
import tqdm

def process_data(dir_path):
    '''
    Process all pkl data saved in dir_path and output a dataframe. The pkl data are saved by reformat_evaluation_data.
    '''
    all_data = []
    rollout_files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    for filename in rollout_files:
        policy = filename.split('.')[0]
        df = pd.DataFrame()
        df.name=policy
        filepath = os.path.join(dir_path, filename)

        with open(filepath, 'rb') as f:
            v = pkl.load(f)
        ue_obs = v['UEs']
        cell_obs = v['cells']
        x = v['simulation_time']
        n_ue = ue_obs['position'].shape[1]
        n_cell = cell_obs['position'].shape[1]
        
        total_load = cell_obs["loads"] # this is already normalized, between 0,1
        most_load = np.amax(total_load.reshape(total_load.shape[0], -1), axis=1)
        load_std = np.std(total_load.reshape(total_load.shape[0], -1), axis=1)
        dropped = cell_obs["dropped"]
        dropped_count = np.sum(dropped.reshape(dropped.shape[0], -1), axis=1)
        rsrp = ue_obs['rsrp']
        sinr = ue_obs['sinr']
        worst_rsrp = np.amin(rsrp.reshape(rsrp.shape[0], -1), axis=1)
        mean_rsrp = np.mean(rsrp.reshape(rsrp.shape[0], -1), axis=1)
        worst_sinr = np.amin(sinr.reshape(sinr.shape[0], -1), axis=1)
        mean_sinr = np.mean(sinr.reshape(sinr.shape[0], -1), axis=1)
        serving_cell = ue_obs['serving_cell']
        s_1 = serving_cell[:-1,:]
        s_2 = serving_cell[1:,:]
        tmp = np.equal(s_1, s_2)
        handover = tmp.shape[1] - np.count_nonzero(tmp,axis=1) 
        handover = np.concatenate([[0], handover])

        df['max_load'] = most_load
        df['load_std'] = load_std
        df['worst_rsrp'] = worst_rsrp
        df['mean_rsrp'] = mean_rsrp
        df['worst_sinr'] = worst_sinr
        df['mean_sinr'] = mean_sinr
        df['handover'] = handover
        df['dropped'] = dropped_count
        save_path = os.path.join(dir_path, policy+'.csv')
        df.to_csv(save_path)
        all_data.append(df)

    return all_data

class MlbEvaluator(EvaluatorBase):
    """
    Basic evaluator for classic MLB problems.

    """
    name = 'mlb-evaluator'

    def evaluate(self, solver, out_dir):
        mean_rewards = []
        mean_load_std = []
        mean_dropped = []
        print('rolling out:', solver.name)

        for seed in tqdm.tqdm(self.get_seed_iterable()):
            env = self.get_env()
            if not self.skip_vis:
                env.log_movies = True
            solver.pre_eval(env)
            env.seed(seed)
            env.sls_seed = seed
            rewards, load_stds, dropped = self.get_rollouts(env, solver)
            mean_rewards.append(rewards)
            mean_load_std.append(load_stds)
            mean_dropped.append(dropped)
        

        os.makedirs(out_dir, exist_ok=True)
        if not self.skip_vis:
            env.make_movie(os.path.join(out_dir, '%s.mp4' % solver.name), which='last')

        cell_capacity = env.scenario.radio.cell_capacity(0, env.n_cells)
        env_info =  {
            'cell_capacity': cell_capacity
        }
        data = reformat_evaluation_data(env.get_last_complete_run(), env_info)
        pkl_path = os.path.join(out_dir, 'history.pkl')
        with open(pkl_path, 'wb') as f:
            pkl.dump(data, f)
        if not self.skip_vis:
            plot_data(process_data(out_dir), out_dir)

        # Average KPIs over seeds
        return {'performance': np.mean(mean_rewards),
                'load_deviation': np.mean(mean_load_std),
                'dropped': np.mean(mean_dropped)}

    def get_rollouts(self, env, solver):
        """
        Perform rollout and compute average KPIs
        """
        episode_rewards = []
        episode_dropped = []
        episode_load_std = []

        obs = env.reset()
        done = False
        ep_rew = 0
        it = 0
        load_std = []

        while not done:
            it += 1
            action = solver.get_action(obs)
            obs, rew, done, info = env.step(action)
            episode_dropped.append(np.sum(info['cells']['dropped']))
            mean_load = np.mean(info['cells']['loads'])
            std_load = np.mean(np.abs(info['cells']['loads']-mean_load))
            ep_rew += rew
            load_std.append(std_load)

        ep_rew /= it
        episode_rewards = np.mean(ep_rew)
        episode_load_std = np.mean(load_std)
        episode_dropped = np.mean(episode_dropped)

        return episode_rewards, episode_load_std, episode_dropped
