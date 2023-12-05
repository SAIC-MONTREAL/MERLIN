import os
import pickle as pkl
import numpy as np
import pandas as pd

from saic5g.evaluators.evaluators import EvaluatorBase, rollout
from saic5g.utils.format import reformat_evaluation_data
from saic5g.vis.timeseries import plot_data

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
        try:
            cell_capacity = v['env_info']['cell_capacity']
        except:
            cell_capacity = np.array([50*7])
        unsat_load = cell_obs['demand'] * cell_obs['packet_loss']
        sat_load = cell_obs['demand'] - unsat_load
        total_load = cell_obs['demand']
        most_load = np.max(total_load, axis=1)
        load_std = np.std(total_load, axis=1)
        total_load_norm = total_load / cell_capacity 
        most_load_norm = np.max(total_load_norm, axis=1)
        load_std_norm = np.std(total_load_norm, axis=1)

        rsrp = ue_obs['rsrp']
        rsrq = ue_obs['rsrq']
        sinr = ue_obs['sinr']
        packet_loss = ue_obs['packet_loss']

        worst_rsrp = np.amin(rsrp.reshape(rsrp.shape[0], -1), axis=1)
        mean_rsrp = np.mean(rsrp.reshape(rsrp.shape[0], -1), axis=1)
        worst_rsrq = np.amin(rsrq.reshape(rsrq.shape[0], -1), axis=1)
        mean_rsrq = np.mean(rsrq.reshape(rsrq.shape[0], -1), axis=1)
        worst_sinr = np.amin(sinr.reshape(sinr.shape[0], -1), axis=1)
        mean_sinr = np.mean(sinr.reshape(sinr.shape[0], -1), axis=1)
        worst_packet_loss = np.amax(packet_loss.reshape(packet_loss.shape[0], -1), axis=1)
        mean_packet_loss = np.mean(packet_loss.reshape(packet_loss.shape[0], -1), axis=1)
        
        serving_cell = ue_obs['serving_cell']
        s_1 = serving_cell[:-1,:]
        s_2 = serving_cell[1:,:]
        tmp = np.equal(s_1, s_2)
        handover = tmp.shape[1] - np.count_nonzero(tmp,axis=1) 
        handover = np.concatenate([[0], handover])

        df['most_load'] = most_load_norm
        df['load_std'] = load_std_norm
        df['worst_rsrp'] = worst_rsrp
        df['mean_rsrp'] = mean_rsrp
        df['worst_rsrq'] = worst_rsrq
        df['mean_rsrq'] = mean_rsrq
        df['worst_sinr'] = worst_sinr
        df['mean_sinr'] = mean_sinr
        df['worst_packet_loss'] = worst_packet_loss
        df['mean_packet_loss'] = mean_packet_loss
        df['handover'] = handover

        for cell in range(total_load.shape[1]):
            l = total_load[:,cell] / np.mean(cell_capacity)
            df['load_cell_'+str(cell)] = l

        save_path = os.path.join(dir_path, policy+'.csv')
        df.to_csv(save_path)
        all_data.append(df)

    return all_data

class PerUeHandoverEvaluator(EvaluatorBase):
    """
    Basic evaluator for problems whose envs are derived from the PerUeHandover env.
    """
    name = 'per-ue-handover-evaluator'

    def evaluate(self, solver, out_dir):
        mean_rewards = []
        print('rolling out:', solver.name)
        for seed in self.get_seed_iterable():
            env = self.get_env()
            env.log_movies = True
            solver.pre_eval(env)
            env.seed(seed)
            env.sls_seed = seed
            rewards, _, _, _ = rollout(env, solver)
            mean_rewards.append(np.array(rewards).mean())

        # Log stats for the last rollout for analysis
        cell_capacity = env.scenario.radio.cell_capacity(0, env.n_cells)
        env_info =  {
            'cell_capacity': cell_capacity
        }

        os.makedirs(out_dir, exist_ok=True)
        if not self.skip_vis:
            env.make_movie(os.path.join(out_dir, '%s.mp4' % solver.name),which='last')
        data = reformat_evaluation_data(env.get_last_complete_run(), env_info)
        pkl_path = os.path.join(out_dir, 'history.pkl')
        with open(pkl_path, 'wb') as f:
            pkl.dump(data, f)
        if not self.skip_vis:
            plot_data(process_data(out_dir), out_dir)
        r = np.array(mean_rewards)
        return r.mean()