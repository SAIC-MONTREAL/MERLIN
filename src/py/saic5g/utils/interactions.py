import glob
import os
import json
import numpy as np
from saic5g.utils.format import np2json

class InteractionBase:
    '''
    Base class for loading and saving interaction experiences
    '''
    
    def get(self, path):
        '''
        Get all interaction experiences from the given path, including observations, actions,
        and rewards in the following format::

            [
                # episode 1
                {
                    'reset': {'obs': [...]},
                    'steps':[
                        {
                            'obs': [...],
                            'action': [...],
                            'reward': 1.2
                        }
                        ...
                        ...
                    ]
                }, 

                # episode 2
                {...}

                ...

                # episode n
                {...}
            ]
        '''
        raise NotImplementedError
    
    def get_obs(self, path):
        '''
        Get all observations encountered throughout all episodes. This is
        particularly useful for distillation.

        Returns:
            List of observation vectors as a numpy array
        '''
        raise NotImplementedError
    
    def save(self, path, data):
        '''
        Save interaction data.

        Args:
            path (str): Path where the interaction data will be saved
            data (dict): The data to be saved
        '''
        raise NotImplementedError
    
    def format_arrays(self, observations, actions, rewards):
        '''
        Sometimes the rollout function will accumulate the observations,
        actions, and rewards in separate arrays. This helper function reformats the data
        to be consistent with the save format.

        Note that we expect observations to have one more element than actions and rewards.
        The first element of observations is assumed to be from reset.
        '''
        data = {
            'reset': {'obs': observations[0]},
            'steps': [],
        }
        for obs, act, rew in zip(observations[1:], actions, rewards):
            data['steps'].append({
                'obs': obs,
                'action': act,
                'reward': rew
            })
        return data

class JsonInteractions(InteractionBase):
    '''
    Handles interaction experiences saved in json files,
    where each json file contains the experiences for one episode.
    Each json file is expected to have the following structure::

         {
            'reset': {'obs': [...]},
            'steps':[
                {
                    'obs': [...],
                    'action': [...],
                    'reward': 1.2
                }
                ...
                ...
            ]
        }
    '''
    
    def get(self, path):
        '''
        Load interactions form the given path.
        
        Args:
            path (str): If this is a directory, look for an "interactions" folder under the given 
                path and glob for all json files under this folder. If this is a file, assume the
                file is a json file and load interaction experiences from it.

        Returns:
            See super class docstring for the return format
        '''
        if os.path.isdir(path):
            json_files = glob.glob(os.path.join(path, '**', 'interactions', '*.json'), recursive=True)
            json_files += glob.glob(os.path.join(path, '**', 'dataset', '*.json'), recursive=True)
        else:
            json_files = [path]
        assert len(json_files) > 0, 'No interaction file(s) found at %s' % path

        episodes = []
        for json_file in json_files:
            with open(json_file, 'r') as fh:
                data = json.load(fh)
                episodes.append(data)
        return episodes
    
    def get_obs(self, path):
        episodes = self.get(path)
        all_obs = []
        for eps in episodes:
            all_obs += [eps['reset']['obs']] + [step['obs'] for step in eps['steps']]
        return np.array(all_obs)
    
    def save(self, path, data):
        with open(path, 'w') as fh:
            json.dump(np2json(data), fh, indent=4)

