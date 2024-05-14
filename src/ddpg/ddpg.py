class DDPG:
    def __init__(self):
        raise NotImplementedError

    def add_to_replay_buffer(self, obs, action, reward, next_obs, done):
        '''Store the transition into the replay buffer'''
        raise NotImplementedError

    def select_action(self, obs):
        '''Select an action from the policy network'''
        raise NotImplementedError

    def optimize(self):
        '''Updates policy and value network using a batch from the replay buffer'''
        raise NotImplementedError

    def set_device(self):
        raise NotImplementedError
