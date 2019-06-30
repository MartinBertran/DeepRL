#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
from deep_rl.utils import *
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave

#Reward agents assume states and rewards are prenormalized
class RewardAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.reward_model' % (filename))
        with open('%s.reward_stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.reward_model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.reward_stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, transitions):
        raise NotImplementedError

    def step(self, transitions):
        raise NotImplementedError




