#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .RewardAgent import *



class IcmAgent(RewardAgent):
    def __init__(self, config):
        RewardAgent.__init__(self, config)
        self.config = config
        self.network = config.reward_network_fn()
        self.replay = config.reward_replay_fn()
        self.optimizer = config.reward_optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.CE_loss=nn.CrossEntropyLoss()


    def _compute(self, transitions, eval_flag):


        # process transitions
        states, actions, rewards, next_states, dones, infos = transitions
        prediction_error, predicted_action_logits = self.network(states, actions, next_states)

        # add rewards to normal environment rewards, end-of-episode predictions are discarded
        returned_rewards = to_np(self.config.reward_eta/2 *prediction_error) +rewards
        original_prediction_error = prediction_error

        if not eval_flag:
            # store experiences in replay buffer
            experiences = []
            for idx in np.arange(states.shape[0]):
                self.total_steps += 1
                experiences.append([states[idx,...], actions[idx,...], rewards[idx,...], next_states[idx,...], dones[idx,...]])
            self.replay.feed_batch(experiences)

            #TODO ADD STORAGE FOR ICM LOSSES..., REPLAY SMARTER
            if self.total_steps> self.config.reward_exploration_steps: # If I have enough samples in buffer, sample
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
            prediction_error, predicted_action_logits = self.network(states, rewards, next_states)

            ce_loss = self.CE_loss(predicted_action_logits,tensor(actions).long())
            prediction_loss = prediction_error
            loss = self.config.reward_beta * prediction_loss + \
                   (1-self.config.reward_beta)*ce_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            try:
                self.config.writer.add_scalar('ICM_reward_training',prediction_error)
                self.config.writer.add_scalar('ICM_inverse_crossentropy_training', ce_loss)
                self.config.writer.add_scalar('ICM_reward', original_prediction_error)
            except AttributeError:
                pass

        return returned_rewards

    def close(self):
        close_obj(self.replay)

    def compute(self, transitions):
        return self._compute(transitions, False)

    def evaluate(self, transitions):
        return self._compute(transitions, True)