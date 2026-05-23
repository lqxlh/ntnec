import collections
import random
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        # 如果 DQN 经验里额外保存了下一状态动作掩码，就用这个列表把它们一起采样出来。
        next_mask_batch = []
        for experience in mini_batch:
            # 旧的 SAC 经验是 5 个元素；新的 DQN 经验是 6 个元素，最后一个是 next_action_mask。
            if len(experience) == 6:
                s, a, r, s_p, done, next_mask = experience
                next_mask_batch.append(next_mask)
            else:
                s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
        sampled_data = (
            np.array(obs_batch).astype('float32'),
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')
        )
        # 只有全部样本都带有 next_mask 时，才返回第 6 个数组；这样不会影响 SAC 的旧用法。
        if next_mask_batch:
            return sampled_data + (np.array(next_mask_batch).astype('bool'),)
        return sampled_data

    def __len__(self):
        return len(self.buffer)
