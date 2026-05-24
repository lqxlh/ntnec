"""
ReplayMemory.py — 经验回放池

支持两种经验格式（向后兼容旧 5/6 元组，新增 7 元组供 HD3QN 使用）：

旧格式（D3QN，6 元组）：
    (obs, disc_act, reward, next_obs, done, next_action_mask)

新格式（HD3QN，7 元组）：
    (obs, disc_act, cont_act, reward, next_obs, done, next_action_mask)

sample() 返回规则
-----------------
- 若 buffer 中的样本均为 7 元组，返回 7 个数组：
    obs, disc_act, cont_act, reward, next_obs, done, next_action_mask
- 若均为 6 元组（旧 DQN 格式），返回 6 个数组（保持原有兼容性）
- 若均为 5 元组（旧 SAC 格式），返回 5 个数组
"""

import collections
import random

import numpy as np


class ReplayMemory:
    def __init__(self, max_size: int):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        """
        exp 可以是 5/6/7 元组，统一 push 进去。
        HD3QN 调用方应传入 7 元组：
            (obs, disc_act, cont_act, reward, next_obs, done, next_action_mask)
        """
        self.buffer.append(exp)

    def sample(self, batch_size: int):
        mini_batch = random.sample(self.buffer, batch_size)

        obs_batch, disc_batch, cont_batch = [], [], []
        reward_batch, next_obs_batch, done_batch = [], [], []
        next_mask_batch = []

        has_cont = False   # 是否出现了 7 元组（含连续动作）
        has_mask = False   # 是否出现了 next_action_mask

        for exp in mini_batch:
            if len(exp) == 7:
                # HD3QN 格式：(obs, disc_act, cont_act, reward, next_obs, done, next_mask)
                s, da, ca, r, sp, done, nm = exp
                cont_batch.append(ca)
                next_mask_batch.append(nm)
                has_cont = True
                has_mask = True
            elif len(exp) == 6:
                # 旧 DQN 格式：(obs, disc_act, reward, next_obs, done, next_mask)
                s, da, r, sp, done, nm = exp
                ca = 0.5
                cont_batch.append(ca)
                next_mask_batch.append(nm)
                has_mask = True
            else:
                # 旧 SAC 5 元组：(obs, act, reward, next_obs, done)
                s, da, r, sp, done = exp
                ca = 0.5
                cont_batch.append(ca)

            obs_batch.append(s)
            disc_batch.append(da)
            reward_batch.append(r)
            next_obs_batch.append(sp)
            done_batch.append(done)

        base = (
            np.array(obs_batch,      dtype="float32"),
            np.array(disc_batch,     dtype="float32"),
            np.array(reward_batch,   dtype="float32"),
            np.array(next_obs_batch, dtype="float32"),
            np.array(done_batch,     dtype="float32"),
        )

        if has_cont and has_mask:
            # HD3QN 7 元组：返回 7 个数组
            return (
                base[0],                                      # obs
                base[1],                                      # disc_act
                np.array(cont_batch, dtype="float32"),        # cont_act
                base[2],                                      # reward
                base[3],                                      # next_obs
                base[4],                                      # done
                np.array(next_mask_batch, dtype=bool),        # next_action_mask
            )
        elif has_mask:
            # 旧 DQN 6 元组
            return base + (np.array(next_mask_batch, dtype=bool),)
        else:
            # 旧 SAC 5 元组
            return base

    def __len__(self):
        return len(self.buffer)