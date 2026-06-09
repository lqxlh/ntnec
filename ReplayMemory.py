"""
PERReplayMemory.py — 经验回放池

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

#新加的PER
class SumTree:
    """PER 核心数据结构：求和二叉树"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayMemory:
    """优先经验回放池 (PER)，完美对齐 7 元组格式"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.n_entries

    def append(self, exp):
        """存入一条完整的 7 元组经验"""
        self.tree.add(self.max_priority, exp)

    def sample(self, batch_size: int):
        mini_batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            idxs.append(idx)
            mini_batch.append(data)

        # 计算重要度采样权重 (Importance Sampling Weights)
        sampling_probabilities = np.array(priorities) / (self.tree.total() + 1e-9)
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        # 修改后：使用 Mean 归一化，保持 Batch 权重的期望值为 1.0
        is_weights /= (is_weights.mean() + 1e-9)

        obs_batch, disc_batch, cont_batch = [], [], []
        reward_batch, next_obs_batch, done_batch = [], [], []
        next_mask_batch = []

        # 严格按照你原先 7 元组的解包顺序进行拼装
        for exp in mini_batch:
            s, da, ca, r, sp, done, nm = exp
            obs_batch.append(s)
            disc_batch.append(da)
            cont_batch.append(ca)
            reward_batch.append(r)
            next_obs_batch.append(sp)
            done_batch.append(done)
            next_mask_batch.append(nm)

        batched_data = (
            np.array(obs_batch,      dtype="float32"),
            np.array(disc_batch,     dtype="float32"),
            np.array(cont_batch,     dtype="float32"),
            np.array(reward_batch,   dtype="float32"),
            np.array(next_obs_batch, dtype="float32"),
            np.array(done_batch,     dtype="float32"),
            np.array(next_mask_batch, dtype=bool),
        )

        return batched_data, idxs, is_weights.astype("float32")

    def update_priorities(self, idxs, td_errors):
        """根据最新的 TD-Error 更新树节点的优先级"""
        for idx, error in zip(idxs, td_errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)