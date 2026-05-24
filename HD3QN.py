"""
HD3QN.py — Hybrid Dueling Double DQN

结构说明
--------
共享特征层  →  离散 Q 分支（Dueling，输出每个离散动作的 Q 值）
           →  连续参数分支（输出每个离散动作对应的连续资源分配值，范围 [0,1]）

动作执行
--------
1. 离散 Q 分支 argmax（含 action_mask 屏蔽非法动作）→ discrete_action
2. 取连续参数分支中 discrete_action 对应的那一维 → continuous_action ∈ [0,1]

训练
----
- 离散分支：Double DQN 方式更新 Q 值
- 连续参数分支：通过"对被选中离散动作的 Q 目标"间接监督；
  具体地，用 MSE(cont_output[selected], sigmoid(td_target.detach())) 约束，
  使连续分支逐步向"Q 值高的目标动作收敛时对应的连续参数"对齐。

经验格式（7 元组）
-----------------
(obs, discrete_act, cont_act, reward, next_obs, done, next_action_mask)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import para

device = torch.device("cuda" if para.USE_CUDA and torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. 网络结构
# ---------------------------------------------------------------------------

class HD3QNModel(nn.Module):
    """
    共享特征层 + 离散 Q 分支（Dueling）+ 连续参数分支。

    参数
    ----
    obs_dim   : 状态观测维度
    act_dim   : 离散动作数量（local=1, BS=N, SAT=S → N+S+1）
    hid_size  : 隐藏层宽度
    """

    def __init__(self, obs_dim: int, act_dim: int, hid_size: int = 128):
        super().__init__()
        self.act_dim = act_dim

        # ---- 共享特征层 ----
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
        )

        # ---- 离散 Q 分支（Dueling：Advantage + Value） ----
        self.adv_fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, act_dim),
        )
        self.val_fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

        # ---- 连续参数分支 ----
        # 对每个离散动作输出一个连续值 ∈ [0,1]（用 Sigmoid 约束）
        self.cont_fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, act_dim),
            nn.Sigmoid(),
        )

    def forward(self, obs):
        """
        返回
        ----
        q_values  : [batch, act_dim]  每个离散动作的 Q 值
        cont_params: [batch, act_dim]  每个离散动作对应的连续参数 ∈ (0,1)
        """
        feat = self.shared(obs)

        # Dueling Q
        adv = self.adv_fc(feat)                              # [B, act_dim]
        val = self.val_fc(feat)                              # [B, 1]
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))  # [B, act_dim]

        # 连续参数
        cont_params = self.cont_fc(feat)                     # [B, act_dim]

        return q_values, cont_params


# ---------------------------------------------------------------------------
# 2. 算法（Double DQN + 连续分支间接监督）
# ---------------------------------------------------------------------------

class HD3QN(nn.Module):
    """
    封装主网络 + 目标网络，实现 Double DQN 更新逻辑，
    同时附带连续参数分支的间接监督。
    """

    def __init__(self, model: HD3QNModel, act_dim: int, gamma: float, lr: float):
        super().__init__()
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # 推断接口
    # ------------------------------------------------------------------

    def predict_q(self, obs: torch.Tensor):
        """返回主网络 Q 值（eval 模式，无梯度）。"""
        self.model.eval()
        with torch.no_grad():
            q, _ = self.model(obs)
        return q

    def predict_both(self, obs: torch.Tensor):
        """返回主网络 Q 值与连续参数（eval 模式，无梯度）。"""
        self.model.eval()
        with torch.no_grad():
            q, cont = self.model(obs)
        return q, cont

    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------

    def learn(
        self,
        obs: torch.Tensor,           # [B, obs_dim]
        disc_action: torch.Tensor,   # [B]  long
        cont_action: torch.Tensor,   # [B]  float，经验中记录的连续动作（可选监督）
        reward: torch.Tensor,        # [B]
        next_obs: torch.Tensor,      # [B, obs_dim]
        terminal: torch.Tensor,      # [B]  bool
        next_action_mask=None,       # [B, act_dim] bool 或 None
    ):
        self.model.train()

        # ---------- 1. 主网络当前 Q 值 ----------
        q_pred, cont_pred = self.model(obs)           # [B, act_dim], [B, act_dim]
        # 取被选中离散动作的 Q 值
        disc_action_long = disc_action.long()
        q_selected = q_pred.gather(1, disc_action_long.unsqueeze(1)).squeeze(1)  # [B]

        # ---------- 2. Double DQN target ----------
        with torch.no_grad():
            # 主网络选下一步动作（含 mask）
            next_q_main, _ = self.model(next_obs)     # [B, act_dim]
            if next_action_mask is not None:
                mask = next_action_mask.to(device=next_q_main.device, dtype=torch.bool)
                next_q_main = next_q_main.masked_fill(~mask, -1e9)
            next_greedy_action = next_q_main.argmax(dim=1)  # [B]

            # 目标网络评估该动作的 Q 值
            next_q_target, _ = self.target_model(next_obs)  # [B, act_dim]
            next_q_val = next_q_target.gather(
                1, next_greedy_action.unsqueeze(1).long()
            ).squeeze(1)                                      # [B]

            td_target = reward + (1.0 - terminal.float()) * self.gamma * next_q_val  # [B]

        # ---------- 3. 离散 Q 损失 ----------
        q_loss = F.mse_loss(q_selected, td_target)

        # ---------- 4. 连续参数分支损失 ----------
        # 策略：让被选中动作对应的连续参数向"经验中记录的 cont_action"靠近。
        # 这样连续分支通过高 Q 样本的"成功连续动作"间接学习好的资源分配。
        cont_selected = cont_pred.gather(
            1, disc_action_long.unsqueeze(1)
        ).squeeze(1)                                          # [B]
        cont_target = cont_action.float().clamp(0.0, 1.0)    # [B]
        cont_loss = F.mse_loss(cont_selected, cont_target)

        # ---------- 5. 联合损失 & 反向传播 ----------
        # cont_loss 权重略小，避免主导 Q 学习；可按实验调整。
        total_loss = q_loss + 0.5 * cont_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪：防止卫星奖励稀疏时出现梯度爆炸
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return total_loss, q_loss, cont_loss

    def sync_target(self):
        """硬更新目标网络（完全复制）。"""
        self.target_model.load_state_dict(self.model.state_dict())

    # ------------------------------------------------------------------
    # 模型持久化
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.target_model.load_state_dict(ckpt["target_model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.model.eval()
        self.target_model.eval()


# ---------------------------------------------------------------------------
# 3. Agent（对外统一接口，兼容原 D3QN.Agent + SAC.Agent 的组合调用方式）
# ---------------------------------------------------------------------------

class Agent:
    """
    HD3QN 智能体。

    对外接口与原 D3QN.Agent 兼容：
      sample(obs, action_mask) → discrete_action (int)
      predict(obs, action_mask) → discrete_action (int)
      get_continuous(obs, discrete_action) → continuous_action (float, [0,1])
      learn(obs, disc_act, cont_act, reward, next_obs, terminal, next_action_mask)
        → (total_loss, q_loss, cont_loss)

    说明
    ----
    - sample / predict 只返回离散动作，调用方再调 get_continuous 拿连续动作。
    - 这样 EdgeEnv.step 接口保持不变：仍传入 discrete_action 和 continuous_action。
    """

    def __init__(
        self,
        algorithm: HD3QN,
        obs_dim: int,
        act_dim: int,
        e_greed: float = 0.9,
        e_greed_decrement: float = 0.0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alg = algorithm
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
        self.global_step = 0
        self.update_target_steps = 300

    # ------------------------------------------------------------------
    # 辅助：obs → tensor
    # ------------------------------------------------------------------

    def _to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    # ------------------------------------------------------------------
    # 离散动作选择
    # ------------------------------------------------------------------

    def sample(self, obs, action_mask=None) -> int:
        """ε-greedy 采样（训练用）。只在合法动作中随机/贪心。"""
        valid_actions = np.arange(self.act_dim)
        if action_mask is not None:
            valid_actions = valid_actions[np.array(action_mask, dtype=bool)]
            if len(valid_actions) == 0:
                valid_actions = np.arange(self.act_dim)

        if np.random.rand() < self.e_greed:
            act = int(np.random.choice(valid_actions))
        else:
            act = self.predict(obs, action_mask=action_mask)

        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs, action_mask=None) -> int:
        """贪心预测（评估用）。"""
        obs_t = self._to_tensor(obs)
        q = self.alg.predict_q(obs_t)          # [1, act_dim]
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=q.device).unsqueeze(0)
            q = q.masked_fill(~mask_t, -1e9)
        return int(q.argmax(dim=-1).item())

    # ------------------------------------------------------------------
    # 连续动作读取
    # ------------------------------------------------------------------

    def get_continuous(self, obs, discrete_action: int) -> float:
        """
        从连续参数分支中取 discrete_action 对应的连续值。
        返回标量 float ∈ [0, 1]，供 EdgeEnv.decode_frequency 使用。
        """
        obs_t = self._to_tensor(obs)
        _, cont = self.alg.predict_both(obs_t)   # [1, act_dim]
        return float(cont[0, discrete_action].item())

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def learn(
        self,
        obs,            # np [B, obs_dim]
        disc_act,       # np [B]
        cont_act,       # np [B]   ← 经验中保存的连续动作
        reward,         # np [B]
        next_obs,       # np [B, obs_dim]
        terminal,       # np [B]
        next_action_mask=None,  # np [B, act_dim] bool 或 None
    ):
        self.global_step += 1

        obs_t      = torch.tensor(obs,      dtype=torch.float32).to(device)
        disc_act_t = torch.tensor(disc_act, dtype=torch.long).to(device)
        cont_act_t = torch.tensor(cont_act, dtype=torch.float32).to(device)
        reward_t   = torch.tensor(reward,   dtype=torch.float32).to(device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
        terminal_t = torch.tensor(terminal, dtype=torch.bool).to(device)

        if next_action_mask is not None:
            next_mask_t = torch.tensor(next_action_mask, dtype=torch.bool).to(device)
        else:
            next_mask_t = None

        total_loss, q_loss, cont_loss = self.alg.learn(
            obs_t, disc_act_t, cont_act_t,
            reward_t, next_obs_t, terminal_t,
            next_action_mask=next_mask_t,
        )

        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()

        return total_loss.item(), q_loss.item(), cont_loss.item()