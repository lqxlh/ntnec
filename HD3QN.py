"""
HD3QN.py - Hybrid Dueling Double DQN.

修复版（cont_loss 改为 DDPG 风格的 -Q）：
1. Q-loss 部分：cont_pred 从计算图里 detach 掉，让 Q 分支只通过 TD 误差更新自己。
2. Cont-loss 部分：重新前向一次，冻结 adv_fc/val_fc 的梯度，只让 cont_fc 通过 -Q 学习。
   这样连续分支的目标是"提升被选动作的 Q 值"，而不是"模仿自己上一次的输出"。
3. 两步分别构造 loss，加权求和后一次 backward。
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import para

device = torch.device("cuda" if para.USE_CUDA and torch.cuda.is_available() else "cpu")


class HD3QNModel(nn.Module):
    """Hybrid D3QN 网络：共享特征层 + 连续参数分支 + Dueling Q 分支。"""

    def __init__(self, obs_dim: int, act_dim: int, hid_size: int = 128):
        super().__init__()
        self.act_dim = act_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
        )

        self.cont_fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, act_dim),
            nn.Sigmoid(),
        )

        q_input_dim = hid_size + act_dim
        self.adv_fc = nn.Sequential(
            nn.Linear(q_input_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, act_dim),
        )
        self.val_fc = nn.Sequential(
            nn.Linear(q_input_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def _compute_q(self, q_input: torch.Tensor) -> torch.Tensor:
        """共用的 Dueling Q 计算逻辑。"""
        adv = self.adv_fc(q_input)
        val = self.val_fc(q_input)
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

    def forward(self, obs):
        """对外接口保持不变：返回 Q 值和连续参数。"""
        feat = self.shared(obs)
        cont_params = self.cont_fc(feat)
        q_input = torch.cat([feat, cont_params], dim=1)
        q_values = self._compute_q(q_input)
        return q_values, cont_params


class HD3QN(nn.Module):
    """封装主网络、目标网络和 HD3QN 学习逻辑。"""

    def __init__(self, model: HD3QNModel, act_dim: int, gamma: float, lr: float):
        super().__init__()
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.act_dim = act_dim
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.reward_clip = 5.0
        self.td_target_clip = 20.0
        # cont_loss 在总 loss 中的权重。0.1 是 P-DQN 论文里的常用值，
        # 既能让连续分支真正学习，也不会冲垮 Q 分支的 TD 训练。
        self.cont_loss_weight = 0.1

    def predict_q(self, obs: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            q_values, _ = self.model(obs)
        return q_values

    def predict_both(self, obs: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            q_values, cont_params = self.model(obs)
        return q_values, cont_params

    def _set_q_head_requires_grad(self, flag: bool):
        """临时开关 adv_fc/val_fc 的梯度，用于第二步只更新 cont_fc。"""
        for p in self.model.adv_fc.parameters():
            p.requires_grad = flag
        for p in self.model.val_fc.parameters():
            p.requires_grad = flag

    def learn(
        self,
        obs: torch.Tensor,
        disc_action: torch.Tensor,
        cont_action: torch.Tensor,  # 保留入参以兼容主流程，但不再用于 cont_loss
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminal: torch.Tensor,
        next_action_mask=None,
    ):
        """两阶段构造 loss：
        - 阶段 1：q_loss，让 cont_fc 输出脱离梯度图。
        - 阶段 2：cont_loss = -Q，冻结 Q 分支头，只让 cont_fc 学习。
        然后把两个 loss 一起 backward。
        """
        self.model.train()
        disc_action_long = disc_action.long()

        # ========== 阶段 1：Q-loss ==========
        # 前向：cont_pred 被 detach 掉，避免 Q-loss 反传时影响 cont_fc。
        feat_q = self.model.shared(obs)
        cont_pred_q = self.model.cont_fc(feat_q).detach()
        q_input_q = torch.cat([feat_q, cont_pred_q], dim=1)
        q_pred_q = self.model._compute_q(q_input_q)
        q_selected = q_pred_q.gather(1, disc_action_long.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_main, _ = self.model(next_obs)
            if next_action_mask is not None:
                mask = next_action_mask.to(device=next_q_main.device, dtype=torch.bool)
                next_q_main = next_q_main.masked_fill(~mask, -1e9)
            next_greedy_action = next_q_main.argmax(dim=1)

            next_q_target, _ = self.target_model(next_obs)
            next_q_val = next_q_target.gather(1, next_greedy_action.unsqueeze(1)).squeeze(1)
            clipped_reward = reward.clamp(-self.reward_clip, self.reward_clip)
            td_target = clipped_reward + (1.0 - terminal.float()) * self.gamma * next_q_val
            td_target = td_target.clamp(-self.td_target_clip, self.td_target_clip)

        q_loss = F.smooth_l1_loss(q_selected, td_target)

        # ========== 阶段 2：Cont-loss = -Q ==========
        # 冻结 Q 头，让梯度只能流过 cont_fc。
        self._set_q_head_requires_grad(False)

        feat_c = self.model.shared(obs)
        cont_pred_c = self.model.cont_fc(feat_c)
        # feat 部分 detach，避免 cont_loss 通过 shared 层影响 Q 学习的特征表示。
        # 这样 shared 层主要由 q_loss 驱动，cont_fc 由 -Q 驱动，分工明确。
        q_input_c = torch.cat([feat_c.detach(), cont_pred_c], dim=1)
        q_for_cont = self.model._compute_q(q_input_c)
        # 让被选动作的 Q 值最大化 → 取负作为 loss
        cont_loss = -q_for_cont.gather(1, disc_action_long.unsqueeze(1)).squeeze(1).mean()

        # 恢复 Q 头的梯度
        self._set_q_head_requires_grad(True)

        # ========== 联合 backward ==========
        total_loss = q_loss + self.cont_loss_weight * cont_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return total_loss, q_loss, cont_loss

    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

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


class Agent:
    """HD3QN 智能体，对外接口完全保留。"""

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

    def _to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    def sample(self, obs, action_mask=None) -> int:
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
        obs_t = self._to_tensor(obs)
        q_values = self.alg.predict_q(obs_t)
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=q_values.device).unsqueeze(0)
            q_values = q_values.masked_fill(~mask_t, -1e9)
        return int(q_values.argmax(dim=-1).item())

    def get_continuous(self, obs, discrete_action: int) -> float:
        obs_t = self._to_tensor(obs)
        _, cont_params = self.alg.predict_both(obs_t)
        return float(cont_params[0, discrete_action].item())

    def learn(
        self,
        obs,
        disc_act,
        cont_act,
        reward,
        next_obs,
        terminal,
        next_action_mask=None,
    ):
        self.global_step += 1

        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        disc_act_t = torch.tensor(disc_act, dtype=torch.long).to(device)
        cont_act_t = torch.tensor(cont_act, dtype=torch.float32).to(device)
        reward_t = torch.tensor(reward, dtype=torch.float32).to(device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
        terminal_t = torch.tensor(terminal, dtype=torch.bool).to(device)

        if next_action_mask is not None:
            next_mask_t = torch.tensor(next_action_mask, dtype=torch.bool).to(device)
        else:
            next_mask_t = None

        total_loss, q_loss, cont_loss = self.alg.learn(
            obs_t,
            disc_act_t,
            cont_act_t,
            reward_t,
            next_obs_t,
            terminal_t,
            next_action_mask=next_mask_t,
        )

        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()

        return total_loss.item(), q_loss.item(), cont_loss.item()

    def predict_q_values(self, obs, action_mask=None):
        """诊断用：返回三个离散动作的 Q 值数组，不做 argmax。"""
        obs_t = self._to_tensor(obs)
        q_values = self.alg.predict_q(obs_t)
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=q_values.device).unsqueeze(0)
            q_values = q_values.masked_fill(~mask_t, -1e9)
        return q_values.squeeze(0).cpu().numpy()