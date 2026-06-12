"""
HD3QN.py - Hybrid Dueling Double DQN for NTN-MEC.

当前版本支持参数化离散动作：
- 离散动作决定任务执行位置：local / BS / single SAT / two-SAT split。
- 连续动作固定为 CONT_ACTION_DIM 维。
  普通动作只使用第 0 维；两星分片动作使用 0/1/2 三维。
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
    """共享状态编码 + 连续参数分支 + Dueling Q 分支。"""

    def __init__(self, obs_dim: int, act_dim: int, hid_size: int = 128):
        super().__init__()
        self.act_dim = act_dim
        self.cont_dim = int(getattr(para, "CONT_ACTION_DIM", 1))

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
        )

        # 每个离散动作都输出一组连续参数。
        # 形状会在 forward 中整理为 [batch, act_dim, cont_dim]。
        self.cont_fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, act_dim * self.cont_dim),
            nn.Sigmoid(),
        )

        # Q(s, a, x_a)：状态特征 + 动作 one-hot + 当前动作自己的连续参数。
        q_input_dim = hid_size + act_dim + self.cont_dim
        self.register_buffer("action_eye", torch.eye(act_dim))
        self.adv_fc = nn.Sequential(
            nn.Linear(q_input_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )
        self.val_fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def _format_cont_params(self, cont_params: torch.Tensor, batch_size: int) -> torch.Tensor:
        """把连续参数统一整理成 [batch, act_dim, cont_dim]。"""
        if cont_params.dim() == 3:
            return cont_params
        if cont_params.dim() == 2 and cont_params.shape[1] == self.act_dim:
            return cont_params.unsqueeze(-1).expand(-1, -1, self.cont_dim)
        return cont_params.view(batch_size, self.act_dim, self.cont_dim)

    def _compute_q(self, feat: torch.Tensor, cont_params: torch.Tensor) -> torch.Tensor:
        """根据状态特征和每个动作对应的连续参数计算所有离散动作 Q 值。"""
        batch_size = feat.shape[0]
        cont_params = self._format_cont_params(cont_params, batch_size)

        action_eye = self.action_eye.to(device=feat.device, dtype=feat.dtype)
        action_eye = action_eye.unsqueeze(0).expand(batch_size, -1, -1)
        feat_rep = feat.unsqueeze(1).expand(-1, self.act_dim, -1)

        q_input = torch.cat([feat_rep, action_eye, cont_params], dim=-1)
        q_input = q_input.reshape(batch_size * self.act_dim, -1)
        adv = self.adv_fc(q_input).view(batch_size, self.act_dim)
        val = self.val_fc(feat)
        # Dueling 聚合公式：减去优势均值，避免 V(s) 和 A(s,a) 互相吸收同一部分 Q 值。
        return val + adv - adv.mean(dim=1, keepdim=True)

    def forward(self, obs):
        """返回 Q 值和连续动作参数。"""
        feat = self.shared(obs)
        cont_params = self.cont_fc(feat).view(obs.shape[0], self.act_dim, self.cont_dim)
        q_values = self._compute_q(feat, cont_params)
        return q_values, cont_params


class HD3QN(nn.Module):
    """封装主网络、目标网络、TD 学习和连续分支优化。"""

    def __init__(self, model: HD3QNModel, act_dim: int, gamma: float, lr: float):
        super().__init__()
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.act_dim = act_dim
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=para.lr_step_size,
            gamma=para.lr_gamma,
        )
        self.td_target_clip = 20.0
        self.cont_loss_weight = 0.04
        # 分片比例边界惩罚只约束 split 动作的连续第 0 维，避免比例输出长期贴在 0 或 1。
        self.split_ratio_boundary_weight = float(getattr(para, "SPLIT_RATIO_BOUNDARY_WEIGHT", 0.0))

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
        """训练连续分支时临时冻结 Q 头，避免 Q 头被 -Q 目标带偏。"""
        for p in self.model.adv_fc.parameters():
            p.requires_grad = flag
        for p in self.model.val_fc.parameters():
            p.requires_grad = flag

    def _format_batch_cont_action(self, cont_action: torch.Tensor) -> torch.Tensor:
        """把 replay buffer 中的连续动作整理成 [batch, cont_dim]。"""
        cont_action = cont_action.float()
        if cont_action.dim() == 1:
            cont_action = cont_action.unsqueeze(1)
        if cont_action.shape[1] < self.model.cont_dim:
            pad_width = self.model.cont_dim - cont_action.shape[1]
            cont_action = F.pad(cont_action, (0, pad_width), value=0.5)
        return cont_action[:, : self.model.cont_dim]

    def learn(
        self,
        obs: torch.Tensor,
        disc_action: torch.Tensor,
        cont_action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminal: torch.Tensor,
        next_action_mask=None,
        is_weights=None,
    ):
        self.model.train()
        disc_action_long = disc_action.long()
        cont_action = self._format_batch_cont_action(cont_action)

        # 阶段 1：Q-loss。这里使用 replay 中真实执行过的连续动作。
        feat_q = self.model.shared(obs)
        with torch.no_grad():
            # 未执行动作没有 replay 里的真实连续参数，用当前连续分支的预测值填充更贴近网络实际输出。
            cont_for_q = self.model.cont_fc(feat_q).view(obs.shape[0], self.act_dim, self.model.cont_dim)
        # clone 后再写入真实执行动作，避免原张量被原地 scatter 修改导致调试时难以追踪。
        cont_for_q = cont_for_q.clone()
        action_index = disc_action_long.view(-1, 1, 1).expand(-1, 1, self.model.cont_dim)
        cont_for_q.scatter_(1, action_index, cont_action.unsqueeze(1))
        q_pred_q = self.model._compute_q(feat_q, cont_for_q)
        q_selected = q_pred_q.gather(1, disc_action_long.unsqueeze(1)).squeeze(1)

        # Double DQN target：主网络选动作，目标网络估值。
        with torch.no_grad():
            next_q_main, _ = self.model(next_obs)
            if next_action_mask is not None:
                mask = next_action_mask.to(device=next_q_main.device, dtype=torch.bool)
                next_q_main = next_q_main.masked_fill(~mask, -1e9)
            next_greedy_action = next_q_main.argmax(dim=1)

            next_q_target, _ = self.target_model(next_obs)
            next_q_val = next_q_target.gather(1, next_greedy_action.unsqueeze(1)).squeeze(1)
            # 环境里已经对单步 reward 做过下限保护，这里只保留 TD target 的整体数值保护。
            td_target = reward + (1.0 - terminal.float()) * self.gamma * next_q_val
            td_target = td_target.clamp(-self.td_target_clip, self.td_target_clip)

        td_errors = torch.abs(q_selected - td_target).detach().cpu().numpy()
        elementwise_loss = F.smooth_l1_loss(q_selected, td_target, reduction="none")
        if is_weights is not None:
            is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=q_selected.device)
            q_loss = torch.mean(elementwise_loss * is_weights_t)
        else:
            q_loss = torch.mean(elementwise_loss)

        # 阶段 2：cont-loss。连续分支通过提升当前动作 Q 值学习资源分配。
        self._set_q_head_requires_grad(False)
        # 复用阶段 1 已经算出的 shared 特征；detach 后 cont-loss 不会更新 shared encoder。
        feat_c = feat_q.detach()
        cont_pred_c = self.model.cont_fc(feat_c).view(obs.shape[0], self.act_dim, self.model.cont_dim)
        q_for_cont = self.model._compute_q(feat_c, cont_pred_c)
        cont_loss = -q_for_cont.gather(1, disc_action_long.unsqueeze(1)).squeeze(1).mean()
        # 只在两星分片样本上加很轻的边界惩罚，让 raw ratio 不要塌缩到 0/1 输出边界。
        ratio_boundary_loss = torch.zeros((), dtype=cont_loss.dtype, device=cont_loss.device)
        split_action_start = int(para.N + para.S + 1)
        split_mask = disc_action_long >= split_action_start
        if self.split_ratio_boundary_weight > 0 and split_mask.any() and self.model.cont_dim > 0:
            selected_cont = cont_pred_c.gather(
                1,
                disc_action_long.view(-1, 1, 1).expand(-1, 1, self.model.cont_dim),
            ).squeeze(1)
            ratio_raw = selected_cont[split_mask, 0].clamp(1e-4, 1.0 - 1e-4)
            ratio_boundary_loss = -(torch.log(ratio_raw) + torch.log(1.0 - ratio_raw)).mean()
        self._set_q_head_requires_grad(True)

        total_loss = q_loss + self.cont_loss_weight * cont_loss + self.split_ratio_boundary_weight * ratio_boundary_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return total_loss, q_loss, cont_loss, td_errors

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
    """HD3QN 智能体接口。"""

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

    def get_continuous(self, obs, discrete_action: int, explore: bool = False):
        """返回当前离散动作对应的连续动作向量。"""
        obs_t = self._to_tensor(obs)
        _, cont_params = self.alg.predict_both(obs_t)
        val = cont_params[0, discrete_action].detach().cpu().numpy().astype(np.float32)

        # 训练阶段加高斯噪声，评估阶段保持确定性输出。
        if explore:
            noise_scale = 0.15 * self.e_greed
            noise = np.random.normal(0, noise_scale, size=val.shape)
            # 分片动作的第 0 维是任务比例，单独给更稳定的探索强度，避免一开始贴住最小比例。
            split_action_start = int(para.N + para.S + 1)
            if discrete_action >= split_action_start and val.shape[0] > 0:
                ratio_noise_scale = max(float(para.SPLIT_RATIO_NOISE_FLOOR), noise_scale)
                noise[0] = np.random.normal(0, ratio_noise_scale)
                if np.random.rand() < float(para.SPLIT_RATIO_EXPLORE_PROB) * max(self.e_greed, 0.1):
                    val[0] = np.random.uniform(0.1, 0.9)
            val = np.clip(val + noise, 0.0, 1.0).astype(np.float32)
        return val

    def learn(
        self,
        obs,
        disc_act,
        cont_act,
        reward,
        next_obs,
        terminal,
        next_action_mask=None,
        is_weights=None,
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

        total_loss, q_loss, cont_loss, td_errors = self.alg.learn(
            obs_t,
            disc_act_t,
            cont_act_t,
            reward_t,
            next_obs_t,
            terminal_t,
            next_action_mask=next_mask_t,
            is_weights=is_weights,
        )

        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()

        return total_loss.item(), q_loss.item(), cont_loss.item(), td_errors

    def predict_q_values(self, obs, action_mask=None):
        """诊断用：返回当前状态下所有离散动作的 Q 值。"""
        obs_t = self._to_tensor(obs)
        q_values = self.alg.predict_q(obs_t)
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=q_values.device).unsqueeze(0)
            q_values = q_values.masked_fill(~mask_t, -1e9)
        return q_values.squeeze(0).cpu().numpy()
