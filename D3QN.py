import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
import para

# 设备选择遵循 para.py 里的统一配置：
# 正式训练时会优先尝试 CUDA，调试时也可以手动切回 CPU。
device = torch.device("cuda" if para.USE_CUDA and torch.cuda.is_available() else "cpu")

#Dueling 神经网络
class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        # 神经网络隐藏层大小
        hid_size = 128
        # 分支1：优势函数 A → 衡量「每个动作」的好坏
        self.fc1_adv = nn.Linear(in_features=obs_dim, out_features=hid_size)
        self.fc2_adv = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc3_adv = nn.Linear(in_features=hid_size, out_features=act_dim)
        # 分支2：价值函数 V → 衡量「当前状态」的好坏
        self.fc1_val = nn.Linear(in_features=obs_dim, out_features=hid_size)
        self.fc2_val = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc3_val = nn.Linear(in_features=hid_size, out_features=1)

    def forward(self, obs):
        # 计算优势A
        adv = F.relu(self.fc1_adv(obs))
        adv = F.relu(self.fc2_adv(adv))
        As = self.fc3_adv(adv)
        # 计算价值V
        val = F.relu(self.fc1_val(obs))
        val = F.relu(self.fc2_val(val))
        V = self.fc3_val(val)
        # 合并：最终Q值 = 状态价值 + 动作优势（Dueling核心公式）
        Q = As + (V - As.mean(dim=1, keepdim=True))
        # 输出每个动作的Q值（Q越大，动作越好）
        return Q

#双 DQN 算法，解决普通 DQN高估 Q 值的问题
class DDQN(nn.Module):
    def __init__(self, model, act_dim, gamma, lr):
        super(DDQN, self).__init__()

        self.model = model                      # 主网络（实时训练、选动作）
        self.target_model = copy.deepcopy(model)# 目标网络（计算目标值，稳定训练）
        self.model.to(device)
        self.target_model.to(device)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)# 优化器

    # 预测：输入状态，输出所有动作的Q值
    def predict(self, obs):
        self.model.eval()
        return self.model(obs)

    # 学习：用经验训练网络，更新参数
    def learn(self, obs, action, reward, next_obs, terminal, next_action_mask=None, learning_rate=None):
        self.model.train()
        if learning_rate is None:
            learning_rate = self.lr
        # 1. 主网络预测当前Q值
        pred_value = self.model(obs)
        # 2. 计算选中动作的Q值
        action_onehot = F.one_hot(action, num_classes=self.act_dim).float()
        pred_action_value = (action_onehot * pred_value).sum(dim=1)
        next_action_value = self.model(next_obs)
        if next_action_mask is not None:
            # 训练 target 也必须屏蔽下一状态的非法动作，否则 DQN 会从“评估时根本不能选”的动作上借到虚高 Q 值。
            next_action_mask = torch.as_tensor(next_action_mask, dtype=torch.bool, device=next_action_value.device)
            next_action_value = next_action_value.masked_fill(~next_action_mask, -1e9)
        # 3. DDQN核心：主网络选动作，目标网络算Q值
        greedy_action = next_action_value.argmax(dim=1)
        next_pred_value = self.target_model(next_obs)
        max_v = next_pred_value.gather(1, greedy_action.unsqueeze(1).long())
        max_v = max_v.squeeze(1)
        # 4. 计算目标Q值
        target = reward + (1.0 - terminal.float()) * self.gamma * max_v.detach()
        # 5. 计算损失 + 更新网络
        loss = F.mse_loss(pred_action_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    # 同步目标网络：定期复制主网络权重
    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),              # 保存 主神经网络 的所有参数（决策核心）
            'target_model_state_dict': self.target_model.state_dict(),# 保存 目标神经网络 的所有参数（DDQN训练专用）
            'optimizer_state_dict': self.optimizer.state_dict(),      # 保存 优化器 的状态（方便下次接着训练，不用从头开始）
        }, path)# path = 保存的文件路径

    def load_model(self, path):
        checkpoint = torch.load(path)                                           # 从文件中读取保存的所有数据
        self.model.load_state_dict(checkpoint['model_state_dict'])              # 加载主模型权重
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])# 加载目标模型权重
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])      # 加载优化器状态
        # 切换为【评估模式】
        self.model.eval()
        self.target_model.eval()


class Agent:
    def __init__(self, algorithm, obs_dim, act_dim, e_greed=0.1, e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alg = algorithm# 绑定DDQN算法
        self.global_step = 0
        self.update_target_steps = 300
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

    # 采样动作：训练用（带探索：随机尝试+最优选择）
    def sample(self, obs, action_mask=None):
        # action_mask 里 True 表示动作合法，False 表示当前时刻不允许选择这个动作。
        sample = np.random.rand()
        valid_actions = np.arange(self.act_dim)
        if action_mask is not None:
            valid_actions = valid_actions[np.array(action_mask, dtype=bool)]
            if len(valid_actions) == 0:
                valid_actions = np.arange(self.act_dim)
        if sample < self.e_greed:
            act = int(np.random.choice(valid_actions))# 随机探索时也只在合法动作里选
        else:
            act = self.predict(obs, action_mask=action_mask)              # 选最优动作
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    # 预测动作：测试用（纯最优选择，无探索）
    def predict(self, obs, action_mask=None):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_Q = self.alg.predict(obs)
        if action_mask is not None:
            # 对不可见卫星动作加上一个极小值，等价于论文里的可见性掩码。
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=pred_Q.device).unsqueeze(0)
            pred_Q = pred_Q.masked_fill(~mask_tensor, -1e9)
        act = torch.argmax(pred_Q, dim=-1).item()
        return act

    # 触发学习 + 定期同步目标网络
    def learn(self, obs, act, reward, next_obs, terminal, next_action_mask=None):
        self.global_step += 1
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        act = torch.tensor(act, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(device)
        if next_action_mask is not None:
            # next_action_mask 的形状是 [batch_size, action_dim]，True 表示下一状态下这个动作允许参与 target 计算。
            next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool).to(device)
        loss = self.alg.learn(obs, act, reward, next_obs, terminal, next_action_mask=next_action_mask)
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        return loss.item()
