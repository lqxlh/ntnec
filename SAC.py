import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import math
import numpy as np
import para

# SAC 和 D3QN 共用同一套设备配置，避免一个在 CPU、一个在 GPU。
device = torch.device("cuda" if para.USE_CUDA and torch.cuda.is_available() else "cpu")

# 演员网络（策略网络）：输出高斯分布，生成动作
class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorModel, self).__init__()
        # 隐藏层神经元数量
        hid_size = 128

        # 全连接层定义：状态 -> 隐藏层
        self.fc1 = nn.Linear(in_features=obs_dim, out_features=hid_size)
        self.fc2 = nn.Linear(in_features=hid_size, out_features=hid_size)
        # 输出层：动作均值（高斯分布核心参数）
        self.fc_mean = nn.Linear(in_features=hid_size, out_features=act_dim)
        # 输出层：动作标准差（高斯分布核心参数）
        self.fc_std = nn.Linear(in_features=hid_size, out_features=act_dim)

    # 核心方法：根据状态生成带探索的随机动作（训练用）
    def policy(self, obs):
        # 前向传播：两层ReLU激活的隐藏层
        hid1 = F.relu(self.fc1(obs)).squeeze(0)
        hid2 = F.relu(self.fc2(hid1))
        # 计算高斯分布的均值
        act_mean = self.fc_mean(hid2)
        # 计算标准差：softplus保证标准差>0，clamp防止数值过小/除零
        act_std = F.softplus(self.fc_std(hid2))
        act_std = torch.clamp(act_std, min=1e-6)
        # 构建正态分布（高斯策略）
        dist = torch.distributions.Normal(act_mean, act_std)
        # 重参数化采样：保证梯度可回传（SAC核心技巧）
        sample = dist.rsample()
        # tanh压缩：将动作限制在[-1,1]区间
        action = torch.tanh(sample)

        # 计算动作的对数概率（修正tanh的雅可比行列式，保证概率计算正确）
        log_prob = dist.log_prob(sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        # 归一化：将动作从[-1,1]缩放到[0,1]，适配环境动作空间
        action = (action + 1) / 2
        return action, log_prob, sample

    # 预测方法：根据状态生成确定性动作（测试/部署用，无探索）
    def predict(self, obs):
        # 前向传播
        hid1 = F.relu(self.fc1(obs)).squeeze(0)
        hid2 = F.relu(self.fc2(hid1))
        # 直接用均值作为动作（去掉随机采样）
        act_mean = self.fc_mean(hid2)
        # 压缩+归一化
        action = torch.tanh(act_mean)
        action = (action + 1) / 2
        return action

#评论家网络（价值网络）
# 作用：输入【状态+动作】，输出Q值（评估动作的好坏）
# SAC使用双Q网络：防止价值过估计，提升训练稳定性
# Critic Model (Value Function)
class CriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CriticModel, self).__init__()

        hid_size = 128

        # 第一个Q网络（Q1）
        self.fc1 = nn.Linear(in_features=obs_dim + act_dim, out_features=hid_size)  
        self.fc2 = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc3 = nn.Linear(in_features=hid_size, out_features=1)

        # 第二个Q网络（Q2）
        self.fc4 = nn.Linear(in_features=obs_dim + act_dim, out_features=hid_size)  
        self.fc5 = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc6 = nn.Linear(in_features=hid_size, out_features=1)

    # 计算双Q值
    def value(self, obs, act):
        # 拼接状态和动作作为网络输入
        concat = torch.cat((obs, act), dim=1)

        # 计算第一个Q网络的输出
        hid1 = F.relu(self.fc1(concat))
        hid2 = F.relu(self.fc2(hid1))
        Q1 = self.fc3(hid2)

        # 计算第二个Q网络的输出
        hid3 = F.relu(self.fc4(concat))
        hid4 = F.relu(self.fc5(hid3))
        Q2 = self.fc6(hid4)

        return Q1, Q2

#作用：封装评论家网络，统一接口，方便管理参数
class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.critic_model = CriticModel(obs_dim, act_dim)

    # 调用评论家网络计算Q值
    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    # 获取评论家网络的参数
    def get_critic_params(self):
        return self.critic_model.parameters()


# 软Actor-Critic：最大化奖励+策略熵，平衡探索与利用
class SAC(nn.Module):
    def __init__(self,
                 model,         # 主价值网络
                 actor_model,   # 演员网络
                 gamma=None,    # 折扣因子：未来奖励的权重
                 tau=None,      # 软更新系数：目标网络更新幅度
                 actor_lr=None, # 演员网络学习率
                 critic_lr=None,# 评论家网络学习率
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2):

        super(SAC, self).__init__()

        # 超参数赋值
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        # 温度系数α：自动调整（控制熵的权重，SAC核心）
        self.log_alpha = torch.tensor(math.log(0.01),device=device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -1
        self.actor_model = actor_model.to(device)
        self.model = model
        # 目标价值网络：深拷贝主网络，延迟更新，提升训练稳定性
        self.target_model = copy.deepcopy(self.model)
        self.model.to(device)
        self.target_model.to(device)
        # 优化器：分别优化演员、评论家、温度系数α
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.model.get_critic_params(), lr=self.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.actor_lr)

    # 训练时：带探索的动作预测
    def predict(self, obs):
        self.model.eval()       # 切换评估模式
        with torch.no_grad():   # 关闭梯度计算
            obs = torch.tensor(obs, dtype=torch.float32)
            action, _, _ = self.actor_model.policy(obs)
        return action.cpu().numpy()

    # 测试时：确定性动作预测（无探索）
    def predict_e(self, obs):
        self.model.eval()
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            action = self.actor_model.predict(obs)
        return action.cpu().numpy()

    # 总学习入口：更新评论家 → 更新演员
    def learn(self, obs, action, reward, next_obs, terminal, global_step):
        self.model.train()# 切换训练模式
        # 更新评论家网络
        critic_loss = self._critic_learn(obs, action, reward, next_obs, terminal)
        # 更新演员网络 + 温度系数α
        actor_loss = self._actor_learn(obs)
        return actor_loss, critic_loss

    # 评论家网络学习：最小化Q值预测误差
    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():# 目标网络无梯度
            # 维度扩展，适配网络输入
            reward = reward.unsqueeze(1)
            terminal = terminal.unsqueeze(1).float()
            # 下一状态的动作 + 对数概率
            next_action, next_log_pro, _ = self.actor_model.policy(next_obs)
            next_entropy = -next_log_pro

            # 目标网络计算下一状态的Q值（取双Q最小值）
            target_q1, target_q2 = self.target_model.value(next_obs, next_action)
            # 目标Q值 = 奖励 + 折扣*(最小Q值 + α*熵)
            target_q = torch.min(target_q1, target_q2) + self.log_alpha.exp() * next_entropy
            # 终止状态无未来奖励
            target_q = reward + (1.0 - terminal) * self.gamma * target_q
            target_q = target_q.detach()

        # 主网络计算当前Q值
        current_q1, current_q2 = self.model.value(obs, action.unsqueeze(1))
        # 损失函数：均方误差（双Q网络损失相加）
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 反向传播更新评论家
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    # 演员网络学习：最大化 Q值 + 策略熵
    def _actor_learn(self, obs):
        # 当前状态的动作 + 对数概率
        action, log_pro, pre_act = self.actor_model.policy(obs)
        entropy = - log_pro
        # 取双Q最小值
        Q1, Q2 = self.model.value(obs, action)
        Q = torch.min(Q1, Q2)
        # 演员损失：最大化 -(α*熵 + Q) → 等价于最小化损失
        actor_loss = (-self.log_alpha.exp() * entropy - Q).mean()
        # 温度系数损失：自动调整α，让熵接近目标值
        loss_alpha = self.log_alpha.exp() * (entropy - self.target_entropy).detach()
        loss_alpha = loss_alpha.mean()

        # 优化actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新温度系数α
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        return actor_loss.item()

    # 软更新目标网络：指数移动平均（EMA）
    def sync_target(self, decay=None):
        if decay is None:
            decay = 1. - self.tau
        # 逐参数更新：目标参数 = 旧目标参数*衰减 + 主参数*(1-衰减)
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(decay * target_param.data + (1.0 - decay) * param.data)

    # 保存模型：网络参数 + 优化器参数 + 超参数
    def save_model(self, path):
        torch.save({
            'actor_model_state_dict': self.actor_model.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    # 加载模型
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor_model.load_state_dict(checkpoint['actor_model_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        # 切换评估模式
        self.actor_model.eval()
        self.model.eval()
        self.target_model.eval()

# 作用：统一对外接口，封装SAC算法，处理数据格式、目标网络同步
class Agent(nn.Module):
    def __init__(self, algorithm, obs_dim, act_dim):
        super(Agent, self).__init__()
        # 参数校验
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alg = algorithm

        self.global_step = 0
        self.update_target_steps = 2
        # 初始化目标网络（完全复制主网络）
        self.alg.sync_target(decay=0)
        self.a = 0

    # 训练：带探索的动作预测
    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.alg.predict(obs)
        action = np.squeeze(action)
        return action

    # 测试：确定性动作预测
    def predict_e(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.alg.predict_e(obs)
        action = np.squeeze(action)
        return action

    # 智能体学习：数据格式转换 + 目标网络同步 + 调用SAC学习
    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        # 数据转换：numpy数组 → PyTorch张量
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        act = torch.tensor(act, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(device)

        # 调用SAC核心学习逻辑
        actor_cost, critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal, self.global_step)
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        return critic_cost, self.alg.log_alpha.exp()
