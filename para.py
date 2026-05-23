import math

# 运行模式配置：
# 这里用于在“快速调试”和“论文正式训练”之间切换。
RUN_MODE = "formal"

# 硬件配置：
# 这里统一管理 PyTorch 使用的线程数和是否优先启用 CUDA。
CPU_THREADS = 6
USE_CUDA = True

# 预设训练档位：
# - debug：优先验证代码是否能跑通；
# - formal：按论文实验规模运行。
RUN_PROFILES = {
    "debug": {
        "memory_size": 20000,
        "sac_memory_size": 20000,
        "memory_warmup_size": 128,
        "sac_memory_warmup_size": 128,
        "steps": 40,
        "max_episode": 12,
        "seed": [1],
        "batch_size": 64,
        "sac_batch_size": 64,
        "eval_interval": 1,
        "eval_rounds": 1,
    },
    "formal": {
        "memory_size": 100000,
        "sac_memory_size": 100000,
        "memory_warmup_size": 2000,
        "sac_memory_warmup_size": 2000,
        "steps": 200,
        "max_episode": 300,
        "seed": [1],
        "batch_size": 256,
        "sac_batch_size": 256,
        "eval_interval": 1,
        "eval_rounds": 2,
    },
}

if RUN_MODE not in RUN_PROFILES:
    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")

ACTIVE_PROFILE = RUN_PROFILES[RUN_MODE]

# 强化学习超参数：
# 这里直接对应论文的 DRL 参数表。
LEARN_FREQ = 1
SAC_LEARN_FREQ = 1
MEMORY_SIZE = ACTIVE_PROFILE["memory_size"]
SAC_MEMORY_SIZE = ACTIVE_PROFILE["sac_memory_size"]
MEMORY_WARMUP_SIZE = ACTIVE_PROFILE["memory_warmup_size"]
SAC_MEMORY_WARMUP_SIZE = ACTIVE_PROFILE["sac_memory_warmup_size"]
BATCH_SIZE = ACTIVE_PROFILE["batch_size"]
SAC_BATCH_SIZE = ACTIVE_PROFILE["sac_batch_size"]
LEARNING_RATE = 1e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
DQN_GAMMA = 0.99
SAC_GAMMA = 0.99
TAU = 0.005
EVAL_INTERVAL = ACTIVE_PROFILE["eval_interval"]
EVAL_ROUNDS = ACTIVE_PROFILE["eval_rounds"]

# 训练稳定性分析参数：
# 下面这组参数不改变论文里的 MDP 定义，也不改变状态/动作/奖励本身，
# 只是用于“如何判断当前联合策略是否已经进入稳定有效区间”以及“如何避免过了最佳轮次后继续学坏”。
# 其中：
# 1. REWARD_SMOOTH_WINDOW 用于对评估回报做滑动平均，减小单次随机轨道/任务采样造成的抖动；
# 2. STABILITY_* 指标把“高回报、低超时、适度卫星参与”综合成一个稳定性分数；
# 3. EARLY_STOP_* 用于当评估长期不再改善时停止训练，并保留最佳 checkpoint。
REWARD_SMOOTH_WINDOW = 5

# 场景基础参数：
# 这里对应论文中的 N 个地面 ES、S 颗卫星、M 个移动设备。
N = 3
S = 3
M = 10

# 奖励函数中的权重：
# 这里直接对应论文代价函数 phi_{m,t} 里的 w_D、w_E、w_V。
w_t = 0.9
w_e = 0.1
# 这里参考目标论文的优化框架，在总时延和总能耗之外，
# 再加入一个较小的资源占用代价权重，用于度量 BS/卫星算力资源使用强度。
w_v = 1.0

# 资源配置：
# 1. 地面 ES 总算力取 10 GHz；
# 2. 本地设备1 GHz
# 3. 星载节点这里先上调到 6 GHz，目标是先让 NTN 卫星分支具备“存在可行解”的条件；
# 4. 连续动作映射范围分别对应论文中的地面/卫星频率范围。
# 这一版进一步加入“削弱本地优势”的设定，目标是避免评估策略稳定退化成“始终本地执行”。
F_BS = [10e9, 10e9, 10e9]
F_MD = 1e9
F_MD_MIN = 0.1
F_MD_MAX = 1.0
SAT_F = [6e9, 6e9, 6e9]
SAT_F_MIN = 0.8e9
SAT_F_MAX = 3.0e9
BS_F_MIN = 0.5e9
BS_F_MAX = 3.0e9

# 当前模式对应的训练规模。
steps = ACTIVE_PROFILE["steps"]
zeta = 1
A_max = 2
max_episode = ACTIVE_PROFILE["max_episode"]
alpha = math.pi / 6
SEED = ACTIVE_PROFILE["seed"]

# 地图与移动模型：
# 这里保留原代码的二维仿真区域，用于第一阶段 NTN 简化实验。
MAP_WIDTH = 1600
MAP_HEIGHT = 1200
MAX_MD_SPEED = 10

# 任务参数：
# 这里直接对应论文任务向量 l_{m,t} = [B, C, Gamma, V]^T 的取值范围。
# 这一步先做卫星可行性校准：
# 1. 适度减小任务数据量上界和计算强度上界，降低星载计算与传输压力；
# 2. 放宽任务最大容忍时延 Gamma，给“传播 + ISL + 传输 + 计算”留下实际可行窗口。
# 在此基础上，为了削弱“本地永远最优”的趋势，这里再略微抬高任务计算强度，
# 使部分任务在终端本地执行时更容易暴露出时延劣势。
TASK_B_MIN = int(0.8e6)
TASK_B_MAX = int(1.6e6)
TASK_C_MIN = 100
TASK_C_MAX = 850
TASK_GAMMA_MIN = 1.2
TASK_GAMMA_MAX = 1.5
TASK_PRIORITY_MIN = 1
TASK_PRIORITY_MAX = 3

# 地面链路参数：
# 这些值对应论文地面 MEC 基线部分的带宽、噪声、路径损耗和发射功率设定。
BS_HEIGHT = 30
GROUND_BW = 2e7
GROUND_NOISE = 1e-13
GROUND_GAIN_BETA = 1e-4
GROUND_PATHLOSS = 3.0
MD_MAX_POWER = 0.1

# 地面 MEC 拥塞建模：
# 这里保留地面排队/切换时延近似，用来模拟论文中的“地面侧拥塞成本”。
# 当前先恢复为较温和的地面拥塞设定，方便单独观察“削弱本地优势”这一步的影响，
# 避免地面拥塞和本地算力变化同时生效，导致实验解释混在一起。
GROUND_QUEUE_DELAY_MIN = 0.00
GROUND_QUEUE_DELAY_MAX = 0.08
GROUND_HOTSPOT_PROB = 0.35
GROUND_HOTSPOT_DELAY_EXTRA = 0.10
GROUND_HANDOVER_DELAY_SCALE = 1.25e9
ENABLE_GROUND_CONGESTION = 1

# NTN 参数：
# 这里对应论文卫星链路、可见性和传播时延部分的核心参数。
# 当前这组值优先服务于“先让卫星链路可行、让强化学习真正接触到卫星动作”：
# 1. 增大卫星带宽，降低为满足时延约束所需的传输时长与发射功率；
# 2. 增大卫星天线增益，提升卫星链路有效信道增益；
# 3. 后续若卫星动作已经能稳定出现，再逐步把这些参数收紧回论文正式设定。
SAT_HEIGHT = 550e3
SAT_MIN_ELEVATION_DEG = 10.0
SAT_BW = 6e7
SAT_NOISE = 3.1622776601683796e-14
SAT_GAIN = 1e5
MD_GAIN = 3.162
ATM_LOSS_LINEAR = 1.4125
SAT_ETA_D = 0.95
SAT_RICIAN_K = 10.0
LIGHT_SPEED = 3e8
SAT_CARRIER_FREQ = 26e9
SAT_WAVELENGTH = LIGHT_SPEED / SAT_CARRIER_FREQ

# 卫星链路增益归一化：
# g_{s,m}^t 的量级很小，这里用 log10 区间归一化后再送入神经网络状态。
SAT_GAIN_LOG_MIN = -20.0
SAT_GAIN_LOG_MAX = -10.0

# 卫星二维投影与移动近似：
# 虽然当前已经切到 SGP4 传播，但环境状态里仍然使用“局部二维投影坐标 + 固定轨道高度”
# 来和现有 EdgeEnv / MD 接口兼容。
SAT_MAX_HORIZ_SPEED = 7500.0
SAT_PROJECTION_X_MIN = -3.5e6
SAT_PROJECTION_X_MAX = 3.5e6
SAT_PROJECTION_Y_MIN = -3.5e6
SAT_PROJECTION_Y_MAX = 3.5e6

# 卫星时间推进参数：
# 1. SAT_DECISION_DT 表示一次设备级决策对应的物理时间推进；
# 2. SAT_TRAJECTORY_STEPS 表示一个 episode 里预计算的卫星传播步数。
SAT_DECISION_DT = 1.0 / M
SAT_TRAJECTORY_STEPS = steps * M + 2

# SGP4 默认轨道参数：
# 这里对应论文里的“550 km 级 LEO 卫星”。
# 当前仓库还没有外部 TLE 文件，因此先用这些默认轨道根数构造 3 颗示例卫星：
# - inclination_deg：轨道倾角，决定卫星覆盖带方向；
# - raan_deg：升交点赤经，用来区分不同轨道平面；
# - mean_anomaly_deg：初始平近点角，用来区分同轨道平面内的相位；
# - eccentricity：偏心率，论文里默认近圆轨道，因此设得很小；
# - arg_perigee_deg：近地点幅角；
# - mean_motion_rev_per_day：平均角速度，550 km 高度附近约 15 rev/day。
SAT_EPOCH = (2026, 1, 1, 0, 0, 0)
SAT_DEFAULT_ORBITS = [
    {
        "inclination_deg": 53.0,
        "raan_deg": 0.0,
        "mean_anomaly_deg": 0.0,
        "eccentricity": 0.0001,
        "arg_perigee_deg": 0.0,
        "mean_motion_rev_per_day": 15.05,
    },
    {
        "inclination_deg": 53.0,
        "raan_deg": 120.0,
        "mean_anomaly_deg": 120.0,
        "eccentricity": 0.0001,
        "arg_perigee_deg": 0.0,
        "mean_motion_rev_per_day": 15.05,
    },
    {
        "inclination_deg": 53.0,
        "raan_deg": 240.0,
        "mean_anomaly_deg": 240.0,
        "eccentricity": 0.0001,
        "arg_perigee_deg": 0.0,
        "mean_motion_rev_per_day": 15.05,
    },
]

# 局部投影偏移：
# SGP4 直接给出的是地心惯性系位置。为了和当前二维仿真区域兼容，
# 我们把卫星相对其初始时刻的位置变化投影到局部平面，并叠加一个固定偏移，
# 这样既保留真实轨道传播带来的速度/方向变化，也能形成有区分度的可见窗口。
SAT_LOCAL_OFFSETS = [
    (0.0, 0.0),
    (1.2e6, 6.0e5),
    (-1.2e6, -6.0e5),
]

# 星间链路参数：
# 这里对应论文中的 T_{m,t}^{ISL} = B_{m,t} / R_{s,s'}^{ISL}。
SAT_ISL_MAX_RATE = 1e9
SAT_ISL_PARETO_SHAPE = 1.5
SAT_ISL_LOAD_CLIP = 0.9

# 传播与链路的归一化辅助上界：
# 这些量只用于把距离、传播时延等输入状态归一化到 [0,1]。
MAX_SAT_HORIZONTAL_DISTANCE = math.sqrt(
    (SAT_PROJECTION_X_MAX - SAT_PROJECTION_X_MIN) ** 2 +
    (SAT_PROJECTION_Y_MAX - SAT_PROJECTION_Y_MIN) ** 2
)
MAX_SAT_DISTANCE = math.sqrt(MAX_SAT_HORIZONTAL_DISTANCE ** 2 + SAT_HEIGHT ** 2)
MAX_PROP_DELAY = MAX_SAT_DISTANCE / LIGHT_SPEED

# 约束惩罚：
# 这里对应论文奖励函数中的 r_time、r_fre、r_vis、r_prop。
PENALTY_TIME = -1.0
PENALTY_RESOURCE = -1.0
PENALTY_VISIBILITY = -1.0
PENALTY_PROPAGATION = -1.0
PENALTY_ZERO_ALLOCATION = -1.0

# 卫星软约束参数：
# 这里不是替换论文中的可见性、资源、时延预算、功率可行性等硬约束，
# 而是在“卫星动作已经可行”时，再给策略一个“避免单时隙过度依赖卫星”的平滑引导。
# 这样更适合解释当前实验里观察到的现象：
# 卫星适度参与可以改善性能，但卫星使用过猛时，系统奖励和成功率会明显波动。
ENABLE_SAT_LOAD_SOFT_PENALTY = 1

# SAT_TARGET_USAGE 表示单颗卫星在一个时隙内更稳妥的目标负载比例。
# 当某颗卫星的累计占用比例超过这个阈值后，奖励里会开始增加软惩罚。
SAT_TARGET_USAGE = 0.55

# SAT_LOAD_PENALTY_WEIGHT 控制这项“卫星过载软惩罚”的强度。
# 这里先取较小值，保证论文主目标仍然是时延-能耗-任务价值权衡，
# 软惩罚只作为稳定两阶段联合训练的辅助项。
SAT_LOAD_PENALTY_WEIGHT = 0.5
