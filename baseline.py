"""
run_baselines.py — NTN 系统六组基线对比实验
==============================================

维度一：物理拓扑消融实验（证明 LEO 卫星在 6G 中的必要性）
  BL1 · local_only      全本地计算：证明移动终端在密集任务下会毁灭性超时
  BL2 · no_satellite    纯地面 MEC：证明地面拥塞/边缘时系统性能剧烈下降
  BL3 · no_gnb          纯卫星+本地：证明卫星空间时延对极敏感任务的局限性

维度二：算法对比实验（证明 HD3QN 的先进性）
  BL4 · random_equal    随机分流 + 算力平分：性能下限
  BL5 · greedy_delay    贪心时延最优：证明多目标联合优化的必要性
  BL6 · dqn_heuristic   D3QN + 启发式频率：证明混合动作联合学习的收敛优势

使用方法：
  python run_baselines.py                  # 运行全部 6 个基线
  python run_baselines.py --only bl1 bl4   # 只运行指定基线
  python run_baselines.py --skip bl2 bl3   # 跳过指定基线

结果保存格式与 HD3QN_t.py 一致，可直接用同一套绘图脚本比较。
"""

import argparse
import math
import os
import pickle
import random
import warnings

import numpy as np
import torch

import BS
import D3QN
import EdgeEnv
import HD3QN
import MD
import ReplayMemory
import ReplayMemory_SAC
import SAT
import experiment_config
import para

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 超参数（与主实验对齐，保证公平比较）
# ─────────────────────────────────────────────────────────────────────────────
N = para.N
S = para.S
M = para.M
F_BS = para.F_BS
F_MD = para.F_MD
SAT_F = para.SAT_F
steps = para.steps
max_episode = para.max_episode
SEED = para.SEED
BATCH_SIZE = para.BATCH_SIZE
MEMORY_SIZE = para.MEMORY_SIZE
WARMUP_SIZE = para.MEMORY_WARMUP_SIZE
LEARNING_RATE = para.LEARNING_RATE
DQN_GAMMA = para.DQN_GAMMA
EVAL_INTERVAL = para.EVAL_INTERVAL
EVAL_ROUNDS = para.EVAL_ROUNDS
REWARD_SMOOTH_WINDOW = para.REWARD_SMOOTH_WINDOW
LEARN_FREQ = para.LEARN_FREQ

# ─────────────────────────────────────────────────────────────────────────────
# 拓扑掩码常量
# ─────────────────────────────────────────────────────────────────────────────
# 动作索引：0=本地, 1..N=地面BS, N+1..N+S=卫星
# N=1, S=2, action_dim=4: [local, BS, SAT0, SAT1]
TOPO_ALL = np.ones(N + S + 1, dtype=bool)  # 完整拓扑
TOPO_LOCAL_ONLY = np.array([True] + [False] * (N + S), dtype=bool)  # BL1: 只本地
TOPO_NO_SAT = np.array([True] * (N + 1) + [False] * S, dtype=bool)  # BL2: 无卫星
TOPO_NO_GNB = np.array([True] + [False] * N + [True] * S, dtype=bool)  # BL3: 无BS

# 结果前缀映射
BL_PREFIX = {
    "bl1": "baselines_local_only",
    "bl2": "baselines_no_satellite",
    "bl3": "baselines_no_gnb",
    "bl4": "baselines_random_equal",
    "bl5": "baselines_greedy_delay",
    "bl6": "baselines_dqn_heuristic",
}


# ═════════════════════════════════════════════════════════════════════════════
# 一、通用工具函数
# ═════════════════════════════════════════════════════════════════════════════

def build_entities(env):
    """初始化 BS / SAT / MD 列表（与 HD3QN_t.py 完全一致）。"""
    bs_list, md_list, sat_list = [], [], []
    for n in range(N):
        bs = BS.BS(n, F_BS[n],
                   random.uniform(0, para.MAP_WIDTH),
                   random.uniform(0, para.MAP_HEIGHT),
                   para.BS_HEIGHT)
        bs_list.append(bs)
    for s_idx in range(S):
        sat_list.append(SAT.SAT(s_idx, SAT_F[s_idx]))
    for m in range(M):
        md_list.append(MD.MD(m, F_MD, env, bs_list))
    return bs_list, md_list, sat_list


def apply_topology_mask(env_mask: np.ndarray, topo_mask: np.ndarray) -> np.ndarray:
    """将环境生成的动作掩码与拓扑约束取交集。

    若交集为空（极端边缘情况，如 BL3 卫星全不可见），强制保留本地动作作为最后退路。
    """
    combined = env_mask & topo_mask
    if not combined.any():
        combined[0] = True  # 本地计算永远可以作为 fallback
    return combined


def compute_smoothed_value(history, window=REWARD_SMOOTH_WINDOW):
    if not history:
        return 0.0
    return float(np.mean(history[-max(window, 1):]))


def summarize_debug(debug_info):
    """复用 HD3QN_t.py / D3SAC.py 的 summarize_debug 逻辑，保证字段一致。"""
    n = max(float(debug_info.get("steps", 0.0)), 1.0)
    sat_act = float(debug_info.get("sat_actions", 0.0))
    sat_ms = max(float(debug_info.get("sat_metric_steps", 0.0)), 1.0)
    sat_exec = float(debug_info.get("sat_exec_success_actions", 0.0))
    sat_dead = float(debug_info.get("sat_deadline_success_actions", 0.0))
    sat_to = float(debug_info.get("sat_timeout_actions", 0.0))
    sat_mc = max(float(debug_info.get("mask_sat_total_checks", 0.0)), 1.0)
    sat_fms = max(float(debug_info.get("sat_feasible_metric_steps", 0.0)), 1.0)
    sat_lpc = max(float(debug_info.get("sat_load_penalty_count", 0.0)), 1.0)
    sat_bfc = max(float(debug_info.get("mask_sat_remaining_budget_fail_count", 0.0)), 1.0)
    sat_pfc = max(float(debug_info.get("mask_sat_power_excess_ratio_count", 0.0)), 1.0)
    vis_d = float(debug_info.get("visible_sat_decisions", 0.0))

    return {
        "avg_total_delay": float(debug_info.get("avg_total_delay_sum", 0.0)) / n,
        "avg_prop_delay": float(debug_info.get("avg_prop_delay_sum", 0.0)) / n,
        "avg_total_energy": float(debug_info.get("avg_total_energy_sum", 0.0)) / n,
        "avg_reward": float(debug_info.get("reward_sum", 0.0)) / n,
        "avg_phi_cost": float(debug_info.get("avg_cost_sum", 0.0)) / n,
        "avg_task_value": float(debug_info.get("avg_value_sum", 0.0)) / n,
        "avg_delay_cost": float(debug_info.get("avg_delay_cost_sum", 0.0)) / n,
        "avg_energy_cost": float(debug_info.get("avg_energy_cost_sum", 0.0)) / n,
        "avg_selected_power": float(debug_info.get("avg_selected_power_sum", 0.0)) / n,
        "avg_g_bar_s": float(debug_info.get("avg_g_bar_s_sum", 0.0)) / n,
        "avg_remaining_budget_sat": float(debug_info.get("avg_remaining_budget_sat_sum", 0.0)) / n,
        "avg_P_min_sat": float(debug_info.get("avg_P_min_sat_sum", 0.0)) / n,
        "avg_p_star_sat": float(debug_info.get("avg_p_star_sat_sum", 0.0)) / n,
        "avg_g_sat_raw": float(debug_info.get("avg_g_sat_raw_sum", 0.0)) / n,
        "avg_penalty_time": float(debug_info.get("penalty_time_sum", 0.0)) / n,
        "avg_penalty_resource": float(debug_info.get("penalty_resource_sum", 0.0)) / n,
        "avg_penalty_visibility": float(debug_info.get("penalty_visibility_sum", 0.0)) / n,
        "avg_penalty_propagation": float(debug_info.get("penalty_propagation_sum", 0.0)) / n,
        "avg_penalty_zero_alloc": float(debug_info.get("penalty_zero_alloc_sum", 0.0)) / n,
        "sat_usage_rate": sat_act / n,
        "sat_exec_success_rate": sat_exec / max(sat_act, 1.0),
        "sat_deadline_success_rate": sat_dead / max(sat_act, 1.0),
        "sat_timeout_rate": sat_to / max(sat_act, 1.0),
        "avg_sat_delay_over_gamma": float(debug_info.get("sat_delay_over_gamma_sum", 0.0)) / max(sat_to, 1.0),
        "avg_sat_load_penalty": float(debug_info.get("sat_load_penalty_sum", 0.0)) / n,
        "avg_sat_usage_ratio_on_feasible_sat": float(debug_info.get("sat_usage_ratio_sum", 0.0)) / sat_fms,
        "avg_sat_peak_usage_ratio_on_feasible_sat": float(debug_info.get("sat_peak_usage_ratio_sum", 0.0)) / sat_fms,
        "max_sat_peak_usage_ratio": float(debug_info.get("sat_peak_usage_ratio_max", 0.0)),
        "sat_mask_feasible_rate": float(debug_info.get("mask_sat_feasible", 0.0)) / sat_mc,
        "sat_mask_not_visible_rate": float(debug_info.get("mask_sat_not_visible", 0.0)) / sat_mc,
        "sat_mask_no_resource_rate": float(debug_info.get("mask_sat_no_resource", 0.0)) / sat_mc,
        "sat_mask_non_positive_budget_rate": float(debug_info.get("mask_sat_non_positive_budget", 0.0)) / sat_mc,
        "sat_mask_power_infeasible_rate": float(debug_info.get("mask_sat_power_infeasible", 0.0)) / sat_mc,
        "visible_sat_decision_rate": vis_d / n,
    }


def save_results(prefix: str, train_rewards: list, eval_rewards: list,
                 debug_history: list, train_metrics: list, eval_metrics: list):
    """以与主实验完全相同的格式保存结果，方便统一绘图。"""
    os.makedirs("result", exist_ok=True)
    seed = SEED[0]
    with open(f"result/{prefix}_train_rewards.pickle", "wb") as f:
        pickle.dump([train_rewards], f)
    with open(f"result/{prefix}_eval_rewards.pickle", "wb") as f:
        pickle.dump([eval_rewards], f)
    with open(f"result/{prefix}_debug_history.pickle", "wb") as f:
        pickle.dump([debug_history], f)
    with open(f"result/{prefix}_train_metrics_seed_{seed}.pickle", "wb") as f:
        pickle.dump(train_metrics, f)
    with open(f"result/{prefix}_eval_metrics_seed_{seed}.pickle", "wb") as f:
        pickle.dump(eval_metrics, f)
    print(f"  → 结果已保存至 result/{prefix}_*.pickle")


# ═════════════════════════════════════════════════════════════════════════════
# 二、BL5 专用 — 乐观时延估算（Greedy Delay Oracle）
# ═════════════════════════════════════════════════════════════════════════════

def _estimate_optimistic_delay(md, action_idx: int, bs_list, sat_list) -> float:
    """对每种卸载路径计算乐观（最优信道、最大频率）时延估算。

    只用于 BL5 的贪心选择，不修改任何环境状态。

    BL5 设计原则：不考虑能耗和资源开销，以最短预估总时延为唯一指标。
    """
    B, C = md.B, md.C

    if action_idx == 0:
        # ── 本地计算 ──────────────────────────────────────────────────────
        # T_local = B * C / (F_MD_MAX * F_MD)  (最大算力比 × 真实频率)
        return B * C / (para.F_MD_MAX * para.F_MD)

    elif 1 <= action_idx <= N:
        # ── 地面 BS 卸载（乐观：最大发射功率 + SNR≈30dB + 最大BS算力）────
        # 地面传输速率：R = BW * log2(1 + SNR_30dB) ≈ 20MHz * log2(1001)
        rate_bs = para.GROUND_BW * math.log2(1.0 + 1000.0)  # SNR 30dB
        t_tran = B / rate_bs
        t_comp = B * C / para.BS_F_MAX
        # 忽略排队和切换时延（最乐观假设）
        return t_tran + t_comp

    else:
        # ── 卫星卸载（乐观：可见性已满足 + 直接接入 + 最大卫星算力）────
        sat_idx = action_idx - (N + 1)
        sat = sat_list[sat_idx]
        if not md.is_sat_visible(sat):
            return float("inf")  # 不可见卫星在贪心选择中直接排除

        t_prop = md.sat_distance(sat) / para.LIGHT_SPEED
        t_isl = 0.0  # 乐观假设：直接接入，无 ISL 转发时延
        # 卫星传输速率：R = BW * log2(1 + SNR_20dB) ≈ 60MHz * log2(101)
        rate_sat = para.SAT_BW * math.log2(1.0 + 100.0)  # SNR 20dB
        t_tran = B / rate_sat
        t_comp = B * C / para.SAT_F_MAX
        return t_prop + t_isl + t_tran + t_comp


# ═════════════════════════════════════════════════════════════════════════════
# 三、BL6 专用 — 启发式频率分配
# ═════════════════════════════════════════════════════════════════════════════

def _heuristic_freq(md, action_idx: int) -> float:
    """根据任务数据量 B 在 [B_MIN, B_MAX] 的相对位置线性插值分配频率。

    返回 cont_action ∈ [0.0, 1.0]，由 env.decode_frequency() 映射到实际频率。

    设计原理：
      任务越大（B 接近 B_MAX） → 分配更高频率来赶 deadline；
      任务越小（B 接近 B_MIN） → 分配较低频率节省能耗。
    公式：cont = (B - B_MIN) / (B_MAX - B_MIN)
    """
    b_ratio = (md.B - para.TASK_B_MIN) / max(para.TASK_B_MAX - para.TASK_B_MIN, 1)
    return float(np.clip(b_ratio, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# 四、无训练基线：BL1 / BL4 / BL5 的策略函数
# ═════════════════════════════════════════════════════════════════════════════

def _policy_local_only(obs, md, mask, bs_list, sat_list):
    """BL1 · Local-Only 策略：所有任务全部本地计算，频率拉满（减少超时）。"""
    return 0, 1.0


def _policy_random_equal(obs, md, mask, bs_list, sat_list):
    """BL4 · Random + Equal 策略：
    - 离散：在合法动作集中均匀随机选择（包括卫星）。
    - 连续：cont=0.5 → 对应 (F_MIN + F_MAX) / 2 的频率（平分策略）。
    """
    valid = np.where(mask)[0]
    disc = int(np.random.choice(valid))
    return disc, 0.5


def _policy_greedy_delay(obs, md, mask, bs_list, sat_list):
    """BL5 · Greedy Delay-Optimal 策略：
    - 离散：选乐观总时延最短的合法卸载目标。
    - 连续：cont=1.0 → 对应 F_MAX 的最大频率（配合贪心目标一致）。
    """
    best_action = -1
    best_delay = float("inf")
    for a in range(len(mask)):
        if not mask[a]:
            continue
        d = _estimate_optimistic_delay(md, a, bs_list, sat_list)
        if d < best_delay:
            best_delay = d
            best_action = a
    if best_action < 0:
        best_action = 0  # 极端情况 fallback：本地
    return best_action, 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 五、无训练基线通用回合执行器 (BL1 / BL4 / BL5)
# ═════════════════════════════════════════════════════════════════════════════

def _run_heuristic_episode(env, policy_fn, md_list, bs_list, sat_list,
                           topo_mask: np.ndarray = None):
    """运行一个回合（无学习）。

    policy_fn(obs_row, md, action_mask, bs_list, sat_list) → (disc, cont)
    topo_mask: 叠加拓扑约束（BL1 已内嵌于 policy_fn，此处主要给 BL2/BL3 测试用）
    """
    if topo_mask is None:
        topo_mask = TOPO_ALL

    total_reward = 0.0
    env.reset(md_list, bs_list, sat_list)
    step_idx = 0

    while step_idx < steps:
        obs = env.get_state()
        tcn = [0] * 6
        for md_idx in range(M):
            env_mask = env.get_action_mask(md_list[md_idx], sat_list)
            mask = apply_topology_mask(env_mask, topo_mask)
            disc, cont = policy_fn(obs[md_idx], md_list[md_idx], mask, bs_list, sat_list)
            obs, reward, _, _, _, _, _, _, _, _, _ = env.step(
                md_idx, disc, cont, tcn, bs_list, md_list, sat_list
            )
            total_reward += reward
        step_idx += 1
        env.reset_state(md_list, bs_list, sat_list)

    return total_reward, dict(env.last_debug)


def _eval_heuristic(env, policy_fn, md_list, bs_list, sat_list,
                    eval_rounds: int = 1, topo_mask: np.ndarray = None):
    """评估无学习策略，多轮取均值。"""
    if topo_mask is None:
        topo_mask = TOPO_ALL

    rewards, debug_list = [], []
    for _ in range(eval_rounds):
        env.reset(md_list, bs_list, sat_list)
        env.reset_debug_stats()
        ep_reward = 0.0
        step_idx = 0

        while step_idx < steps:
            obs = env.get_state()
            for bs_idx, bs in enumerate(bs_list):
                bs.res_F = F_BS[bs_idx]
            for sat_idx, sat in enumerate(sat_list):
                sat.res_F = SAT_F[sat_idx]
            tcn = [0] * 6
            for md_idx in range(M):
                env_mask = env.get_action_mask(md_list[md_idx], sat_list)
                mask = apply_topology_mask(env_mask, topo_mask)
                disc, cont = policy_fn(obs[md_idx], md_list[md_idx], mask, bs_list, sat_list)
                obs, reward, _, _, _, _, _, _, _, _, _ = env.step(
                    md_idx, disc, cont, tcn, bs_list, md_list, sat_list
                )
                ep_reward += reward
            step_idx += 1
            env.reset_state(md_list, bs_list, sat_list)

        rewards.append(ep_reward)
        debug_list.append(dict(env.last_debug))

    merged = {}
    if debug_list:
        for k in debug_list[0].keys():
            merged[k] = float(np.mean([d[k] for d in debug_list]))

    return float(np.mean(rewards)), merged


def run_heuristic_experiment(baseline_name: str, policy_fn,
                             topo_mask: np.ndarray = None):
    """通用无训练基线实验主函数（BL1 / BL4 / BL5）。

    运行 max_episode 轮 evaluation，保存与主实验相同格式的结果文件。
    由于策略是固定的，train ≈ eval（只是没有网络更新）。
    """
    prefix = BL_PREFIX[baseline_name]
    topo_mask = TOPO_ALL if topo_mask is None else topo_mask

    print(f"\n{'=' * 60}")
    print(f"[{baseline_name.upper()}] 开始: {prefix}")
    print(f"  Policy: {policy_fn.__name__}  |  Topo-mask: {topo_mask.astype(int)}")
    print(f"  Episodes: {max_episode}  |  Steps/ep: {steps}  |  Seed: {SEED}")
    print("=" * 60)

    for seed in SEED:
        random.seed(seed);
        np.random.seed(seed);
        torch.manual_seed(seed)

        env = EdgeEnv.EdgeEnv()
        bs_list, md_list, sat_list = build_entities(env)

        train_rewards, eval_rewards = [], []
        debug_history = []
        train_metric_history, eval_metric_history = [], []
        eval_ran_only = []

        for episode in range(max_episode):
            # 训练回合 = 评估回合（无学习，策略固定）
            ep_reward, ep_debug = _run_heuristic_episode(
                env, policy_fn, md_list, bs_list, sat_list, topo_mask
            )

            # 评估（间隔控制与主实验一致）
            should_eval = (
                    episode == 0
                    or (episode + 1) % max(EVAL_INTERVAL, 1) == 0
                    or episode == max_episode - 1
            )
            if should_eval:
                eval_r, eval_dbg = _eval_heuristic(
                    env, policy_fn, md_list, bs_list, sat_list,
                    eval_rounds=EVAL_ROUNDS, topo_mask=topo_mask
                )
                eval_ran_only.append(eval_r)
                smoothed = compute_smoothed_value(eval_ran_only)
                last_eval_r = eval_r
                last_eval_dbg = eval_dbg
            else:
                eval_r = last_eval_r
                eval_dbg = last_eval_dbg

            train_metrics = summarize_debug(ep_debug)
            eval_metrics = summarize_debug(eval_dbg)
            eval_metrics["smoothed_eval_reward"] = compute_smoothed_value(eval_ran_only)

            train_rewards.append(ep_reward)
            eval_rewards.append(eval_r)
            train_metric_history.append(train_metrics)
            eval_metric_history.append(eval_metrics)
            debug_history.append({
                "episode": episode + 1,
                "train": ep_debug,
                "eval": eval_dbg,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "eval_ran_this_episode": should_eval,
            })

            if (episode + 1) % 50 == 0 or episode == 0:
                tm = train_metrics
                print(
                    f"  [{baseline_name.upper()}] ep {episode + 1:>4}/{max_episode} | "
                    f"train={ep_reward:>8.2f} eval={eval_r:>8.2f} | "
                    f"local={ep_debug['local_actions']:.0f} "
                    f"bs={ep_debug['bs_actions']:.0f} "
                    f"sat={ep_debug['sat_actions']:.0f} | "
                    f"delay={tm['avg_total_delay']:.4f}s "
                    f"energy={tm['avg_total_energy']:.3f}J"
                )

        save_results(prefix, train_rewards, eval_rewards,
                     debug_history, train_metric_history, eval_metric_history)

    print(f"[{baseline_name.upper()}] 完成。最终 eval 均值 = {np.mean(eval_rewards[-20:]):.2f}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 六、HD3QN + 拓扑掩码训练 (BL2 · no_satellite / BL3 · no_gnb)
# ═════════════════════════════════════════════════════════════════════════════

def _run_hd3qn_episode_masked(env, rpm, agent, md_list, bs_list, sat_list,
                              topo_mask: np.ndarray):
    """BL2 / BL3 的训练回合：HD3QN 学习，但动作空间受拓扑掩码限制。

    除 apply_topology_mask 调用之外，逻辑与 HD3QN_t.run_episode 完全相同。
    """
    total_reward = 0.0
    env.reset(md_list, bs_list, sat_list)
    step_idx = 0
    total_losses, q_losses, cont_losses = [], [], []

    while step_idx < steps:
        slot_reward = 0.0
        obs = env.get_state()
        tcn = [0] * 6

        for md_idx in range(M):
            env_mask = env.get_action_mask(md_list[md_idx], sat_list)
            mask = apply_topology_mask(env_mask, topo_mask)

            # ① ε-greedy（在受限动作集中选择）
            disc = agent.sample(obs[md_idx], action_mask=mask)
            # ② 连续参数（HD3QN 连续分支 + 高斯探索噪声）
            cont = agent.get_continuous(obs[md_idx], disc, explore=True)

            pre_obs = obs[md_idx].copy()

            obs, reward, done, _, _, _, _, _, _, _, _ = env.step(
                md_idx, disc, cont, tcn, bs_list, md_list, sat_list
            )

            # ── 下一状态掩码（同样叠加拓扑约束）────────────────────────
            if md_idx == M - 1:
                next_env_mask = env.get_action_mask(md_list[md_idx], sat_list)
                next_mask = apply_topology_mask(next_env_mask, topo_mask)
                next_obs_row = obs[md_idx]
                is_terminal = True
            else:
                nxt = md_idx + 1
                next_env_mask = env.get_action_mask(md_list[nxt], sat_list)
                next_mask = apply_topology_mask(next_env_mask, topo_mask)
                next_obs_row = obs[nxt]
                is_terminal = done

            rpm.append((pre_obs, disc, cont, reward, next_obs_row, is_terminal, next_mask))
            slot_reward += reward

            # ── 学习 ────────────────────────────────────────────────────
            if len(rpm) > WARMUP_SIZE and (md_idx % LEARN_FREQ == 0):
                b_obs, b_disc, b_cont, b_rew, b_nobs, b_done, b_nmask = rpm.sample(BATCH_SIZE)
                tl, ql, cl = agent.learn(b_obs, b_disc, b_cont, b_rew,
                                         b_nobs, b_done, next_action_mask=b_nmask)
                total_losses.append(tl)
                q_losses.append(ql)
                cont_losses.append(cl)

        total_reward += slot_reward
        step_idx += 1
        env.reset_state(md_list, bs_list, sat_list)

    debug_info = dict(env.last_debug)
    debug_info["mean_dqn_loss"] = float(np.mean(total_losses)) if total_losses else 0.0
    debug_info["mean_q_loss"] = float(np.mean(q_losses)) if q_losses else 0.0
    debug_info["mean_cont_loss"] = float(np.mean(cont_losses)) if cont_losses else 0.0
    debug_info["mean_sac_loss"] = 0.0
    debug_info["mean_alpha"] = 0.0
    return total_reward, debug_info


def _eval_hd3qn_masked(env, agent, md_list, bs_list, sat_list,
                       eval_rounds: int, topo_mask: np.ndarray):
    """BL2 / BL3 的评估函数（纯贪心，无探索，应用拓扑掩码）。"""
    rewards, debug_list = [], []
    for _ in range(eval_rounds):
        env.reset(md_list, bs_list, sat_list)
        env.reset_debug_stats()
        ep_reward = 0.0
        step_idx = 0
        while step_idx < steps:
            obs = env.get_state()
            for bs_idx, bs in enumerate(bs_list):
                bs.res_F = F_BS[bs_idx]
            for sat_idx, sat in enumerate(sat_list):
                sat.res_F = SAT_F[sat_idx]
            tcn = [0] * 6
            for md_idx in range(M):
                env_mask = env.get_action_mask(md_list[md_idx], sat_list)
                mask = apply_topology_mask(env_mask, topo_mask)
                disc = agent.predict(obs[md_idx], action_mask=mask)
                cont = agent.get_continuous(obs[md_idx], disc, explore=False)
                obs, reward, _, _, _, _, _, _, _, _, _ = env.step(
                    md_idx, disc, cont, tcn, bs_list, md_list, sat_list
                )
                ep_reward += reward
            step_idx += 1
            env.reset_state(md_list, bs_list, sat_list)
        rewards.append(ep_reward)
        debug_list.append(dict(env.last_debug))

    merged = {}
    if debug_list:
        for k in debug_list[0].keys():
            merged[k] = float(np.mean([d[k] for d in debug_list]))
    return float(np.mean(rewards)), merged


def run_masked_hd3qn_experiment(baseline_name: str, topo_mask: np.ndarray):
    """BL2 · no_satellite 和 BL3 · no_gnb 的完整训练实验。

    使用 HD3QN（与主实验相同架构），强制屏蔽指定卸载路径。
    """
    prefix = BL_PREFIX[baseline_name]
    disabled = [i for i, v in enumerate(topo_mask) if not v]
    mask_desc = {
        "bl2": "SAT actions disabled [2,3]",
        "bl3": "BS  actions disabled  [1]",
    }.get(baseline_name, str(topo_mask.astype(int)))

    print(f"\n{'=' * 60}")
    print(f"[{baseline_name.upper()}] 开始: {prefix}")
    print(f"  Topology: {mask_desc}  |  Disabled action indices: {disabled}")
    print(f"  Episodes: {max_episode}  |  Steps/ep: {steps}  |  Seed: {SEED}")
    print("=" * 60)

    for seed in SEED:
        random.seed(seed);
        np.random.seed(seed);
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = EdgeEnv.EdgeEnv()
        bs_list, md_list, sat_list = build_entities(env)

        # 与主实验相同的 HD3QN 架构
        action_dim = env.action_space.n
        obs_shape = env.observation_space.shape[1]
        model = HD3QN.HD3QNModel(obs_dim=obs_shape, act_dim=action_dim)
        algorithm = HD3QN.HD3QN(model, act_dim=action_dim, gamma=DQN_GAMMA, lr=LEARNING_RATE)
        agent = HD3QN.Agent(
            algorithm, obs_dim=obs_shape, act_dim=action_dim,
            e_greed=0.9,
            e_greed_decrement=0.89 / max(max_episode * steps * M, 1),
        )
        rpm = ReplayMemory.ReplayMemory(MEMORY_SIZE)

        # Warmup
        print(f"  Warmup 中... (需要 {WARMUP_SIZE} 条经验)")
        while len(rpm) < WARMUP_SIZE:
            _run_hd3qn_episode_masked(env, rpm, agent, md_list, bs_list, sat_list, topo_mask)
        print(f"  Warmup 完成，buffer = {len(rpm)} 条")

        train_rewards, eval_rewards = [], []
        debug_history = []
        train_metric_history, eval_metric_history = [], []
        eval_ran_only = []
        best_reward = -float("inf")
        last_eval_r = None
        last_eval_dbg = None
        last_eval_metrics = None

        for episode in range(max_episode):
            train_r, train_dbg = _run_hd3qn_episode_masked(
                env, rpm, agent, md_list, bs_list, sat_list, topo_mask
            )

            # ── 评估 ────────────────────────────────────────────────────
            should_eval = (
                    episode == 0
                    or (episode + 1) % max(EVAL_INTERVAL, 1) == 0
                    or episode == max_episode - 1
            )
            if should_eval:
                eval_r, eval_dbg = _eval_hd3qn_masked(
                    env, agent, md_list, bs_list, sat_list, EVAL_ROUNDS, topo_mask
                )
                eval_ran_only.append(eval_r)
                last_eval_r = eval_r
                last_eval_dbg = eval_dbg
            else:
                eval_r = last_eval_r
                eval_dbg = last_eval_dbg

            train_metrics = summarize_debug(train_dbg)
            eval_metrics = summarize_debug(eval_dbg)
            eval_metrics["smoothed_eval_reward"] = compute_smoothed_value(eval_ran_only)

            train_rewards.append(train_r)
            eval_rewards.append(eval_r)
            train_metric_history.append(train_metrics)
            eval_metric_history.append(eval_metrics)
            debug_history.append({
                "episode": episode + 1,
                "train": train_dbg,
                "eval": eval_dbg,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "eval_ran_this_episode": should_eval,
            })

            if eval_r > best_reward:
                best_reward = eval_r
                os.makedirs("models", exist_ok=True)
                agent.alg.save_model(f"models/{prefix}_best_hd3qn_model.pth")

            if (episode + 1) % 50 == 0 or episode == 0:
                e_greed = agent.e_greed
                dqn_l = train_dbg.get("mean_dqn_loss", 0.0)
                q_l = train_dbg.get("mean_q_loss", 0.0)
                c_l = train_dbg.get("mean_cont_loss", 0.0)
                print(
                    f"  [{baseline_name.upper()}] ep {episode + 1:>4}/{max_episode} | "
                    f"train={train_r:>8.2f} eval={eval_r:>8.2f} | "
                    f"ε={e_greed:.3f} | "
                    f"q_loss={q_l:.4f} cont_loss={c_l:.4f} | "
                    f"local={train_dbg['local_actions']:.0f} "
                    f"bs={train_dbg['bs_actions']:.0f} "
                    f"sat={train_dbg['sat_actions']:.0f}"
                )

        save_results(prefix, train_rewards, eval_rewards,
                     debug_history, train_metric_history, eval_metric_history)

    print(f"[{baseline_name.upper()}] 完成。最优 eval = {best_reward:.2f}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 七、BL6 · D3QN + 启发式频率（算法对比）
# ═════════════════════════════════════════════════════════════════════════════

def _run_dqn_heuristic_episode(env, rpm, agent_dqn, md_list, bs_list, sat_list):
    """BL6 训练回合：D3QN 学习离散动作，频率用启发式公式计算（无 SAC）。"""
    total_reward = 0.0
    env.reset(md_list, bs_list, sat_list)
    step_idx = 0
    dqn_losses = []

    while step_idx < steps:
        slot_reward = 0.0
        obs = env.get_state()
        tcn = [0] * 6

        for md_idx in range(M):
            mask = env.get_action_mask(md_list[md_idx], sat_list)
            # ① 离散：ε-greedy D3QN
            disc = agent_dqn.sample(obs[md_idx], action_mask=mask)
            # ② 连续：启发式（根据任务大小比例分配频率，不学习）
            cont = _heuristic_freq(md_list[md_idx], disc)

            pre_obs = obs[md_idx].copy()
            obs, reward, done, _, _, _, _, _, _, _, _ = env.step(
                md_idx, disc, cont, tcn, bs_list, md_list, sat_list
            )

            # ── 下一状态 ────────────────────────────────────────────────
            if md_idx == M - 1:
                next_mask = env.get_action_mask(md_list[md_idx], sat_list)
                next_obs_row = obs[md_idx]
                is_terminal = True
            else:
                nxt = md_idx + 1
                next_mask = env.get_action_mask(md_list[nxt], sat_list)
                next_obs_row = obs[nxt]
                is_terminal = done

            # D3QN 使用 6 元组格式（无连续动作字段）
            rpm.append((pre_obs, disc, reward, next_obs_row, is_terminal, next_mask))
            slot_reward += reward

            # ── D3QN 学习 ────────────────────────────────────────────────
            if len(rpm) > WARMUP_SIZE and (md_idx % LEARN_FREQ == 0):
                b_obs, b_disc, b_rew, b_nobs, b_done, b_nmask = rpm.sample(BATCH_SIZE)
                loss = agent_dqn.learn(b_obs, b_disc, b_rew, b_nobs, b_done,
                                       next_action_mask=b_nmask)
                dqn_losses.append(loss)

        total_reward += slot_reward
        step_idx += 1
        env.reset_state(md_list, bs_list, sat_list)

    debug_info = dict(env.last_debug)
    debug_info["mean_dqn_loss"] = float(np.mean(dqn_losses)) if dqn_losses else 0.0
    debug_info["mean_sac_loss"] = 0.0
    debug_info["mean_alpha"] = 0.0
    debug_info["mean_q_loss"] = float(np.mean(dqn_losses)) if dqn_losses else 0.0
    debug_info["mean_cont_loss"] = 0.0  # 无 cont_loss，启发式不参与梯度
    return total_reward, debug_info


def _eval_dqn_heuristic(env, agent_dqn, md_list, bs_list, sat_list, eval_rounds=1):
    """BL6 评估：D3QN 贪心选动作，频率用启发式公式。"""
    rewards, debug_list = [], []
    for _ in range(eval_rounds):
        env.reset(md_list, bs_list, sat_list)
        env.reset_debug_stats()
        ep_reward = 0.0
        step_idx = 0
        while step_idx < steps:
            obs = env.get_state()
            for bs_idx, bs in enumerate(bs_list):
                bs.res_F = F_BS[bs_idx]
            for sat_idx, sat in enumerate(sat_list):
                sat.res_F = SAT_F[sat_idx]
            tcn = [0] * 6
            for md_idx in range(M):
                mask = env.get_action_mask(md_list[md_idx], sat_list)
                disc = agent_dqn.predict(obs[md_idx], action_mask=mask)
                cont = _heuristic_freq(md_list[md_idx], disc)
                obs, reward, _, _, _, _, _, _, _, _, _ = env.step(
                    md_idx, disc, cont, tcn, bs_list, md_list, sat_list
                )
                ep_reward += reward
            step_idx += 1
            env.reset_state(md_list, bs_list, sat_list)
        rewards.append(ep_reward)
        debug_list.append(dict(env.last_debug))

    merged = {}
    if debug_list:
        for k in debug_list[0].keys():
            merged[k] = float(np.mean([d[k] for d in debug_list]))
    return float(np.mean(rewards)), merged


def run_dqn_heuristic_experiment():
    """BL6 · DQN + Heuristic Frequency 完整训练实验。"""
    prefix = BL_PREFIX["bl6"]
    print(f"\n{'=' * 60}")
    print(f"[BL6] 开始: {prefix}")
    print(f"  D3QN（离散）+ 启发式频率（连续，不学习）")
    print(f"  证明目标：混合动作联合学习 >> 割裂的离散+启发式策略")
    print(f"  Episodes: {max_episode}  |  Steps/ep: {steps}  |  Seed: {SEED}")
    print("=" * 60)

    for seed in SEED:
        random.seed(seed);
        np.random.seed(seed);
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = EdgeEnv.EdgeEnv()
        bs_list, md_list, sat_list = build_entities(env)

        obs_shape = env.observation_space.shape[1]
        action_dim = env.action_space.n

        # 使用 D3QN（原版 Dueling DDQN，非 HD3QN）
        model_dqn = D3QN.Model(obs_dim=obs_shape, act_dim=action_dim)
        alg_dqn = D3QN.DDQN(model_dqn, act_dim=action_dim, gamma=DQN_GAMMA, lr=LEARNING_RATE)
        agent_dqn = D3QN.Agent(
            alg_dqn, obs_dim=obs_shape, act_dim=action_dim,
            e_greed=0.9,
            e_greed_decrement=0.89 / max(max_episode * steps * M, 1),
        )
        # D3QN 用 6 元组格式的 ReplayMemory_SAC
        rpm = ReplayMemory_SAC.ReplayMemory(MEMORY_SIZE)

        # Warmup
        print(f"  Warmup 中... (需要 {WARMUP_SIZE} 条经验)")
        while len(rpm) < WARMUP_SIZE:
            _run_dqn_heuristic_episode(env, rpm, agent_dqn, md_list, bs_list, sat_list)
        print(f"  Warmup 完成，buffer = {len(rpm)} 条")

        train_rewards, eval_rewards = [], []
        debug_history = []
        train_metric_history, eval_metric_history = [], []
        eval_ran_only = []
        best_reward = -float("inf")
        last_eval_r = None
        last_eval_dbg = None

        for episode in range(max_episode):
            train_r, train_dbg = _run_dqn_heuristic_episode(
                env, rpm, agent_dqn, md_list, bs_list, sat_list
            )

            should_eval = (
                    episode == 0
                    or (episode + 1) % max(EVAL_INTERVAL, 1) == 0
                    or episode == max_episode - 1
            )
            if should_eval:
                eval_r, eval_dbg = _eval_dqn_heuristic(
                    env, agent_dqn, md_list, bs_list, sat_list, EVAL_ROUNDS
                )
                eval_ran_only.append(eval_r)
                last_eval_r = eval_r
                last_eval_dbg = eval_dbg
            else:
                eval_r = last_eval_r
                eval_dbg = last_eval_dbg

            train_metrics = summarize_debug(train_dbg)
            eval_metrics = summarize_debug(eval_dbg)
            eval_metrics["smoothed_eval_reward"] = compute_smoothed_value(eval_ran_only)

            train_rewards.append(train_r)
            eval_rewards.append(eval_r)
            train_metric_history.append(train_metrics)
            eval_metric_history.append(eval_metrics)
            debug_history.append({
                "episode": episode + 1,
                "train": train_dbg,
                "eval": eval_dbg,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "eval_ran_this_episode": should_eval,
            })

            if eval_r > best_reward:
                best_reward = eval_r
                os.makedirs("models", exist_ok=True)
                alg_dqn.save_model(f"models/{prefix}_best_dqn_model.pth")

            if (episode + 1) % 50 == 0 or episode == 0:
                dqn_l = train_dbg.get("mean_dqn_loss", 0.0)
                print(
                    f"  [BL6] ep {episode + 1:>4}/{max_episode} | "
                    f"train={train_r:>8.2f} eval={eval_r:>8.2f} | "
                    f"ε={agent_dqn.e_greed:.3f} dqn_loss={dqn_l:.4f} | "
                    f"local={train_dbg['local_actions']:.0f} "
                    f"bs={train_dbg['bs_actions']:.0f} "
                    f"sat={train_dbg['sat_actions']:.0f}"
                )

        save_results(prefix, train_rewards, eval_rewards,
                     debug_history, train_metric_history, eval_metric_history)

    print(f"[BL6] 完成。最优 eval = {best_reward:.2f}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 八、统一入口
# ═════════════════════════════════════════════════════════════════════════════

ALL_BASELINES = {
    "bl1": lambda: run_heuristic_experiment(
        "bl1", _policy_local_only, topo_mask=TOPO_LOCAL_ONLY
    ),
    "bl2": lambda: run_masked_hd3qn_experiment(
        "bl2", topo_mask=TOPO_NO_SAT
    ),
    "bl3": lambda: run_masked_hd3qn_experiment(
        "bl3", topo_mask=TOPO_NO_GNB
    ),
    "bl4": lambda: run_heuristic_experiment(
        "bl4", _policy_random_equal, topo_mask=TOPO_ALL
    ),
    "bl5": lambda: run_heuristic_experiment(
        "bl5", _policy_greedy_delay, topo_mask=TOPO_ALL
    ),
    "bl6": lambda: run_dqn_heuristic_experiment(),
}

BL_DESCRIPTIONS = {
    "bl1": "Local-Only     — 全本地计算（拓扑消融）",
    "bl2": "No-Satellite   — 无卫星纯地面MEC（拓扑消融）",
    "bl3": "No-gNB         — 无基站纯卫星+本地（拓扑消融）",
    "bl4": "Random+Equal   — 随机分流+算力平分（算法对比）",
    "bl5": "Greedy-Delay   — 贪心时延最优（算法对比）",
    "bl6": "DQN+Heuristic  — D3QN离散+启发式频率（算法对比）",
}


def main():
    parser = argparse.ArgumentParser(description="NTN 基线实验控制台")
    parser.add_argument("--only", nargs="+", choices=list(ALL_BASELINES.keys()),
                        help="只运行指定基线，例如 --only bl1 bl4")
    parser.add_argument("--skip", nargs="+", choices=list(ALL_BASELINES.keys()),
                        help="跳过指定基线，例如 --skip bl2 bl3")
    args = parser.parse_args()

    to_run = list(ALL_BASELINES.keys())
    if args.only:
        to_run = [b for b in to_run if b in args.only]
    if args.skip:
        to_run = [b for b in to_run if b not in args.skip]

    torch.set_num_threads(para.CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, para.CPU_THREADS))
    except RuntimeError:
        pass

    device = "cuda" if para.USE_CUDA and torch.cuda.is_available() else "cpu"
    print(f"\n{'#' * 60}")
    print(f"  NTN 基线实验启动")
    print(f"  Device: {device}  |  Seed: {SEED}  |  Mode: {para.RUN_MODE}")
    print(f"  将运行 {len(to_run)} 个基线: {to_run}")
    for b in to_run:
        print(f"    {b}: {BL_DESCRIPTIONS[b]}")
    print(f"{'#' * 60}\n")

    for bl_name in to_run:
        ALL_BASELINES[bl_name]()

    print(f"\n{'#' * 60}")
    print(f"  全部基线实验完成。结果保存于 result/baselines_*.pickle")
    print(f"  可直接与 HD3QN 主实验结果一起绘图对比。")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()