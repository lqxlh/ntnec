"""
D3SAC.py — 主训练入口（HD3QN 版）

改造要点
--------
1. 移除 SAC 智能体（agent2），改用 HD3QN.Agent 统一处理离散+连续动作。
2. 经验格式改为 7 元组：
       (obs, disc_act, cont_act, reward, next_obs, done, next_action_mask)
3. 每步：
       discrete_action  = agent.sample(obs, action_mask)   # ε-greedy
       continuous_action = agent.get_continuous(obs, discrete_action)  # 连续分支
   评估时改用 agent.predict() + agent.get_continuous()（纯贪心）。
4. 删除 rpm2 / SAC_MEMORY_SIZE / SAC_WARMUP / SAC_LEARN_FREQ 等 SAC 相关逻辑。
5. 日志字段保持与原版一致（mean_dqn_loss 沿用，mean_sac_loss/mean_alpha 置 0）。
6. 模型保存路径：{prefix}_best_hd3qn_model.pth（兼容 D3SAC_Gen.py 加载）。

不改变内容
----------
- EdgeEnv.step(m, b, f, tcn, bs_list, md_list, sat_list) 接口
- summarize_debug / build_entities / evaluate / run_training_experiment 框架
- 所有日志打印字段名称
- result/*.pickle 导出格式
"""

import os
import pickle
import random
import warnings

import numpy as np
import torch

import BS
import HD3QN
import EdgeEnv
import MD
import ReplayMemory
import SAT
import experiment_config
import para

warnings.filterwarnings("ignore", category=Warning)

# ---------------------------------------------------------------------------
# 超参数（复用 para.py，SAC 相关沿用作 HD3QN 学习率参考，实际只用 LEARNING_RATE）
# ---------------------------------------------------------------------------
LEARN_FREQ         = para.LEARN_FREQ
MEMORY_SIZE        = para.MEMORY_SIZE
MEMORY_WARMUP_SIZE = para.MEMORY_WARMUP_SIZE
BATCH_SIZE         = para.BATCH_SIZE
LEARNING_RATE      = para.LEARNING_RATE
DQN_GAMMA          = para.DQN_GAMMA
N                  = para.N
S                  = para.S
M                  = para.M
F_BS               = para.F_BS
F_MD               = para.F_MD
SAT_F              = para.SAT_F
steps              = para.steps
max_episode        = para.max_episode
SEED               = para.SEED
EVAL_INTERVAL      = para.EVAL_INTERVAL
EVAL_ROUNDS        = para.EVAL_ROUNDS
REWARD_SMOOTH_WINDOW = para.REWARD_SMOOTH_WINDOW


# ---------------------------------------------------------------------------
# 辅助函数（完全保留原版逻辑，供 D3SAC_Gen.py 等调用）
# ---------------------------------------------------------------------------

def summarize_debug(debug_info):
    steps_count    = max(float(debug_info.get("steps", 0.0)), 1.0)
    sat_actions    = float(debug_info.get("sat_actions", 0.0))
    split_actions  = float(debug_info.get("split_actions", 0.0))
    split_metric_steps = max(float(debug_info.get("split_metric_steps", 0.0)), 1.0)
    split_timeout_actions = float(debug_info.get("split_timeout_actions", 0.0))
    split_success_actions = float(debug_info.get("split_deadline_success_actions", 0.0))
    sat_metric_steps = max(float(debug_info.get("sat_metric_steps", 0.0)), 1.0)
    visible_decisions = float(debug_info.get("visible_sat_decisions", 0.0))
    visible_sat_total_count = float(debug_info.get("visible_sat_total_count", 0.0))
    sat_exec_success_actions    = float(debug_info.get("sat_exec_success_actions", 0.0))
    sat_deadline_success_actions = float(debug_info.get("sat_deadline_success_actions", 0.0))
    sat_timeout_actions         = float(debug_info.get("sat_timeout_actions", 0.0))
    sat_mask_checks   = max(float(debug_info.get("mask_sat_total_checks", 0.0)), 1.0)
    sat_budget_fail_count = max(float(debug_info.get("mask_sat_remaining_budget_fail_count", 0.0)), 1.0)
    sat_power_fail_count  = max(float(debug_info.get("mask_sat_power_excess_ratio_count", 0.0)), 1.0)
    sat_feasible_metric_steps = max(float(debug_info.get("sat_feasible_metric_steps", 0.0)), 1.0)
    sat_load_penalty_count    = max(float(debug_info.get("sat_load_penalty_count", 0.0)), 1.0)

    return {
        "avg_prop_delay":            float(debug_info.get("avg_prop_delay_sum", 0.0)) / steps_count,
        "avg_total_delay":           float(debug_info.get("avg_total_delay_sum", 0.0)) / steps_count,
        "avg_reward":                float(debug_info.get("reward_sum", 0.0)) / steps_count,
        "avg_phi_cost":              float(debug_info.get("avg_cost_sum", 0.0)) / steps_count,
        "avg_task_value":            float(debug_info.get("avg_value_sum", 0.0)) / steps_count,
        "avg_delay_norm":            float(debug_info.get("avg_delay_norm_sum", 0.0)) / steps_count,
        "avg_energy_norm":           float(debug_info.get("avg_energy_norm_sum", 0.0)) / steps_count,
        "avg_value_norm":            float(debug_info.get("avg_value_norm_sum", 0.0)) / steps_count,
        "avg_delay_cost":            float(debug_info.get("avg_delay_cost_sum", 0.0)) / steps_count,
        "avg_energy_cost":           float(debug_info.get("avg_energy_cost_sum", 0.0)) / steps_count,
        "avg_total_energy":          float(debug_info.get("avg_total_energy_sum", 0.0)) / steps_count,
        "avg_selected_power":        float(debug_info.get("avg_selected_power_sum", 0.0)) / steps_count,
        "avg_min_power":             float(debug_info.get("avg_min_power_sum", 0.0)) / steps_count,
        "avg_unconstrained_power":   float(debug_info.get("avg_unconstrained_power_sum", 0.0)) / steps_count,
        "avg_g_bar_s":               float(debug_info.get("avg_g_bar_s_sum", 0.0)) / steps_count,
        "avg_remaining_budget_sat":  float(debug_info.get("avg_remaining_budget_sat_sum", 0.0)) / steps_count,
        "avg_P_min_sat":             float(debug_info.get("avg_P_min_sat_sum", 0.0)) / steps_count,
        "avg_p_star_sat":            float(debug_info.get("avg_p_star_sat_sum", 0.0)) / steps_count,
        "avg_g_sat_raw":             float(debug_info.get("avg_g_sat_raw_sum", 0.0)) / steps_count,
        "avg_sat_snr":               float(debug_info.get("avg_sat_snr_sum", 0.0)) / steps_count,
        "avg_sat_rate":              float(debug_info.get("avg_sat_rate_sum", 0.0)) / steps_count,
        "avg_relative_velocity_sat": float(debug_info.get("avg_relative_velocity_sat_sum", 0.0)) / steps_count,
        "avg_doppler_shift_sat":     float(debug_info.get("avg_doppler_shift_sat_sum", 0.0)) / steps_count,
        "avg_eta_d_sat":             float(debug_info.get("avg_eta_d_sat_sum", 0.0)) / steps_count,
        "avg_wavelength_sat":        float(debug_info.get("avg_wavelength_sat_sum", 0.0)) / steps_count,
        "avg_g_bar_s_on_sat":                   float(debug_info.get("avg_g_bar_s_sum", 0.0)) / sat_metric_steps,
        "avg_remaining_budget_sat_on_sat":       float(debug_info.get("avg_remaining_budget_sat_sum", 0.0)) / sat_metric_steps,
        "avg_P_min_sat_on_sat":                  float(debug_info.get("avg_P_min_sat_sum", 0.0)) / sat_metric_steps,
        "avg_p_star_sat_on_sat":                 float(debug_info.get("avg_p_star_sat_sum", 0.0)) / sat_metric_steps,
        "avg_g_bar_s_on_feasible_sat":           float(debug_info.get("avg_g_bar_s_feasible_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_remaining_budget_sat_on_feasible_sat": float(debug_info.get("avg_remaining_budget_sat_feasible_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_P_min_sat_on_feasible_sat":         float(debug_info.get("avg_P_min_sat_feasible_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_p_star_sat_on_feasible_sat":        float(debug_info.get("avg_p_star_sat_feasible_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_g_sat_raw_on_sat":             float(debug_info.get("avg_g_sat_raw_sum", 0.0)) / sat_metric_steps,
        "avg_sat_snr_on_sat":               float(debug_info.get("avg_sat_snr_sum", 0.0)) / sat_metric_steps,
        "avg_sat_rate_on_sat":              float(debug_info.get("avg_sat_rate_sum", 0.0)) / sat_metric_steps,
        "avg_relative_velocity_sat_on_sat": float(debug_info.get("avg_relative_velocity_sat_sum", 0.0)) / sat_metric_steps,
        "avg_doppler_shift_sat_on_sat":     float(debug_info.get("avg_doppler_shift_sat_sum", 0.0)) / sat_metric_steps,
        "avg_eta_d_sat_on_sat":             float(debug_info.get("avg_eta_d_sat_sum", 0.0)) / sat_metric_steps,
        "avg_wavelength_sat_on_sat":        float(debug_info.get("avg_wavelength_sat_sum", 0.0)) / sat_metric_steps,
        "avg_penalty_time":        float(debug_info.get("penalty_time_sum", 0.0)) / steps_count,
        "avg_penalty_resource":    float(debug_info.get("penalty_resource_sum", 0.0)) / steps_count,
        "avg_penalty_visibility":  float(debug_info.get("penalty_visibility_sum", 0.0)) / steps_count,
        "avg_penalty_propagation": float(debug_info.get("penalty_propagation_sum", 0.0)) / steps_count,
        "avg_penalty_zero_alloc":  float(debug_info.get("penalty_zero_alloc_sum", 0.0)) / steps_count,
        "avg_sat_load_penalty":    float(debug_info.get("sat_load_penalty_sum", 0.0)) / steps_count,
        "avg_sat_load_penalty_on_triggered_sat":  float(debug_info.get("sat_load_penalty_sum", 0.0)) / sat_load_penalty_count,
        "avg_sat_usage_ratio_on_feasible_sat":    float(debug_info.get("sat_usage_ratio_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_sat_peak_usage_ratio_on_feasible_sat": float(debug_info.get("sat_peak_usage_ratio_sum", 0.0)) / sat_feasible_metric_steps,
        "max_sat_peak_usage_ratio": float(debug_info.get("sat_peak_usage_ratio_max", 0.0)),
        "sat_usage_rate":           sat_actions / steps_count,
        "split_usage_rate":         split_actions / steps_count,
        "split_in_sat_rate":        split_actions / max(sat_actions, 1.0),
        "split_deadline_success_rate": split_success_actions / max(split_actions, 1.0),
        "split_timeout_rate":       split_timeout_actions / max(split_actions, 1.0),
        "avg_split_delay_over_gamma": float(debug_info.get("split_delay_over_gamma_sum", 0.0)) / max(split_timeout_actions, 1.0),
        "avg_split_ratio":          float(debug_info.get("split_ratio_sum", 0.0)) / split_metric_steps,
        "sat_exec_success_rate":    sat_exec_success_actions / max(sat_actions, 1.0),
        "sat_deadline_success_rate": sat_deadline_success_actions / max(sat_actions, 1.0),
        "sat_timeout_rate":         sat_timeout_actions / max(sat_actions, 1.0),
        "avg_sat_delay_over_gamma": float(debug_info.get("sat_delay_over_gamma_sum", 0.0)) / max(sat_timeout_actions, 1.0),
        "visible_sat_decision_rate": visible_decisions / steps_count,
        "avg_visible_satellites":   visible_sat_total_count / steps_count,
        "visible_but_not_selected_rate": float(debug_info.get("sat_visible_not_selected", 0.0)) / max(visible_decisions, 1.0),
        "sat_mask_feasible_rate":          float(debug_info.get("mask_sat_feasible", 0.0)) / sat_mask_checks,
        "sat_mask_not_visible_rate":       float(debug_info.get("mask_sat_not_visible", 0.0)) / sat_mask_checks,
        "sat_mask_no_resource_rate":       float(debug_info.get("mask_sat_no_resource", 0.0)) / sat_mask_checks,
        "sat_mask_non_positive_budget_rate": float(debug_info.get("mask_sat_non_positive_budget", 0.0)) / sat_mask_checks,
        "sat_mask_power_infeasible_rate":  float(debug_info.get("mask_sat_power_infeasible", 0.0)) / sat_mask_checks,
        "sat_mask_remaining_budget_fail_avg": float(debug_info.get("mask_sat_remaining_budget_fail_sum", 0.0)) / sat_budget_fail_count,
        "sat_mask_power_excess_ratio_avg":    float(debug_info.get("mask_sat_power_excess_ratio_sum", 0.0)) / sat_power_fail_count,
        # ===== 新增字段：让 pickle 保留 loss 和动作计数，方便画图脚本读取 =====
        "mean_dqn_loss":  float(debug_info.get("mean_dqn_loss", 0.0)),
        "mean_q_loss":    float(debug_info.get("mean_q_loss", 0.0)),
        "mean_cont_loss": float(debug_info.get("mean_cont_loss", 0.0)),
        "local_actions":  float(debug_info.get("local_actions", 0.0)),
        "bs_actions":     float(debug_info.get("bs_actions", 0.0)),
        "sat_actions_count": float(debug_info.get("sat_actions", 0.0)),
        "split_actions":  split_actions,
        # ===== 新增结束 =====
    }


def compute_smoothed_value(value_history, window_size):
    if not value_history:
        return 0.0
    tail = value_history[-max(int(window_size), 1):]
    return float(np.mean(tail))


# ---------------------------------------------------------------------------
# 实体构建（完全保留）
# ---------------------------------------------------------------------------

def build_entities(env):
    bs_list, md_list, sat_list = [], [], []
    for n in range(N):
        bs = BS.BS(n, F_BS[n], random.uniform(0, para.MAP_WIDTH), random.uniform(0, para.MAP_HEIGHT), para.BS_HEIGHT)
        bs_list.append(bs)
    for s_idx in range(S):
        sat_list.append(SAT.SAT(s_idx, SAT_F[s_idx]))
    for m in range(M):
        md_list.append(MD.MD(m, F_MD, env, bs_list))
    return bs_list, md_list, sat_list


# ---------------------------------------------------------------------------
# 构建 HD3QN 智能体（替代原 D3QN + SAC 双智能体）
# ---------------------------------------------------------------------------

def build_hd3qn_agent(env):
    """
    构建 HD3QN 智能体。
    obs_dim 与原 D3QN 保持一致（不再需要 SAC 的 obs+2 增强）。
    """
    action_dim = env.action_space.n
    obs_shape  = env.observation_space.shape[1]

    model     = HD3QN.HD3QNModel(obs_dim=obs_shape, act_dim=action_dim)
    algorithm = HD3QN.HD3QN(model, act_dim=action_dim, gamma=DQN_GAMMA, lr=LEARNING_RATE)
    agent     = HD3QN.Agent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.9,
        e_greed_decrement=0.89 / max(100 * steps * M, 1),#调探索率衰减
    )
    return agent


# ---------------------------------------------------------------------------
# 单回合训练（HD3QN 版）
# ---------------------------------------------------------------------------

def run_episode(env, rpm, agent, md_list, bs_list, sat_list):
    """
    HD3QN 版 run_episode：
    - 只有一个回放池 rpm（7 元组）
    - 只有一个智能体 agent（HD3QN）
    - 日志字段保持原版（mean_dqn_loss / mean_sac_loss=0 / mean_alpha=0）
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
            action_mask  = env.get_action_mask(md_list[md_idx], sat_list)
            # ① 离散动作：ε-greedy
            discrete_action = agent.sample(obs[md_idx], action_mask=action_mask)
            # ② 连续动作：HD3QN 连续参数分支（修改：传入 explore=True，激活高斯探索噪声）
            cont_action = agent.get_continuous(obs[md_idx], discrete_action, explore=True)

            pre_obs = obs[md_idx].copy()

            obs, reward, done, _, _, _, _, _, _, _, _ = env.step(
                md_idx, discrete_action, cont_action, tcn, bs_list, md_list, sat_list
            )

            # ---- 构造下一状态信息 ----
            if md_idx == M - 1:
                next_mask    = env.get_action_mask(md_list[md_idx], sat_list)
                next_obs_row = obs[md_idx]
                is_terminal  = True
            else:
                next_md_idx  = md_idx + 1
                next_mask    = env.get_action_mask(md_list[next_md_idx], sat_list)
                next_obs_row = obs[next_md_idx]
                is_terminal  = done

            # ---- 7 元组经验存储 ----
            rpm.append((
                pre_obs,
                discrete_action,
                cont_action,      # ← 新增：连续动作
                reward,
                next_obs_row,
                is_terminal,
                next_mask,
            ))

            slot_reward += reward

            # ---- 学习 ----
            if len(rpm) > MEMORY_WARMUP_SIZE and (md_idx % LEARN_FREQ == 0):
                (b_obs, b_disc, b_cont, b_rew,
                 b_nobs, b_done, b_nmask), idxs, is_weights = rpm.sample(BATCH_SIZE)

                # 将权重传入 agent，并接收当前 batch 真实的 td_errors
                tl, ql, cl, td_errors = agent.learn(
                    b_obs, b_disc, b_cont, b_rew,
                    b_nobs, b_done,
                    next_action_mask=b_nmask,
                    is_weights=is_weights  # 传入权重
                )

                # 【新增】将计算得到的最新误差反馈给回放池，在线更新二叉树的优先级分布
                rpm.update_priorities(idxs, td_errors)
                total_losses.append(tl)
                q_losses.append(ql)
                cont_losses.append(cl)

        total_reward += slot_reward
        step_idx += 1
        env.reset_state(md_list, bs_list, sat_list)

    debug_info = dict(env.last_debug)
    # 保留原版日志字段名：mean_dqn_loss / mean_sac_loss / mean_alpha
    debug_info["mean_dqn_loss"]  = float(np.mean(total_losses)) if total_losses else 0.0
    debug_info["mean_sac_loss"]  = 0.0   # HD3QN 无 SAC，置 0 保持字段兼容
    debug_info["mean_alpha"]     = 0.0
    debug_info["mean_q_loss"]    = float(np.mean(q_losses))    if q_losses    else 0.0
    debug_info["mean_cont_loss"] = float(np.mean(cont_losses)) if cont_losses else 0.0
    return total_reward, debug_info


# ---------------------------------------------------------------------------
# 评估（HD3QN 版）
# ---------------------------------------------------------------------------

def evaluate(env, agent, agent2, md_list, bs_list, sat_list, eval_rounds=2):
    eval_rewards = []
    eval_debug_list = []

    # Q ???????????? BS?? SAT?????????????????
    q_diag = {
        "q_local_sum": 0.0, "q_bs_sum": 0.0, "q_sat_sum": 0.0, "q_split_sum": 0.0,
        "q_local_count": 0, "q_bs_count": 0, "q_sat_count": 0, "q_split_count": 0,
        "q_max_minus_min_sum": 0.0,
        "q_local_minus_max_sum": 0.0,
        "decision_count": 0,
        "local_argmax_count": 0,
        "bs_argmax_count": 0,
        "sat_argmax_count": 0,
        "split_argmax_count": 0,
        "q_local_when_legal_sum": 0.0,
        "q_local_when_legal_count": 0,
        "q_max_when_local_legal_sum": 0.0,
        "local_legal_not_chosen": 0,
        "local_legal_total": 0,
    }

    for _ in range(eval_rounds):
        env.reset(md_list, bs_list, sat_list)
        env.reset_debug_stats()
        episode_reward = 0.0
        tcn = [0] * 6
        step_idx = 0

        while step_idx < steps:
            obs = env.get_state()
            for bs_idx, bs in enumerate(bs_list):
                bs.res_F = F_BS[bs_idx]
            for sat_idx, sat in enumerate(sat_list):
                sat.res_F = SAT_F[sat_idx]

            for md_idx in range(M):
                action_mask = env.get_action_mask(md_list[md_idx], sat_list)
                q_vals = agent.predict_q_values(obs[md_idx], action_mask=action_mask)

                if action_mask[0]:
                    q_diag["q_local_sum"] += float(q_vals[0])
                    q_diag["q_local_count"] += 1
                for act_idx in range(1, N + 1):
                    if action_mask[act_idx]:
                        q_diag["q_bs_sum"] += float(q_vals[act_idx])
                        q_diag["q_bs_count"] += 1
                for act_idx in range(N + 1, N + S + 1):
                    if action_mask[act_idx]:
                        q_diag["q_sat_sum"] += float(q_vals[act_idx])
                        q_diag["q_sat_count"] += 1
                for act_idx in range(N + S + 1, len(action_mask)):
                    if action_mask[act_idx]:
                        q_diag["q_split_sum"] += float(q_vals[act_idx])
                        q_diag["q_split_count"] += 1

                legal_qs = [float(q_vals[i]) for i in range(len(action_mask)) if action_mask[i]]
                if len(legal_qs) >= 2:
                    q_diag["q_max_minus_min_sum"] += max(legal_qs) - min(legal_qs)
                    q_diag["decision_count"] += 1
                    legal_argmax_scores = [float(q_vals[i]) if action_mask[i] else -1e9 for i in range(len(action_mask))]
                    argmax_action = int(np.argmax(legal_argmax_scores))
                    if argmax_action == 0:
                        q_diag["local_argmax_count"] += 1
                    elif 1 <= argmax_action <= N:
                        q_diag["bs_argmax_count"] += 1
                    elif N + 1 <= argmax_action <= N + S:
                        q_diag["sat_argmax_count"] += 1
                    else:
                        q_diag["split_argmax_count"] += 1

                if action_mask[0]:
                    q_diag["local_legal_total"] += 1
                    q_diag["q_local_when_legal_sum"] += float(q_vals[0])
                    q_diag["q_local_when_legal_count"] += 1
                    if legal_qs:
                        q_diag["q_max_when_local_legal_sum"] += max(legal_qs)
                        q_diag["q_local_minus_max_sum"] += float(q_vals[0]) - max(legal_qs)

                discrete_action = agent.predict(obs[md_idx], action_mask=action_mask)
                cont_action = agent.get_continuous(obs[md_idx], discrete_action, explore=False)

                if action_mask[0] and discrete_action != 0:
                    q_diag["local_legal_not_chosen"] += 1

                obs, reward, _, tcn, _, _, _, _, _, _, _ = env.step(
                    md_idx, discrete_action, cont_action, tcn, bs_list, md_list, sat_list
                )
                episode_reward += reward

            step_idx += 1
            env.reset_state(md_list, bs_list, sat_list)

        eval_rewards.append(episode_reward)
        eval_debug_list.append(dict(env.last_debug))

    merged_debug = {}
    if eval_debug_list:
        for key in eval_debug_list[0].keys():
            merged_debug[key] = float(np.mean([item[key] for item in eval_debug_list]))

    def _safe_avg(s, c):
        return s / c if c > 0 else 0.0

    merged_debug["q_local_avg"] = _safe_avg(q_diag["q_local_sum"], q_diag["q_local_count"])
    merged_debug["q_bs_avg"] = _safe_avg(q_diag["q_bs_sum"], q_diag["q_bs_count"])
    merged_debug["q_sat_avg"] = _safe_avg(q_diag["q_sat_sum"], q_diag["q_sat_count"])
    merged_debug["q_split_avg"] = _safe_avg(q_diag["q_split_sum"], q_diag["q_split_count"])
    merged_debug["q_spread_avg"] = _safe_avg(q_diag["q_max_minus_min_sum"], q_diag["decision_count"])
    merged_debug["q_local_argmax_rate"] = _safe_avg(q_diag["local_argmax_count"], q_diag["decision_count"])
    merged_debug["q_bs_argmax_rate"] = _safe_avg(q_diag["bs_argmax_count"], q_diag["decision_count"])
    merged_debug["q_sat_argmax_rate"] = _safe_avg(q_diag["sat_argmax_count"], q_diag["decision_count"])
    merged_debug["q_split_argmax_rate"] = _safe_avg(q_diag["split_argmax_count"], q_diag["decision_count"])
    merged_debug["q_local_gap_to_max_avg"] = _safe_avg(
        q_diag["q_local_minus_max_sum"], q_diag["q_local_when_legal_count"]
    )
    merged_debug["local_legal_not_chosen_rate"] = _safe_avg(
        q_diag["local_legal_not_chosen"], q_diag["local_legal_total"]
    )
    merged_debug["local_legal_rate"] = _safe_avg(
        q_diag["local_legal_total"], q_diag["decision_count"]
    )

    return float(np.mean(eval_rewards)), merged_debug


def run_training_experiment(result_prefix=experiment_config.MAIN_SIM_PREFIX):
    os.makedirs("models", exist_ok=True)
    os.makedirs("result", exist_ok=True)

    train_rewards_all  = []
    eval_rewards_all   = []
    debug_history_all  = []

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env   = EdgeEnv.EdgeEnv()
        agent = build_hd3qn_agent(env)

        bs_list, md_list, sat_list = build_entities(env)
        # 改后：所有 PER 参数从 para.py 统一读取
        rpm = ReplayMemory.PrioritizedReplayMemory(
            MEMORY_SIZE,
            alpha=para.PER_ALPHA,
            beta=para.PER_BETA_INIT,
            beta_increment=para.PER_BETA_INCREMENT,
            epsilon=para.PER_EPSILON,
        )

        # ---- Warmup：先填满回放池 ----
        while len(rpm) < MEMORY_WARMUP_SIZE:
            run_episode(env, rpm, agent, md_list, bs_list, sat_list)

        train_rewards        = []
        eval_rewards         = []
        eval_rewards_ran_only = []
        debug_history        = []
        train_metric_history = []
        eval_metric_history  = []
        best_reward  = -float("inf")
        best_episode = 0
        last_eval_reward  = None
        last_eval_debug   = None
        last_eval_metrics = None

        for episode in range(max_episode):
            train_reward, train_debug = run_episode(
                env, rpm, agent, md_list, bs_list, sat_list
            )

            should_run_eval = (
                episode == 0 or
                (episode + 1) % max(EVAL_INTERVAL, 1) == 0 or
                episode == max_episode - 1
            )

            if should_run_eval:
                # agent2=None：evaluate 忽略 agent2
                eval_reward, eval_debug = evaluate(
                    env, agent, None,
                    md_list, bs_list, sat_list,
                    eval_rounds=EVAL_ROUNDS,
                )
                eval_metrics = summarize_debug(eval_debug)
                eval_rewards_ran_only.append(eval_reward)
                smoothed_eval_reward = compute_smoothed_value(
                    eval_rewards_ran_only, REWARD_SMOOTH_WINDOW
                )
                eval_metrics["smoothed_eval_reward"] = smoothed_eval_reward
                eval_metrics["eval_index"]           = len(eval_rewards_ran_only)
                last_eval_reward  = eval_reward
                last_eval_debug   = dict(eval_debug)
                last_eval_metrics = dict(eval_metrics)
                last_eval_metrics["smoothed_eval_reward"] = smoothed_eval_reward
                last_eval_metrics["eval_index"]           = len(eval_rewards_ran_only)
            else:
                eval_reward  = float(last_eval_reward)
                eval_debug   = dict(last_eval_debug)
                eval_metrics = dict(last_eval_metrics)

            train_rewards.append(train_reward)
            eval_rewards.append(eval_reward)
            train_metrics = summarize_debug(train_debug)
            train_metric_history.append(train_metrics)
            eval_metric_history.append(eval_metrics)
            debug_history.append({
                "episode":              episode + 1,
                "train":                train_debug,
                "eval":                 eval_debug,
                "train_metrics":        train_metrics,
                "eval_metrics":         eval_metrics,
                "eval_ran_this_episode": should_run_eval,
            })

            # ---- 最优模型保存 ----
            if eval_reward > best_reward:
                best_reward  = eval_reward
                best_episode = episode + 1
                # 文件名兼容 D3SAC_Gen.py 里的加载路径（见下方 D3SAC_Gen.py 改造说明）
                agent.alg.save_model(f"models/{result_prefix}_best_hd3qn_model.pth")

            # ---- 日志打印（字段与原版一致）----
            avg_prop_delay  = eval_metrics["avg_prop_delay"]
            avg_total_delay = eval_metrics["avg_total_delay"]
            avg_reward      = eval_metrics["avg_reward"]
            train_avg_prop_delay  = train_metrics["avg_prop_delay"]
            train_avg_total_delay = train_metrics["avg_total_delay"]
            eval_sat_exec_success_rate    = eval_metrics["sat_exec_success_rate"]
            eval_sat_deadline_success_rate = eval_metrics["sat_deadline_success_rate"]
            eval_sat_timeout_rate         = eval_metrics["sat_timeout_rate"]
            eval_avg_sat_delay_over_gamma = eval_metrics["avg_sat_delay_over_gamma"]
            eval_avg_sat_load_penalty     = eval_metrics["avg_sat_load_penalty"]
            eval_avg_sat_usage_ratio_on_feasible_sat      = eval_metrics["avg_sat_usage_ratio_on_feasible_sat"]
            eval_avg_sat_peak_usage_ratio_on_feasible_sat = eval_metrics["avg_sat_peak_usage_ratio_on_feasible_sat"]
            eval_max_sat_peak_usage_ratio = eval_metrics["max_sat_peak_usage_ratio"]
            eval_smoothed_reward          = eval_metrics.get("smoothed_eval_reward", 0.0)
            current_e_greed               = agent.e_greed

            print(
                f"[{result_prefix}] Episode {episode + 1}/{max_episode} | "
                f"train_reward={train_reward:.3f} | eval_reward={eval_reward:.3f} | "
                f"eval_ran={int(should_run_eval)} | "
                f"e_greed={current_e_greed:.3f} | "
                f"train(local={train_debug['local_actions']:.1f},"
                f"bs={train_debug['bs_actions']:.1f},"
                f"sat={train_debug['sat_actions']:.1f},"
                f"split={train_debug.get('split_actions', 0.0):.1f},"
                f"avg_prop={train_avg_prop_delay:.6f}s,"
                f"avg_total={train_avg_total_delay:.6f}s) | "
                f"eval(local={eval_debug['local_actions']:.1f},"
                f"bs={eval_debug['bs_actions']:.1f},"
                f"sat={eval_debug['sat_actions']:.1f},"
                f"split={eval_debug.get('split_actions', 0.0):.1f}) | "
                f"avg_prop={avg_prop_delay:.6f}s avg_total={avg_total_delay:.6f}s "
                f"avg_energy={eval_metrics['avg_total_energy']:.6f}J "
                f"avg_reward={avg_reward:.3f} | "
                f"smooth_eval={eval_smoothed_reward:.3f} | "
                f"sat_usage={eval_metrics['sat_usage_rate']:.3f} "
                f"sat_exec={eval_sat_exec_success_rate:.3f} "
                f"sat_deadline={eval_sat_deadline_success_rate:.3f} "
                f"sat_timeout={eval_sat_timeout_rate:.3f} "
                f"over_gamma={eval_avg_sat_delay_over_gamma:.6f}s | "
                f"sat_load(avg_pen={eval_avg_sat_load_penalty:.4f},"
                f"avg_use={eval_avg_sat_usage_ratio_on_feasible_sat:.3f},"
                f"peak={eval_avg_sat_peak_usage_ratio_on_feasible_sat:.3f},"
                f"peak_max={eval_max_sat_peak_usage_ratio:.3f}) | "
                f"hd3qn_loss={train_debug.get('mean_dqn_loss', 0.0):.5f} "
                f"q_loss={train_debug.get('mean_q_loss', 0.0):.5f} "
                f"cont_loss={train_debug.get('mean_cont_loss', 0.0):.5f}"
            )
            # === 新增：Q 值诊断 ===
            print(
                f"  [Q-diag] "
                f"Q_local={eval_debug.get('q_local_avg', 0.0):.3f} "
                f"Q_bs={eval_debug.get('q_bs_avg', 0.0):.3f} "
                f"Q_sat={eval_debug.get('q_sat_avg', 0.0):.3f} "
                f"Q_split={eval_debug.get('q_split_avg', 0.0):.3f} | "
                f"spread={eval_debug.get('q_spread_avg', 0.0):.3f} | "
                f"argmax(local={eval_debug.get('q_local_argmax_rate', 0.0):.3f},"
                f"bs={eval_debug.get('q_bs_argmax_rate', 0.0):.3f},"
                f"sat={eval_debug.get('q_sat_argmax_rate', 0.0):.3f},"
                f"split={eval_debug.get('q_split_argmax_rate', 0.0):.3f}) | "
                f"local_legal_rate={eval_debug.get('local_legal_rate', 0.0):.3f} "
                f"local_gap_to_max={eval_debug.get('q_local_gap_to_max_avg', 0.0):.3f} "
                f"local_skip_rate={eval_debug.get('local_legal_not_chosen_rate', 0.0):.3f}"
            )

            print(
                f"  [Split] "
                f"usage={eval_metrics.get('split_usage_rate', 0.0):.3f} "
                f"in_sat={eval_metrics.get('split_in_sat_rate', 0.0):.3f} "
                f"deadline={eval_metrics.get('split_deadline_success_rate', 0.0):.3f} "
                f"timeout={eval_metrics.get('split_timeout_rate', 0.0):.3f} "
                f"avg_ratio={eval_metrics.get('avg_split_ratio', 0.0):.3f} "
                f"over_gamma={eval_metrics.get('avg_split_delay_over_gamma', 0.0):.6f}s"
            )

            # 👇 【在这里添加下面这一行，保持 12 个空格缩进】
            print(
                f"  [SAT-mask] "
                f"feasible={eval_metrics.get('sat_mask_feasible_rate', 0.0):.3f} "
                f"not_visible={eval_metrics.get('sat_mask_not_visible_rate', 0.0):.3f} "
                f"no_resource={eval_metrics.get('sat_mask_no_resource_rate', 0.0):.3f} "
                f"budget_fail={eval_metrics.get('sat_mask_non_positive_budget_rate', 0.0):.3f} "
                f"power_gap={eval_metrics.get('sat_mask_power_infeasible_rate', 0.0):.3f} | "
                f"budget_gap={eval_metrics.get('sat_mask_remaining_budget_fail_avg', 0.0):.6f}s "
                f"power_excess={eval_metrics.get('sat_mask_power_excess_ratio_avg', 0.0):.3f} "
                f"visible_rate={eval_metrics.get('visible_sat_decision_rate', 0.0):.3f} "
                f"avg_visible={eval_metrics.get('avg_visible_satellites', 0.0):.3f} "
                f"visible_not_selected={eval_metrics.get('visible_but_not_selected_rate', 0.0):.3f}"
            )
            agent.alg.scheduler.step()
        train_rewards_all.append(train_rewards)
        eval_rewards_all.append(eval_rewards)
        debug_history_all.append(debug_history)

        # ---- pickle 导出（格式与原版完全一致）----
        with open(f"result/{result_prefix}_train_metrics_seed_{seed}.pickle", "wb") as f:
            pickle.dump(train_metric_history, f)
        with open(f"result/{result_prefix}_eval_metrics_seed_{seed}.pickle", "wb") as f:
            pickle.dump(eval_metric_history, f)
        with open(f"result/{result_prefix}_summary_seed_{seed}.pickle", "wb") as f:
            pickle.dump({
                "seed": seed,
                "best_eval_reward":  best_reward,
                "best_eval_episode": best_episode,
                "final_episode":     len(train_rewards),
                "eval_times":        len(eval_rewards_ran_only),
            }, f)

    with open(f"result/{result_prefix}_train_rewards.pickle", "wb") as f:
        pickle.dump(train_rewards_all, f)
    with open(f"result/{result_prefix}_eval_rewards.pickle", "wb") as f:
        pickle.dump(eval_rewards_all, f)
    with open(f"result/{result_prefix}_debug_history.pickle", "wb") as f:
        pickle.dump(debug_history_all, f)

    # ---- 卫星统计曲线（格式与原版一致）----
    train_sat_usage_all = [
        [em["sat_usage_rate"] for em in [item["train_metrics"] for item in seed_d]]
        for seed_d in debug_history_all
    ]
    eval_sat_usage_all = [
        [em["sat_usage_rate"] for em in [item["eval_metrics"] for item in seed_d]]
        for seed_d in debug_history_all
    ]
    eval_sat_success_all = [
        [em["sat_deadline_success_rate"] for em in [item["eval_metrics"] for item in seed_d]]
        for seed_d in debug_history_all
    ]
    eval_prop_delay_all = [
        [em["avg_prop_delay"] for em in [item["eval_metrics"] for item in seed_d]]
        for seed_d in debug_history_all
    ]

    with open(f"result/{result_prefix}_train_sat_usage.pickle", "wb") as f:
        pickle.dump(train_sat_usage_all, f)
    with open(f"result/{result_prefix}_eval_sat_usage.pickle", "wb") as f:
        pickle.dump(eval_sat_usage_all, f)
    with open(f"result/{result_prefix}_eval_sat_success.pickle", "wb") as f:
        pickle.dump(eval_sat_success_all, f)
    with open(f"result/{result_prefix}_eval_prop_delay.pickle", "wb") as f:
        pickle.dump(eval_prop_delay_all, f)

    return {
        "train_rewards_all": train_rewards_all,
        "eval_rewards_all":  eval_rewards_all,
        "debug_history_all": debug_history_all,
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_num_threads(para.CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, para.CPU_THREADS))
    except RuntimeError:
        pass
    current_device = "cuda" if para.USE_CUDA and torch.cuda.is_available() else "cpu"
    print(
        f"Run mode: {para.RUN_MODE} | device: {current_device} | "
        f"cpu_threads: {para.CPU_THREADS} | episodes: {para.max_episode} | "
        f"steps: {para.steps} | seeds: {para.SEED} | algorithm: HD3QN"
    )
    run_training_experiment(result_prefix=experiment_config.MAIN_SIM_PREFIX)
