import os
import pickle
import random
import warnings

import numpy as np
import torch

import BS
import D3QN
import EdgeEnv
import MD
import ReplayMemory
import SAC
import SAT
import experiment_config
import para

warnings.filterwarnings("ignore", category=Warning)

LEARN_FREQ = para.LEARN_FREQ
SAC_LEARN_FREQ = para.SAC_LEARN_FREQ
MEMORY_SIZE = para.MEMORY_SIZE
SAC_MEMORY_SIZE = para.SAC_MEMORY_SIZE
MEMORY_WARMUP_SIZE = para.MEMORY_WARMUP_SIZE
SAC_MEMORY_WARMUP_SIZE = para.SAC_MEMORY_WARMUP_SIZE
BATCH_SIZE = para.BATCH_SIZE
SAC_BATCH_SIZE = para.SAC_BATCH_SIZE
LEARNING_RATE = para.LEARNING_RATE
ACTOR_LR = para.ACTOR_LR
CRITIC_LR = para.CRITIC_LR
DQN_GAMMA = para.DQN_GAMMA
SAC_GAMMA = para.SAC_GAMMA
TAU = para.TAU
N = para.N
S = para.S
M = para.M
F_BS = para.F_BS
F_MD = para.F_MD
SAT_F = para.SAT_F
steps = para.steps
max_episode = para.max_episode
SEED = para.SEED
EVAL_INTERVAL = para.EVAL_INTERVAL
EVAL_ROUNDS = para.EVAL_ROUNDS
REWARD_SMOOTH_WINDOW = para.REWARD_SMOOTH_WINDOW


def get_augmented_obs(obs_row, action_idx, action_dim):
    action_feature = np.array([action_idx / max(action_dim - 1, 1)], dtype=np.float32)
    gamma_feature = np.array([float(obs_row[2])], dtype=np.float32)  # obs_row[2] 是归一化后的 Gamma
    return np.concatenate([obs_row.astype(np.float32), action_feature, gamma_feature], axis=0)


def summarize_debug(debug_info):
    # 这里把环境侧的原始计数转换成更适合论文和画图的指标。
    steps_count = max(float(debug_info.get("steps", 0.0)), 1.0)
    sat_actions = float(debug_info.get("sat_actions", 0.0))
    sat_metric_steps = max(float(debug_info.get("sat_metric_steps", 0.0)), 1.0)
    visible_decisions = float(debug_info.get("visible_sat_decisions", 0.0))
    visible_sat_total_count = float(debug_info.get("visible_sat_total_count", 0.0))
    sat_exec_success_actions = float(debug_info.get("sat_exec_success_actions", 0.0))
    sat_deadline_success_actions = float(debug_info.get("sat_deadline_success_actions", 0.0))
    sat_timeout_actions = float(debug_info.get("sat_timeout_actions", 0.0))
    sat_mask_checks = max(float(debug_info.get("mask_sat_total_checks", 0.0)), 1.0)
    sat_budget_fail_count = max(float(debug_info.get("mask_sat_remaining_budget_fail_count", 0.0)), 1.0)
    sat_power_fail_count = max(float(debug_info.get("mask_sat_power_excess_ratio_count", 0.0)), 1.0)
    sat_feasible_metric_steps = max(float(debug_info.get("sat_feasible_metric_steps", 0.0)), 1.0)
    sat_load_penalty_count = max(float(debug_info.get("sat_load_penalty_count", 0.0)), 1.0)

    return {
        "avg_prop_delay": float(debug_info.get("avg_prop_delay_sum", 0.0)) / steps_count,
        "avg_total_delay": float(debug_info.get("avg_total_delay_sum", 0.0)) / steps_count,
        "avg_reward": float(debug_info.get("reward_sum", 0.0)) / steps_count,
        "avg_phi_cost": float(debug_info.get("avg_cost_sum", 0.0)) / steps_count,
        "avg_task_value": float(debug_info.get("avg_value_sum", 0.0)) / steps_count,
        "avg_delay_cost": float(debug_info.get("avg_delay_cost_sum", 0.0)) / steps_count,
        "avg_energy_cost": float(debug_info.get("avg_energy_cost_sum", 0.0)) / steps_count,
        "avg_selected_power": float(debug_info.get("avg_selected_power_sum", 0.0)) / steps_count,
        "avg_min_power": float(debug_info.get("avg_min_power_sum", 0.0)) / steps_count,
        "avg_unconstrained_power": float(debug_info.get("avg_unconstrained_power_sum", 0.0)) / steps_count,
        # 这些量对应论文中的卫星版 Lemma 2 中间符号，便于后续直接导出写论文。
        "avg_g_bar_s": float(debug_info.get("avg_g_bar_s_sum", 0.0)) / steps_count,
        "avg_remaining_budget_sat": float(debug_info.get("avg_remaining_budget_sat_sum", 0.0)) / steps_count,
        "avg_P_min_sat": float(debug_info.get("avg_P_min_sat_sum", 0.0)) / steps_count,
        "avg_p_star_sat": float(debug_info.get("avg_p_star_sat_sum", 0.0)) / steps_count,
        "avg_g_sat_raw": float(debug_info.get("avg_g_sat_raw_sum", 0.0)) / steps_count,
        "avg_sat_snr": float(debug_info.get("avg_sat_snr_sum", 0.0)) / steps_count,
        "avg_sat_rate": float(debug_info.get("avg_sat_rate_sum", 0.0)) / steps_count,
        "avg_relative_velocity_sat": float(debug_info.get("avg_relative_velocity_sat_sum", 0.0)) / steps_count,
        "avg_doppler_shift_sat": float(debug_info.get("avg_doppler_shift_sat_sum", 0.0)) / steps_count,
        "avg_eta_d_sat": float(debug_info.get("avg_eta_d_sat_sum", 0.0)) / steps_count,
        "avg_wavelength_sat": float(debug_info.get("avg_wavelength_sat_sum", 0.0)) / steps_count,
        # 这一组是“仅在实际选择卫星动作时”的均值，更适合解释论文中的卫星专属中间量。
        "avg_g_bar_s_on_sat": float(debug_info.get("avg_g_bar_s_sum", 0.0)) / sat_metric_steps,
        "avg_remaining_budget_sat_on_sat": float(debug_info.get("avg_remaining_budget_sat_sum", 0.0)) / sat_metric_steps,
        "avg_P_min_sat_on_sat": float(debug_info.get("avg_P_min_sat_sum", 0.0)) / sat_metric_steps,
        "avg_p_star_sat_on_sat": float(debug_info.get("avg_p_star_sat_sum", 0.0)) / sat_metric_steps,
        # 这一组是“仅在卫星动作且真实可行时”的均值，
        # 更适合用来解释论文里的闭式功率控制结果，避免被不可行样本的极端值污染。
        "avg_g_bar_s_on_feasible_sat": float(debug_info.get("avg_g_bar_s_feasible_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_remaining_budget_sat_on_feasible_sat": float(debug_info.get("avg_remaining_budget_sat_feasible_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_P_min_sat_on_feasible_sat": float(debug_info.get("avg_P_min_sat_feasible_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_p_star_sat_on_feasible_sat": float(debug_info.get("avg_p_star_sat_feasible_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_g_sat_raw_on_sat": float(debug_info.get("avg_g_sat_raw_sum", 0.0)) / sat_metric_steps,
        "avg_sat_snr_on_sat": float(debug_info.get("avg_sat_snr_sum", 0.0)) / sat_metric_steps,
        "avg_sat_rate_on_sat": float(debug_info.get("avg_sat_rate_sum", 0.0)) / sat_metric_steps,
        "avg_relative_velocity_sat_on_sat": float(debug_info.get("avg_relative_velocity_sat_sum", 0.0)) / sat_metric_steps,
        "avg_doppler_shift_sat_on_sat": float(debug_info.get("avg_doppler_shift_sat_sum", 0.0)) / sat_metric_steps,
        "avg_eta_d_sat_on_sat": float(debug_info.get("avg_eta_d_sat_sum", 0.0)) / sat_metric_steps,
        "avg_wavelength_sat_on_sat": float(debug_info.get("avg_wavelength_sat_sum", 0.0)) / sat_metric_steps,
        "avg_penalty_time": float(debug_info.get("penalty_time_sum", 0.0)) / steps_count,
        "avg_penalty_resource": float(debug_info.get("penalty_resource_sum", 0.0)) / steps_count,
        "avg_penalty_visibility": float(debug_info.get("penalty_visibility_sum", 0.0)) / steps_count,
        "avg_penalty_propagation": float(debug_info.get("penalty_propagation_sum", 0.0)) / steps_count,
        "avg_penalty_zero_alloc": float(debug_info.get("penalty_zero_alloc_sum", 0.0)) / steps_count,
        # 这组指标对应论文里的“卫星参与度-性能折中”分析。
        # 它们不是新的硬约束，而是用来判断：
        # 当前奖励波动，是否和“单时隙内卫星被集中使用过多”有关。
        "avg_sat_load_penalty": float(debug_info.get("sat_load_penalty_sum", 0.0)) / steps_count,
        "avg_sat_load_penalty_on_triggered_sat": float(debug_info.get("sat_load_penalty_sum", 0.0)) / sat_load_penalty_count,
        "avg_sat_usage_ratio_on_feasible_sat": float(debug_info.get("sat_usage_ratio_sum", 0.0)) / sat_feasible_metric_steps,
        "avg_sat_peak_usage_ratio_on_feasible_sat": float(debug_info.get("sat_peak_usage_ratio_sum", 0.0)) / sat_feasible_metric_steps,
        "max_sat_peak_usage_ratio": float(debug_info.get("sat_peak_usage_ratio_max", 0.0)),
        "sat_usage_rate": sat_actions / steps_count,
        # 这里拆分卫星相关成功率：
        # 1. sat_exec_success_rate 对应“卫星动作在物理层面可执行”的比例；
        # 2. sat_deadline_success_rate 对应“卫星动作同时满足任务 deadline”的比例；
        # 3. sat_timeout_rate 对应“卫星动作被执行后仍然超时”的比例。
        "sat_exec_success_rate": sat_exec_success_actions / max(sat_actions, 1.0),
        "sat_deadline_success_rate": sat_deadline_success_actions / max(sat_actions, 1.0),
        "sat_timeout_rate": sat_timeout_actions / max(sat_actions, 1.0),
        "avg_sat_delay_over_gamma": float(debug_info.get("sat_delay_over_gamma_sum", 0.0)) / max(sat_timeout_actions, 1.0),
        "visible_sat_decision_rate": visible_decisions / steps_count,
        "avg_visible_satellites": visible_sat_total_count / steps_count,
        "visible_but_not_selected_rate": float(debug_info.get("sat_visible_not_selected", 0.0)) / max(visible_decisions, 1.0),
        # 这一组比率直接对应“卫星动作在进入 D3QN 候选集合之前，被哪条论文约束挡掉了”。
        "sat_mask_feasible_rate": float(debug_info.get("mask_sat_feasible", 0.0)) / sat_mask_checks,
        "sat_mask_not_visible_rate": float(debug_info.get("mask_sat_not_visible", 0.0)) / sat_mask_checks,
        "sat_mask_no_resource_rate": float(debug_info.get("mask_sat_no_resource", 0.0)) / sat_mask_checks,
        "sat_mask_non_positive_budget_rate": float(debug_info.get("mask_sat_non_positive_budget", 0.0)) / sat_mask_checks,
        "sat_mask_power_infeasible_rate": float(debug_info.get("mask_sat_power_infeasible", 0.0)) / sat_mask_checks,
        # 这一组量用于刻画“失败程度”，帮助判断是轻微不可行还是严重不可行。
        "sat_mask_remaining_budget_fail_avg": float(debug_info.get("mask_sat_remaining_budget_fail_sum", 0.0)) / sat_budget_fail_count,
        "sat_mask_power_excess_ratio_avg": float(debug_info.get("mask_sat_power_excess_ratio_sum", 0.0)) / sat_power_fail_count,
    }


def compute_smoothed_value(value_history, window_size):
    # 这里做的是“评估结果后处理平滑”，不是修改环境奖励函数。
    # 论文里的状态、动作、奖励定义保持不变：
    # 1. 状态仍然是 EdgeEnv 给出的任务、链路、资源等观测；
    # 2. 动作仍然是 D3QN 选离散卸载目标，SAC 选连续频率分配；
    # 3. 奖励仍然是环境里的时延-能耗-任务价值综合回报。
    # 我们只是在训练分析阶段，对最近若干次 eval_reward 求滑动平均，
    # 用来更稳地判断“当前策略是否真的进入稳定较优区间”。
    if not value_history:
        return 0.0
    effective_window = max(int(window_size), 1)
    tail_values = value_history[-effective_window:]
    return float(np.mean(tail_values))


def run_episode(env, rpm, rpm2, agent, agent2, md_list, bs_list, sat_list):
    total_reward = 0.0
    obs = env.reset(md_list, bs_list, sat_list)
    step_idx = 0
    critic_losses = []
    actor_losses = []
    alpha_values = []

    while step_idx < steps:
        slot_reward = 0.0
        obs = env.get_state()
        tcn = [0] * 6

        for md_idx in range(M):
            action_mask = env.get_action_mask(md_list[md_idx], sat_list)
            discrete_action = agent.sample(obs[md_idx], action_mask=action_mask)
            pre_obs = obs[md_idx].copy()
            sac_obs = get_augmented_obs(pre_obs, discrete_action, env.action_space.n)
            cont_action = agent2.predict(sac_obs.astype("float32"))

            obs, reward, done, _, _, _, _, _, _, _, _ = env.step(
                md_idx, discrete_action, cont_action, tcn, bs_list, md_list, sat_list
            )

            if md_idx == M - 1:
                # 最后一个设备：下一状态未知，done=True，next_obs 复用自身
                next_mask = env.get_action_mask(md_list[md_idx], sat_list)
                next_sac_obs = sac_obs
                rpm.append((pre_obs, discrete_action, reward, obs[md_idx], True, next_mask))
                rpm2.append((sac_obs, cont_action, reward, next_sac_obs, True))
            else:
                next_md_idx = md_idx + 1
                next_mask = env.get_action_mask(md_list[next_md_idx], sat_list)
                next_discrete_action = agent.predict(obs[next_md_idx], action_mask=next_mask)
                next_sac_obs = get_augmented_obs(obs[next_md_idx], next_discrete_action, env.action_space.n)
                rpm.append((pre_obs, discrete_action, reward, obs[next_md_idx], done, next_mask))
                rpm2.append((sac_obs, cont_action, reward, next_sac_obs, done))
            slot_reward += reward

            if len(rpm) > MEMORY_WARMUP_SIZE and (md_idx % LEARN_FREQ == 0):
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_next_mask = rpm.sample(BATCH_SIZE)
                dqn_loss = agent.learn(
                    batch_obs,
                    batch_action,
                    batch_reward,
                    batch_next_obs,
                    batch_done,
                    next_action_mask=batch_next_mask,
                )
                critic_losses.append(dqn_loss)

            if len(rpm2) > SAC_MEMORY_WARMUP_SIZE and (md_idx % SAC_LEARN_FREQ == 0):
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = rpm2.sample(SAC_BATCH_SIZE)
                sac_loss, alpha = agent2.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                actor_losses.append(sac_loss)
                alpha_values.append(float(alpha.detach().cpu().numpy()))

        total_reward += slot_reward
        step_idx += 1
        env.reset_state(md_list, bs_list, sat_list)

    debug_info = dict(env.last_debug)
    debug_info["mean_dqn_loss"] = float(np.mean(critic_losses)) if critic_losses else 0.0
    debug_info["mean_sac_loss"] = float(np.mean(actor_losses)) if actor_losses else 0.0
    debug_info["mean_alpha"] = float(np.mean(alpha_values)) if alpha_values else 0.0
    return total_reward, debug_info


def evaluate(env, agent, agent2, md_list, bs_list, sat_list, eval_rounds=2):
    eval_rewards = []
    eval_debug_list = []

    for _ in range(eval_rounds):
        obs = env.reset(md_list, bs_list, sat_list)
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
                discrete_action = agent.predict(obs[md_idx], action_mask=action_mask)
                sac_obs = get_augmented_obs(obs[md_idx], discrete_action, env.action_space.n)
                cont_action = agent2.predict_e(sac_obs.astype("float32"))
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
    return float(np.mean(eval_rewards)), merged_debug


def build_entities(env):
    bs_list = []
    md_list = []
    sat_list = []

    for n in range(N):
        bs = BS.BS(n, F_BS[n], random.uniform(0, para.MAP_WIDTH), random.uniform(0, para.MAP_HEIGHT), para.BS_HEIGHT)
        bs_list.append(bs)

    for s_idx in range(S):
        sat_list.append(SAT.SAT(s_idx, SAT_F[s_idx]))

    for m in range(M):
        md_list.append(MD.MD(m, F_MD, env, bs_list))

    return bs_list, md_list, sat_list


def run_training_experiment(result_prefix=experiment_config.MAIN_SIM_PREFIX):
    # 这个函数把训练主流程抽出来，方便后续做“有/无拥塞”等对比实验时复用。
    os.makedirs("models", exist_ok=True)
    os.makedirs("result", exist_ok=True)

    train_rewards_all = []
    eval_rewards_all = []
    debug_history_all = []

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = EdgeEnv.EdgeEnv()
        action_dim = env.action_space.n
        obs_shape = env.observation_space.shape[1]
        action_dim2 = env.action_space2.shape[0]

        model = D3QN.Model(obs_dim=obs_shape, act_dim=action_dim)
        algorithm = D3QN.DDQN(model, act_dim=action_dim, gamma=DQN_GAMMA, lr=LEARNING_RATE)
        agent = D3QN.Agent(
            algorithm,
            obs_dim=obs_shape,
            act_dim=action_dim,
            # 这里把离散卸载决策的探索强度调得更温和一些。
            # 论文模型里 D3QN 负责“选本地/地面/卫星”，如果初始探索率过高，
            # 训练阶段会过早、过猛地尝试大量卫星动作，容易在 SAC 还没学稳时把评估策略带偏。
            # 因此这里改成“中等探索 + 更快衰减”，先提升两阶段联合训练的稳定性。
            e_greed=0.9,
            e_greed_decrement=0.89 / max(max_episode * steps * M, 1),
        )

        model2 = SAC.Model(obs_dim=obs_shape + 2, act_dim=action_dim2)
        actor_model = SAC.ActorModel(obs_dim=obs_shape + 2, act_dim=action_dim2)
        algorithm2 = SAC.SAC(model2, actor_model, ...)
        agent2 = SAC.Agent(algorithm2, obs_dim=obs_shape + 2, act_dim=action_dim2)

        bs_list, md_list, sat_list = build_entities(env)
        rpm = ReplayMemory.ReplayMemory(MEMORY_SIZE)
        rpm2 = ReplayMemory.ReplayMemory(SAC_MEMORY_SIZE)

        while len(rpm) < MEMORY_WARMUP_SIZE or len(rpm2) < SAC_MEMORY_WARMUP_SIZE:
            run_episode(env, rpm, rpm2, agent, agent2, md_list, bs_list, sat_list)

        train_rewards = []
        eval_rewards = []
        eval_rewards_ran_only = []
        debug_history = []
        train_metric_history = []
        eval_metric_history = []
        best_reward = -float("inf")
        best_episode = 0
        # 这里缓存最近一次“真实执行过”的评估结果。
        # 论文正式实验不需要每个 episode 都完整评估一次，否则大量时间会消耗在重复测试上。
        # 因此我们采用“间隔评估 + 结果复用”的方式，在保证导出曲线长度不变的同时缩短训练时间。
        last_eval_reward = None
        last_eval_debug = None
        last_eval_metrics = None

        for episode in range(max_episode):
            train_reward, train_debug = run_episode(env, rpm, rpm2, agent, agent2, md_list, bs_list, sat_list)
            # 这里把评估流程改成“首轮评估 + 固定间隔评估 + 最后一轮评估”，
            # 对应投稿阶段常用的正式实验组织方式。
            should_run_eval = (
                episode == 0 or
                (episode + 1) % max(EVAL_INTERVAL, 1) == 0 or
                episode == max_episode - 1
            )
            if should_run_eval:
                eval_reward, eval_debug = evaluate(
                    env,
                    agent,
                    agent2,
                    md_list,
                    bs_list,
                    sat_list,
                    eval_rounds=EVAL_ROUNDS,
                )
                eval_metrics = summarize_debug(eval_debug)
                # 这里单独记录真实执行过的评估点，避免复用值重复进入平滑窗口。
                # 对论文实验解释来说，smoothed_eval_reward 表示“最近若干次真实评估的平均性能”。
                eval_rewards_ran_only.append(eval_reward)
                smoothed_eval_reward = compute_smoothed_value(eval_rewards_ran_only, REWARD_SMOOTH_WINDOW)
                # 这里把平滑后的评估回报和稳定性分数直接写回当前 eval_metrics。
                # 这样同一轮 episode 后续的日志打印、最优模型判定、early stopping
                # 都能直接读取，不会出现字段缺失。
                eval_metrics["smoothed_eval_reward"] = smoothed_eval_reward
                eval_metrics["eval_index"] = len(eval_rewards_ran_only)
                last_eval_reward = eval_reward
                last_eval_debug = dict(eval_debug)
                last_eval_metrics = dict(eval_metrics)
                last_eval_metrics["smoothed_eval_reward"] = smoothed_eval_reward
                last_eval_metrics["eval_index"] = len(eval_rewards_ran_only)
            else:
                # 这里沿用最近一次评估结果，只是为了保持每个 episode 都有一条可导出的记录。
                eval_reward = float(last_eval_reward)
                eval_debug = dict(last_eval_debug)
                eval_metrics = dict(last_eval_metrics)

            train_rewards.append(train_reward)
            eval_rewards.append(eval_reward)
            train_metrics = summarize_debug(train_debug)
            train_metric_history.append(train_metrics)
            eval_metric_history.append(eval_metrics)
            debug_history.append({
                "episode": episode + 1,
                "train": train_debug,
                "eval": eval_debug,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                # 这个标记方便后续区分：
                # 当前 episode 的评估结果是“真实跑出来的”，还是“复用最近一次评估”的。
                "eval_ran_this_episode": should_run_eval,
            })

            if eval_reward > best_reward:
                # 这里仍然保留 best checkpoint 机制。
                # 对应论文实验里的含义是：虽然训练完整跑满设定轮次，
                # 但最终报告和复现实验时，仍可回到评估回报最高的模型参数。
                best_reward = eval_reward
                best_episode = episode + 1
                agent.alg.save_model(f"models/{result_prefix}_best_d3qn_model.pth")
                agent2.alg.save_model(f"models/{result_prefix}_best_sac_model.pth")


            avg_prop_delay = eval_metrics["avg_prop_delay"]
            avg_total_delay = eval_metrics["avg_total_delay"]
            avg_reward = eval_metrics["avg_reward"]
            train_avg_prop_delay = train_metrics["avg_prop_delay"]
            train_avg_total_delay = train_metrics["avg_total_delay"]
            eval_sat_exec_success_rate = eval_metrics["sat_exec_success_rate"]
            eval_sat_deadline_success_rate = eval_metrics["sat_deadline_success_rate"]
            eval_sat_timeout_rate = eval_metrics["sat_timeout_rate"]
            eval_avg_sat_delay_over_gamma = eval_metrics["avg_sat_delay_over_gamma"]
            eval_avg_sat_load_penalty = eval_metrics["avg_sat_load_penalty"]
            eval_avg_sat_usage_ratio_on_feasible_sat = eval_metrics["avg_sat_usage_ratio_on_feasible_sat"]
            eval_avg_sat_peak_usage_ratio_on_feasible_sat = eval_metrics["avg_sat_peak_usage_ratio_on_feasible_sat"]
            eval_max_sat_peak_usage_ratio = eval_metrics["max_sat_peak_usage_ratio"]
            eval_smoothed_reward = eval_metrics.get("smoothed_eval_reward", 0.0)
            current_e_greed = agent.e_greed
            print(
                f"[{result_prefix}] Episode {episode + 1}/{max_episode} | "
                f"train_reward={train_reward:.3f} | eval_reward={eval_reward:.3f} | "
                f"eval_ran={int(should_run_eval)} | "
                f"e_greed={current_e_greed:.3f} | "
                f"train(local={train_debug['local_actions']:.1f},bs={train_debug['bs_actions']:.1f},sat={train_debug['sat_actions']:.1f},"
                f"avg_prop={train_avg_prop_delay:.6f}s,avg_total={train_avg_total_delay:.6f}s) | "
                f"eval(local={eval_debug['local_actions']:.1f},bs={eval_debug['bs_actions']:.1f},sat={eval_debug['sat_actions']:.1f}) | "
                f"avg_prop={avg_prop_delay:.6f}s avg_total={avg_total_delay:.6f}s avg_reward={avg_reward:.3f} | "
                f"smooth_eval={eval_smoothed_reward:.3f} | "
                f"sat_usage={eval_metrics['sat_usage_rate']:.3f} "
                f"sat_exec={eval_sat_exec_success_rate:.3f} "
                f"sat_deadline={eval_sat_deadline_success_rate:.3f} "
                f"sat_timeout={eval_sat_timeout_rate:.3f} "
                f"over_gamma={eval_avg_sat_delay_over_gamma:.6f}s | "
                f"sat_load(avg_pen={eval_avg_sat_load_penalty:.4f},avg_use={eval_avg_sat_usage_ratio_on_feasible_sat:.3f},"
                f"peak={eval_avg_sat_peak_usage_ratio_on_feasible_sat:.3f},peak_max={eval_max_sat_peak_usage_ratio:.3f})"
            )


        train_rewards_all.append(train_rewards)
        eval_rewards_all.append(eval_rewards)
        debug_history_all.append(debug_history)

        with open(f"result/{result_prefix}_train_metrics_seed_{seed}.pickle", "wb") as file_obj:
            pickle.dump(train_metric_history, file_obj)
        with open(f"result/{result_prefix}_eval_metrics_seed_{seed}.pickle", "wb") as file_obj:
            pickle.dump(eval_metric_history, file_obj)
        with open(f"result/{result_prefix}_summary_seed_{seed}.pickle", "wb") as file_obj:
            # 这里额外导出每个 seed 的训练摘要，方便后续论文表格和结果说明直接引用。
            # 其中 best_stability_episode 往往比 best_eval_episode 更适合作为“最终采用模型轮次”，
            # 因为它同时考虑了回报、超时率和卫星参与度是否适中。
            pickle.dump({
                "seed": seed,
                "best_eval_reward": best_reward,
                "best_eval_episode": best_episode,
                "final_episode": len(train_rewards),
                "eval_times": len(eval_rewards_ran_only),
            }, file_obj)

    with open(f"result/{result_prefix}_train_rewards.pickle", "wb") as file_obj:
        pickle.dump(train_rewards_all, file_obj)
    with open(f"result/{result_prefix}_eval_rewards.pickle", "wb") as file_obj:
        pickle.dump(eval_rewards_all, file_obj)
    with open(f"result/{result_prefix}_debug_history.pickle", "wb") as file_obj:
        pickle.dump(debug_history_all, file_obj)

    train_sat_usage_all = [[episode_metrics["sat_usage_rate"] for episode_metrics in seed_history]
                           for seed_history in [[item["train_metrics"] for item in seed_debug] for seed_debug in debug_history_all]]
    eval_sat_usage_all = [[episode_metrics["sat_usage_rate"] for episode_metrics in seed_history]
                          for seed_history in [[item["eval_metrics"] for item in seed_debug] for seed_debug in debug_history_all]]
    # 这里导出“卫星 deadline 成功率”，更符合论文里“任务按时完成”的定义。
    eval_sat_success_all = [[episode_metrics["sat_deadline_success_rate"] for episode_metrics in seed_history]
                            for seed_history in [[item["eval_metrics"] for item in seed_debug] for seed_debug in debug_history_all]]
    eval_prop_delay_all = [[episode_metrics["avg_prop_delay"] for episode_metrics in seed_history]
                           for seed_history in [[item["eval_metrics"] for item in seed_debug] for seed_debug in debug_history_all]]

    with open(f"result/{result_prefix}_train_sat_usage.pickle", "wb") as file_obj:
        pickle.dump(train_sat_usage_all, file_obj)
    with open(f"result/{result_prefix}_eval_sat_usage.pickle", "wb") as file_obj:
        pickle.dump(eval_sat_usage_all, file_obj)
    with open(f"result/{result_prefix}_eval_sat_success.pickle", "wb") as file_obj:
        pickle.dump(eval_sat_success_all, file_obj)
    with open(f"result/{result_prefix}_eval_prop_delay.pickle", "wb") as file_obj:
        pickle.dump(eval_prop_delay_all, file_obj)

    return {
        "train_rewards_all": train_rewards_all,
        "eval_rewards_all": eval_rewards_all,
        "debug_history_all": debug_history_all,
    }


if __name__ == "__main__":
    # 在主脚本入口统一设置线程数，避免多个模块重复设置 PyTorch 线程导致问题。
    torch.set_num_threads(para.CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, para.CPU_THREADS))
    except RuntimeError:
        # 如果当前 PyTorch 运行时已经初始化过 interop 线程，这里就跳过，不影响训练。
        pass
    current_device = "cuda" if para.USE_CUDA and torch.cuda.is_available() else "cpu"
    print(
        f"Run mode: {para.RUN_MODE} | device: {current_device} | "
        f"cpu_threads: {para.CPU_THREADS} | episodes: {para.max_episode} | "
        f"steps: {para.steps} | seeds: {para.SEED}"
    )
    run_training_experiment(result_prefix=experiment_config.MAIN_SIM_PREFIX)
