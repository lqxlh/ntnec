import math
import random

import gym
import numpy as np
from gym import spaces
from itertools import combinations

import para
import power_control

N = para.N
S = para.S
M = para.M
F_BS = para.F_BS
SAT_F = para.SAT_F
w_t = para.w_t
w_e = para.w_e
w_v = para.w_v


class EdgeEnv:
    def __init__(self):
        # 状态维度对应论文里的“任务特征 + 地面链路 + 卫星链路 + 剩余算力”。
        # 当前仍采用增强版状态：
        # 任务(4) + 地面连接(2) + 每颗卫星的可见性/仰角/距离/eta_d/g_raw(5S)
        # + 接入卫星索引(1) + 地面剩余算力(N) + 卫星剩余算力(S)
        self.task_feature_dim = 4
        self.ground_link_feature_dim = 2
        self.sat_link_feature_groups = 5
        self.sat_access_feature_dim = 1
        self.obs_dim = N + 6 * S + 7
        self.resource_base = (
            self.task_feature_dim
            + self.ground_link_feature_dim
            + self.sat_link_feature_groups * S
            + self.sat_access_feature_dim
        )
        low = np.tile(np.zeros(self.obs_dim), (M, 1))
        high = np.tile(np.ones(self.obs_dim), (M, 1))
        self.observation_space = spaces.Box(low=low, high=high, shape=(M, self.obs_dim))
        self.split_sat_pairs = list(combinations(range(S), 2)) if para.ENABLE_TWO_SAT_SPLIT else []
        self.split_action_start = N + S + 1
        self.action_space = gym.spaces.Discrete(N + S + 1 + len(self.split_sat_pairs))
        self.action_space2 = gym.spaces.Box(low=np.zeros(para.CONT_ACTION_DIM), high=np.ones(para.CONT_ACTION_DIM))
        self.state = None
        # 这里缓存“当前状态下每个终端-目标卫星”的随机链路样本。
        # 这样做是为了解决论文模型实现里的一个关键一致性问题：
        # 动作掩码判断卫星可行性时，会用到随机 ISL 负载；
        # 真正执行卫星动作时，也会用到同一个随机量。
        # 如果两边各抽一次，就会出现“判掩码”和“真执行”不属于同一时隙 realization 的问题。
        self._satellite_link_cache = {}
        # 这里记录“当前时隙内每颗卫星已经被累计分配了多少计算资源”。
        # 对应论文模型里的含义是：
        # D3QN+SAC 是按终端逐个决策的，但一个时隙内多终端会共享同一颗卫星资源，
        # 因此需要显式记录时隙内的累计卫星负载，才能判断“卫星是否被用得过猛”。
        self._slot_sat_usage = [0.0 for _ in range(S)]
        self.last_debug = self._empty_debug()

    def _empty_debug(self):
        # 这些统计量用于分析“论文模型改造后训练为什么成功/失败”。
        # 其中新增的 mask_sat_* 指标，专门对应论文里的卫星卸载硬约束诊断：
        # 可见性约束、资源约束、剩余时延预算约束、最小可行功率约束。
        return {
            "local_actions": 0,
            "bs_actions": 0,
            "sat_actions": 0,
            "split_actions": 0,
            "split_deadline_success_actions": 0,
            "split_timeout_actions": 0,
            "split_delay_over_gamma_sum": 0.0,
            "split_ratio_sum": 0.0,
            "split_metric_steps": 0,
            # 这里把原来单一的 sat_success 拆成两类：
            # 1. sat_exec_success_actions：卫星动作在物理层和资源层面真实可执行；
            # 2. sat_deadline_success_actions：卫星动作不仅可执行，而且总时延满足任务时延约束 Gamma。
            # 这样更符合论文里的“可行性”和“任务完成质量”两个不同概念。
            "sat_exec_success_actions": 0,
            "sat_deadline_success_actions": 0,
            "sat_timeout_actions": 0,
            "sat_delay_over_gamma_sum": 0.0,
            "sat_metric_steps": 0,
            "deadline_bonus_sum": 0.0,
            "deadline_overrun_penalty_sum": 0.0,
            "sat_visible_not_selected": 0,
            "sat_invalid_visibility": 0,
            "visible_sat_decisions": 0,
            "visible_sat_total_count": 0,
            "mask_sat_total_checks": 0,
            "mask_sat_feasible": 0,
            "mask_sat_not_visible": 0,
            "mask_sat_no_resource": 0,
            "mask_sat_non_positive_budget": 0,
            "mask_sat_power_infeasible": 0,
            "mask_sat_remaining_budget_fail_sum": 0.0,
            "mask_sat_remaining_budget_fail_count": 0,
            "mask_sat_power_excess_ratio_sum": 0.0,
            "mask_sat_power_excess_ratio_count": 0,
            # 这里单独统计“P_min_sat > MD_MAX_POWER”带来的 reward 软惩罚，方便确认它已经不再只是 mask 诊断项。
            "sat_power_penalty_count": 0,
            "sat_power_penalty_sum": 0.0,
            "sat_power_excess_ratio_sum": 0.0,
            # 这组统计量专门用于分析“卫星使用率高但收益反而下降”是否由过载引起。
            # 论文解释上，它们刻画的是一个时隙内卫星资源占用强度与策略稳定性的关系。
            "sat_load_penalty_count": 0,
            "sat_load_penalty_sum": 0.0,
            "sat_usage_ratio_sum": 0.0,
            "sat_peak_usage_ratio_sum": 0.0,
            "sat_peak_usage_ratio_max": 0.0,
            "resource_penalty": 0,
            "time_penalty": 0,
            "propagation_penalty": 0,
            "avg_selected_power_sum": 0.0,
            "avg_min_power_sum": 0.0,
            "avg_unconstrained_power_sum": 0.0,
            "avg_g_bar_s_sum": 0.0,
            "avg_remaining_budget_sat_sum": 0.0,
            "avg_P_min_sat_sum": 0.0,
            "avg_p_star_sat_sum": 0.0,
            "avg_g_bar_s_feasible_sum": 0.0,
            "avg_remaining_budget_sat_feasible_sum": 0.0,
            "avg_P_min_sat_feasible_sum": 0.0,
            "avg_p_star_sat_feasible_sum": 0.0,
            "sat_feasible_metric_steps": 0,
            "avg_g_sat_raw_sum": 0.0,
            "avg_sat_snr_sum": 0.0,
            "avg_sat_rate_sum": 0.0,
            "avg_relative_velocity_sat_sum": 0.0,
            "avg_doppler_shift_sat_sum": 0.0,
            "avg_eta_d_sat_sum": 0.0,
            "avg_wavelength_sat_sum": 0.0,
            "avg_prop_delay_sum": 0.0,
            "avg_total_delay_sum": 0.0,
            "avg_cost_sum": 0.0,
            "avg_value_sum": 0.0,
            "avg_delay_norm_sum": 0.0,
            "avg_energy_norm_sum": 0.0,
            "avg_value_norm_sum": 0.0,
            "avg_delay_cost_sum": 0.0,
            "avg_energy_cost_sum": 0.0,
            "penalty_time_sum": 0.0,
            "penalty_resource_sum": 0.0,
            "penalty_visibility_sum": 0.0,
            "penalty_propagation_sum": 0.0,
            "penalty_zero_alloc_sum": 0.0,
            # 这里汇总超功率软惩罚本身，便于和总 reward、传播惩罚分开分析。
            "penalty_sat_power_sum": 0.0,
            "reward_sum": 0.0,
            "steps": 0,
            "avg_total_energy_sum": 0.0,
        }

    def reset_debug_stats(self):
        self.last_debug = self._empty_debug()

    def task(self):
        b_bits = random.randint(para.TASK_B_MIN, para.TASK_B_MAX)
        cycles_per_bit = random.randint(para.TASK_C_MIN, para.TASK_C_MAX)
        gamma = random.uniform(para.TASK_GAMMA_MIN, para.TASK_GAMMA_MAX)
        priority = random.randint(para.TASK_PRIORITY_MIN, para.TASK_PRIORITY_MAX)
        return b_bits, cycles_per_bit, gamma, priority

    def get_action_mask(self, md, sat_list):
        # 动作索引：
        # 0 = 本地，1..N = 地面基站，N+1..N+S = 卫星。
        # 新版论文里卫星动作不仅要满足可见性，还要满足时延预算和功率可行域。
        # 这里在生成动作掩码的同时，也把“卫星为什么被挡掉”记录下来，便于论文调试。
        mask = np.ones(N + S + 1, dtype=bool)
        if self.split_sat_pairs:
            mask = np.ones(self.action_space.n, dtype=bool)
        for sat_idx, sat in enumerate(sat_list):
            mask[N + 1 + sat_idx] = self._is_satellite_action_feasible(md, sat, sat_idx, sat_list)
        for pair_idx, (sat_a, sat_b) in enumerate(self.split_sat_pairs):
            split_action_idx = self.split_action_start + pair_idx
            # 分片只要求“两颗卫星各自承担一部分任务”可行，不再要求两颗卫星都能独立跑完整任务。
            mask[split_action_idx] = self._is_split_action_feasible(md, sat_list, sat_a, sat_b)
        return mask

    def decode_frequency(self, action_idx, action_value):
        # 连续动作第一维用于普通动作的资源比例；分片动作会额外使用分片比例和第二颗卫星资源比例。
        action_vec = np.asarray(action_value, dtype=np.float32).reshape(-1)
        if action_vec.size == 0:
            action_vec = np.array([0.5], dtype=np.float32)
        action_value = float(np.clip(action_vec[0], 0.0, 1.0))
        if action_idx == 0:
            return max(action_value, para.F_MD_MIN)
        if 1 <= action_idx <= N:
            return para.BS_F_MIN + action_value * (para.BS_F_MAX - para.BS_F_MIN)
        if self._is_split_action(action_idx):
            ratio_raw = float(action_vec[0]) if action_vec.size > 0 else 0.5
            freq_a_raw = float(action_vec[1]) if action_vec.size > 1 else ratio_raw
            freq_b_raw = float(action_vec[2]) if action_vec.size > 2 else freq_a_raw
            split_ratio = para.SPLIT_MIN_RATIO + np.clip(ratio_raw, 0.0, 1.0) * (1.0 - 2.0 * para.SPLIT_MIN_RATIO)
            freq_a = para.SAT_F_MIN + np.clip(freq_a_raw, 0.0, 1.0) * (para.SAT_F_MAX - para.SAT_F_MIN)
            freq_b = para.SAT_F_MIN + np.clip(freq_b_raw, 0.0, 1.0) * (para.SAT_F_MAX - para.SAT_F_MIN)
            return float(split_ratio), float(freq_a), float(freq_b)
        return para.SAT_F_MIN + action_value * (para.SAT_F_MAX - para.SAT_F_MIN)

    def _is_split_action(self, action_idx):
        return self.split_action_start <= action_idx < self.action_space.n

    def _decode_split_pair(self, action_idx):
        pair_idx = action_idx - self.split_action_start
        return self.split_sat_pairs[pair_idx]

    def _calc_sat_load_penalty(self, sat_idx, actual_freq):
        # 这里构造一个“卫星过载软惩罚”：
        # 1. 只在卫星动作且分配频率有效时生效；
        # 2. 惩罚的对象不是“是否可行”，而是“当前时隙内是否把某颗卫星推到过高负载区”；
        # 3. 用二次型超过阈值部分来平滑惩罚，避免策略在阈值附近出现剧烈抖动。
        if not para.ENABLE_SAT_LOAD_SOFT_PENALTY:
            return 0.0, 0.0, 0.0
        if sat_idx < 0 or sat_idx >= S or actual_freq <= 0:
            return 0.0, 0.0, 0.0

        sat_capacity = max(float(SAT_F[sat_idx]), 1e-9)
        prev_ratio = float(np.clip(self._slot_sat_usage[sat_idx] / sat_capacity, 0.0, 1.5))
        next_ratio = float(np.clip((self._slot_sat_usage[sat_idx] + actual_freq) / sat_capacity, 0.0, 1.5))
        excess_ratio = max(0.0, next_ratio - para.SAT_TARGET_USAGE)
        soft_penalty = -para.SAT_LOAD_PENALTY_WEIGHT * (excess_ratio ** 2)
        return soft_penalty, prev_ratio, next_ratio

    def _calc_sat_power_excess_penalty(self, power_result):
        # 这里把 P_min_sat > MD_MAX_POWER 从“直接禁止动作”改成“按超出比例扣 reward”。
        # excess_ratio = 0 表示功率上限内可行；excess_ratio = 1 表示所需最小功率比上限多 100%。
        p_min_sat = float(power_result.get("P_min_sat", 0.0))
        if not np.isfinite(p_min_sat):
            # 当 P_min_sat 为 inf 时，说明链路在当前时延预算下非常差，用截断值代表一个很强但有限的惩罚。
            excess_ratio = float(para.SAT_POWER_EXCESS_RATIO_CLIP)
        else:
            # 只惩罚超过终端最大发射功率的部分，未超过时保持 0，不影响正常卫星动作。
            excess_ratio = max(p_min_sat / max(float(para.MD_MAX_POWER), 1e-12) - 1.0, 0.0)
        # 截断超功率比例后再平方，既保留“越超越罚”的趋势，也避免异常链路把 reward 数值拉爆。
        clipped_ratio = float(np.clip(excess_ratio, 0.0, para.SAT_POWER_EXCESS_RATIO_CLIP))
        penalty = -float(para.SAT_POWER_EXCESS_PENALTY_WEIGHT) * (clipped_ratio ** 2)
        return penalty, clipped_ratio

    def reset(self, md_list, bs_list, sat_list):
        for md in md_list:
            md.reset()
        for bs in bs_list:
            bs.reset()
        for sat in sat_list:
            sat.reset()

        self.state = np.zeros((M, self.obs_dim), dtype=np.float32)
        self._satellite_link_cache = {}
        # reset 时把时隙内卫星累计占用清零，对应一个新 episode 的开始。
        self._slot_sat_usage = [0.0 for _ in range(S)]
        self.reset_debug_stats()
        self._refresh_full_state(md_list, bs_list, sat_list)
        return np.array(self.state)

    def reset_state(self, md_list, bs_list, sat_list):
        # 每个时隙结束后恢复边缘节点剩余算力，并统一推进卫星到下一个时隙。
        for bs_idx, bs in enumerate(bs_list):
            bs.res_F = F_BS[bs_idx]
        for sat_idx, sat in enumerate(sat_list):
            sat.res_F = SAT_F[sat_idx]
            # 卫星位置按时隙同步更新，保证同一时隙内所有 MD 共享同一个 NTN 几何状态。
            sat.move()
        # 这里对应论文中的“进入下一个决策时隙”：
        # 卫星总算力恢复为下一时隙的初始可分配资源，因此时隙级累计负载也要同步清零。
        self._slot_sat_usage = [0.0 for _ in range(S)]
        self._refresh_full_state(md_list, bs_list, sat_list)

    def _normalize_task_b(self, b_bits):
        return (b_bits - para.TASK_B_MIN) / max(para.TASK_B_MAX - para.TASK_B_MIN, 1)

    def _normalize_task_c(self, cycles_per_bit):
        return (cycles_per_bit - para.TASK_C_MIN) / max(para.TASK_C_MAX - para.TASK_C_MIN, 1)

    def _normalize_gamma(self, gamma):
        return (gamma - para.TASK_GAMMA_MIN) / max(para.TASK_GAMMA_MAX - para.TASK_GAMMA_MIN, 1e-6)

    def _normalize_priority(self, priority):
        return (priority - para.TASK_PRIORITY_MIN) / max(para.TASK_PRIORITY_MAX - para.TASK_PRIORITY_MIN, 1)

    def _safe_ratio(self, val, max_val):
        return float(np.clip(val / max(max_val, 1e-9), 0.0, 1.0))

    def _normalize_sat_gain_raw(self, gain_val):
        # 卫星原始信道增益量级很小，这里按 log10 归一化后再放进状态。
        log_gain = math.log10(max(gain_val, 1e-20))
        return float(
            np.clip(
                (log_gain - para.SAT_GAIN_LOG_MIN) / max(para.SAT_GAIN_LOG_MAX - para.SAT_GAIN_LOG_MIN, 1e-9),
                0.0,
                1.0,
            )
        )

    def _get_best_visible_satellite(self, md, sat_list):
        # 对应论文里的接入卫星 c_{m,t}^{SAT}。
        best_idx = 0
        best_elevation = -1.0
        for sat_idx, sat in enumerate(sat_list):
            elevation = md.sat_elevation(sat)
            if elevation >= para.SAT_MIN_ELEVATION_DEG and elevation > best_elevation:
                best_idx = sat_idx + 1
                best_elevation = elevation
        return best_idx

    def _build_state_for_md(self, md, bs_list, sat_list):
        state_row = []
        state_row.append(self._normalize_task_b(md.B))
        state_row.append(self._normalize_task_c(md.C))
        state_row.append(self._normalize_gamma(md.Gamma))
        state_row.append(self._normalize_priority(md.Priority))

        state_row.append(md.connect_BS / max(N - 1, 1))
        state_row.append(self._safe_ratio(md.distance(bs_list[md.connect_BS]), para.MAP_WIDTH))

        visibility_vals = []
        elevation_vals = []
        distance_vals = []
        doppler_loss_vals = []
        sat_gain_vals = []
        for sat in sat_list:
            channel_info = md.sat_channel_components(sat)
            visibility_vals.append(1.0 if md.is_sat_visible(sat) else 0.0)
            elevation_vals.append(np.clip(md.sat_elevation(sat) / 90.0, 0.0, 1.0))
            distance_vals.append(self._safe_ratio(md.sat_distance(sat), para.MAX_SAT_DISTANCE))
            doppler_loss_vals.append(float(np.clip(channel_info["doppler_loss"], 0.0, 1.0)))
            sat_gain_vals.append(self._normalize_sat_gain_raw(channel_info["g_sat_raw"]))
        state_row.extend(visibility_vals)
        state_row.extend(elevation_vals)
        state_row.extend(distance_vals)
        state_row.extend(doppler_loss_vals)
        state_row.extend(sat_gain_vals)

        best_sat_idx = self._get_best_visible_satellite(md, sat_list)
        state_row.append(best_sat_idx / max(S, 1))

        for bs_idx, bs in enumerate(bs_list):
            state_row.append(self._safe_ratio(bs.res_F, F_BS[bs_idx]))
        for sat_idx, sat in enumerate(sat_list):
            state_row.append(self._safe_ratio(sat.res_F, SAT_F[sat_idx]))

        return np.array(state_row, dtype=np.float32)

    def _refresh_full_state(self, md_list, bs_list, sat_list):
        # 论文里每次状态刷新都意味着“任务/位置/可见性”可能已经变化。
        # 因此这里要清空上一状态缓存的卫星链路随机样本，避免把旧时隙的样本错用到新时隙。
        self._satellite_link_cache = {}
        for md_idx, md in enumerate(md_list):
            md.connect_BS = md.connect_choice()
            self.state[md_idx] = self._build_state_for_md(md, bs_list, sat_list)

    def _record_success(self, priority, tcn):
        if priority == 1:
            tcn[0] += 1
        elif priority == 2:
            tcn[1] += 1
        else:
            tcn[2] += 1

    def _record_total(self, priority, tcn):
        if priority == 1:
            tcn[3] += 1
        elif priority == 2:
            tcn[4] += 1
        else:
            tcn[5] += 1

    def _calc_ground_power(self, md, t_non_transmission):
        # 地面闭式功率控制：把排队、切换、计算时延都当作非传输预算消耗。
        gain_val = md.gain()
        power_result = power_control.solve_ground_power(
            task_bits=md.B,
            weight_delay=w_t,
            weight_energy=w_e,
            bandwidth=para.GROUND_BW,
            gain_value=gain_val,
            noise_power=para.GROUND_NOISE,
            max_power=para.MD_MAX_POWER,
            remaining_budget=md.Gamma - t_non_transmission,
        )
        return gain_val, power_result

    def _calc_sat_power(self, md, sat, t_comp, t_isl):
        # 卫星闭式功率控制对应论文中的：
        # bar_Gamma^SAT、g_bar_s、P_min^SAT、p_star^SAT。
        channel_info = md.sat_channel_components(sat)
        gain_val = channel_info["g_sat_raw"]
        t_prop = md.sat_distance(sat) / para.LIGHT_SPEED
        power_result = power_control.solve_satellite_power(
            task_bits=md.B,
            weight_delay=w_t,
            weight_energy=w_e,
            bandwidth=para.SAT_BW,
            gain_value=gain_val,
            doppler_loss=channel_info["doppler_loss"],
            noise_power=para.SAT_NOISE,
            max_power=para.MD_MAX_POWER,
            remaining_budget=md.Gamma - t_prop - t_isl - t_comp,
        )
        power_result["T_prop_s"] = t_prop
        power_result["channel_info"] = channel_info
        return gain_val, power_result

    def _get_access_satellite_index(self, md, sat_list):
        return self._get_best_visible_satellite(md, sat_list)

    def _calc_satellite_pair_distance(self, sat_a, sat_b):
        # 对应论文里的星间距离 d_{s,s'}^t。
        delta_x = sat_a.x - sat_b.x
        delta_y = sat_a.y - sat_b.y
        delta_h = sat_a.height - sat_b.height
        return math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_h ** 2)

    def _sample_sat_isl_delay(self, md, sat_list, target_sat_idx):
        # 对应论文中的：
        # T_ISL = 1(target != access) * [ d_{s,s'}^t / c + B / (R_max * (1-rho)) ]
        # 这里特别按新版论文修正了方向：
        # 任务从接入卫星 c_{m,t}^{SAT} 转发到目标计算卫星 b_{m,t}-N。
        access_sat_idx = self._get_access_satellite_index(md, sat_list)
        if access_sat_idx == 0:
            return 0.0, 0, 0.0
        if access_sat_idx - 1 == target_sat_idx:
            return 0.0, access_sat_idx, 0.0

        access_sat = sat_list[access_sat_idx - 1]
        target_sat = sat_list[target_sat_idx]
        d_isl = self._calc_satellite_pair_distance(target_sat, access_sat)
        t_isl_prop = d_isl / para.LIGHT_SPEED
        pareto_value = np.random.pareto(para.SAT_ISL_PARETO_SHAPE) + 1.0
        load_ratio = min(pareto_value * 0.01, para.SAT_ISL_LOAD_CLIP)
        isl_rate = para.SAT_ISL_MAX_RATE * max(1e-3, 1.0 - load_ratio)
        t_isl_trans = md.B / isl_rate
        return t_isl_prop + t_isl_trans, access_sat_idx, t_isl_prop

    def _get_satellite_link_sample(self, md, sat_list, sat_idx):
        # 这个缓存函数对应论文里的“同一时隙、同一终端、同一目标卫星”的固定链路 realization。
        # 一旦在当前状态下为某个卫星动作抽到了 ISL 负载样本，
        # 后续无论是动作掩码判断还是实际执行该动作，都复用同一份样本，
        # 从而保证决策依据和执行环境是一致的。
        # ISL 传输时延包含任务 bit 数，分片子任务不能复用完整任务的缓存结果。
        cache_key = (md.id, sat_idx, int(md.B))
        if cache_key not in self._satellite_link_cache:
            t_isl, access_sat_idx, t_isl_prop = self._sample_sat_isl_delay(md, sat_list, sat_idx)
            self._satellite_link_cache[cache_key] = {
                "T_isl_s": t_isl,
                "access_sat_id": access_sat_idx,
                "T_isl_prop_s": t_isl_prop,
            }
        return dict(self._satellite_link_cache[cache_key])

    def _estimate_satellite_part(self, md, sat, sat_idx, sat_list, task_bits, actual_freq):
        # 只用于动作掩码的乐观估算，不扣减卫星资源，也不改写任务本体。
        original_b = md.B
        try:
            md.B = max(int(task_bits), 1)
            t_isl = self._get_satellite_link_sample(md, sat_list, sat_idx)["T_isl_s"]
            t_comp, _ = sat.computing(md.B, md.C, actual_freq)
            _, power_result = self._calc_sat_power(md, sat, t_comp, t_isl)
            # 分片 mask 阶段只保留“没有剩余传输预算”这类前置硬失败，P_min_sat 超功率交给 reward 软惩罚处理。
            if power_result["remaining_budget_sat"] <= 0:
                return False, float("inf")
            t_tran, _ = md.sat_offloading(
                md.B,
                power_result["selected_power"],
                power_result["channel_info"]["g_sat_raw"],
                doppler_loss=power_result["channel_info"]["doppler_loss"],
            )
            total_delay = power_result["T_prop_s"] + t_isl + t_tran + t_comp
            if power_result["power_limit_exceeded"]:
                # 分片动作的 mask 也不能再因为 P_min_sat 超过 MD_MAX_POWER 被挡掉；这里给外层一个可通过的乐观时延。
                return True, max(md.Gamma - para.SPLIT_MERGE_DELAY, 0.0)
            return total_delay <= md.Gamma, total_delay
        finally:
            # 掩码阶段不能污染真实任务大小，否则后续执行和日志都会偏掉。
            md.B = original_b

    def _is_split_action_feasible(self, md, sat_list, sat_a_idx, sat_b_idx):
        # 分片动作判断“是否存在一个比例，让两颗卫星合起来满足 deadline”。
        sat_a = sat_list[sat_a_idx]
        sat_b = sat_list[sat_b_idx]
        if not md.is_sat_visible(sat_a) or not md.is_sat_visible(sat_b):
            return False
        if sat_a.res_F < para.SAT_F_MIN or sat_b.res_F < para.SAT_F_MIN:
            return False

        freq_a = min(float(sat_a.res_F), float(para.SAT_F_MAX))
        freq_b = min(float(sat_b.res_F), float(para.SAT_F_MAX))
        for split_ratio in para.SPLIT_FEASIBILITY_RATIOS:
            part_a_bits = max(int(md.B * split_ratio), 1)
            part_b_bits = max(int(md.B) - part_a_bits, 1)
            ok_a, delay_a = self._estimate_satellite_part(md, sat_a, sat_a_idx, sat_list, part_a_bits, freq_a)
            ok_b, delay_b = self._estimate_satellite_part(md, sat_b, sat_b_idx, sat_list, part_b_bits, freq_b)
            if ok_a and ok_b and max(delay_a, delay_b) + para.SPLIT_MERGE_DELAY <= md.Gamma:
                return True
        return False

    def _is_satellite_action_feasible(self, md, sat, sat_idx, sat_list):
        # 这里把新版论文的卫星卸载可行条件直接做成动作掩码：
        # 1. 卫星可见；
        # 2. 卫星剩余算力至少能容纳最小分配频率；
        # 3. 在“当前卫星可提供的较优频率”下，剩余时延预算 bar_Gamma^SAT > 0；
        # 4. P_min^SAT > P_max^MD 只记录诊断并进入 reward 软惩罚，不再作为硬 mask。
        # 这里特意不用 SAT_F_MIN 去判可行性。
        # 原因是：论文中的 D3QN 只负责决定“是否选这颗卫星”，
        # 真正的连续频率分配是后续 SAC 再做。
        # 如果掩码阶段直接按最小频率计算，就会把很多“本可通过更高频率满足时延”的卫星动作过早过滤掉。
        self.last_debug["mask_sat_total_checks"] += 1
        if not md.is_sat_visible(sat):
            # 对应论文里的可见性约束不满足。
            self.last_debug["mask_sat_not_visible"] += 1
            return False
        if sat.res_F < para.SAT_F_MIN:
            # 对应论文里的卫星计算资源约束不满足。
            self.last_debug["mask_sat_no_resource"] += 1
            return False

        sat_link_sample = self._get_satellite_link_sample(md, sat_list, sat_idx)
        t_isl = sat_link_sample["T_isl_s"]
        # 这里用“当前卫星此刻最多能拿出的可分配频率”来判断是否存在可行连续动作，
        # 对应论文里“离散目标可选后，连续资源分配仍有解”的存在性判定。
        candidate_freq = min(sat.res_F, para.SAT_F_MAX)
        t_comp_candidate, _ = sat.computing(md.B, md.C, candidate_freq)
        _, power_result = self._calc_sat_power(md, sat, t_comp_candidate, t_isl)
        if power_result["remaining_budget_sat"] <= 0:
            # 对应论文里的剩余时延预算 bar_Gamma^SAT 非正。
            # 这里表示：即便给这颗卫星当前尽量高的频率，任务仍然没有剩余传输预算。
            self.last_debug["mask_sat_non_positive_budget"] += 1
            # 这里记录“预算失败的平均缺口”，方便判断是只差一点点，还是卫星路径整体严重超时。
            self.last_debug["mask_sat_remaining_budget_fail_sum"] += power_result["remaining_budget_sat"]
            self.last_debug["mask_sat_remaining_budget_fail_count"] += 1
            return False
        power_limit_exceeded = bool(power_result.get("power_limit_exceeded", False))
        if power_limit_exceeded:
            # 对应论文里的最小可行功率 P_min^SAT 超过终端最大功率上限。
            # 现在这里只记录诊断信息，不再把该卫星动作从 DQN 候选集中硬删除。
            self.last_debug["mask_sat_power_infeasible"] += 1
            # 这里只裁剪诊断日志里的功率缺口比例，避免 inf 或极端值把平均数拉爆。
            # 真正的训练信号会在 step() 中通过 sat_power_penalty 进入 reward。
            raw_power_excess_ratio = power_result["P_min_sat"] / max(para.MD_MAX_POWER, 1e-12)
            power_excess_ratio = float(np.clip(raw_power_excess_ratio, 0.0, 100.0))
            self.last_debug["mask_sat_power_excess_ratio_sum"] += power_excess_ratio
            self.last_debug["mask_sat_power_excess_ratio_count"] += 1

        # 问题 1 的关键修复：卫星动作进入 DQN 候选集之前，也要估算最终总时延是否满足 deadline。
        # 这里使用 candidate_freq 做乐观估算；如果乐观情况下仍然超过 Gamma，就说明该卫星动作不适合作为候选动作。
        t_tran_candidate, _ = md.sat_offloading(
            md.B,
            power_result["selected_power"],
            power_result["channel_info"]["g_sat_raw"],
            doppler_loss=power_result["channel_info"]["doppler_loss"],
        )
        total_delay_candidate = power_result["T_prop_s"] + t_isl + t_tran_candidate + t_comp_candidate
        if total_delay_candidate > md.Gamma:
            if power_limit_exceeded:
                # 如果超时是由 P_min_sat 超过 MD_MAX_POWER 导致的，就让动作通过 mask，并在 reward 里承担超功率和 deadline 惩罚。
                self.last_debug["mask_sat_feasible"] += 1
                return True
            self.last_debug["mask_sat_remaining_budget_fail_sum"] += md.Gamma - total_delay_candidate
            self.last_debug["mask_sat_remaining_budget_fail_count"] += 1
            return False

        # 通过可见性、资源和剩余预算检查后，这个卫星动作进入 D3QN 候选集合；超功率问题留给 reward 学习。
        self.last_debug["mask_sat_feasible"] += 1
        return True

    def _sample_ground_queue_delay(self, bs_idx):
        if not para.ENABLE_GROUND_CONGESTION:
            return 0.0
        queue_delay = random.uniform(para.GROUND_QUEUE_DELAY_MIN, para.GROUND_QUEUE_DELAY_MAX)
        if random.random() < para.GROUND_HOTSPOT_PROB:
            queue_delay += para.GROUND_HOTSPOT_DELAY_EXTRA * (1.0 + 0.1 * bs_idx)
        return queue_delay

    def _build_paper_reward(
        self,
        priority,
        total_delay,
        total_energy,
        penalty_time,
        penalty_resource,
        penalty_visibility,
        penalty_propagation,
        penalty_zero_alloc,
        sat_load_penalty,
        sat_power_penalty,
        deadline_violated,
        deadline_overrun_ratio,
    ):
        # 这里把奖励函数恢复成论文大纲里的形式：
        # 1. 代价函数主干为 phi = w_D * T + w_E * E - w_V * V；
        # 2. 强化学习奖励写成 reward = -phi + 各类约束惩罚；
        # 3. 其中 V 对应任务价值，这里沿用当前任务优先级 Priority 作为任务价值表征。
        # 奖励主项先做量纲归一化，再加权求和。
        # 这样 w_t / w_e / w_v 反映真实偏好，而不是让秒、焦耳和优先级数字直接竞争。
        delay_norm = total_delay / max(float(para.REWARD_DELAY_NORM), 1e-9)
        energy_norm = total_energy / max(float(para.REWARD_ENERGY_NORM), 1e-9)
        value_norm = priority / max(float(para.REWARD_VALUE_NORM), 1e-9)
        delay_cost = w_t * delay_norm
        energy_cost = w_e * energy_norm
        task_value = w_v * value_norm
        phi_cost = delay_cost + energy_cost - task_value
        # 成功 bonus 让“满足 deadline”在 reward 上有明确正反馈；超时按超出比例连续惩罚。
        deadline_bonus = 0.0 if deadline_violated else para.REWARD_SUCCESS_BONUS
        deadline_overrun_penalty = -para.REWARD_OVERRUN_WEIGHT * deadline_overrun_ratio
        # 这里恢复成论文大纲里的主奖励结构：
        # reward = -phi + r_time + r_fre + r_vis + r_prop
        # 同时保留你后续加入的零分配惩罚、卫星负载软惩罚和卫星超功率软惩罚，
        # 因为它们属于训练实现层面的辅助约束，不改变主干目标表达式。
        reward_val = (
            -phi_cost
            + penalty_time
            + penalty_resource
            + penalty_visibility
            + penalty_propagation
            + penalty_zero_alloc
            + sat_load_penalty
            + sat_power_penalty
            + deadline_bonus
            + deadline_overrun_penalty
        )
        return reward_val, phi_cost, delay_cost, energy_cost, task_value, deadline_bonus, deadline_overrun_penalty

    def _empty_transition_metrics(self):
        return {
            "T_loc": 0.0,
            "E_loc": 0.0,
            "T_tran_g": 0.0,
            "E_tran_g": 0.0,
            "T_queue_g": 0.0,
            "T_switch_g": 0.0,
            "T_comp_g": 0.0,
            "E_comp_g": 0.0,
            "T_prop_s": 0.0,
            "T_isl_s": 0.0,
            "T_isl_prop_s": 0.0,
            "T_tran_s": 0.0,
            "E_tran_s": 0.0,
            "T_comp_s": 0.0,
            "E_comp_s": 0.0,
            "T_tran_pmax": 0.0,
            "E_tran_pmax": 0.0,
            "selected_power": 0.0,
            "unconstrained_power": 0.0,
            "min_feasible_power": 0.0,
            "g_bar_s": 0.0,
            "remaining_budget_sat": 0.0,
            "P_min_sat": 0.0,
            "p_star_sat": 0.0,
            # 这两个字段记录卫星最小发射功率超出上限的比例和对应 reward 软惩罚。
            "sat_power_excess_ratio": 0.0,
            "sat_power_penalty": 0.0,
            "bar_w_delay_s": 0.0,
            "bar_w_energy_s": 0.0,
            "g_sat_raw": 0.0,
            "sat_snr": 0.0,
            "sat_rate": 0.0,
            "relative_velocity_sat": 0.0,
            "doppler_shift_sat": 0.0,
            "eta_d_sat": 0.0,
            "wavelength_sat": 0.0,
            "free_space_gain": 0.0,
            "free_space_denominator": 0.0,
            "rician_fading": 0.0,
            "access_sat_id": 0,
            "T_split_s": 0.0,
            "E_split_s": 0.0,
            "split_ratio": 0.0,
            "split_sat_a": -1,
            "split_sat_b": -1,
            "split_T_part_a": 0.0,
            "split_T_part_b": 0.0,
            "split_E_part_a": 0.0,
            "split_E_part_b": 0.0,
        }

    def _compute_local_metrics(self, md, actual_freq):
        metrics = self._empty_transition_metrics()
        if actual_freq > 0:
            metrics["T_loc"], metrics["E_loc"] = md.local_com(md.B, md.C, actual_freq)
        return metrics

    def _compute_ground_metrics(self, md, bs, action_idx, actual_freq):
        metrics = self._empty_transition_metrics()
        penalty_resource = 0.0
        penalty_propagation = 0.0

        if bs.res_F < actual_freq:
            penalty_resource = para.PENALTY_RESOURCE
            return metrics, penalty_resource, penalty_propagation, False

        if action_idx != md.connect_BS + 1:
            pareto_value = np.random.pareto(1.5) + 1
            metrics["T_switch_g"] = md.B / (
                para.GROUND_HANDOVER_DELAY_SCALE * max(1e-3, 1 - min(pareto_value * 0.01, 0.9))
            )

        metrics["T_comp_g"], metrics["E_comp_g"] = bs.computing(md.B, md.C, actual_freq)
        metrics["T_queue_g"] = self._sample_ground_queue_delay(action_idx - 1)
        gain_val, power_result = self._calc_ground_power(md, metrics["T_comp_g"] + metrics["T_switch_g"] + metrics["T_queue_g"])
        metrics["selected_power"] = power_result["selected_power"]
        metrics["unconstrained_power"] = power_result["unconstrained_power"]
        metrics["min_feasible_power"] = power_result["min_feasible_power"]

        if power_result["invalid_budget"]:
            penalty_propagation = para.PENALTY_PROPAGATION

        metrics["T_tran_g"], metrics["E_tran_g"] = md.offloading(md.B, metrics["selected_power"], gain_val)
        metrics["T_tran_pmax"], metrics["E_tran_pmax"] = md.offloading(md.B, para.MD_MAX_POWER, gain_val)
        bs.res_F -= actual_freq
        return metrics, penalty_resource, penalty_propagation, True

    def _compute_satellite_metrics(self, md, sat, sat_idx, sat_list, actual_freq):
        metrics = self._empty_transition_metrics()
        penalty_resource = 0.0
        penalty_visibility = 0.0
        penalty_propagation = 0.0

        if not md.is_sat_visible(sat):
            penalty_visibility = para.PENALTY_VISIBILITY
            return metrics, penalty_resource, penalty_visibility, penalty_propagation, False

        if sat.res_F < actual_freq:
            penalty_resource = para.PENALTY_RESOURCE
            return metrics, penalty_resource, penalty_visibility, penalty_propagation, False

        # 这里复用动作掩码阶段已经抽到的同一份链路样本，
        # 保证“卫星动作能不能选”和“选了之后实际经历什么链路条件”是同一时隙 realization。
        sat_link_sample = self._get_satellite_link_sample(md, sat_list, sat_idx)
        metrics["T_isl_s"] = sat_link_sample["T_isl_s"]
        metrics["access_sat_id"] = sat_link_sample["access_sat_id"]
        metrics["T_isl_prop_s"] = sat_link_sample["T_isl_prop_s"]
        metrics["T_comp_s"], metrics["E_comp_s"] = sat.computing(md.B, md.C, actual_freq)
        gain_val, power_result = self._calc_sat_power(md, sat, metrics["T_comp_s"], metrics["T_isl_s"])

        metrics["selected_power"] = power_result["selected_power"]
        metrics["unconstrained_power"] = power_result["unconstrained_power"]
        metrics["min_feasible_power"] = power_result["min_feasible_power"]
        metrics["T_prop_s"] = power_result["T_prop_s"]
        metrics["g_bar_s"] = power_result["g_bar_s"]
        metrics["remaining_budget_sat"] = power_result["remaining_budget_sat"]
        metrics["P_min_sat"] = power_result["P_min_sat"]
        metrics["p_star_sat"] = power_result["p_star_sat"]
        metrics["bar_w_delay_s"] = power_result["bar_w_delay_s"]
        metrics["bar_w_energy_s"] = power_result["bar_w_energy_s"]

        if power_result["invalid_budget"]:
            # 剩余传输预算非正仍属于传播/时延结构上的硬失败，继续沿用原来的传播惩罚。
            penalty_propagation = para.PENALTY_PROPAGATION
        if power_result["power_limit_exceeded"]:
            # P_min_sat 超过 MD_MAX_POWER 时不再把动作判死，而是把超出比例换成 reward 软惩罚。
            metrics["sat_power_penalty"], metrics["sat_power_excess_ratio"] = self._calc_sat_power_excess_penalty(
                power_result
            )

        sat_link_info = dict(power_result["channel_info"])
        sat_link_info["snr_sat"] = (
            sat_link_info["doppler_loss"] * sat_link_info["g_sat_raw"] * metrics["selected_power"] / para.SAT_NOISE
        )
        sat_link_info["rate_sat"] = para.SAT_BW * math.log2(1 + max(sat_link_info["snr_sat"], 0.0))
        metrics["g_sat_raw"] = sat_link_info["g_sat_raw"]
        metrics["sat_snr"] = max(sat_link_info["snr_sat"], 0.0)
        metrics["sat_rate"] = max(sat_link_info["rate_sat"], 1e-3)
        metrics["relative_velocity_sat"] = sat_link_info["relative_radial_velocity"]
        metrics["doppler_shift_sat"] = sat_link_info["doppler_shift"]
        metrics["eta_d_sat"] = sat_link_info["doppler_loss"]
        metrics["wavelength_sat"] = sat_link_info["wavelength"]
        metrics["free_space_gain"] = sat_link_info["free_space_gain"]
        metrics["free_space_denominator"] = sat_link_info["free_space_denominator"]
        metrics["rician_fading"] = sat_link_info["rician_fading"]
        metrics["T_tran_s"], metrics["E_tran_s"] = md.sat_offloading(
            md.B,
            metrics["selected_power"],
            gain_val,
            doppler_loss=metrics["eta_d_sat"],
        )
        metrics["T_tran_pmax"], metrics["E_tran_pmax"] = md.sat_offloading(
            md.B,
            para.MD_MAX_POWER,
            gain_val,
            doppler_loss=metrics["eta_d_sat"],
        )

        sat.res_F -= actual_freq
        return metrics, penalty_resource, penalty_visibility, penalty_propagation, True

    def _compute_split_satellite_metrics(self, md, sat_list, sat_a_idx, sat_b_idx, split_ratio, freq_a, freq_b):
        metrics = self._empty_transition_metrics()
        penalty_resource = 0.0
        penalty_visibility = 0.0
        penalty_propagation = 0.0

        original_b = md.B
        original_c = md.C
        part_a_bits = max(int(original_b * split_ratio), 1)
        part_b_bits = max(original_b - part_a_bits, 1)

        md.B = part_a_bits
        metrics_a, res_pen_a, vis_pen_a, prop_pen_a, ok_a = self._compute_satellite_metrics(
            md, sat_list[sat_a_idx], sat_a_idx, sat_list, freq_a
        )

        md.B = part_b_bits
        metrics_b, res_pen_b, vis_pen_b, prop_pen_b, ok_b = self._compute_satellite_metrics(
            md, sat_list[sat_b_idx], sat_b_idx, sat_list, freq_b
        )
        md.B = original_b
        md.C = original_c

        penalty_resource += res_pen_a + res_pen_b
        penalty_visibility += vis_pen_a + vis_pen_b
        penalty_propagation += prop_pen_a + prop_pen_b

        delay_a = metrics_a["T_prop_s"] + metrics_a["T_isl_s"] + metrics_a["T_tran_s"] + metrics_a["T_comp_s"]
        delay_b = metrics_b["T_prop_s"] + metrics_b["T_isl_s"] + metrics_b["T_tran_s"] + metrics_b["T_comp_s"]
        energy_a = metrics_a["E_tran_s"] + metrics_a["E_comp_s"]
        energy_b = metrics_b["E_tran_s"] + metrics_b["E_comp_s"]

        metrics.update(metrics_a)
        metrics["T_split_s"] = max(delay_a, delay_b) + para.SPLIT_MERGE_DELAY
        metrics["E_split_s"] = energy_a + energy_b + para.SPLIT_EXTRA_ENERGY
        metrics["split_ratio"] = float(split_ratio)
        metrics["split_sat_a"] = int(sat_a_idx)
        metrics["split_sat_b"] = int(sat_b_idx)
        metrics["split_T_part_a"] = delay_a
        metrics["split_T_part_b"] = delay_b
        metrics["split_E_part_a"] = energy_a
        metrics["split_E_part_b"] = energy_b
        metrics["T_tran_s"] = metrics_a["T_tran_s"] + metrics_b["T_tran_s"]
        metrics["E_tran_s"] = metrics_a["E_tran_s"] + metrics_b["E_tran_s"]
        metrics["T_comp_s"] = max(metrics_a["T_comp_s"], metrics_b["T_comp_s"])
        metrics["E_comp_s"] = metrics_a["E_comp_s"] + metrics_b["E_comp_s"] + para.SPLIT_EXTRA_ENERGY
        metrics["T_prop_s"] = max(metrics_a["T_prop_s"], metrics_b["T_prop_s"])
        metrics["T_isl_s"] = max(metrics_a["T_isl_s"], metrics_b["T_isl_s"])
        # 分片动作会同时使用两颗卫星，因此超功率 reward 惩罚需要把两个子任务的惩罚相加。
        metrics["sat_power_penalty"] = metrics_a["sat_power_penalty"] + metrics_b["sat_power_penalty"]
        # 分片动作的超功率比例用两颗卫星中的最大值，便于日志反映最严重的那条链路。
        metrics["sat_power_excess_ratio"] = max(metrics_a["sat_power_excess_ratio"], metrics_b["sat_power_excess_ratio"])
        return metrics, penalty_resource, penalty_visibility, penalty_propagation, bool(ok_a and ok_b)

    def _aggregate_total_delay_energy(self, action_idx, metrics):
        if action_idx == 0:
            total_delay = metrics["T_loc"]
            total_energy = metrics["E_loc"]
        elif 1 <= action_idx <= N:
            total_delay = metrics["T_tran_g"] + metrics["T_switch_g"] + metrics["T_queue_g"] + metrics["T_comp_g"]
            total_energy = metrics["E_tran_g"] + metrics["E_comp_g"]
        elif self._is_split_action(action_idx):
            total_delay = metrics["T_split_s"]
            total_energy = metrics["E_split_s"]
        else:
            # 新版卫星总时延：
            # T_SEC = T_prop + T_tran,S + T_ISL + T_comp,S
            total_delay = metrics["T_prop_s"] + metrics["T_isl_s"] + metrics["T_tran_s"] + metrics["T_comp_s"]
            # 新版卫星总能耗：
            # E_SEC = E_tran,S + E_comp,S
            total_energy = metrics["E_tran_s"] + metrics["E_comp_s"]
        return total_delay, total_energy

    def step(self, m, b, f, tcn, bs_list, md_list, sat_list):
        md = md_list[m]
        priority = md.Priority
        self._record_total(priority, tcn)

        metrics = self._empty_transition_metrics()
        penalty_time = 0.0
        penalty_resource = 0.0
        penalty_visibility = 0.0
        penalty_propagation = 0.0
        penalty_zero_alloc = 0.0
        # 这个变量专门承接 P_min_sat > MD_MAX_POWER 的软惩罚，避免继续把它当作动作 mask。
        penalty_sat_power = 0.0

        selected_visible_sat_count = sum(1 for sat in sat_list if md.is_sat_visible(sat))
        self.last_debug["visible_sat_total_count"] += selected_visible_sat_count
        if selected_visible_sat_count > 0:
            self.last_debug["visible_sat_decisions"] += 1
        if selected_visible_sat_count > 0 and b <= N:
            self.last_debug["sat_visible_not_selected"] += 1

        actual_freq = self.decode_frequency(b, f)

        if b == 0:
            self.last_debug["local_actions"] += 1
            if actual_freq <= 0:
                penalty_zero_alloc += para.PENALTY_ZERO_ALLOCATION
            metrics = self._compute_local_metrics(md, actual_freq)
        elif 1 <= b <= N:
            self.last_debug["bs_actions"] += 1
            bs = bs_list[b - 1]
            metrics, resource_penalty, propagation_penalty, _ = self._compute_ground_metrics(md, bs, b, actual_freq)
            penalty_resource += resource_penalty
            penalty_propagation += propagation_penalty
            if resource_penalty != 0:
                self.last_debug["resource_penalty"] += 1
            if propagation_penalty != 0:
                self.last_debug["propagation_penalty"] += 1
        elif self._is_split_action(b):
            self.last_debug["split_actions"] += 1
            self.last_debug["sat_actions"] += 1
            sat_a_idx, sat_b_idx = self._decode_split_pair(b)
            split_ratio, freq_a, freq_b = actual_freq
            metrics, resource_penalty, visibility_penalty, propagation_penalty, _ = self._compute_split_satellite_metrics(
                md, sat_list, sat_a_idx, sat_b_idx, split_ratio, freq_a, freq_b
            )
            penalty_resource += resource_penalty
            penalty_visibility += visibility_penalty
            penalty_propagation += propagation_penalty + para.SPLIT_OVERHEAD_PENALTY
            self.last_debug["split_ratio_sum"] += split_ratio
            self.last_debug["split_metric_steps"] += 1
            if visibility_penalty != 0:
                self.last_debug["sat_invalid_visibility"] += 1
            if resource_penalty != 0:
                self.last_debug["resource_penalty"] += 1
            if propagation_penalty != 0:
                self.last_debug["propagation_penalty"] += 1
        else:
            self.last_debug["sat_actions"] += 1
            sat_idx = b - (N + 1)
            sat = sat_list[sat_idx]
            metrics, resource_penalty, visibility_penalty, propagation_penalty, _ = self._compute_satellite_metrics(
                md, sat, sat_idx, sat_list, actual_freq
            )
            penalty_resource += resource_penalty
            penalty_visibility += visibility_penalty
            penalty_propagation += propagation_penalty
            if visibility_penalty != 0:
                self.last_debug["sat_invalid_visibility"] += 1
            if resource_penalty != 0:
                self.last_debug["resource_penalty"] += 1
            if propagation_penalty != 0:
                self.last_debug["propagation_penalty"] += 1

        # 卫星普通动作和分片动作都会把超功率惩罚写入 metrics；非卫星动作该值保持 0。
        penalty_sat_power = metrics["sat_power_penalty"]
        if penalty_sat_power != 0.0:
            # 记录执行阶段真实产生的超功率惩罚，和 mask 阶段诊断计数分开看。
            self.last_debug["sat_power_penalty_count"] += 1
            self.last_debug["sat_power_penalty_sum"] += penalty_sat_power
            self.last_debug["sat_power_excess_ratio_sum"] += metrics["sat_power_excess_ratio"]

        total_delay, total_energy = self._aggregate_total_delay_energy(b, metrics)
        # ================== [新增] 物理指标安全截断与丢弃保护 ==================
        # 实际通信系统中，任务超时后会被直接丢弃（Aborted），不会无休止占用计算/传输资源。
        # 这里设定一个合理的物理安全上限（最大截止时间 Gamma_MAX 的 1.5 倍，即 2.0s * 1.5 = 3.0s）
        DELAY_CEILING = para.TASK_GAMMA_MAX * 1.5

        # 设定单步能耗的安全上限
        ENERGY_CEILING = 10

        if total_delay > DELAY_CEILING or total_delay <= 0:
            total_delay = DELAY_CEILING
            total_energy = min(total_energy, ENERGY_CEILING)
        else:
            total_energy = min(total_energy, ENERGY_CEILING)
        # ===================================================================
        sat_load_penalty = 0.0
        sat_usage_ratio_after = 0.0
        sat_peak_usage_ratio = 0.0

        if b > N and not self._is_split_action(b):
            sat_idx = b - (N + 1)
            # 这里仅在卫星动作上评估“是否过度占用当前时隙内的卫星资源”。
            # 论文含义上，它反映的是多终端在同一时隙竞争同一卫星算力时的负载均衡程度。
            sat_load_penalty, _, sat_usage_ratio_after = self._calc_sat_load_penalty(sat_idx, actual_freq)
            if penalty_visibility == 0 and penalty_resource == 0 and actual_freq > 0:
                # 只有当卫星动作真实可执行时，才把这次频率占用计入时隙内累计负载。
                self._slot_sat_usage[sat_idx] += actual_freq
                sat_capacity = max(float(SAT_F[sat_idx]), 1e-9)
                sat_usage_ratio_after = float(np.clip(self._slot_sat_usage[sat_idx] / sat_capacity, 0.0, 1.5))
                sat_peak_usage_ratio = max(
                    float(np.clip(self._slot_sat_usage[idx] / max(float(SAT_F[idx]), 1e-9), 0.0, 1.5))
                    for idx in range(S)
                )
                self.last_debug["sat_usage_ratio_sum"] += sat_usage_ratio_after
                self.last_debug["sat_peak_usage_ratio_sum"] += sat_peak_usage_ratio
                self.last_debug["sat_peak_usage_ratio_max"] = max(
                    self.last_debug["sat_peak_usage_ratio_max"],
                    sat_peak_usage_ratio,
                )
                if sat_load_penalty != 0.0:
                    self.last_debug["sat_load_penalty_count"] += 1
                    self.last_debug["sat_load_penalty_sum"] += sat_load_penalty
        elif self._is_split_action(b):
            sat_a_idx, sat_b_idx = self._decode_split_pair(b)
            split_ratio, freq_a, freq_b = actual_freq
            # 两星分片会同时占用两颗卫星的时隙算力，因此负载软惩罚也要分别计算。
            sat_load_penalty_a, _, usage_a_after = self._calc_sat_load_penalty(sat_a_idx, freq_a)
            sat_load_penalty_b, _, usage_b_after = self._calc_sat_load_penalty(sat_b_idx, freq_b)
            sat_load_penalty = sat_load_penalty_a + sat_load_penalty_b
            if penalty_visibility == 0 and penalty_resource == 0 and freq_a > 0 and freq_b > 0:
                self._slot_sat_usage[sat_a_idx] += freq_a
                self._slot_sat_usage[sat_b_idx] += freq_b
                sat_usage_ratio_after = float((usage_a_after + usage_b_after) / 2.0)
                sat_peak_usage_ratio = max(
                    float(np.clip(self._slot_sat_usage[idx] / max(float(SAT_F[idx]), 1e-9), 0.0, 1.5))
                    for idx in range(S)
                )
                self.last_debug["sat_usage_ratio_sum"] += sat_usage_ratio_after
                self.last_debug["sat_peak_usage_ratio_sum"] += sat_peak_usage_ratio
                self.last_debug["sat_peak_usage_ratio_max"] = max(
                    self.last_debug["sat_peak_usage_ratio_max"],
                    sat_peak_usage_ratio,
                )
                if sat_load_penalty != 0.0:
                    self.last_debug["sat_load_penalty_count"] += 1
                    self.last_debug["sat_load_penalty_sum"] += sat_load_penalty

        # 新版论文里时间违约惩罚只显式作用在本地/地面路径；
        # 卫星的两类不可行都通过可见性/传播可行性惩罚体现。
        # 这里统一把“总时延是否超过任务时延约束 Gamma”作为三类路径共同的 deadline 约束。
        # 对应论文模型的含义是：无论任务在本地、地面还是卫星执行，只要最终完成时延超过 Gamma，
        # 都应该被视为一次任务级失败，而不是只把卫星看成“链路可见即可”。
        deadline_violated = (total_delay > md.Gamma) or (total_delay == 0)
        if deadline_violated:
            penalty_time += para.PENALTY_TIME
            self.last_debug["time_penalty"] += 1
            if b > N:
                # 这里单独记录“卫星动作虽然被执行，但没有满足任务 deadline”的情况，
                # 便于论文里区分“卫星可执行成功率”和“卫星时延成功率”。
                self.last_debug["sat_timeout_actions"] += 1
                self.last_debug["sat_delay_over_gamma_sum"] += max(total_delay - md.Gamma, 0.0)
                if self._is_split_action(b):
                    self.last_debug["split_timeout_actions"] += 1
                    self.last_debug["split_delay_over_gamma_sum"] += max(total_delay - md.Gamma, 0.0)
        else:
            self._record_success(priority, tcn)
            if b > N and penalty_visibility == 0 and penalty_resource == 0:
                self.last_debug["sat_deadline_success_actions"] += 1
                if self._is_split_action(b):
                    self.last_debug["split_deadline_success_actions"] += 1

        if b > N and penalty_visibility == 0 and penalty_resource == 0:
            # 这里表示卫星动作在可见性、传播可行性、算力资源等层面是“可执行”的。
            # 它不保证一定满足 Gamma，只表示该离散动作不是一个物理不可行的坏动作。
            self.last_debug["sat_exec_success_actions"] += 1

        deadline_overrun_ratio = max((total_delay - md.Gamma) / max(float(md.Gamma), 1e-9), 0.0)
        delay_norm = total_delay / max(float(para.REWARD_DELAY_NORM), 1e-9)
        energy_norm = total_energy / max(float(para.REWARD_ENERGY_NORM), 1e-9)
        value_norm = priority / max(float(para.REWARD_VALUE_NORM), 1e-9)

        reward, phi_cost, delay_cost, energy_cost, task_value, deadline_bonus, deadline_overrun_penalty = self._build_paper_reward(
            priority=priority,
            total_delay=total_delay,
            total_energy=total_energy,
            penalty_time=penalty_time,
            penalty_resource=penalty_resource,
            penalty_visibility=penalty_visibility,
            penalty_propagation=penalty_propagation,
            penalty_zero_alloc=penalty_zero_alloc,
            sat_load_penalty=sat_load_penalty,
            sat_power_penalty=penalty_sat_power,
            deadline_violated=deadline_violated,
            deadline_overrun_ratio=deadline_overrun_ratio,
        )

        md.B, md.C, md.Gamma, md.Priority = self.task()
        md.move()
        md.connect_BS = md.connect_choice()
        # 卫星不在单个 MD 决策后移动，避免同一时隙内不同 MD 看到不同卫星位置。
        # 统一的卫星移动放在 reset_state() 中，由外层训练循环在一个时隙结束后调用。
        self._refresh_full_state(md_list, bs_list, sat_list)

        if b == 0:
            phi_p = 0.0
            phi_pmax = 0.0
            tran_delay = 0.0
            tran_energy = 0.0
        elif 1 <= b <= N:
            tran_delay = metrics["T_tran_g"]
            tran_energy = metrics["E_tran_g"]
            phi_p = w_t * tran_delay + w_e * tran_energy
            phi_pmax = w_t * metrics["T_tran_pmax"] + w_e * metrics["E_tran_pmax"]
        elif self._is_split_action(b):
            tran_delay = metrics["T_tran_s"]
            tran_energy = metrics["E_tran_s"]
            phi_p = w_t * tran_delay + w_e * tran_energy
            phi_pmax = phi_p
        else:
            tran_delay = metrics["T_tran_s"]
            tran_energy = metrics["E_tran_s"]
            phi_p = w_t * tran_delay + w_e * tran_energy
            phi_pmax = w_t * metrics["T_tran_pmax"] + w_e * metrics["E_tran_pmax"]

        self.last_debug["avg_prop_delay_sum"] += metrics["T_prop_s"]
        self.last_debug["avg_total_delay_sum"] += total_delay
        self.last_debug["avg_selected_power_sum"] += metrics["selected_power"]
        self.last_debug["avg_min_power_sum"] += metrics["min_feasible_power"]
        self.last_debug["avg_unconstrained_power_sum"] += metrics["unconstrained_power"]
        self.last_debug["avg_g_bar_s_sum"] += metrics["g_bar_s"]
        self.last_debug["avg_remaining_budget_sat_sum"] += metrics["remaining_budget_sat"]
        self.last_debug["avg_P_min_sat_sum"] += metrics["P_min_sat"]
        self.last_debug["avg_p_star_sat_sum"] += metrics["p_star_sat"]
        self.last_debug["avg_g_sat_raw_sum"] += metrics["g_sat_raw"]
        self.last_debug["avg_sat_snr_sum"] += metrics["sat_snr"]
        self.last_debug["avg_sat_rate_sum"] += metrics["sat_rate"]
        self.last_debug["avg_relative_velocity_sat_sum"] += metrics["relative_velocity_sat"]
        self.last_debug["avg_doppler_shift_sat_sum"] += metrics["doppler_shift_sat"]
        self.last_debug["avg_eta_d_sat_sum"] += metrics["eta_d_sat"]
        self.last_debug["avg_wavelength_sat_sum"] += metrics["wavelength_sat"]
        if b > N and not self._is_split_action(b):
            self.last_debug["sat_metric_steps"] += 1
            # 这里单独记录“卫星动作且真实可行”的样本统计。
            # 论文里像 g_bar_s、P_min_sat、p_star_sat 这类量更适合在“真正可执行的卫星卸载样本”上解释，
            # 否则少量不可行样本的极端数值会把均值严重污染，影响我们判断策略是否稳定。
            # 解释 P_min_sat 等物理量的“可行样本均值”时，仍排除已经触发超功率软惩罚的样本。
            if penalty_visibility == 0 and penalty_propagation == 0 and penalty_resource == 0 and penalty_sat_power == 0:
                self.last_debug["avg_g_bar_s_feasible_sum"] += metrics["g_bar_s"]
                self.last_debug["avg_remaining_budget_sat_feasible_sum"] += metrics["remaining_budget_sat"]
                self.last_debug["avg_P_min_sat_feasible_sum"] += metrics["P_min_sat"]
                self.last_debug["avg_p_star_sat_feasible_sum"] += metrics["p_star_sat"]
                self.last_debug["sat_feasible_metric_steps"] += 1
        self.last_debug["avg_cost_sum"] += phi_cost
        self.last_debug["avg_value_sum"] += task_value
        self.last_debug["avg_delay_norm_sum"] += delay_norm
        self.last_debug["avg_energy_norm_sum"] += energy_norm
        self.last_debug["avg_value_norm_sum"] += value_norm
        self.last_debug["avg_delay_cost_sum"] += delay_cost
        self.last_debug["avg_energy_cost_sum"] += energy_cost
        self.last_debug["avg_total_energy_sum"] +=total_energy
        self.last_debug["penalty_time_sum"] += penalty_time
        self.last_debug["penalty_resource_sum"] += penalty_resource
        self.last_debug["penalty_visibility_sum"] += penalty_visibility
        self.last_debug["penalty_propagation_sum"] += penalty_propagation
        self.last_debug["penalty_zero_alloc_sum"] += penalty_zero_alloc
        # 汇总 P_min_sat 超过 MD_MAX_POWER 带来的 reward 软惩罚，方便训练后算平均值。
        self.last_debug["penalty_sat_power_sum"] += penalty_sat_power
        self.last_debug["deadline_bonus_sum"] += deadline_bonus
        self.last_debug["deadline_overrun_penalty_sum"] += deadline_overrun_penalty
        # 在 reward 计算完成的最后
        reward = max(reward, -8.0)  # 单步 reward 下限截断到 -5
        self.last_debug["reward_sum"] += reward
        self.last_debug["steps"] += 1

        return (
            np.array(self.state),
            reward,
            False,
            tcn,
            tran_delay,
            tran_energy,
            metrics["T_tran_pmax"],
            metrics["E_tran_pmax"],
            phi_p,
            phi_pmax,
            metrics["selected_power"],
        )

    def step_real(self, m, b, f, tcn, bs_list, md_list, sat_list, step):
        md_list[m].move_real(step)
        md_list[m].connect_BS = md_list[m].connect_choice()
        self._refresh_full_state(md_list, bs_list, sat_list)
        result = self.step(m, b, f, tcn, bs_list, md_list, sat_list)
        return result

    def get_state(self):
        return np.array(self.state)
