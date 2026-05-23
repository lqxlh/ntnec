import math

from scipy.special import lambertw


def _safe_real(value):
    # `lambertw` 返回的是复数对象；论文这里使用主值分支的实部即可。
    if hasattr(value, "real"):
        return float(value.real)
    return float(value)


def _safe_min_power_from_budget(task_bits, bandwidth, remaining_budget, effective_gain):
    # 这里对应论文里的最小可行功率下界：
    #   P_min = (2^(B / (W * bar_Gamma)) - 1) / g_bar
    # 新版论文把 “P_min 是否超过终端最大发射功率” 单独作为不可行条件，
    # 因此这里不再把它裁剪到 P_max，而是保留原始结果给环境层判断。
    exponent_value = task_bits / max(bandwidth * remaining_budget, 1e-12)
    if exponent_value >= 60.0:
        return float("inf")

    numerator = (2.0 ** exponent_value) - 1.0
    min_power = numerator / max(effective_gain, 1e-12)
    return float(max(min_power, 0.0))


def solve_lambert_power(task_bits, weight_delay, weight_energy, bandwidth, effective_gain, max_power, remaining_budget):
    # 这里实现论文里基于 Lambert W 的闭式功率控制。
    # 目标函数对应：
    #   phi_bar(p) = (bar_w_delay + bar_w_energy * p) / ln(1 + g_bar * p)
    effective_gain = max(float(effective_gain), 1e-12)
    max_power = max(float(max_power), 1e-6)
    remaining_budget = float(remaining_budget)
    bar_w_delay = math.log(2.0) * task_bits * weight_delay / bandwidth
    bar_w_energy = math.log(2.0) * task_bits * weight_energy / bandwidth
    lambert_arg = None
    lambert_val = None

    if remaining_budget <= 0:
        # 论文里的第一类不可行条件：bar_Gamma <= 0。
        return {
            "selected_power": max_power,
            "unconstrained_power": max_power,
            "min_feasible_power": max_power,
            "p_star": max_power,
            "p_min": max_power,
            "remaining_budget": remaining_budget,
            "g_bar": effective_gain,
            "bar_w_delay": bar_w_delay,
            "bar_w_energy": bar_w_energy,
            "lambert_arg": lambert_arg,
            "lambert_val": lambert_val,
            "invalid_budget": True,
            "power_limit_exceeded": True,
            "infeasible_reason": "remaining_budget_non_positive",
        }

    remaining_budget = max(remaining_budget, 1e-6)

    if bar_w_energy <= 0:
        unconstrained_power = max_power
    else:
        lambert_arg = effective_gain * bar_w_delay / (math.e * bar_w_energy) - 1.0 / math.e
        lambert_val = _safe_real(lambertw(lambert_arg, k=0))
        unconstrained_power = (math.exp(lambert_val + 1.0) - 1.0) / effective_gain

    unconstrained_power = float(max(unconstrained_power, 0.0))
    min_feasible_power = _safe_min_power_from_budget(
        task_bits=task_bits,
        bandwidth=bandwidth,
        remaining_budget=remaining_budget,
        effective_gain=effective_gain,
    )

    if unconstrained_power <= min_feasible_power:
        selected_power = min_feasible_power
    elif unconstrained_power >= max_power:
        selected_power = max_power
    else:
        selected_power = unconstrained_power

    selected_power = float(min(max(selected_power, 0.0), max_power))
    power_limit_exceeded = min_feasible_power > max_power
    return {
        "selected_power": selected_power,
        "unconstrained_power": unconstrained_power,
        "min_feasible_power": min_feasible_power,
        "p_star": unconstrained_power,
        "p_min": min_feasible_power,
        "remaining_budget": remaining_budget,
        "g_bar": effective_gain,
        "bar_w_delay": bar_w_delay,
        "bar_w_energy": bar_w_energy,
        "lambert_arg": lambert_arg,
        "lambert_val": lambert_val,
        "invalid_budget": False,
        "power_limit_exceeded": power_limit_exceeded,
        "infeasible_reason": "",
    }


def solve_ground_power(task_bits, weight_delay, weight_energy, bandwidth, gain_value, noise_power, max_power, remaining_budget):
    # 地面链路等效信道增益：g_bar = g / sigma^2
    effective_gain = gain_value / max(noise_power, 1e-18)
    return solve_lambert_power(
        task_bits=task_bits,
        weight_delay=weight_delay,
        weight_energy=weight_energy,
        bandwidth=bandwidth,
        effective_gain=effective_gain,
        max_power=max_power,
        remaining_budget=remaining_budget,
    )


def solve_satellite_power(task_bits, weight_delay, weight_energy, bandwidth, gain_value, doppler_loss, noise_power, max_power, remaining_budget):
    # 卫星链路等效信道增益对应论文中的：
    #   g_bar_s = eta_d * g_{s,m}^t / sigma_s^2
    effective_gain = doppler_loss * gain_value / max(noise_power, 1e-18)
    result = solve_lambert_power(
        task_bits=task_bits,
        weight_delay=weight_delay,
        weight_energy=weight_energy,
        bandwidth=bandwidth,
        effective_gain=effective_gain,
        max_power=max_power,
        remaining_budget=remaining_budget,
    )
    result["g_bar_s"] = result["g_bar"]
    result["remaining_budget_sat"] = result["remaining_budget"]
    result["P_min_sat"] = result["p_min"]
    result["p_star_sat"] = result["p_star"]
    result["bar_w_delay_s"] = result["bar_w_delay"]
    result["bar_w_energy_s"] = result["bar_w_energy"]
    # 新版论文里卫星卸载可行当且仅当：
    # 1. bar_Gamma^SAT > 0
    # 2. P_min^SAT <= P_max^MD
    result["satellite_infeasible"] = bool(
        result["invalid_budget"] or result["power_limit_exceeded"]
    )
    if result["invalid_budget"]:
        result["satellite_infeasible_reason"] = "remaining_budget_non_positive"
    elif result["power_limit_exceeded"]:
        result["satellite_infeasible_reason"] = "min_power_exceeds_md_power"
    else:
        result["satellite_infeasible_reason"] = ""
    return result
