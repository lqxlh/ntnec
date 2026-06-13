import glob
import os
import pickle
import sys

import numpy as np

import experiment_config
import para


def load_pickle(path):
    # 统一读取 pickle 结果文件，避免每个指标都重复写 open/load。
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


def get_script_dir():
    # 画图脚本可能从 D3SAC-main 内部运行，也可能从项目根目录运行，所以先定位脚本自己的目录。
    return os.path.dirname(os.path.abspath(__file__))


def get_result_dir():
    # result 文件夹固定在 D3SAC-main/result，使用绝对路径能减少运行目录变化带来的错误。
    return os.path.join(get_script_dir(), "result")


def metric_get(metric_dict, metric_key, default_value=0.0):
    # 某些旧结果文件可能没有新字段，这里给默认值，保证画图脚本能兼容旧实验。
    return float(metric_dict.get(metric_key, default_value))


def smooth_curve(values, window_size=4):
    # 对曲线做简单滑动平均，让收敛趋势更清楚；窗口不足时使用已有前缀数据求均值。
    values = np.asarray(values, dtype=np.float64)
    if window_size <= 1 or len(values) == 0:
        return values
    smoothed_values = np.zeros_like(values)
    for idx in range(len(values)):
        start_idx = max(0, idx - window_size + 1)
        smoothed_values[idx] = np.mean(values[start_idx:idx + 1])
    return smoothed_values


def load_metric_histories(result_dir, result_prefix, split_name):
    # 读取 train/eval 的逐 seed 指标历史，例如 main_ntn_sim_eval_metrics_seed_1.pickle。
    pattern = os.path.join(result_dir, f"{result_prefix}_{split_name}_metrics_seed_*.pickle")
    metric_paths = sorted(glob.glob(pattern))
    if not metric_paths:
        raise FileNotFoundError(f"没有找到指标文件: {pattern}")
    return [load_pickle(path) for path in metric_paths]


def build_curve(metric_histories, metric_key, default_value=0.0):
    # 把多个 seed 的同名指标按 episode 对齐后求平均，得到一条最终曲线。
    min_len = min(len(history) for history in metric_histories)
    curves = []
    for history in metric_histories:
        curves.append([metric_get(item, metric_key, default_value) for item in history[:min_len]])
    return np.mean(np.asarray(curves, dtype=np.float64), axis=0)


def load_return_curve(result_dir, result_prefix, split_name):
    # 收敛曲线使用总回合奖励，和训练日志里的 eval_reward/train_reward 保持同一口径。
    return_path = os.path.join(result_dir, f"{result_prefix}_{split_name}_rewards.pickle")
    return_histories = np.asarray(load_pickle(return_path), dtype=np.float64)
    if return_histories.ndim == 1:
        return return_histories
    return np.mean(return_histories, axis=0)


def build_task_success_curve(metric_histories):
    # 成功率归一化
    penalty_time_curve = build_curve(metric_histories, "avg_penalty_time", 0.0)
    return np.clip(1.0 + penalty_time_curve, 0.0, 1.0)


def build_action_usage_curves(metric_histories):
    # 三种动作的使用率
    local_curve = build_curve(metric_histories, "local_actions", 0.0)
    bs_curve = build_curve(metric_histories, "bs_actions", 0.0)
    sat_curve = build_curve(metric_histories, "sat_actions_count", 0.0)
    total = local_curve + bs_curve + sat_curve
    total_safe = np.where(total > 0, total, 1.0)
    return {
        "local": local_curve / total_safe,
        "bs": bs_curve / total_safe,
        "sat": sat_curve / total_safe,
    }


def draw_metric_overview(plt, episodes, curves, output_path):
    # 2 行 3 列：5 个核心指标 + 1 个动作使用率分布。
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("NTN Evaluation Metrics", fontsize=15)
    axes = axes.reshape(-1)

    # 前 4 个子图：常规单线指标
    plot_items = [
        ("Average Task Success Rate", curves["task_success"], "Success Rate", "#2ca02c"),
        ("Average Total Delay", curves["total_delay"], "Delay (s)", "#d62728"),
        ("Average Total Energy", curves["total_energy"], "Energy (J)", "#9467bd"),
        ("Average Reward", curves["reward"], "Reward / Step", "#1f77b4"),
    ]
    for axis, (title, values, ylabel, color) in zip(axes[:4], plot_items):
        axis.plot(episodes, values, color=color, linewidth=1.5, alpha=0.5, label="Raw")
        axis.plot(episodes, smooth_curve(values), color=color, linewidth=2.4, label="Smoothed")
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()

    # 第 5 个子图：动作使用率分布（三条曲线同图）
    usage = curves["action_usage"]
    axis_usage = axes[4]
    axis_usage.plot(episodes, smooth_curve(usage["local"]), color="#2ca02c",
                    linewidth=2.4, label="Local")
    axis_usage.plot(episodes, smooth_curve(usage["bs"]), color="#1f77b4",
                    linewidth=2.4, label="BS")
    axis_usage.plot(episodes, smooth_curve(usage["sat"]), color="#ff7f0e",
                    linewidth=2.4, label="SAT")
    # 用半透明细线叠 raw 数据
    axis_usage.plot(episodes, usage["local"], color="#2ca02c", linewidth=0.8, alpha=0.3)
    axis_usage.plot(episodes, usage["bs"], color="#1f77b4", linewidth=0.8, alpha=0.3)
    axis_usage.plot(episodes, usage["sat"], color="#ff7f0e", linewidth=0.8, alpha=0.3)
    axis_usage.set_title("Action Usage Distribution")
    axis_usage.set_xlabel("Episode")
    axis_usage.set_ylabel("Usage Ratio")
    axis_usage.set_ylim(0.0, 1.0)
    axis_usage.grid(alpha=0.3)
    axis_usage.legend()

    # 第 6 格留空（隐藏边框）
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_convergence_curve(plt, episodes, eval_return_curve, output_path, train_return_curve=None):
    # 单独的收敛曲线
    fig, axis = plt.subplots(figsize=(10, 5))
    if train_return_curve is not None:
        axis.plot(episodes, train_return_curve, color="#ff7f0e", linewidth=1.0, alpha=0.25, label="Train Return")
        axis.plot(episodes, smooth_curve(train_return_curve), color="#ff7f0e", linewidth=1.8, alpha=0.8,
                  label="Smoothed Train Return")
    axis.plot(episodes, eval_return_curve, color="#1f77b4", linewidth=1.3, alpha=0.45, label="Eval Return")
    axis.plot(episodes, smooth_curve(eval_return_curve), color="#1f77b4", linewidth=2.5, label="Smoothed Eval Return")
    axis.set_title("Convergence Curve")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Episode Return")
    axis.grid(alpha=0.3)
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 新增功能一：多算法收敛曲线对比图 (Convergence Analysis)
# ═════════════════════════════════════════════════════════════════════════════
def draw_baseline_convergence_comparison(plt, result_dir, result_prefix, output_path):
    """在同一张图内画出主实验、DQN+Heuristic、以及 Greedy 三种算法的收敛曲线对比。"""
    fig, axis = plt.subplots(figsize=(10, 5))

    # 定义需要对比的算法前缀和对应 label
    comparison_targets = [

        ("PHD3QN",
         result_prefix,
         "#1f77b4"),

        ("BL1 Local",
         "baselines_local_only",
         "#ff7f0e"),

        ("BL2 NoSat",
         "baselines_no_satellite",
         "#2ca02c"),

        ("BL3 NoBS",
         "baselines_no_gnb",
         "#d62728"),

        ("BL4 Random",
         "baselines_random_equal",
         "#9467bd"),

        ("BL5 Greedy",
         "baselines_greedy_delay",
         "#8c564b"),

        ("BL6 DQN",
         "baselines_ddqn_fixed",
         "#e377c2")
    ]

    has_data = False
    for label, prefix, color in comparison_targets:
        try:
            curve = load_return_curve(result_dir, prefix, "eval")
            episodes = np.arange(1, len(curve) + 1)
            # 绘制 Raw 细线
            axis.plot(episodes, curve, color=color, linewidth=0.8, alpha=0.25)
            # 绘制 Smoothed 粗线
            axis.plot(episodes, smooth_curve(curve), color=color, linewidth=2.2, label=label)
            has_data = True
        except FileNotFoundError:
            print(f"  [提示] 未找到 {label} 的收益数据，跳过该曲线绘制。")

    if not has_data:
        plt.close(fig)
        return

    axis.set_title("Convergence Analysis (Proposed vs. Baselines)")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Episode Return")
    axis.grid(alpha=0.3)
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"收敛对比图已保存到: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 新增功能二：物理性能直方图 (Bar Charts of QoS)
# ═════════════════════════════════════════════════════════════════════════════
def draw_qos_bar_charts(plt,
                        result_dir,
                        result_prefix,
                        output_path,
                        steady_episodes=50):
    """
    7种算法性能对比柱状图

    Reward
    Delay
    Energy
    Success Rate
    """

    algo_dict = {
        "PHD3QN": result_prefix,
        "BL1\nLocal": "baselines_local_only",
        "BL2\nNoSat": "baselines_no_satellite",
        "BL3\nNoBS": "baselines_no_gnb",
        "BL4\nRandom": "baselines_random_equal",
        "BL5\nGreedy": "baselines_greedy_delay",
        "BL6\nDQN": "baselines_ddqn_fixed"
    }

    active_algos = []

    rewards = []
    delays = []
    energies = []
    successes = []

    for algo_name, prefix in algo_dict.items():

        try:
            histories = load_metric_histories(
                result_dir,
                prefix,
                "eval"
            )

            reward = build_curve(
                histories,
                "avg_reward",
                0.0
            )[-steady_episodes:].mean()

            delay = build_curve(
                histories,
                "avg_total_delay",
                0.0
            )[-steady_episodes:].mean()

            energy = build_curve(
                histories,
                "avg_total_energy",
                0.0
            )[-steady_episodes:].mean()

            penalty_time = build_curve(
                histories,
                "avg_penalty_time",
                0.0
            )[-steady_episodes:].mean()

            success = np.clip(
                1.0 + penalty_time,
                0.0,
                1.0
            )

            active_algos.append(algo_name)

            rewards.append(reward)
            delays.append(delay)
            energies.append(energy)
            successes.append(success * 100)

        except FileNotFoundError:
            print(f"{algo_name} 数据不存在，跳过")
            continue

    if len(active_algos) == 0:
        print("没有可绘制的数据")
        return

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2"
    ][:len(active_algos)]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10)
    )

    fig.suptitle(
        "Performance Comparison of Algorithms",
        fontsize=16
    )

    metric_info = [

        (
            "Average Reward",
            rewards,
            axes[0, 0],
            ".2f"
        ),

        (
            "Average Delay (s)",
            delays,
            axes[0, 1],
            ".3f"
        ),

        (
            "Average Energy (J)",
            energies,
            axes[1, 0],
            ".3f"
        ),

        (
            "Task Success Rate (%)",
            successes,
            axes[1, 1],
            ".1f"
        )
    ]

    for title, values, ax, fmt in metric_info:

        bars = ax.bar(
            active_algos,
            values,
            color=colors,
            edgecolor="black",
            alpha=0.85
        )

        ax.set_title(title)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        for bar in bars:

            height = bar.get_height()

            ax.annotate(
                format(height, fmt),
                xy=(
                    bar.get_x() + bar.get_width()/2,
                    height
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

        plt.setp(
            ax.get_xticklabels(),
            rotation=15,
            ha="right"
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"QoS柱状图已保存: {output_path}")

def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("当前环境没有安装 matplotlib，无法直接出图。")
        print("训练结果已经保存在 result/*.pickle，可以安装 matplotlib 后重新运行本脚本。")
        return

    result_dir = get_result_dir()
    result_prefix = sys.argv[1] if len(sys.argv) > 1 else experiment_config.DEFAULT_PLOT_PREFIX

    # ── 1. 基础总览图与基础收敛曲线 ────────────────────────────────────
    eval_histories = load_metric_histories(result_dir, result_prefix, "eval")
    eval_return_curve = load_return_curve(result_dir, result_prefix, "eval")
    train_return_curve = load_return_curve(result_dir, result_prefix, "train")

    reward_curve = build_curve(eval_histories, "avg_reward")
    curves = {
        "task_success": build_task_success_curve(eval_histories),
        "total_delay": build_curve(eval_histories, "avg_total_delay"),
        "total_energy": build_curve(eval_histories, "avg_total_energy", 0.0),  # 用真实能耗
        "reward": reward_curve,
        "action_usage": build_action_usage_curves(eval_histories),
    }
    metric_episodes = np.arange(1, len(reward_curve) + 1)
    return_episodes = np.arange(1, len(eval_return_curve) + 1)

    overview_path = os.path.join(result_dir, f"{result_prefix}_metrics_overview.png")
    convergence_path = os.path.join(result_dir, f"{result_prefix}_convergence.png")
    draw_metric_overview(plt, metric_episodes, curves, overview_path)
    draw_convergence_curve(plt, return_episodes, eval_return_curve, convergence_path,
                           train_return_curve=train_return_curve)

    print(f"指标总览图已保存到: {overview_path}")
    print(f"收敛曲线已保存到: {convergence_path}")

    # ── 2. 新增：多算法收敛对比与 QoS 性能直方图对比 ─────────────────────
    comparison_conv_path = os.path.join(result_dir, f"baselines_convergence_comparison.png")
    qos_bar_path = os.path.join(result_dir, f"baselines_qos_bar_comparison.png")

    draw_baseline_convergence_comparison(plt, result_dir, result_prefix, comparison_conv_path)
    draw_qos_bar_charts(plt, result_dir, result_prefix, qos_bar_path)


if __name__ == "__main__":
    main()
