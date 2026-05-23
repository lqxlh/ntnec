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


def smooth_curve(values, window_size=5):
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
    # 任务成功率没有单独保存；deadline 违反会写入 avg_penalty_time，惩罚为 0 表示成功，为 -1 表示失败。
    # 因此 success_rate = 1 + avg_penalty_time，最后裁剪到 [0, 1]，防止旧结果或极端值越界。
    penalty_time_curve = build_curve(metric_histories, "avg_penalty_time", 0.0)
    return np.clip(1.0 + penalty_time_curve, 0.0, 1.0)


def build_energy_curve(metric_histories):
    # 环境里保存的是 w_e * energy，这里除以 w_e 还原平均总能耗。
    weighted_energy_curve = build_curve(metric_histories, "avg_energy_cost", 0.0)
    weight_energy = max(float(para.w_e), 1e-12)
    return weighted_energy_curve / weight_energy


def draw_metric_overview(plt, episodes, curves, output_path):
    # 2 行 3 列分别展示用户要求的六个核心指标。
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("NTN Evaluation Metrics", fontsize=15)
    axes = axes.reshape(-1)

    plot_items = [
        ("Average Task Success Rate", curves["task_success"], "Success Rate", "#2ca02c"),
        ("Average Total Delay", curves["total_delay"], "Delay (s)", "#d62728"),
        ("Average Total Energy", curves["total_energy"], "Energy (J)", "#9467bd"),
        ("Average Reward", curves["reward"], "Reward / Step", "#1f77b4"),
        ("Satellite Usage Rate", curves["sat_usage"], "Usage Rate", "#ff7f0e"),
        ("Satellite Success Rate", curves["sat_success"], "Success Rate", "#17becf"),
    ]

    for axis, (title, values, ylabel, color) in zip(axes, plot_items):
        axis.plot(episodes, values, color=color, linewidth=1.5, alpha=0.5, label="Raw")
        axis.plot(episodes, smooth_curve(values), color=color, linewidth=2.4, label="Smoothed")
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_convergence_curve(plt, episodes, eval_return_curve, output_path, train_return_curve=None):
    # 收敛曲线通常看总回合奖励；这里和日志里的 eval_reward/train_reward 保持一致。
    fig, axis = plt.subplots(figsize=(10, 5))
    if train_return_curve is not None:
        axis.plot(episodes, train_return_curve, color="#ff7f0e", linewidth=1.0, alpha=0.25, label="Train Return")
        axis.plot(episodes, smooth_curve(train_return_curve), color="#ff7f0e", linewidth=1.8, alpha=0.8, label="Smoothed Train Return")
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


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("当前环境没有安装 matplotlib，无法直接出图。")
        print("训练结果已经保存在 result/*.pickle，可以安装 matplotlib 后重新运行本脚本。")
        return

    result_dir = get_result_dir()
    result_prefix = sys.argv[1] if len(sys.argv) > 1 else experiment_config.DEFAULT_PLOT_PREFIX
    eval_histories = load_metric_histories(result_dir, result_prefix, "eval")
    eval_return_curve = load_return_curve(result_dir, result_prefix, "eval")
    train_return_curve = load_return_curve(result_dir, result_prefix, "train")

    reward_curve = build_curve(eval_histories, "avg_reward")
    curves = {
        "task_success": build_task_success_curve(eval_histories),
        "total_delay": build_curve(eval_histories, "avg_total_delay"),
        "total_energy": build_energy_curve(eval_histories),
        "reward": reward_curve,
        "sat_usage": build_curve(eval_histories, "sat_usage_rate"),
        "sat_success": build_curve(eval_histories, "sat_deadline_success_rate"),
    }
    metric_episodes = np.arange(1, len(reward_curve) + 1)
    return_episodes = np.arange(1, len(eval_return_curve) + 1)

    overview_path = os.path.join(result_dir, f"{result_prefix}_metrics_overview.png")
    convergence_path = os.path.join(result_dir, f"{result_prefix}_convergence.png")
    draw_metric_overview(plt, metric_episodes, curves, overview_path)
    draw_convergence_curve(plt, return_episodes, eval_return_curve, convergence_path, train_return_curve=train_return_curve)

    print(f"指标总览图已保存到: {overview_path}")
    print(f"收敛曲线已保存到: {convergence_path}")


if __name__ == "__main__":
    main()
