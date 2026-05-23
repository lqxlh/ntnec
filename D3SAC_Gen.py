import os
import pickle
import random
import warnings

import numpy as np
import torch

import D3QN
import D3SAC
import EdgeEnv
import SAC
import experiment_config
import para

warnings.filterwarnings("ignore", category=Warning)

LEARNING_RATE = para.LEARNING_RATE
ACTOR_LR = para.ACTOR_LR
CRITIC_LR = para.CRITIC_LR
DQN_GAMMA = para.DQN_GAMMA
SAC_GAMMA = para.SAC_GAMMA
TAU = para.TAU
SEED = para.SEED


def build_agents(env):
    # 这里重建与主训练阶段完全一致的网络结构，然后加载已经训练好的 NTN 模型参数。
    action_dim = env.action_space.n
    obs_shape = env.observation_space.shape[1]
    action_dim2 = env.action_space2.shape[0]

    model = D3QN.Model(obs_dim=obs_shape, act_dim=action_dim)
    algorithm = D3QN.DDQN(model, act_dim=action_dim, gamma=DQN_GAMMA, lr=LEARNING_RATE)
    agent = D3QN.Agent(algorithm, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.0, e_greed_decrement=0.0)

    model2 = SAC.Model(obs_dim=obs_shape + 1, act_dim=action_dim2)
    actor_model = SAC.ActorModel(obs_dim=obs_shape + 1, act_dim=action_dim2)
    algorithm2 = SAC.SAC(model2, actor_model, gamma=SAC_GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent2 = SAC.Agent(algorithm2, obs_dim=obs_shape + 1, act_dim=action_dim2)
    return agent, agent2


def load_trained_agents(env, result_prefix):
    # 这里加载当前主 NTN 训练入口保存的最佳模型。
    # 文件名改成和 D3SAC.py 一致的 result_prefix 风格，避免旧版 best_*_20.pth 这种过时命名继续混用。
    agent, agent2 = build_agents(env)
    agent.alg.load_model(f"models/{result_prefix}_best_d3qn_model.pth")
    agent2.alg.load_model(f"models/{result_prefix}_best_sac_model.pth")
    return agent, agent2


def evaluate_loaded_model(result_prefix=experiment_config.MAIN_SIM_PREFIX, eval_rounds=5):
    # 这里对应“加载已训练好的 NTN 模型，在当前合成场景下重复评估若干轮”。
    # 它主要用于泛化测试、绘图前复核，以及和其他实验前缀做并排比较。
    eval_rewards = []
    eval_metric_list = []

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = EdgeEnv.EdgeEnv()
        agent, agent2 = load_trained_agents(env, result_prefix)
        bs_list, md_list, sat_list = D3SAC.build_entities(env)
        eval_reward, eval_debug = D3SAC.evaluate(env, agent, agent2, md_list, bs_list, sat_list, eval_rounds=eval_rounds)
        eval_rewards.append(eval_reward)
        eval_metric_list.append(D3SAC.summarize_debug(eval_debug))

    output = {
        "mean_eval_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
        "std_eval_reward": float(np.std(eval_rewards)) if eval_rewards else 0.0,
        "per_seed_eval_reward": eval_rewards,
        "per_seed_eval_metrics": eval_metric_list,
    }
    return output


def save_generation_summary(result_prefix=experiment_config.MAIN_SIM_PREFIX, output_prefix=None):
    # 这里把加载评估后的结果重新存成 pickle，方便后续统一画图或导出论文表格。
    if output_prefix is None:
        output_prefix = f"{result_prefix}_gen_eval"

    os.makedirs("result", exist_ok=True)
    summary = evaluate_loaded_model(result_prefix=result_prefix)

    with open(f"result/{output_prefix}_summary.pickle", "wb") as file_obj:
        pickle.dump(summary, file_obj)

    print(
        f"[{output_prefix}] mean_eval_reward={summary['mean_eval_reward']:.3f} | "
        f"std_eval_reward={summary['std_eval_reward']:.3f}"
    )


if __name__ == "__main__":
    # 这里默认评估当前主训练脚本保存的最佳 NTN 模型。
    save_generation_summary(result_prefix=experiment_config.MAIN_SIM_PREFIX)