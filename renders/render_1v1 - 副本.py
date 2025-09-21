# region ############## 环境初始化与路径配置 ############## [参考网页1]
import sys
import os
# 获取当前脚本所在目录（renders目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到系统路径（假设父目录包含算法和环境模块）
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# region ############## 核心模块导入 ##############
import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv  # JSBSim空战环境
from algorithms.ppo.ppo_actor import PPOActor  # PPO策略网络
import logging
logging.basicConfig(level=logging.DEBUG)  # 启用DEBUG级日志

# region ############## 网络参数配置类 ##############
class Args:
    def __init__(self) -> None:
        self.gain = 0.01  # 动作分布初始化增益因子
        self.hidden_size = '128 128'  # MLP基础网络隐藏层结构
        self.act_hidden_size = '128 128'  # 动作网络隐藏层结构
        self.activation_id = 1  # 激活函数类型标识（如ReLU）
        self.use_feature_normalization = False  # 禁用特征归一化
        self.use_recurrent_policy = True  # 启用GRU循环策略
        self.recurrent_hidden_size = 128  # GRU隐藏层维度
        self.recurrent_hidden_layers = 1  # GRU堆叠层数
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))  # 张量配置
        self.use_prior = True  # 启用先验知识调整（如导弹发射条件）

# region ############## 张量转换函数 ##############
def _t2n(x):
    """将GPU张量转换为numpy数组，用于环境交互"""
    return x.detach().cpu().numpy()

# region ############## 对抗测试参数 ##############
num_agents = 2  # 总智能体数（1v1对抗）
render = True  # 启用ACMI格式渲染
ego_policy_index = 1  # 主策略模型版本号
enm_policy_index = 1040  # 对手策略模型版本号
episode_rewards = 0  # 累计回合奖励

# 模型加载路径（示例路径）
ego_run_dir = "C:/Users/avic/Desktop/lag--old/.../run-20250305_150841-9bdk22m8/files"
enm_run_dir = "C:/Users/avic/Desktop/lag--old/.../run-20250305_150841-9bdk22m8/files"

# region ############## 环境与策略初始化 ############## [参考网页1]
env = SingleCombatEnv("1v1/NoWeapon/Selfplay")  # 创建1v1无武器自博弈环境
env.seed(0)  # 固定随机种子
args = Args()  # 实例化网络参数

# 初始化双策略网络（GPU加速）
ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()  # 设置为评估模式
enm_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))  # 加载训练好的策略参数

# region ############## 仿真循环控制 ##############
obs = env.reset()  # 重置环境初始状态
if render:
    env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')  # 创建ACMI记录文件[参考网页1]

# 初始化RNN隐藏状态（匹配GRU层结构）
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)  # (num_layers, batch, hidden_size)
masks = np.ones((num_agents // 2, 1))  # 状态更新掩码（控制RNN状态重置）

# region ############## 主循环流程 ##############
while True:
    # 主策略动作生成
    ego_actions, _, ego_rnn_states = ego_policy(
        ego_obs, 
        ego_rnn_states, 
        masks, 
        deterministic=True  # 启用确定性策略（不采样）
    )
    ego_actions = _t2n(ego_actions)  # 转换为numpy数组
    ego_rnn_states = _t2n(ego_rnn_states)  # 更新RNN状态
    
    # 对手策略动作生成（逻辑同上）
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    
    # 合并双方动作（环境执行需要完整动作数组）
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    
    # 环境交互（获取新状态和奖励）
    obs, rewards, dones, infos = env.step(actions)
    
    # 渲染更新（每步写入ACMI文件）
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')  # 参考网页1的CSV记录方式
    
    # 终止条件判断（所有智能体完成回合）
    if dones.all():
        print(infos)  # 输出终止信息（如击落状态）
        break
    
    # 状态更新（分割双方观察值）
    enm_obs = obs[num_agents // 2:, ...]  # 对手新观察
    ego_obs = obs[:num_agents // 2, ...]  # 己方新观察