import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict


class BaseRewardFunction(ABC):
    """
    强化学习奖励函数基类
    功能：
    - 定义奖励函数通用接口
    - 实现奖励缩放、潜在奖励计算等基础功能
    - 记录奖励轨迹数据
    
    属性：
    config: 配置参数对象
    reward_scale: 奖励缩放系数（从配置中读取或默认1.0）
    is_potential: 是否启用潜在奖励计算模式（从配置中读取或默认False）
    pre_rewards: 前一步的潜在奖励值缓存（字典结构，按agent_id存储）
    reward_trajectory: 轨迹记录（字典结构，按agent_id存储奖励序列）
    reward_item_names: 奖励项名称列表（默认为类名）
    """
    def __init__(self, config):
        self.config = config
        # 内部状态变量初始化
        self.reward_scale = getattr(self.config, f'{self.__class__.__name__}_scale', 1.0)  # 奖励缩放系数
        self.is_potential = getattr(self.config, f'{self.__class__.__name__}_potential', False)  # 潜在奖励模式标志位
        self.pre_rewards = defaultdict(float)  # 存储各agent的上一步奖励值
        self.reward_trajectory = defaultdict(list)  # 轨迹数据记录器
        self.reward_item_names = [self.__class__.__name__]  # 奖励项标识符

    def reset(self, task, env):
        """环境重置时执行奖励函数特定重置
        功能：
        - 清除潜在奖励缓存（当启用潜在奖励模式时）
        - 重置轨迹记录器
        
        参数：
        task: 任务实例，包含环境状态信息
        env: 环境实例，包含智能体交互接口
        
        处理逻辑：
        当启用潜在奖励模式时，预计算初始奖励值并缓存
        """
        if self.is_potential:
            self.pre_rewards.clear()  # 清空潜在奖励缓存
            for agent_id in env.agents.keys():
                # 初始化潜在奖励基准值
                self.pre_rewards[agent_id] = self.get_reward(task, env, agent_id)
        self.reward_trajectory.clear()  # 清空历史轨迹

    @abstractmethod
    def get_reward(self, task, env, agent_id):
        """抽象方法：计算当前时刻奖励值
        子类必须实现的具体奖励计算逻辑
        
        参数：
        task: 任务实例，提供状态观测信息
        env: 环境实例，提供智能体交互接口
        agent_id: 智能体唯一标识符
        
        返回值：
        float: 原始奖励值（未经缩放和处理）
        
        注意：
        该方法应由子类根据具体奖励逻辑实现
        """
        raise NotImplementedError

    def _process(self, new_reward, agent_id, render_items=()):
        """奖励后处理方法
        功能：
        - 应用奖励缩放系数
        - 计算潜在奖励差值（当启用潜在奖励模式时）
        - 记录奖励轨迹数据
        
        参数：
        new_reward: 原始奖励值
        agent_id: 智能体唯一标识符
        render_items: 可视化项元组（当奖励项多于1个时必须设置）
        
        返回值：
        float: 处理后的最终奖励值
        
        处理流程：
        1. 应用缩放系数得到基础奖励
        2. 若启用潜在奖励模式，计算差值并更新缓存
        3. 记录当前时刻奖励数据到轨迹
        """
        reward = new_reward * self.reward_scale  # 应用缩放
        if self.is_potential:
            # 计算潜在奖励差值（当前值 - 前值）
            reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
        # 记录奖励轨迹（包含奖励值和可视化项）
        self.reward_trajectory[agent_id].append([reward, *render_items])
        return reward

    def get_reward_trajectory(self):
        """获取当前回合的完整奖励轨迹
        返回值：
        dict: 按奖励项分组的轨迹数据字典
        数据结构：
        {
            "reward_name": np.array([时间步序列][智能体序列])
        }
        
        注意：
        返回数组维度为（奖励项数量 × 智能体数量 × 时间步数）
        """
        return dict(zip(self.reward_item_names, np.array(self.reward_trajectory.values()).transpose(2, 0, 1)))