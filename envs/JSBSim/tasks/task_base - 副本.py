import numpy as np
from gymnasium import spaces
from typing import List, Tuple
from abc import ABC, abstractmethod
from ..core.catalog import Catalog as c

class BaseTask(ABC):
    """
    任务系统的抽象基类，用于创建自定义任务环境[7,8](@ref)
    
    子类必须实现：
    1. 状态/动作变量定义
    2. 观测/动作空间配置
    3. 任务特定逻辑
    
    属性：
    config (dict): 环境配置参数
    reward_functions (list): 奖励函数集合
    termination_conditions (list): 终止条件集合
    state_var (list): 状态变量集合
    action_var (list): 动作变量集合
    observation_space (gym.Space): 观测空间定义
    action_space (gym.Space): 动作空间定义
    """
    def __init__(self, config):
        """初始化任务环境[6](@ref)
        
        Args:
            config (dict): 包含环境配置参数的字典
        """
        self.config = config
        self.reward_functions = []  # 奖励函数容器
        self.termination_conditions = []  # 终止条件容器
        self.load_variables()  # 加载状态/动作变量
        self.load_observation_space()  # 初始化观测空间
        self.load_action_space()  # 初始化动作空间

    @property
    def num_agents(self):
        """代理数量（默认单代理系统，子类可重写）[8](@ref)
        
        Returns:
            int: 代理数量
        """
        return 1

    @abstractmethod
    def load_variables(self):
        """抽象方法：加载状态/动作变量[3,7](@ref)
        
        子类必须实现：
        - 定义state_var状态变量列表
        - 定义action_var动作变量列表
        """
        # 示例变量定义（需子类具体实现）
        self.state_var = [
            c.position_long_gc_deg,  # 地理经度(度)
            c.position_lat_geod_deg, # 地理纬度(度) 
            c.position_h_sl_m,       # 海拔高度(米)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,   # 副翼指令（归一化值）
            c.fcs_elevator_cmd_norm,  # 升降舵指令
            c.fcs_rudder_cmd_norm,    # 方向舵指令
            c.fcs_throttle_cmd_norm,  # 油门指令
        ]

    @abstractmethod
    def load_observation_space(self):
        """抽象方法：定义观测空间[7](@ref)
        
        子类必须实现：
        - 设置self.observation_space为gymnasium.spaces对象
        """
        self.observation_space = spaces.Discrete(5)  # 示例离散空间

    @abstractmethod
    def load_action_space(self):
        """抽象方法：定义动作空间[7](@ref)
        
        子类必须实现：
        - 设置self.action_space为gymnasium.spaces对象
        """
        self.action_space = spaces.Discrete(5)  # 示例离散空间

    def reset(self, env):
        """任务特定重置逻辑[6](@ref)
        
        Args:
            env (Env): 环境实例
        """
        # 重置所有奖励函数
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def step(self, env):
        """任务特定步进逻辑（需子类扩展）[8](@ref)
        
        Args:
            env (Env): 环境实例
        """
        pass  # 预留接口供子类实现具体逻辑

    def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        """计算聚合奖励值[6](@ref)
        
        Args:
            env (Env): 环境实例
            agent_id (int): 当前代理ID
            info (dict): 附加信息字典

        Returns:
            Tuple[float, dict]: 
                float: 当前时间步总奖励值
                dict: 更新后的信息字典
        """
        reward = 0.0
        # 累加所有奖励函数的输出
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env, agent_id)
        return reward, info

    def get_termination(self, env, agent_id, info={}) -> Tuple[bool, dict]:
        """判断终止条件[6](@ref)
        
        Args:
            env (Env): 环境实例
            agent_id (int): 当前代理ID
            info (dict): 附加信息字典

        Returns:
            Tuple[bool, dict]:
                bool: 是否终止当前episode
                dict: 包含终止状态的信息字典
        """
        done = False
        success = True
        # 检查所有终止条件
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d          # 任一条件触发即终止
            success = success and s   # 所有条件成功