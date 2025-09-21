from abc import ABC
import sys
import os
# 处理导入路径问题：将项目根目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
from typing import Literal
from abc import ABC, abstractmethod
from ..utils.utils import get_root_dir
from .baseline_actor import BaselineActor

class BaselineAgent(ABC):
    """强化学习智能体基类，实现基础策略模型加载和观察值预处理"""
    def __init__(self, agent_id) -> None:
        """
        初始化基础智能体
        :param agent_id: 智能体标识符，用于多智能体场景区分
        """
        self.model_path = get_root_dir() + '/model/baseline_model.pt'  # 预训练模型路径
        self.actor = BaselineActor()  # 策略网络实例
        # 加载模型参数并设置为评估模式
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.actor.eval()
        self.agent_id = agent_id  # 智能体ID
        self.reset()  # 初始化RNN状态

    def reset(self):
        """重置RNN隐藏状态，用于回合开始时状态初始化"""
        self.rnn_states = np.zeros((1, 1, 128))  # RNN初始状态 (batch_size, seq_len, hidden_size)

    @abstractmethod
    def set_delta_value(self, observation):
        """抽象方法：计算状态差值（由子类实现特定策略）"""
        raise NotImplementedError

    def get_observation(self, observation, delta_value):
        """
        构建标准化观察值向量[6](@ref)
        
        观察值结构：
        0-2: 高度差/航向差/速度差 (delta_value)
        3-11: 原始观察值中的9个维度（单位标准化后）
        
        参数：
            observation : 原始环境观测值
            delta_value : 由子类计算的差值向量
            
        返回：
            norm_obs : 标准化后的观察值向量 (1,12)
        """
        norm_obs = np.zeros(12)
        norm_obs[:3] = delta_value  # 前3维为策略相关差值
        # 单位标准化处理：
        norm_obs[3] = observation[0] * 5   # 高度 (单位：5km)
        norm_obs[4:8] = observation[1:5]   # 姿态角的正余弦值
        norm_obs[8:11] = observation[5:8]   # 机体坐标系速度 (单位：马赫数)
        norm_obs[11] = observation[8]       # 真空速 (单位：马赫数)
        return np.expand_dims(norm_obs, axis=0)  # 增加批次维度

    def get_action(self, observation):
        """
        获取动作决策[8](@ref)
        
        流程：
        1. 计算状态差值
        2. 构建标准化观察值
        3. 通过策略网络生成动作
        4. 维护RNN状态
        
        返回：
            action : 动作向量 (numpy数组)
        """
        delta_value = self.set_delta_value(observation[self.agent_id])  # 策略相关差值计算
        obs = self.get_observation(observation[self.agent_id], delta_value)  # 标准化处理
        # 策略网络推理
        _action, self.rnn_states = self.actor(
            torch.from_numpy(obs).float(),  # 转换为Tensor
            torch.from_numpy(self.rnn_states).float()
        )
        return _action.detach().cpu().numpy().squeeze()  # 转回numpy并移除批次维度

class PursueAgent(BaselineAgent):
    """追击策略智能体，实现基础追踪行为"""
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, observation):
        """
        计算追击策略差值：
        - 高度差：observation[10] (原始单位)
        - 航向差：observation[14] * observation[11] (航向角差 × 距离因子)
        - 速度差：observation[9] (原始单位)
        """
        delta_altitude = observation[10]  # 高度差异值
        delta_heading = observation[14] * observation[11]  # 航向调整量
        delta_velocity = observation[9]  # 速度差异值
        return np.array([delta_altitude, delta_heading, delta_velocity])

class ManeuverAgent(BaselineAgent):
    """机动策略智能体，实现预设战术动作[7](@ref)"""
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n']) -> None:
        """
        :param maneuver: 机动类型
            'l' - 左转机动
            'r' - 右转机动 
            'n' - 复合机动
        """
        super().__init__(agent_id)
        self.turn_interval = 7         # 机动阶段间隔（秒）
        self.env_time_interval = 0.2   # 环境步长时间（秒）
        self.dodge_missile = True       # 导弹规避模式开关
        # 初始化机动参数序列
        if maneuver == 'l':
            self.delta_heading_list = [0, 0, 0, 0]          # 左转航向变化序列
        elif maneuver == 'r':
            self.delta_heading_list = [np.pi/2, 0, 0, 0]    # 右转航向变化序列 
        elif maneuver == 'n':
            self.delta_heading_list = [np.pi/2, np.pi/2, 0, 0]  # 复合机动序列
        self.target_altitude_list = [6096] * 4  # 目标高度序列（单位：米）
        self.target_velocity_list = [243] * 4   # 目标速度序列（单位：米/秒）

    def reset(self):
        """重置步数计数器和RNN状态"""
        self.step = 0  # 时间步计数器
        super().reset()

    def set_delta_value(self, observation):
        """动态计算机动策略差值[6](@ref)"""
        step_thresholds = np.arange(1, len(self.delta_heading_list)+1) * self.turn_interval / self.env_time_interval
        
        # 导弹规避逻辑判断
        if not self.dodge_missile or (len(observation) > 15 and observation[15] != 0):
            # 分阶段执行预设机动
            current_stage = next((i for i, th in enumerate(step_thresholds) if self.step <= th), len(step_thresholds)-1)
            delta_heading = self.delta_heading_list[current_stage]
            # 高度差值计算（目标高度 - 当前高度）标准化到km
            delta_altitude = (self.target_altitude_list[current_stage] - observation[0] * 5000) / 1000
            # 速度差值计算（目标速度 - 当前速度）标准化到马赫数（假设340m/s=1马赫）
            delta_velocity = (self.target_velocity_list[current_stage] - observation[5] * 340) / 340
            self.step += 1
        else:
            # 保持当前状态
            delta_heading = 0
            delta_altitude = (6096 - observation[0] * 5000) / 1000  # 默认高度目标
            delta_velocity = (243 - observation[5] * 340) / 340      # 默认速度目标
            
        return np.array([delta_altitude, delta_heading, delta_velocity])