"""
强化学习空战任务模块
包含单机格斗任务(SingleCombatTask)、分层任务(HierarchicalSingleCombatTask)及多种基线智能体实现
核心功能：定义空战环境的状态空间、动作空间、奖励机制及终止条件
"""

import torch
import numpy as np
from gymnasium import spaces
from typing import Literal
from .task_base import BaseTask
from ..core.simulatior import AircraftSimulator
from ..core.catalog import Catalog as c

class SingleCombatTask(BaseTask):
    """单机格斗任务环境
    属性：
        use_baseline (bool): 是否使用基线策略
        use_artillery (bool): 是否启用武器系统
        baseline_agent (BaselineAgent): 基线策略代理
        reward_functions (list): 奖励函数集合[高度奖励,姿态奖励,事件驱动奖励]
        termination_conditions (list): 终止条件集合[低空,极端状态,过载,安全返航,超时]
    """
    def __init__(self, config):
        super().__init__(config)
        # 配置参数初始化
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        self.use_artillery = getattr(self.config, 'use_artillery', False)
        if self.use_baseline:
            self.baseline_agent = self.load_agent(self.config.baseline_type)

        # 奖励函数初始化（包含3种奖励机制）
        self.reward_functions = [
            AltitudeReward(self.config),    # 高度维持奖励
            PostureReward(self.config),     # 飞行姿态稳定性奖励
            EventDrivenReward(self.config)  # 交战事件触发奖励
        ]

        # 终止条件初始化（5种终止情形）
        self.termination_conditions = [
            LowAltitude(self.config),       # 飞行高度过低
            ExtremeState(self.config),      # 极端姿态角
            Overload(self.config),          # 过载超限
            SafeReturn(self.config),        # 安全返航区域
            Timeout(self.config),           # 任务超时
        ]

    @property
    def num_agents(self) -> int:
        """代理数量动态计算（双机对抗或单机对抗基线）"""
        return 2 if not self.use_baseline else 1

    def load_variables(self):
        """定义状态变量集合（16维飞行状态参数）"""
        self.state_var = [
            c.position_long_gc_deg,             # [0] 经度（度）
            c.position_lat_geod_deg,            # [1] 纬度（度）
            c.position_h_sl_m,                  # [2] 海拔高度（米）
            c.attitude_roll_rad,                # [3] 滚转角（弧度）
            c.attitude_pitch_rad,               # [4] 俯仰角（弧度）
            c.attitude_heading_true_rad,        # [5] 偏航角（弧度）
            c.velocities_v_north_mps,           # [6] 北向速度（m/s）
            c.velocities_v_east_mps,            # [7] 东向速度（m/s）
            c.velocities_v_down_mps,            # [8] 地速（m/s）
            c.velocities_u_mps,                 # [9] 机体X轴速度（m/s）
            c.velocities_v_mps,                 # [10] 机体Y轴速度（m/s）
            c.velocities_w_mps,                 # [11] 机体Z轴速度（m/s）
            c.velocities_vc_mps,                # [12] 真空速（m/s）
            c.accelerations_n_pilot_x_norm,     # [13] 北向加速度（G值）
            c.accelerations_n_pilot_y_norm,     # [14] 东向加速度（G值）
            c.accelerations_n_pilot_z_norm,     # [15] 垂直加速度（G值）
        ]
        
        """定义动作变量集合（4个飞行控制面）"""
        self.action_var = [
            c.fcs_aileron_cmd_norm,    # [0] 副翼指令（归一化值[-1,1]）
            c.fcs_elevator_cmd_norm,    # [1] 升降舵指令
            c.fcs_rudder_cmd_norm,      # [2] 方向舵指令
            c.fcs_throttle_cmd_norm,    # [3] 油门指令（[0.4,0.9]区间）
        ]

        """渲染所需状态变量"""
        self.render_var = [
            c.position_long_gc_deg,     # 经度（可视化用）
            c.position_lat_geod_deg,    # 纬度
            c.position_h_sl_m,          # 高度
            c.attitude_roll_rad,        # 滚转角
            c.attitude_pitch_rad,       # 俯仰角
            c.attitude_heading_true_rad,# 偏航角
        ]

    def load_observation_space(self):
        """定义15维观测空间（数值范围[-10,10]）"""
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    def load_action_space(self):
        """定义多离散动作空间（4个控制面，41/41/41/30离散区间）"""
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """状态空间构建流程：
        1. 提取敌我双方原始状态数据
        2. 转换地理坐标系到东北天(NED)坐标系
        3. 计算相对状态参数（方位角、进入角等）
        4. 归一化处理到[-10,10]区间
        """
        # ...（具体计算逻辑略，保持原注释）

    def normalize_action(self, env, agent_id, action):
        """动作空间归一化处理：
        - 基线代理：直接获取预定义动作
        - 学习代理：将离散动作映射到连续控制量
        转换公式：
            舵面：discrete[0-40] -> [-1.0, 1.0]
            油门：discrete[0-29] -> [0.4, 0.9]
        """
        # ...（具体转换逻辑略）

    def step(self, env):
        """仿真步进处理：
        1. 计算方位函数（_orientation_fn）和距离函数（_distance_fn）
        2. 若启用武器系统，计算敌方血量损伤
        """
        # ...（武器系统逻辑略）

class HierarchicalSingleCombatTask(SingleCombatTask):
    """分层强化学习任务（高层策略+底层控制器）
    创新点：
        - 高层动作空间简化为3维[高度变化,航向变化,速度变化]
        - 底层使用预训练策略网络生成具体控制指令
    """
    def __init__(self, config: str):
        super().__init__(config)
        # 加载预训练底层策略模型
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(
            torch.load(get_root_dir() + '/model/baseline_model.pt'))
        self.lowlevel_policy.eval()
        
        # 定义高层动作参数空间
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])  # 高度变化选项
        self.norm_delta_heading = np.array([-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6])  # 航向变化
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])  # 速度变化

    def normalize_action(self, env, agent_id, action):
        """分层动作转换：
        1. 将高层离散动作转换为delta参数
        2. 拼接底层策略输入观测值
        3. 通过策略网络生成底层控制指令
        4. 最终归一化到连续动作空间
        """
        # ...（分层处理逻辑略）

class PursueAgent(BaselineAgent):
    """追击策略代理
    策略逻辑：
        - 保持与目标的相对高度差
        - 计算最优拦截航向
        - 匹配目标速度
    """
    def set_delta_value(self, sim: AircraftSimulator):
        """计算delta参数：
        1. delta_altitude：保持与目标相同高度
        2. delta_heading：计算最优拦截角
        3. delta_velocity：匹配目标速度
        """
        # ...（拦截计算逻辑略）