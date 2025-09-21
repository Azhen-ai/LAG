import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R

class PostureReward(BaseRewardFunction):
    """
    姿态奖励计算模块（继承自基础奖励函数）
    核心公式：PostureReward = Orientation * Range
    - 方向因子(Orientation)：鼓励指向敌机，被敌机指向时惩罚
    - 距离因子(Range)：鼓励接近敌机，超出目标距离时惩罚

    注意：
    - 当前仅支持1v1空战场景
    """
    def __init__(self, config):
        super().__init__(config)
        # 初始化配置参数（从config动态获取或设置默认值）
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')  # 方向因子计算版本
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')  # 距离因子计算版本
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)  # 目标距离阈值（单位：千米）

        # 动态绑定计算函数
        self.orientation_fn = self.get_orientation_function(self.orientation_version)  # 方向因子计算函数
        self.range_fn = self.get_range_funtion(self.range_version)  # 距离因子计算函数
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]  # 奖励项名称扩展

    def get_reward(self, task, env, agent_id):
        """
        核心奖励计算函数
        基于AO（进入角）、TA（目标角）、R（距离）等空战要素计算综合奖励

        参数：
        task - 任务实例，提供环境状态信息
        env - 环境实例，提供智能体交互接口
        agent_id - 当前智能体标识

        返回：
        float - 综合姿态奖励值（经过后处理）
        """
        new_reward = 0
        # 构建本机特征向量：[北向位置, 东向位置, 高度, 北向速度, 东向速度, 垂向速度]
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        
        # 遍历所有敌机计算威胁
        for enm in env.agents[agent_id].enemies:
            # 构建敌机特征向量
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            # 获取空战要素参数（AO:进入角，TA:目标角，R:相对距离）
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            
            # 计算方向因子（基于AO和TA的角度参数）
            orientation_reward = self.orientation_fn(AO, TA)
            # 计算距离因子（R转换为千米单位）
            range_reward = self.range_fn(R / 1000)
            
            # 综合奖励叠加（不同敌机的奖励累加）
            new_reward += orientation_reward * range_reward
        
        # 后处理（包含奖励缩放和轨迹记录）
        return self._process(new_reward, agent_id, (orientation_reward, range_reward))

    def get_orientation_function(self, version):
        """方向因子计算函数工厂（根据版本选择不同计算策略）"""
        if version == 'v0':
            # 版本0：双曲正切函数+角度阈值组合
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            # 版本1：分段函数组合（双曲正切与反双曲正切组合）
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            # 版本2：线性衰减+非线性补偿组合
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            raise NotImplementedError(f"未知方向因子计算版本: {version}")

    def get_range_funtion(self, version):
        """距离因子计算函数工厂（根据版本选择不同计算策略）"""
        if version == 'v0':
            # 版本0：高斯函数与逻辑函数组合
            return lambda R: np.exp(-(R - self.target_dist) ​**​ 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            # 版本1：指数衰减与S型函数组合
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            # 版本2：带符号函数的复合衰减
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            # 版本3：分段多项式与指数函数组合
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"未知距离因子计算版本: {version}")