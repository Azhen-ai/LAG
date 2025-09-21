import logging
import time
import gymnasium
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from ..core.simulatior import AircraftSimulator, BaseSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config

class BaseEnv(gymnasium.Env):
    """
    基于JSBSim飞行动力学模块的强化学习环境基类，遵循OpenAI Gym接口规范
    
    特性：
    - 支持多智能体协同/对抗场景
    - 提供标准化观察空间和动作空间
    - 集成Tacview实时可视化
    - 支持同步/异步策略更新
    
    扩展点：
    - 通过继承BaseTask实现自定义任务
    - 通过AircraftSimulator扩展飞行器模型
    
    [1,3](@ref)
    """
    metadata = {"render.modes": ["human", "txt"]}  # 支持的渲染模式

    def __init__(self, config_name: str):
        # 初始化基础配置
        self.config = parse_config(config_name)  # 解析YAML/JSON配置文件
        self.max_steps = getattr(self.config, 'max_steps', 100)  # 单次episode最大步数
        self.sim_freq = getattr(self.config, 'sim_freq', 60)  # 物理仿真频率(Hz)
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # 策略执行间隔步数
        # 战场中心坐标(经度, 纬度, 海拔)
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))
        self._create_records = False  # 渲染记录文件标记
        self.load()  # 加载任务和仿真器

    @property
    def num_agents(self) -> int:
        """当前激活的RL智能体数量（排除非学习型对手）"""
        return self.task.num_agents

    @property
    def observation_space(self) -> gymnasium.Space:
        """动态获取观察空间（由具体任务定义）"""
        return self.task.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """动态获取动作空间（由具体任务定义）"""
        return self.task.action_space

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        """所有飞行器仿真实例的字典（包含敌我双方）"""
        return self._jsbsims

    @property
    def time_interval(self) -> int:
        """策略执行时间间隔（秒）"""
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        """初始化环境三要素：任务、仿真器、随机种子"""
        self.load_task()
        self.load_simulator()
        self.seed()

    def load_task(self):
        """加载任务模块（需子类实现具体任务）"""
        self.task = BaseTask(self.config)  # 抽象任务基类

    def load_simulator(self):
        """初始化飞行器仿真器集群"""
        self._jsbsims = {}  # 主仿真器字典 {uid: AircraftSimulator}
        for uid, config in self.config.aircraft_configs.items():
            # 实例化每个飞行器的仿真器
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),  # 可视化颜色
                model=config.get("model", "f16"),  # 飞行器模型
                init_state=config.get("init_state"),  # 初始状态(位置、姿态等)
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                sim_freq=self.sim_freq,  # 仿真频率
                num_missiles=config.get("missile", 0))  # 挂载导弹数量
        
        # 根据UID首字母划分敌我阵营（例如'A01'为友方，'B02'为敌方）
        _default_team_uid = list(self._jsbsims.keys())[0][0]
        self.ego_ids = [uid for uid in self._jsbsims.keys() if uid[0] == _default_team_uid]
        self.enm_ids = [uid for uid in self._jsbsims.keys() if uid[0] != _default_team_uid]

        # 建立飞行器间的关联关系（用于传感器模拟）
        for key, sim in self._jsbsims.items():
            for k, s in self._jsbsims.items():
                if k == key:
                    pass  # 跳过自身
                elif k[0] == key[0]:
                    sim.partners.append(s)  # 友军列表
                else:
                    sim.enemies.append(s)  # 敌军列表

        self._tempsims = {}  # 临时仿真器字典（用于导弹等临时物体）

    def add_temp_simulator(self, sim: BaseSimulator):
        """动态添加临时仿真器（如发射的导弹）"""
        self._tempsims[sim.uid] = sim

    def reset(self) -> np.ndarray:
        """
        重置环境状态并返回初始观察
        
        流程：
        1. 重置仿真器状态
        2. 清除临时物体
        3. 重置任务状态
        4. 获取初始观察量
        
        Returns:
            np.ndarray: 拼接后的多智能体观察矩阵
        """
        self.current_step = 0  # 重置步数计数器
        # 重置所有飞行器状态
        for sim in self._jsbsims.values():
            sim.reload()  # 重新加载初始状态
        self._tempsims.clear()  # 清除临时物体
        
        # 任务重置（例如目标点重置）
        self.task.reset(self)
        
        # 获取观察并打包为numpy数组
        obs = self.get_obs()
        return self._pack(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        执行一个时间步长的环境模拟
        
        Args:
            action (np.ndarray): 多维动作数组，形状为(num_agents, action_dim)
            
        Returns:
            tuple: 
                obs: 各智能体的观察值
                rewards: 奖励值
                dones: 终止标记
                info: 调试信息字典
        """
        self.current_step += 1
        info = {"current_step": self.current_step}  # 记录当前步数
        
        # 1. 应用动作到仿真器
        action = self._unpack(action)  # 将数组转换为字典格式
        for agent_id in self.agents.keys():
            # 动作归一化处理（例如将[-1,1]映射到实际控制量）
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            # 设置飞行器控制属性（如舵面偏转、油门等）
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        
        # 2. 运行物理仿真（按指定步长推进）
        for _ in range(self.agent_interaction_steps):
            # 更新所有飞行器状态
            for sim in self._jsbsims.values():
                sim.run()  # 执行单个仿真步
            # 更新临时物体状态（如导弹飞行）
            for sim in self._tempsims.values():
                sim.run()
        
        # 3. 任务逻辑更新（例如检查胜负条件）
        self.task.step(self)
        
        # 4. 获取新观察
        obs = self.get_obs()
        
        # 5. 计算终止条件
        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]
            
        # 6. 计算即时奖励
        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def get_obs(self):
        """获取所有智能体的局部观察（去中心化执行时使用）"""
        return dict([(agent_id, self.task.get_obs(self, agent_id)) 
                    for agent_id in self.agents.keys()])

    def get_state(self):
        """获取全局状态（中心化训练时使用）"""
        state = np.hstack([self.task.get_obs(self, agent_id) 
                         for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id in self.agents.keys()])

    def close(self):
        """释放所有仿真器资源"""
        for sim in self._jsbsims.values():
            sim.close()  # 关闭飞行器仿真
        for sim in self._tempsims.values():
            sim.close()  # 关闭临时物体仿真
        self._jsbsims.clear()
        self._tempsims.clear()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        """
        环境状态可视化
        
        模式：
        - txt: 生成Tacview兼容的ACMI格式日志
        - real_time: 通过Socket实时传输到Tacview客户端
        
        [7,8](@ref)
        """
        if mode == "txt":
            # 初始化ACMI文件头
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            
            # 追加当前状态帧
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval
                f.write(f"#{timestamp:.2f}\n")
                # 记录所有飞行器状态
                for sim in self._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
                # 记录临时物体状态
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        elif mode == "real_time":
            # 实时渲染模式（需要Tacview客户端）
            timestamp = self.current_step * self.time_interval
            data = [f"#{timestamp:.2f}\n"]
            # 收集所有实体状态
            for sim in self._jsbsims.values():
                log_msg = sim.log()
                if log_msg is not None:
                    data.append(log_msg + "\n")
            for sim in self._tempsims.values():
                log_msg = sim.log()
                if log_msg is not None:
                    data.append(log_msg + "\n")
            # 通过Socket发送数据
            tacview.send_data_to_client("".join(data))
        else:
            raise NotImplementedError("不支持的渲染模式")

    def seed(self, seed=None):
        """
        设置随机数种子保证实验可复现
        
        注意：
        - 需要同时设置numpy和仿真器的随机种子
        - 返回种子列表便于调试
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """
        将字典数据打包为numpy数组（用于多智能体数据整合）
        
        处理逻辑：
        1. 分离友军和敌军数据
        2. 拼接为统一数组
        3. 仅返回RL控制的智能体数据
        """
        ego_data = np.array([data[uid] for uid in self.ego_ids])
        enm_data = np.array([data[uid] for uid in self.enm_ids])
        # 拼接敌我数据（保持顺序一致）
        data = np.concatenate((ego_data, enm_data)) if enm_data.shape[0] > 0 else ego_data
        # 数据完整性检查
        try:
            assert np.isnan(data).sum() == 0
        except AssertionError:
            import pdb; pdb.set_trace()  # 触发调试器
        return data[:self.num_agents, ...]

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """
        将numpy数组解包为字典格式（用于动作分发）
        
        处理逻辑：
        1. 根据预设顺序映射到智能体UID
        2. 非RL控制的智能体动作为None
        """
        assert len(data) == self.num_agents, "动作数量与智能体数不匹配"
        # 创建动作字典
        unpack_data = dict(zip((self.ego_ids + self.enm_ids)[:self.num_agents], data))
        # 填充非RL智能体的空动作
        for agent_id in (self.ego_ids + self.enm_ids)[self.num_agents:]:
            unpack_data[agent_id] = None
        return unpack_data