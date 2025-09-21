import logging
import time
import gymnasium
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple, Optional  # 添加 Optional 导入
from ..core.simulatior import AircraftSimulator, BaseSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config
# 导入实时传输器
from .real_time_transmitter import RealTimeACMITransmitter

class BaseEnv(gymnasium.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An BaseEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "txt", "real_time1"]}

    def __init__(self, config_name: str, enable_real_time_transmission: bool = False, 
                 unity_host: str = '127.0.0.1', unity_port: int = 5000):
        """
        初始化环境
        
        Args:
            config_name (str): 配置文件名称
            enable_real_time_transmission (bool): 是否启用实时传输到Unity
            unity_host (str): Unity监听的IP地址
            unity_port (int): Unity监听的端口号
        """
        # basic args
        self.config = parse_config(config_name)
        self.max_steps = getattr(self.config, 'max_steps', 100)  # type: int
        self.sim_freq = getattr(self.config, 'sim_freq', 60)  # type: int
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # type: int
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))
        self._create_records = False
        
        # 实时传输相关配置
        self.enable_real_time_transmission = enable_real_time_transmission
        self.real_time_transmitter: Optional[RealTimeACMITransmitter] = None
        
        # 先加载基础组件
        self.load()
        
        # 如果启用实时传输，初始化传输器
        if self.enable_real_time_transmission:
            self.real_time_transmitter = RealTimeACMITransmitter(unity_host, unity_port)
            # 在单独线程中启动服务器，避免阻塞训练
            import threading
            server_thread = threading.Thread(target=self._start_transmitter, daemon=True)
            server_thread.start()
    
    def _start_transmitter(self):
        """
        在后台线程中启动实时传输服务
        """
        if self.real_time_transmitter:
            success = self.real_time_transmitter.start_server()
            if not success:
                logging.warning("实时传输服务启动失败，将禁用实时传输功能")
                self.enable_real_time_transmission = False

    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def observation_space(self) -> gymnasium.Space:
        return self.task.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self.task.action_space

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        return self._jsbsims

    @property
    def time_interval(self) -> float:  # 修改返回类型为float，因为是除法运算
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        self.load_task()
        self.load_simulator()
        self.seed()

    def load_task(self):
        self.task = BaseTask(self.config)

    def load_simulator(self):
        self._jsbsims = {}     # type: Dict[str, AircraftSimulator]
        for uid, config in self.config.aircraft_configs.items():
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),
                model=config.get("model", "f16"),
                init_state=config.get("init_state"),
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                sim_freq=self.sim_freq,
                num_missiles=config.get("missile", 0))
        # Different teams have different uid[0]
        _default_team_uid = list(self._jsbsims.keys())[0][0]
        self.ego_ids = [uid for uid in self._jsbsims.keys() if uid[0] == _default_team_uid]
        self.enm_ids = [uid for uid in self._jsbsims.keys() if uid[0] != _default_team_uid]

        # Link jsbsims
        for key, sim in self._jsbsims.items():
            for k, s in self._jsbsims.items():
                if k == key:
                    pass
                elif k[0] == key[0]:
                    sim.partners.append(s)
                else:
                    sim.enemies.append(s)

        self._tempsims = {}    # type: Dict[str, BaseSimulator]

    def add_temp_simulator(self, sim: BaseSimulator):
        self._tempsims[sim.uid] = sim

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (np.ndarray): initial observation
        """
        # reset sim
        self.current_step = 0
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()
        # reset task
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.ndarray): the agents' actions, allow opponent's action input

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)

        obs = self.get_obs()

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
            
        # 如果启用实时传输，自动发送当前步骤的数据
        if self.enable_real_time_transmission:
            self.render(mode="real_time1")

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return dict([(agent_id, self.task.get_obs(self, agent_id)) for agent_id in self.agents.keys()])

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.task.get_obs(self, agent_id) for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id in self.agents.keys()])

    def close(self):
        """
        清理环境资源
        
        在原有功能基础上，增加了实时传输服务的清理
        """
        # 停止实时传输服务
        if self.real_time_transmitter:
            self.real_time_transmitter.stop()
            self.real_time_transmitter = None
            
        # 清理仿真器资源
        for sim in self._jsbsims.values():
            sim.close()
        for sim in self._tempsims.values():
            sim.close()
        self._jsbsims.clear()
        self._tempsims.clear()

    def render(self, mode="real_time1", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        """
        渲染环境数据
        
        支持的模式:
        - human: 在终端打印
        - txt: 输出到txt.acmi文件
        - real_time: 实时传输到Unity (新增)
        
        Args:
            mode (str): 渲染模式
            filepath (str): 文件路径（txt模式使用）
            tacview: tacview对象（兼容性保留）
        """
        # 生成当前帧的数据
        timestamp = self.current_step * self.time_interval
        frame_data = [f"#{timestamp:.2f}"]
        
        # 收集所有仿真器的日志数据
        for sim in self._jsbsims.values():
            log_msg = sim.log()
            if log_msg is not None:
                frame_data.append(log_msg)
        
        for sim in self._tempsims.values():
            log_msg = sim.log()
            if log_msg is not None:
                frame_data.append(log_msg)
        
        # 根据模式处理数据
        if mode == "txt":
            # 文件输出模式（保持原有功能）
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                for line in frame_data:
                    f.write(line + "\n")
                    
        elif mode == "real_time1":
            # 实时传输模式（新功能）
            if self.enable_real_time_transmission and self.real_time_transmitter:
                if self.real_time_transmitter.is_connected():
                    success = self.real_time_transmitter.send_frame_data(frame_data)
                    if not success:
                        logging.warning(f"实时传输失败 - 步骤 {self.current_step}")
                else:
                    logging.debug("Unity未连接，跳过实时传输")
            else:
                logging.warning("实时传输未启用或传输器未初始化")
                
            # 如果有tacview参数，保持原有的tacview功能
            if tacview is not None:
                data_str = "\n".join(frame_data) + "\n"  # 修复：使用frame_data而不是未定义的data
                tacview.send_data_to_client(data_str)
                
        elif mode == "human":
            # 终端输出模式
            print("\n".join(frame_data))
            
        else:
            raise NotImplementedError(f"不支持的渲染模式: {mode}")

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """Pack seperated key-value dict into grouped np.ndarray"""
        ego_data = np.array([data[uid] for uid in self.ego_ids])
        enm_data = np.array([data[uid] for uid in self.enm_ids])
        if enm_data.shape[0] > 0:
            data = np.concatenate((ego_data, enm_data))  # type: np.ndarray
        else:
            data = ego_data  # type: np.ndarray
        try:
            assert np.isnan(data).sum() == 0
        except AssertionError:
            import pdb
            pdb.set_trace()
        # only return data that belongs to RL agents
        return data[:self.num_agents, ...]

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """Unpack grouped np.ndarray into seperated key-value dict"""
        assert isinstance(data, (np.ndarray, list, tuple)) and len(data) == self.num_agents
        # unpack data in the same order to packing process
        unpack_data = dict(zip((self.ego_ids + self.enm_ids)[:self.num_agents], data))
        # fill in None for other not-RL agents
        for agent_id in (self.ego_ids + self.enm_ids)[self.num_agents:]:
            unpack_data[agent_id] = None
        return unpack_data