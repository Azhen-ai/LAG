import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask
from ..human_task.HumanSingleCombatTask import HumanSingleCombatTask

class SingleCombatEnv(BaseEnv):
    """
    1v1对抗型强化学习环境，支持多种空战任务配置
    
    Attributes:
        agents (Dict[str, AircraftSimulator]): 包含两个对抗方飞行器实例的字典
        init_states (List[dict]): 初始状态备份，用于环境重置时的随机化
    
    Methods:
        load_task: 根据配置动态加载具体任务
        reset: 重置环境状态并返回初始观测
        reset_simulators: 随机化飞行器初始状态
    
    Example:
        >>> env = SingleCombatEnv("config/aircombat.yaml")
        >>> obs = env.reset()
        >>> action = [0.5, 0.3]  # 示例动作
        >>> next_obs, reward, done, info = env.step(action)
    """
    def __init__(self, config_name: str):
        """
        初始化1v1对抗环境
        
        Args:
            config_name (str): 配置文件路径，包含任务定义和飞行器参数
            
        Raises:
            AssertionError: 当配置的智能体数量不等于2时触发
        """
        super().__init__(config_name)
        # 环境特性校验（强制1v1场景）
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} 仅支持1v1对抗场景！"
        self.init_states = None  # 延迟初始化，首次reset时生成

    def load_task(self):
        """
        动态加载任务模块，支持7种空战任务：
        
        - singlecombat: 基础空战对抗
        - hierarchical_singlecombat: 分层决策空战
        - singlecombat_dodge_missile: 导弹规避专项训练
        - singlecombat_shoot: 导弹攻击专项训练
        - hierarchical_singlecombat_dodge_missile: 分层导弹规避
        - hierarchical_singlecombat_shoot: 分层导弹攻击
        - HumanSingleCombat: 人类演示模式
        
        Raises:
            NotImplementedError: 未知任务类型时触发
        """
        taskname = getattr(self.config, 'task', None)
        # 多分支任务加载器
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'HumanSingleCombat':
            self.task = HumanSingleCombatTask(self.config)
        else:
            raise NotImplementedError(f"未知任务类型: {taskname}")

    def reset(self) -> np.ndarray:
        """
        重置环境状态，包含三个核心步骤：
        
        1. 重置步数计数器
        2. 随机化飞行器初始状态
        3. 重置任务相关状态
        
        Returns:
            np.ndarray: 打包后的多智能体观测矩阵，形状为(2, obs_dim)
        """
        self.current_step = 0  # 重置步数计时器
        self.reset_simulators()  # 飞行器状态随机化
        self.task.reset(self)    # 任务指标重置
        obs = self.get_obs()     # 获取原始观测字典
        return self._pack(obs)   # 转换为标准numpy数组

    def reset_simulators(self):
        """
        飞行器状态随机化策略：
        
        1. 首次运行时备份初始状态
        2. 每次重置时打乱初始状态顺序
        3. 重新加载随机化后的状态到仿真器
        
        注：注释掉的代码段展示航向(270-540度)和高度(17000-23000英尺)的随机化示例
        """
        # 延迟初始化备份（仅首次运行）
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        
        # 状态随机化策略示例（可根据任务需求启用）
        # self.init_states[0].update({
        #     'ic_psi_true_deg': (self.np_random.uniform(270, 540)) % 360,  # 随机初始航向
        #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),            # 随机初始高度
        # })
        
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)  # 打乱初始状态顺序
        
        # 加载随机化状态到每个仿真器
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])  # 重新初始化物理引擎
            
        self._tempsims.clear()  # 清除临时物体（如已发射导弹）