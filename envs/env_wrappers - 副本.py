"""
基于OpenAI Baselines简化的多进程环境并行化框架，支持单/多智能体场景
"""

import os
import contextlib
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

# region 序列化工具类
class CloudpickleWrapper(object):
    """
    使用cloudpickle进行对象序列化（解决multiprocessing模块pickle功能限制问题）
    适用于需要跨进程传递复杂对象（如lambda函数、类实例等）的场景
    """
    def __init__(self, x):
        self.x = x  # 需要序列化的对象

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)  # 序列化为字节流

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)  # 反序列化恢复对象
# endregion

# region MPI环境处理
@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    清除MPI环境变量的上下文管理器
    防止子进程继承父进程的MPI配置导致冲突（常见于HPC环境）
    """
    removed_env = {}
    # 过滤OMPI_和PMI_开头的环境变量
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_env[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_env)  # 恢复原始环境变量
# endregion

# region 基础向量化环境接口
class VecEnv(ABC):
    """
    抽象向量化环境基类，定义多环境并行操作接口
    实现要点：
    - 支持异步环境步进(step_async + step_wait)
    - 统一处理多个环境的观测/动作空间
    - 提供同步模式兼容接口(step)
    """
    closed = False  # 环境关闭状态标记

    def __init__(self, num_envs, obs_space, act_space):
        self.num_envs = num_envs          # 并行环境数量
        self.observation_space = obs_space  # 观测空间定义
        self.action_space = act_space     # 动作空间定义

    @abstractmethod
    def reset(self): 
        """重置所有环境并返回批量观测"""
        pass

    @abstractmethod
    def step_async(self, actions):
        """异步执行环境步进（非阻塞）"""
        pass

    @abstractmethod
    def step_wait(self):
        """等待异步步进完成并获取结果"""
        pass

    def close(self):
        """安全关闭环境及相关资源"""
        if not self.closed:
            self.close_extras()
            self.closed = True

    def step(self, actions):
        """同步执行环境步进（兼容模式）"""
        self.step_async(actions)
        return self.step_wait()
# endregion

# region 同步向量化环境实现
class DummyVecEnv(VecEnv):
    """
    同步向量化环境实现（单进程顺序执行）
    适用场景：
    - 调试环境逻辑
    - 环境计算开销小
    - 需要渲染可视化
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]  # 创建多个环境实例
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
        self.actions = None
        self.num_agents = getattr(env, "num_agents", 1)  # 支持多智能体场景

    def step_async(self, actions):
        """缓存动作以待执行"""
        self.actions = actions

    def step_wait(self):
        """同步执行所有环境步进"""
        results = [env.step(a) for a, env in zip(self.actions, self.envs)]
        # 解包结果并处理环境终止
        obss, rews, dones, infos = map(list, zip(*results))
        for i, done in enumerate(dones):
            # 处理不同类型done标记的复位逻辑
            if isinstance(done, (bool, np.ndarray, dict)) and np.all(done):
                obss[i] = self.envs[i].reset()
        self.actions = None
        # 将结果展平为批量数据
        return self._flatten(obss), self._flatten(rews), self._flatten(dones), np.array(infos)

    def reset(self):
        """批量重置环境"""
        return self._flatten([env.reset() for env in self.envs])

    @classmethod
    def _flatten(cls, data):
        """将多环境数据转换为批量格式"""
        if isinstance(data[0], dict):
            return {k: np.stack([d[k] for d in data]) for k in data[0]}  # 字典空间处理
        return np.stack(data)  # 常规数组空间处理
# endregion

# region 多进程工作进程
def worker(remote: Connection, parent_remote: Connection, env_fn_wrappers):
    """
    子进程工作函数（核心通信逻辑）
    参数：
    - remote: 子进程端连接
    - parent_remote: 父进程端连接（在子进程中需关闭）
    - env_fn_wrappers: 经过封装的云序列化环境创建函数
    """
    def step_env(env, action):
        """执行单环境步进并处理终止状态"""
        obs, rew, done, info = env.step(action)
        if np.all(done):  # 支持多种done类型判断
            obs = env.reset()
        return obs, rew, done, info

    parent_remote.close()  # 关闭父进程端连接
    envs = [fn() for fn in env_fn_wrappers.x]  # 反序列化创建环境实例

    try:
        while True:
            cmd, data = remote.recv()  # 等待主进程指令
            if cmd == 'step':
                # 执行批量环境步进
                remote.send([step_env(env, a) for env, a in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                # 返回环境空间定义（用于主进程初始化）
                remote.send(CloudpickleWrapper((
                    envs[0].observation_space, 
                    envs[0].action_space
                )))
            elif cmd == 'get_num_agents':
                # 多智能体数量支持
                remote.send(CloudpickleWrapper(getattr(envs[0], "num_agents", 1)))
    finally:
        [env.close() for env in envs]  # 确保环境资源释放
# endregion

# region 异步多进程向量化环境
class SubprocVecEnv(VecEnv):
    """
    多进程并行向量化环境实现
    优势：
    - 充分利用多核CPU资源
    - 避免GIL锁对性能的影响
    实现要点：
    - 使用Pipe进行进程间通信
    - 支持批量环境分组并行（in_series参数）
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        self.in_series = in_series  # 单进程连续运行的环境数
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "环境数量必须能被分组数整除"
        
        # 创建进程间通信管道
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs//in_series)])
        # 启动子进程
        self.ps = [
            Process(
                target=worker,
                args=(work, remote, CloudpickleWrapper(fns)),
                daemon=True  # 主进程终止时自动结束子进程
            ) for (work, remote, fns) in zip(
                self.work_remotes, 
                self.remotes, 
                np.array_split(env_fns, len(self.remotes))
            )
        ]
        for p in self.ps:
            with clear_mpi_env_vars():  # 清除MPI环境变量
                p.start()
        [r.close() for r in self.work_remotes]  # 关闭工作端冗余连接

        # 初始化环境参数
        self.remotes[0].send(('get_spaces', None))
        obs_space, act_space = self.remotes[0].recv().x
        super().__init__(nenvs, obs_space, act_space)

    def step_async(self, actions):
        """分发动作到各子进程"""
        actions = np.array_split(actions, len(self.remotes))
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        """收集各子进程执行结果"""
        results = [remote.recv() for remote in self.remotes]
        # 展平分组结果 [[env1,env2], [env3,env4]] -> [env1,env2,env3,env4]
        results = self._flatten_series(results)  
        obss, rews, dones, infos = zip(*results)
        return self._flatten(obss), self._flatten(rews), self._flatten(dones), np.array(infos)
# endregion

# region 多智能体扩展实现
class ShareVecEnv(VecEnv):
    """
    多智能体向量化环境基类
    新增特性：
    - 支持共享观测空间(share_observation_space)
    - 处理智能体间的协同观测
    """
    def __init__(self, num_envs, obs_space, share_obs_space, act_space):
        super().__init__(num_envs, obs_space, act_space)
        self.share_observation_space = share_obs_space  # 共享观测空间定义

class ShareSubprocVecEnv(SubprocVecEnv, ShareVecEnv):
    """
    多智能体多进程实现
    扩展功能：
    - 在step/reset中返回共享观测
    - 支持多智能体环境参数传递
    """
    def step_wait(self):
        """处理含共享观测的多智能体数据"""
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)
        obs, share_obs, rews, dones, infos = zip(*results)
        return (
            self._flatten(obs), 
            self._flatten(share_obs),
            self._flatten(rews),
            self._flatten(dones), 
            np.array(infos)
        )
# endregion