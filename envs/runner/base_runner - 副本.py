import os
import sys
import wandb
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # 添加项目根目录到系统路径
from algorithms.utils.buffer import ReplayBuffer
import logging

def _t2n(x):
    """将PyTorch张量转换为Numpy数组的工具函数"""
    return x.detach().cpu().numpy()

class Runner(object):
    """强化学习训练流程基础运行器（需子类实现具体逻辑）"""
    
    def __init__(self, config):
        """
        初始化基础运行器
        Args:
            config: 配置字典，包含以下关键内容:
                - all_args: 所有命令行参数
                - envs: 训练环境对象
                - eval_envs: 评估环境对象
                - device: 计算设备 (cpu/gpu)
                - run_dir: 运行结果存储目录
        """
        # 基础配置初始化
        self.all_args = config['all_args']          # 所有配置参数
        self.envs = config['envs']                  # 并行训练环境
        self.eval_envs = config['eval_envs']        # 并行评估环境
        self.device = config['device']              # 计算设备
        self.render_mode = config['render_mode']    # 渲染模式
        
        # Tacview实时渲染初始化
        self.tacview = None
        if self.render_mode == "real_time":
            from runner.tacview import Tacview
            self.tacview = Tacview()  # 创建Tacview连接对象

        # 参数解包（来自命令行参数）
        self.env_name = self.all_args.env_name                    # 环境名称
        self.algorithm_name = self.all_args.algorithm_name       # 算法名称
        self.experiment_name = self.all_args.experiment_name      # 实验名称
        self.num_env_steps = int(self.all_args.num_env_steps)     # 总环境步数
        self.n_rollout_threads = self.all_args.n_rollout_threads  # 并行训练线程数
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads  # 并行评估线程数
        self.buffer_size = self.all_args.buffer_size              # 经验回放缓冲区大小
        self.use_wandb = self.all_args.use_wandb                  # 是否使用Wandb记录

        # 流程控制参数
        self.save_interval = self.all_args.save_interval  # 模型保存间隔
        self.log_interval = self.all_args.log_interval    # 日志记录间隔
        self.use_eval = self.all_args.use_eval            # 是否启用评估
        self.eval_interval = self.all_args.eval_interval  # 评估间隔
        self.eval_episodes = self.all_args.eval_episodes  # 每次评估回合数

        # 路径配置
        self.model_dir = self.all_args.model_dir  # 模型加载目录
        self.run_dir = config["run_dir"]          # 运行结果根目录
        
        # 创建模型保存目录
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)    # 使用Wandb自动创建的目录
        else:
            self.save_dir = str(self.run_dir)     # 使用自定义目录
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.load()  # 加载算法组件

    def load(self):
        """加载算法组件（策略网络+训练器）"""
        # 动态导入算法模块
        if self.algorithm_name == "ppo":
            from ..algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from ..algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError(f"不支持的算法: {self.algorithm_name}")
        
        # 初始化策略网络
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space,  # 从环境获取观测空间维度
            self.envs.action_space,       # 从环境获取动作空间维度
            device=self.device
        )
        
        # 初始化训练器
        self.trainer = Trainer(self.all_args, self.policy, device=self.device)

        # 初始化经验回放缓冲区
        self.buffer = ReplayBuffer(
            self.all_args,
            self.envs.observation_space,
            self.envs.action_space
        )

        # 加载预训练模型（如果提供目录）
        if self.model_dir is not None:
            self.restore()

    def run(self):
        """主训练流程（需子类实现）"""
        raise NotImplementedError

    def warmup(self):
        """环境预热初始化（需子类实现）"""
        raise NotImplementedError

    def collect(self, step):
        """经验收集（需子类实现）"""
        raise NotImplementedError

    def rollout(self):
        """策略部署（需子类实现）"""
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """计算GAE优势估计和折扣回报"""
        self.policy.prep_rollout()  # 设置策略为评估模式
        
        # 计算最后一步的观测值
        next_values = self.policy.get_values(
            np.concatenate(self.buffer.obs[-1]),          # 拼接所有线程的最终观测
            np.concatenate(self.buffer.rnn_states_critic[-1]),  # RNN隐藏状态
            np.concatenate(self.buffer.masks[-1])        # 序列终止mask
        )
        
        # 将值函数分割为各线程数据
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        
        # 计算回报和优势估计
        self.buffer.compute_returns(next_values)

    def train(self):
        """执行策略优化"""
        self.policy.prep_training()  # 设置策略为训练模式
        # 调用训练器进行参数更新
        train_infos = self.trainer.train(self.policy, self.buffer)
        # 清空已处理数据
        self.buffer.after_update()
        return train_infos

    def save(self):
        """保存模型参数"""
        # 保存Actor网络
        policy_actor = self.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_latest.pt")
        # 保存Critic网络
        policy_critic = self.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_latest.pt")

    def restore(self):
        """加载预训练模型"""
        # 加载Actor参数
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        # 加载Critic参数
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_info(self, infos, total_num_steps):
        """记录训练信息到Wandb"""
        if self.use_wandb:
            for k, v in infos.items():
                wandb.log({k: v}, step=total_num_steps)  # 按训练步数记录指标
        
    def render_with_tacview(self, data):
        """
        Tacview实时渲染接口
        :param data: 需要渲染的数据，支持字符串或结构化数据
        """
        if self.tacview:
            try:
                self.tacview.send_data_to_client(data)  # 通过UDP发送数据
            except Exception as e:
                logging.error(f"Tacview渲染错误: {e}")