import time
import torch
import logging
import numpy as np
from typing import List
from .base_runner import Runner, ReplayBuffer

def _t2n(x):
    """将PyTorch张量转换为Numpy数组的工具函数"""
    return x.detach().cpu().numpy()

class JSBSimRunner(Runner):
    """JSBSim环境专用的强化学习运行器，继承自基础Runner类"""
    
    def load(self):
        """初始化环境、策略模型和经验回放缓冲区"""
        # 从环境中获取观测和动作空间信息
        self.obs_space = self.envs.observation_space  # 观测空间维度
        self.act_space = self.envs.action_space       # 动作空间维度
        self.num_agents = self.envs.num_agents        # 多智能体数量
        self.use_selfplay = self.all_args.use_selfplay # 是否使用自博弈模式

        # 策略和算法初始化（**核心组件加载**）
        if self.algorithm_name == "ppo":
            # PPO算法相关组件导入
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)  # 策略网络
        self.trainer = Trainer(self.all_args, device=self.device)  # 训练器

        # 初始化经验回放缓冲区（**数据存储核心**）
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)

        # 模型恢复（断点续训）
        if self.model_dir is not None:
            self.restore()

    def run(self):
        """主训练循环"""
        # 数据收集与交互阶段
        self.warmup()  # 环境预热初始化

        start = time.time()
        self.total_num_steps = 0  # 总训练步数计数器
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads  # 计算总训练轮次

        for episode in range(episodes):
            heading_turns_list = []  # 记录飞行器转向次数（**自定义指标**）

            # 数据收集阶段
            for step in range(self.buffer_size):
                # 策略采样（**动作生成**）
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step) #使用collect()方法生成动作

                # 环境交互（**关键步骤**）
                obs, rewards, dones, infos = self.envs.step(actions)

                # 记录自定义指标
                for info in infos:
                    if 'heading_turn_counts' in info:
                        heading_turns_list.append(info['heading_turn_counts'])

                # 数据打包
                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # 存入缓冲区（**经验存储**）
                self.insert(data)

            # 策略更新阶段（**参数优化**）
            self.compute()  # 计算回报
            train_infos = self.train()  # 执行梯度更新

            # 日志记录
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads
            if episode % self.log_interval == 0:
                # 计算性能指标
                end = time.time()
                # 训练指标打印（**关键输出**）
                logging.info(f"\n Scenario {self.all_args.scenario_name} Algo {self.algorithm_name}...")
                
                # 计算平均奖励
                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                logging.info(f"Average episode rewards: {train_infos['average_episode_rewards']}")
                
                # 记录转向次数（**自定义指标处理**）
                if heading_turns_list:
                    train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                    logging.info(f"Average heading turns: {train_infos['average_heading_turns']}")
                self.log_info(train_infos, self.total_num_steps)  # 写入日志系统

            # 模型评估（**策略验证阶段**）
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.total_num_steps)

            # 模型保存（**断点保存**）
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

    def warmup(self):
        """环境预热初始化"""
        obs = self.envs.reset()  # 重置环境调用JSBSim的run_ic()初始化仿真条件[1](@ref)
        self.buffer.step = 0     # 缓冲指针归零
        self.buffer.obs[0] = obs.copy()  # 初始观测存储

    @torch.no_grad()
    def collect(self, step):
        """策略采样：生成动作和状态估值"""
        self.policy.prep_rollout()  # 设置策略为评估模式
        
        # 从策略网络获取动作（**核心采样逻辑**）
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.policy.get_actions(
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states_actor[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )
        
        # 数据分块处理（**多线程数据处理**）
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        """将交互数据插入经验缓冲区"""
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        # 环境终止标志处理（**关键状态重置逻辑**）
        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)  # 判断整个环境是否终止
        
        # RNN状态重置（**序列建模关键**）
        rnn_states_actor[dones_env] = np.zeros_like(rnn_states_actor[dones_env])
        rnn_states_critic[dones_env] = np.zeros_like(rnn_states_critic[dones_env])

        # 生成mask矩阵（**用于计算折扣因子**）使用mask矩阵区分有效/无效数据，避免终止状态影响序列建模
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env] = 0.0  # 终止环境的mask设为0

        # 执行插入操作
        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        """策略评估方法"""
        logging.info("\nStart evaluation...")
        total_episodes = 0
        eval_episode_rewards = []
        eval_cumulative_rewards = np.zeros_like(self.buffer.rewards)  # 累计奖励初始化

        # 评估环境初始化（**与训练环境独立**）
        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones_like(self.buffer.masks)
        eval_rnn_states = np.zeros_like(self.buffer.rnn_states_actor)

        self.timestamp = 0  # Tacview实时渲染时间戳

        while total_episodes < self.eval_episodes:
            # 策略动作生成（**确定性策略**）
            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True  # 评估使用确定性策略
            )
            
            # 数据分块处理
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # 环境交互（**评估步进**）
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            # Tacview实时渲染（**可视化模块**）
            if self.render_mode == "real_time" and self.tacview:
                render_data = [f"#{self.timestamp:.2f}\n"]
                for sim in self.eval_envs.envs[0]._jsbsims.values():
                    log_msg = sim.log()  # 获取JSBSim仿真数据
                    if log_msg:
                        render_data.append(log_msg + "\n")
                try:
                    self.tacview.send_data_to_client("".join(render_data))  # 发送渲染数据
                except Exception as e:
                    logging.error(f"Tacview渲染异常: {e}")
            self.timestamp += 0.2  # 时间步进0.2秒

            # 奖励累计和回合统计
            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            
            # 终止环境处理
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env])
            eval_cumulative_rewards[eval_dones_env] = 0  # 重置累计奖励

            # 状态重置（**与训练逻辑一致**）
            eval_masks[eval_dones_env] = 0.0
            eval_rnn_states[eval_dones_env] = 0.0

        # 评估结果处理
        eval_infos = {
            'eval_average_episode_rewards': np.concatenate(eval_episode_rewards).mean(axis=1)
        }
        logging.info(f"评估平均奖励: {np.mean(eval_infos['eval_average_episode_rewards'])}")
        self.log_info(eval_infos, total_num_steps)  # 记录评估指标

    @torch.no_grad()
    def render(self):
        """环境渲染方法（演示模式）"""
        logging.info(f"\n启动{self.render_mode}模式渲染...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones_like(self.buffer.masks)
        render_rnn_states = np.zeros_like(self.buffer.rnn_states_actor)
        
        # 初始化渲染文件
        self.envs.render(
            mode=self.render_mode,
            filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi',
            tacview=self.tacview
        )

        while True:
            # 生成确定性动作
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(
                np.concatenate(render_obs),
                np.concatenate(render_rnn_states),
                np.concatenate(render_masks),
                deterministic=True
            )
            
            # 环境步进
            render_obs, render_rewards, render_dones, _ = self.envs.step(render_actions)
            
            # 自博弈模式奖励处理
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2]
                
            render_episode_rewards += render_rewards
            
            # 实时渲染更新
            self.envs.render(
                mode='txt',
                filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi'
            )
            
            if render_dones.all():
                break  # 所有环境终止时退出

        # 渲染结果记录
        render_infos = {'render_episode_reward': render_episode_rewards}
        logging.info(f"演示回合奖励: {render_infos['render_episode_reward']}")

    def save(self, episode):
        """模型保存方法"""
        # Actor网络保存
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, f'{self.save_dir}/actor_latest.pt')
        
        # Critic网络保存
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, f'{self.save_dir}/critic_latest.pt')