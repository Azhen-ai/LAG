import torch
import torch.nn as nn
from typing import Union, List
from .ppo_policy import PPOPolicy  # 导入PPO策略模型
from ..utils.buffer import ReplayBuffer  # 经验回放缓冲区
from ..utils.utils import check, get_gard_norm  # 实用工具函数


class PPOTrainer():
    """PPO算法训练器，负责执行策略的优化更新"""
    
    def __init__(self, args, device=torch.device("cpu")):
        """初始化训练器配置
        
        Args:
            args: 包含训练超参数的配置对象
            device: 计算设备，默认CPU
        """
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)  # 张量配置字典
        
        # region PPO算法超参数
        self.ppo_epoch = args.ppo_epoch  # 每次数据采样的训练轮数
        self.clip_param = args.clip_param  # 策略损失裁剪系数
        self.use_clipped_value_loss = args.use_clipped_value_loss  # 是否裁剪价值损失
        self.num_mini_batch = args.num_mini_batch  # 小批量更新次数
        self.value_loss_coef = args.value_loss_coef  # 价值损失权重
        self.entropy_coef = args.entropy_coef  # 熵正则项权重
        self.use_max_grad_norm = args.use_max_grad_norm  # 是否使用梯度裁剪
        self.max_grad_norm = args.max_grad_norm  # 最大梯度范数阈值
        
        # region RNN相关配置
        self.use_recurrent_policy = args.use_recurrent_policy  # 是否使用循环策略
        self.data_chunk_length = args.data_chunk_length  # RNN数据块长度

    def ppo_update(self, policy: PPOPolicy, sample: tuple) -> tuple:
        """执行单次PPO参数更新
        
        Args:
            policy: PPO策略模型，包含Actor和Critic
            sample: 从缓冲区采样的训练数据，包含：
                obs_batch: 观察值             [batch_size, obs_dim]
                actions_batch: 动作           [batch_size, act_dim]
                masks_batch: 终止状态掩码      [batch_size, 1]
                old_action_log_probs_batch: 旧策略的动作对数概率 [batch_size, 1]
                advantages_batch: 优势估计     [batch_size, 1]
                returns_batch: 回报值          [batch_size, 1]
                value_preds_batch: 旧价值预测   [batch_size, 1]
                rnn_states_actor_batch: Actor的RNN状态
                rnn_states_critic_batch: Critic的RNN状态
                
        Returns:
            包含各项损失值和指标的元组:
            policy_loss, value_loss, entropy_loss, ratio, actor_grad_norm, critic_grad_norm
        """
        # 解包样本数据并转换到指定设备
        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # region 评估当前策略的动作概率和状态价值
        values, action_log_probs, dist_entropy = policy.evaluate_actions(
            obs_batch,
            rnn_states_actor_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch
        )
        # endregion

        # region 计算策略损失（Clipped Surrogate Loss）
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)  # 新旧策略概率比
        surr1 = ratio * advantages_batch  # 未裁剪的损失项
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch  # 裁剪后的损失项
        policy_loss = -torch.min(surr1, surr2).mean()  # 取最小值确保更新方向正确
        # endregion

        # region 计算价值函数损失
        if self.use_clipped_value_loss:
            # 裁剪价值预测防止剧烈变化
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)  # 原始损失
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)  # 裁剪后的损失
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()  # 取最大值作为保守估计
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2).mean()  # 普通MSE损失
        # endregion

        # region 计算熵正则项
        policy_entropy_loss = -dist_entropy.mean()  # 鼓励策略多样性
        # endregion

        # 总损失 = 策略损失 + 价值损失 * 权重 + 熵正则 * 权重
        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

        # region 反向传播与参数更新
        policy.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        
        # 梯度裁剪防止爆炸
        if self.use_max_grad_norm:
            # 对Actor和Critic分别进行梯度裁剪
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            # 仅计算梯度范数不做裁剪
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        
        policy.optimizer.step()  # 更新参数
        # endregion

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]) -> dict:
        """执行完整训练周期
        
        Args:
            policy: 待训练的PPO策略
            buffer: 经验回放缓冲区，可能包含多个缓冲区实例
            
        Returns:
            包含平均训练指标的字典，包括各项损失和梯度范数
        """
        # 初始化训练指标统计字典
        train_info = {
            'value_loss': 0,
            'policy_loss': 0,
            'policy_entropy_loss': 0,
            'actor_grad_norm': 0,
            'critic_grad_norm': 0,
            'ratio': 0
        }

        # 按配置的epoch数进行多次数据遍历
        for _ in range(self.ppo_epoch):
            # 生成数据迭代器（当前仅支持循环策略的时序数据块）
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(
                    buffer, 
                    self.num_mini_batch, 
                    self.data_chunk_length  # 将长序列切分为固定长度的块
                )
            else:
                raise NotImplementedError("非循环策略的数据生成器尚未实现")

            # 遍历所有小批量数据
            for sample in data_generator:
                # 执行参数更新并获取指标
                policy_loss, value_loss, entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                # 累加训练指标
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        # 计算平均指标（总更新次数 = epoch数 * 小批量数）
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info