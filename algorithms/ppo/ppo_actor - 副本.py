import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """PPO策略网络中的Actor部分，负责生成动作和评估动作概率
        
        Args:
            args: 命令行参数集合，包含网络配置参数
            obs_space: 观察空间维度信息
            act_space: 动作空间维度信息
            device: 运行设备，默认CPU
        """
        super(PPOActor, self).__init__()
        # region 网络参数初始化
        self.gain = args.gain  # 动作分布参数初始化增益
        self.hidden_size = args.hidden_size  # MLP基础网络隐藏层维度
        self.act_hidden_size = args.act_hidden_size  # 动作网络隐藏层维度
        self.activation_id = args.activation_id  # 激活函数类型标识
        self.use_feature_normalization = args.use_feature_normalization  # 是否使用输入归一化
        self.use_recurrent_policy = args.use_recurrent_policy  # 是否使用循环策略（GRU）
        self.recurrent_hidden_size = args.recurrent_hidden_size  # GRU隐藏层维度
        self.recurrent_hidden_layers = args.recurrent_hidden_layers  # GRU层数
        self.tpdv = dict(dtype=torch.float32, device=device)  # 张量设备与类型配置
        self.use_prior = args.use_prior  # 是否使用先验知识调整动作分布
        # endregion

        # region 网络模块构建
        # (1) 特征提取模块: 将原始观察转换为高级特征
        self.base = MLPBase(
            obs_space, 
            self.hidden_size,
            self.activation_id,
            self.use_feature_normalization
        )
        
        # (2) 循环网络模块: 处理时序依赖（仅在启用循环策略时构建）
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(
                input_size,
                self.recurrent_hidden_size,
                self.recurrent_hidden_layers
            )
            input_size = self.rnn.output_size  # 更新后续输入维度
            
        # (3) 动作生成模块: 基于特征生成动作分布
        self.act = ACTLayer(
            act_space, 
            input_size,
            self.act_hidden_size,
            self.activation_id,
            self.gain
        )
        # endregion

        self.to(device)  # 将网络移动到指定设备

    def forward(self, obs, rnn_states, masks, deterministic=False):
        """前向传播过程，生成动作及其对数概率
        
        Args:
            obs: 当前环境观察值，形状为(batch_size, obs_dim)
            rnn_states: GRU隐藏状态，形状为(num_layers, batch_size, hidden_size)
            masks: 用于重置RNN状态的掩码（通常与episode终止状态相关）
            deterministic: 是否采用确定性策略（不采样）
            
        Returns:
            actions: 生成的动作，形状取决于动作空间
            action_log_probs: 动作的对数概率
            rnn_states: 更新后的RNN隐藏状态
        """
        # 确保输入张量在正确的设备和类型上
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 先验知识处理（导弹发射控制逻辑）
        if self.use_prior:
            # 从观察中提取攻击角度（转换为角度制）和距离（单位转换为米）
            attack_angle = torch.rad2deg(obs[:, 11]) 
            distance = obs[:, 13] * 10000 
            
            # 初始化默认Beta分布参数
            alpha0 = torch.full(size=(obs.shape[0],1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0],1), fill_value=10).to(**self.tpdv)
            
            # 根据距离调整alpha参数（控制发射阈值）
            alpha0[distance<=12000] = 6
            alpha0[distance<=8000] = 10
            
            # 根据攻击角度调整beta参数（控制发射角度容差）
            beta0[attack_angle<=45] = 6
            beta0[attack_angle<=22.5] = 3

        # 特征提取
        actor_features = self.base(obs)

        # 时序特征处理（如果使用循环策略）
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # 动作生成（根据是否使用先验参数）
        if self.use_prior:
            actions, action_log_probs = self.act(
                actor_features, 
                deterministic,
                alpha0=alpha0,  # 传入调整后的alpha参数
                beta0=beta0     # 传入调整后的beta参数
            )
        else:
            actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        """评估给定动作的对数概率和分布熵
        
        Args:
            obs: 环境观察值
            rnn_states: GRU隐藏状态
            action: 待评估的动作
            masks: RNN状态重置掩码
            active_masks: 可选，用于屏蔽无效动作
            
        Returns:
            action_log_probs: 动作的对数概率
            dist_entropy: 动作分布的熵（用于正则化）
        """
        # 确保输入张量在正确的设备和类型上
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # 先验知识处理（与forward逻辑一致）
        if self.use_prior:
            attack_angle = torch.rad2deg(obs[:, 11])
            distance = obs[:, 13] * 10000
            alpha0 = torch.full(size=(obs.shape[0], 1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0], 1), fill_value=10).to(**self.tpdv)
            alpha0[distance<=12000] = 6
            alpha0[distance<=8000] = 10
            beta0[attack_angle<=45] = 6
            beta0[attack_angle<=22.5] = 3

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # 特征提取流程
        actor_features = self.base(obs)

        # 时序特征处理
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # 动作评估（根据是否使用先验参数）
        if self.use_prior:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features, 
                action, 
                active_masks,
                alpha0=alpha0,  # 传入调整后的alpha参数
                beta0=beta0     # 传入调整后的beta参数
            )
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features, 
                action, 
                active_masks
            )

        return action_log_probs, dist_entropy