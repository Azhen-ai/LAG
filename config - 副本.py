import argparse
from tokenize import group

def get_config():
    """主配置解析器入口：整合所有子模块的配置参数"""
    # 使用RawDescriptionHelpFormatter保留帮助信息的原始格式
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # 按功能模块分层添加配置参数（增强可维护性）[6,8](@ref)
    parser = _get_prepare_config(parser)        # 基础训练准备参数
    parser = _get_replaybuffer_config(parser)  # 经验回放缓冲区参数
    parser = _get_network_config(parser)       # 神经网络结构参数
    parser = _get_recurrent_config(parser)      # 循环神经网络参数
    parser = _get_optimizer_config(parser)      # 优化器参数
    parser = _get_ppo_config(parser)            # PPO算法超参数
    parser = _get_selfplay_config(parser)      # 自博弈训练参数
    parser = _get_save_config(parser)           # 模型保存参数
    parser = _get_log_config(parser)            # 日志记录参数
    parser = _get_eval_config(parser)          # 评估策略参数
    parser = _get_render_config(parser)        # 可视化渲染参数
    return parser

def _get_prepare_config(parser: argparse.ArgumentParser):
    """训练准备参数组：控制实验基本设置和硬件配置"""
    group = parser.add_argument_group("Prepare parameters")  # 参数分组提升可读性[6](@ref)
    
    # 核心实验配置
    group.add_argument("--env-name", type=str, default='JSBSim',
                     help="指定训练环境名称（如JSBSim飞行模拟器）")
    group.add_argument("--algorithm-name", type=str, default='ppo', choices=["ppo", "mappo"],
                     help="选择强化学习算法（默认PPO）[6](@ref)")
    group.add_argument("--experiment-name", type=str, default="check",
                     help="实验标识符，用于区分不同实验的日志和输出")
    
    # 可复现性配置
    group.add_argument("--seed", type=int, default=1,
                     help="设置Numpy/PyTorch的随机种子（确保实验可复现）")
    
    # 硬件加速配置
    group.add_argument("--cuda", action='store_true', default=False,
                     help="启用GPU加速（默认使用CPU）[5](@ref)")
    group.add_argument("--n-training-threads", type=int, default=1,
                     help="PyTorch训练线程数（影响CPU并行计算效率）")
    
    # 环境并行配置
    group.add_argument("--n-rollout-threads", type=int, default=4,
                     help="训练时并行环境实例数量（影响数据收集速度）")
    group.add_argument("--num-env-steps", type=float, default=1e7,
                     help="总训练步数（控制训练时长）")
    
    # 模型与监控配置
    group.add_argument("--model-dir", type=str, default=None,
                     help="预训练模型加载路径（用于迁移学习或断点续训）")
    group.add_argument("--use-wandb", action='store_true', default=False,
                     help="启用Weights & Biases实验跟踪（需配置API Key）[5](@ref)")
    group.add_argument("--user-name", type=str, default='liuqh',
                     help="设置进程名称标识（便于系统监控）")
    return parser

def _get_replaybuffer_config(parser: argparse.ArgumentParser):
    """经验回放参数组：控制价值估计和优势计算"""
    group = parser.add_argument_group("Replay Buffer parameters")
    
    # 折扣因子设置
    group.add_argument("--gamma", type=float, default=0.99,
                     help="奖励折扣因子（影响未来奖励的现值计算）[7](@ref)")
    
    # 优势估计配置
    group.add_argument("--use-gae", action='store_false', default=True,
                     help="禁用广义优势估计（GAE）（默认启用）[7](@ref)")
    group.add_argument("--gae-lambda", type=float, default=0.95,
                     help="GAE的λ参数（平衡偏差与方差）")
    
    # 缓冲区管理
    group.add_argument("--buffer-size", type=int, default=200,
                     help="经验回放缓冲区的最大容量（影响样本多样性）")
    return parser

def _get_network_config(parser: argparse.ArgumentParser):
    """神经网络架构参数组：控制策略和价值网络的拓扑结构"""
    group = parser.add_argument_group("Network parameters")
    
    # 网络结构配置
    group.add_argument("--hidden-size", type=str, default='128 128',
                     help="MLP隐藏层维度（多个层用空格分隔，如'256 128'）[8](@ref)")
    group.add_argument("--activation-id", type=int, default=1,
                     help="激活函数选择（0:Tanh,1:ReLU,2:LeakyReLU,3:ELU）")
    
    # 归一化配置
    group.add_argument("--use-feature-normalization", action='store_true', default=False,
                     help="在输入层应用LayerNorm（提升训练稳定性）")
    return parser

def _get_ppo_config(parser: argparse.ArgumentParser):
    """PPO算法超参数组：控制策略优化的核心参数"""
    group = parser.add_argument_group("PPO parameters")
    
    # 策略优化配置
    group.add_argument("--ppo-epoch", type=int, default=10,
                     help="每次采样数据重复使用的优化轮次")
    group.add_argument("--clip-param", type=float, default=0.2,
                     help="策略梯度剪切阈值（限制策略更新幅度）[7](@ref)")
    
    # 损失函数权重
    group.add_argument("--value-loss-coef", type=float, default=1,
                     help="价值函数损失的权重系数")
    group.add_argument("--entropy-coef", type=float, default=0.01,
                     help="策略熵正则项的权重系数（鼓励探索）")
    
    # 梯度管理
    group.add_argument("--max-grad-norm", type=float, default=2,
                     help="梯度裁剪的最大范数（防止梯度爆炸）")
    return parser

def _get_selfplay_config(parser: argparse.ArgumentParser):
    """自博弈训练参数组：控制对抗训练的对手策略管理"""
    group = parser.add_argument_group("Selfplay parameters")
    
    group.add_argument("--use-selfplay", action='store_true', default=False,
                     help="启用自博弈训练模式（如AlphaGo Zero风格训练）")
    group.add_argument("--selfplay-algorithm", type=str, default='sp', choices=["sp", "fsp", "pfsp"],
                     help="选择自博弈算法：sp-标准自博弈, fsp-种群训练")
    group.add_argument('--n-choose-opponents', type=int, default=1,
                     help="每次训练迭代选择的对手策略数量")
    return parser

# （以下省略部分参数组的注释，结构类似）

if __name__ == "__main__":
    # 配置解析器初始化
    parser = get_config()
    # 解析命令行参数（自动处理-h帮助信息）[1,6](@ref)
    all_args = parser.parse_args()
    # 示例：打印解析后的参数命名空间
    print("当前配置参数：", all_args)