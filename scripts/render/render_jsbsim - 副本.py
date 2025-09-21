#!/usr/bin/env python
# -*- coding: utf-8 -*-  # 声明文件编码格式（参考网页5中文注释支持）
"""
强化学习环境主入口脚本（JSBSim飞行模拟器集成）
功能包含：环境配置、参数解析、模型加载、训练/测试流程控制
"""
import sys
import os
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle

# 处理模块导入路径问题（参考网页6/7 JSBSim编译相关路径配置）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# 自定义模块导入
from config import get_config  # 配置文件解析器
from runner.share_jsbsim_runner import ShareJSBSimRunner  # 多智能体运行器
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv  # 不同战斗场景环境（参考网页6/7 JSBSim场景）
from envs.env_wrappers import DummyVecEnv, ShareDummyVecEnv  # 环境向量化封装

def make_render_env(all_args):
    """创建渲染环境的工厂函数（参考网页6/7 JSBSim仿真环境配置）"""
    def get_env_fn(rank):
        def init_env():
            # 根据参数选择不同战斗场景（参考网页6/7 JSBSim场景类型）
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)  # 设置环境随机种子
            return env
        return init_env
    
    # 选择环境向量化方式（参考网页7多线程编译配置）
    return ShareDummyVecEnv([get_env_fn(0)]) if all_args.env_name == "MultipleCombat" else DummyVecEnv([get_env_fn(0)])

def parse_args(args, parser):
    """解析JSBSim环境特定参数（参考网页6/7仿真参数配置）"""
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--episode-length', type=int, default=1000,
                       help="单个训练回合的最大步数")
    group.add_argument('--scenario-name', type=str, default='singlecombat_vsbaseline',
                       help="训练场景名称（对应JSBSim配置文件）")
    group.add_argument('--num-agents', type=int, default=1,
                       help="RL策略控制的战斗机数量")
    return parser.parse_known_args(args)[0]

def main(args):
    """主执行流程（配置初始化->环境创建->运行实验）"""
    # 参数解析（参考网页6/7配置管理）
    parser = get_config()
    all_args = parse_args(args, parser)
    assert all_args.model_dir is not None  # 确保模型路径存在

    # 随机种子设置（保证实验可复现性）
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # 硬件设备配置（参考网页6多线程编译）
    if all_args.cuda and torch.cuda.is_available():
        logging.info("使用GPU加速...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True  # 保证卷积运算确定性
        torch.backends.cudnn.benchmark = False     # 固定网络结构时可设为True加速
    else:
        logging.info("使用CPU运行...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 结果存储目录配置（参考网页7工程目录结构）
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    # 创建当前运行版本目录
    curr_run = 'render'
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 设置进程名称（便于系统监控）
    setproctitle.setproctitle(f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}")

    # 环境初始化（参考网页6/7环境配置）
    envs = make_render_env(all_args)
    num_agents = all_args.num_agents

    # 构建配置字典
    config = {
        "all_args": all_args,
        "eval_envs": None,        # 评估环境（当前未使用）
        "envs": envs,             # 训练环境
        "num_agents": num_agents, # 智能体数量
        "device": device,         # 计算设备
        "run_dir": run_dir        # 存储路径
    }

    # 选择运行器类型（参考网页7静态库调用）
    if all_args.env_name == "MultipleCombat":
        runner = ShareJSBSimRunner(config)  # 多智能体共享参数运行器
    else:
        if all_args.use_selfplay:
            from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            from runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)  # 单智能体运行器
    
    runner.render()  # 启动可视化渲染

    # 后处理
    envs.close()

if __name__ == "__main__":
    # 日志配置（参考网页5调试注释实践）
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    main(sys.argv[1:])  # 命令行参数传递给主函数