#!/usr/bin/env python
# 指定使用Python解释器执行本脚本

# 导入系统库
import sys
import os
import traceback
import socket
from pathlib import Path

# 导入第三方库
import wandb            # 实验跟踪工具
import torch            # PyTorch深度学习框架
import random
import logging
import numpy as np
import setproctitle     # 设置进程名称的工具

# 将项目根目录添加到系统路径中（假设当前文件位于项目子目录中）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# 导入自定义模块
from config import get_config #第一步，加载配置
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
from runner.tacview import Tacview
#第二步，仿真环境构建
def make_train_env(all_args):  #动态生成多线程环境
    """创建训练环境的工厂函数"""
    def get_env_fn(rank):
        """为每个并行环境创建初始化函数"""
        def init_env():
            """实际的环境初始化函数"""
            # 根据env_name参数选择不同的环境类型
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name) #实现多进程并行采样
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("不支持的环境类型: " + all_args.env_name)
                raise NotImplementedError
            # 设置环境种子（不同rank的环境使用不同的种子）
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    # 根据环境类型选择不同的并行环境包装器
    if all_args.env_name == "MultipleCombat":
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])  # 单线程使用虚拟环境
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    else:
        if all_args.n_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    """创建评估环境，结构与训练环境类似但种子不同"""
    def get_env_fn(rank):
        def init_env():
            # 环境类型选择逻辑与训练环境相同
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("不支持的环境类型: " + all_args.env_name)
                raise NotImplementedError
            # 评估环境使用不同的种子范围
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    
    # 并行环境包装逻辑与训练环境相同
    if all_args.env_name == "MultipleCombat":
        if all_args.n_eval_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    """解析JSBSim环境相关参数"""
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="选择要运行的场景名称")
    group.add_argument('--render-mode', type=str, default='txt',
                       help="渲染模式：txt 或 real_time")
    return parser.parse_known_args(args)[0]

def main(args):
    """主程序入口"""
    # 获取配置参数
    parser = get_config()
    all_args = parse_args(args, parser)

    # 设置随机种子保证可重复性
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(all_args.seed)

    # 配置计算设备（GPU/CPU）
    if all_args.cuda and torch.cuda.is_available():
        logging.info("选择使用GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        # 设置CUDA相关参数保证确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("选择使用CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 创建运行结果存储目录 第三步
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():  #创建带时间戳的结果目录
        os.makedirs(str(run_dir))

    # 初始化Wandb实验跟踪
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.experiment_name}_seed{all_args.seed}",
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        # 当不使用Wandb时自动创建运行编号目录
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() 
                            if str(folder.name).startswith('run')]
            curr_run = f'run{max(exst_run_nums)+1}' if exst_run_nums else 'run1'
        run_dir = run_dir / curr_run
        os.makedirs(str(run_dir), exist_ok=True)

    # 设置进程名称便于监控
    setproctitle.setproctitle(f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}")

    # 初始化环境 第四步 核心训练循环阶段
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # 准备运行配置字典
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
        "render_mode": all_args.render_mode
    }

    # 根据环境类型选择运行器 该模块负责：
    #    ​状态封装​：将JSBSim的飞行状态（姿态、速度等）转换为强化学习观测空间
    #    ​动作映射​：将AI策略输出转换为JSBSim控制指令（如副翼偏转角）
    #​奖励计算​：基于场景目标设计奖励函数（如空战场景的敌机击落奖励）

    #
    if all_args.env_name == "MultipleCombat":
        runner = ShareJSBSimRunner(config)
    else:
        Runner = SelfplayJSBSimRunner if all_args.use_selfplay else JSBSimRunner
        runner = Runner(config)

    # 执行训练流程
    try:
        runner.run()
    except Exception as e:
        traceback.print_exc()
    finally:
        # 清理资源
        envs.close()
        if all_args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    # 配置基础日志格式
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])  # 跳过第一个参数（脚本名称）