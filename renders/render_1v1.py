import numpy as np
import torch
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

RESULT_DIR = BASE_DIR / "results"

from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)


class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
    
def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 2
render = True
ego_policy_index = 1040
enm_policy_index = 0
episode_rewards = 0

# 修改为指定的路径
ego_run_dir = r"E:\unityproject\1v1 noweapon\files"
enm_run_dir = r"E:\unityproject\1v1 noweapon\files"
# 修改experiment_name，从新路径获取
experiment_name = os.path.basename(os.path.dirname(ego_run_dir)).replace(" ", "_")
env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
ego_policy.eval()
enm_policy.eval()
#选择模型
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt", map_location="cpu"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt", map_location="cpu"))


print("Start render")
obs = env.reset()

env.start_acmi_stream()

if render:
    env.render(mode='unity3d', filepath=f'{experiment_name}.txt.acmi')
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
enm_obs =  obs[num_agents // 2:, :]
ego_obs =  obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    # Obser reward and next obs
    obs, rewards, dones, infos = env.step(actions)
    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    if dones.all():
        print(infos)
        break
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods}")
    enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]

print(episode_rewards)