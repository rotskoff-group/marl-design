from system import ActiveSystem, LJSystem
from cdql import CDQL
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--active", action='store_true', default=False)
parser.add_argument("--region_num", type=int, default=2)

config = parser.parse_args()
region_num = config.region_num
folder_name = "./"
if (config.active):
    all_actions = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    gamma = 0.9
    num_explore_episodes = 5
    system = ActiveSystem(filename="temp.h5", region_num=region_num)
else:
    all_actions = [0.01, 0.25, 1.0]
    gamma = 0.95
    num_explore_episodes = 25

    system = LJSystem(filename="temp.h5", region_num=region_num)

c = CDQL(system=system, all_actions=all_actions, gamma=gamma,
         num_explore_episodes=num_explore_episodes, device=torch.device("cpu"),
         folder_name=folder_name)
c.train()
