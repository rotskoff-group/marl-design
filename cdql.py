import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from openmm import OpenMMException
from replaybuffer import ReplayBuffer
from cdqlnetwork import Model


class CDQL:
    def __init__(self, system, all_actions, num_explore_episodes, gamma,
                 device=torch.device("cuda:0"), folder_name="./",
                 centralize_states=False, centralize_rewards=False,
                 update_num=20):

        self.gamma = gamma
        self.batch_size = 32
        self.buffer = ReplayBuffer(1e6)

        self.folder_name = folder_name
        self.device = device
        self.centralize_states = centralize_states
        self.centralize_rewards = centralize_rewards
        self.system = system
        self.num_explore_episodes = num_explore_episodes
        self.all_actions = all_actions
        self.num_actions = len(self.all_actions)

        if self.centralize_states:
            num_bins = self.system.num_bins * 2
        else:
            num_bins = self.system.num_bins

        self.model = Model(self.device, num_bins=num_bins,
                           num_actions=self.num_actions)
        self.loss = []
        self.store_Q = []
        self.training_iter = 0
        self.update_freq = 2
        self.update_num = update_num


    def _update(self):
        """Updates q1, q2, q1_target and q2_target networks based on
        clipped Double Q Learning Algorithm
        """
        if (len(self.buffer) < self.batch_size):
            return
        self.training_iter += 1
        # Make sure actor_target and critic_target are in eval mode
        assert not self.model.q_target_1.training
        assert not self.model.q_target_2.training

        assert self.model.q_1.training
        assert self.model.q_2.training
        transitions = self.buffer.sample(self.batch_size)
        batch = self.buffer.transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state, device=self.device).float()
        action_batch = torch.tensor(batch.action,
                                    device=self.device).unsqueeze(-1).long()
        reward_batch = torch.tensor(batch.reward,
                                    device=self.device).unsqueeze(-1).float()
        next_state_batch = torch.tensor(batch.next_state,
                                        device=self.device).float()
        is_done_batch = torch.tensor(batch.done,
                                     device=self.device).unsqueeze(-1).bool()
        with torch.no_grad():
            Q_next_1 = ((~is_done_batch)
                        * (self.model.q_target_1(next_state_batch).min(dim=-1)[0].unsqueeze(-1)))
            Q_next_2 = ((~is_done_batch)
                        * (self.model.q_target_2(next_state_batch).min(dim=-1)[0].unsqueeze(-1)))

            # Use max want to avoid underestimation bias
            Q_next = torch.max(Q_next_1, Q_next_2)
            Q_expected = reward_batch + self.gamma * Q_next

        Q_1 = self.model.q_1(state_batch).gather(-1, action_batch)
        Q_2 = self.model.q_2(state_batch).gather(-1, action_batch)
        L_1 = nn.MSELoss()(Q_1, Q_expected)
        L_2 = nn.MSELoss()(Q_2, Q_expected)
        self.loss.append([L_1.item(), L_2.item()])
        self.model.q_optimizer_1.zero_grad()
        self.model.q_optimizer_2.zero_grad()
        L_1.backward()
        L_2.backward()
        self.model.q_optimizer_1.step()
        self.model.q_optimizer_2.step()
        self.store_Q.append([Q_1.tolist(), Q_2.tolist(), Q_expected.tolist()])
        if (self.training_iter % self.update_freq) == 0:
            self.model.update_target_nn()

    def _get_state(self, grid_dist, surrounding_dist):
        """Gets concatenated state if including state information about surrounding regions
        (i.e. if self.centralize_states). Inputs a list of states for multiple regions/surrounding regions
        Args:
            grid_dist: A 2D List representing the state of the current region
            surrounding_dist: A 2D List representing the state of the surrounding region
        Returns:
            A 2D List representing the final state information to use
        """
        if (self.centralize_states):
            grid_dist = torch.tensor(grid_dist, device=self.device).float()
            surrounding_dist = torch.tensor(surrounding_dist,
                                            device=self.device).float()
            cat_dist = torch.cat((grid_dist, surrounding_dist), dim=1)
            return cat_dist.tolist()
        else:
            return grid_dist

    def _get_reward(self, grid_reward, surrounding_reward):
        """Gets concatenated rewards if including cost information about surrounding regions
        (i.e. if self.centralize_rewards).  Inputs a list of rewards for multiple regions/surrounding regions
        Args:
            grid_reward: A list representing the reward of the current region
            surrounding_reward: A list representing the reward of the surrounding regions
        Returns:
            A list representing the final reward information to use
        """
        if (self.centralize_rewards):
            grid_reward = torch.tensor(grid_reward, device=self.device).float()
            surrounding_reward = torch.tensor(surrounding_reward,
                                              device=self.device).float()
            final_reward = (grid_reward + surrounding_reward) / 2
            return final_reward.tolist()
        else:
            return grid_reward

    def _get_action(self, state, episode):
        """Gets action given some state
        if episode is less than 5 returns a random action for each region
        Args:
            state: List of states (corresponding to each region)
            episode: episode number
        """
        if (episode < self.num_explore_episodes):
            action = [random.choice(list(range(self.num_actions)))
                      for _ in range(len(state))]
            return action

        action = []
        self.model.q_1.eval()
        with torch.no_grad():
            state = torch.tensor(state, device=self.device).float()
            action = torch.argmin(self.model.q_1(state), dim=-1).tolist()
        self.model.q_1.train()
        return action

    def _save_data(self):
        filename = self.folder_name + "replaybuffer"
        np.save(filename, np.array(self.buffer.buffer, dtype=object))

        filename = self.folder_name + "loss"
        np.save(filename, np.array(self.loss))

        filename = self.folder_name + "Q_pair.npy"
        np.save(filename, np.array(self.store_Q, dtype=object))

        self.model.save_networks(self.folder_name)

    def _save_episode_data(self, episode_folder_name):
        filename = episode_folder_name + "replaybuffer"
        np.save(filename, np.array(self.buffer.buffer, dtype=object))

        self.model.save_networks(episode_folder_name)

    def load_data(self):
        self.loss = torch.load(self.folder_name + "loss.pt").tolist()
        self.buffer.load_buffer(self.folder_name + "replaybuffer.npy")
        self.model.load_networks(self.folder_name)

    def train(self, num_decisions=350):
        """Train q networks based on Clipped Double Q Learning
        Args:
            num_decisions: Number of decisions to train algorithm for
        """
        os.system("mkdir " + self.folder_name + "Train")
        for i in range(5000):
            episode_folder_name = self.folder_name + "Train/" + str(i) + "/"
            all_system_states = []
            all_system_rewards = []
            all_system_states_cluster = []
            all_grid_states_cluster = []
            all_surrounding_states_cluster = []
            os.system("mkdir " + episode_folder_name)
            filename = episode_folder_name + str(i) + ".h5"
            self.system.reset_context(filename)
            self.system.run_decorrelation(20)
            grid_dist, surrounding_dist, _, _, _, _ = self.system.get_state_reward()
            state = self._get_state(grid_dist, surrounding_dist)
            for j in range(num_decisions):
                action_index = self._get_action(state, i)
                transition_to_add = [state, action_index]
                tag = "_train_" + str(j)
                actions = [self.all_actions[i] for i in action_index]
                try:
                    self.system.update_action(actions)
                    system_states, system_rewards, system_states_cluster = self.system.run_step(
                        is_detailed=True, tag=tag)
                    all_system_states.append(system_states)
                    all_system_rewards.append(system_rewards)
                    all_system_states_cluster.append(system_states_cluster)

                except OpenMMException:
                    print("Broken Simulation at Episode:",
                          str(i), ", Decision:", str(j))
                    break

                grid_dist, surrounding_dist, grid_reward, surrounding_reward, grid_states_cluster, surrounding_states_cluster = self.system.get_state_reward()
                state = self._get_state(grid_dist, surrounding_dist)
                reward = self._get_reward(grid_reward, surrounding_reward)

                all_grid_states_cluster.append(grid_states_cluster)
                all_surrounding_states_cluster.append(surrounding_states_cluster)

                # Use len_reward for number of grids
                done = [False] * len(reward)  # Never Done
                transition_to_add.extend([reward, state, done])
                rb_decision_samples = 0
                for rb_tuple in zip(*transition_to_add):
                    self.buffer.push(*list(rb_tuple))

                for _ in range(self.update_num):
                    self._update()
            self._save_episode_data(episode_folder_name)
            np.save(episode_folder_name + "system_states",
                    np.array(all_system_states))
            np.save(episode_folder_name + "system_rewards",
                    np.array(all_system_rewards))
            np.save(episode_folder_name + "system_states_cluster",
                    np.array(all_system_states_cluster))
            np.save(episode_folder_name + "grid_states_cluster",
                    np.array(all_grid_states_cluster, dtype=object))
            np.save(episode_folder_name + "all_states_cluster",
                    np.array(all_surrounding_states_cluster))
            self._save_data()


    def test(self, num_decisions=1000):
        """Given trained q networks, generate trajectories
        Saves:
            grid_rewards: Numpy array of all the rewards of each region along traj
            grid_states: Numpy array of all the states (i.e. normalized distibution of cluster sizes)
                         of each region along traj
            grid_states_cluster: Numpy array of all the cluster sizes of each region along traj
            actions: Numpy array of actions taken along trajectory
            dissipation: Total dissipation (not average dissipation rate) along trajectory
            system_states: Numpy array of states of the system along traj:
            system_states_cluster: Numpy array of cluster sizes along traj
            system_rewards: Numpy array of reward of entire system along traj
        """
        all_grid_states = []
        all_grid_rewards = []

        all_grid_states_cluster = []
        all_surrounding_states_cluster = []

        all_system_rewards = []
        all_system_states = []
        all_system_states_cluster = []

        all_actions = []
        all_dissipation = []

        os.system("mkdir " + self.folder_name + "Test/")
        filename = self.folder_name + "Test/" + "TEST.h5"
        self.system.reset_context(filename)
        tag = "_test_init"
        self.system.run_decorrelation(20)

        grid_dist, surrounding_dist, _, _, _, _ = self.system.get_state_reward()
        state = self._get_state(grid_dist, surrounding_dist)

        all_dissipation.append(self.system.get_dissipation())
        for i in range(num_decisions):
            action_index = self._get_action(state, episode=10000)
            tag = "_test_" + str(i)
            actions = [self.all_actions[i] for i in action_index]
            all_actions.append(actions)
            self.system.update_action(actions)
            system_states, system_rewards, system_states_cluster = self.system.run_step(
                is_detailed=True, tag=tag)

            grid_dist, surrounding_dist, grid_reward, surrounding_reward, grid_states_cluster, surrounding_states_cluster = self.system.get_state_reward()

            # The "grid states" and dissipation are recorded at the end of a decision
            # Dissipation here is total entropy production (not epr)
            # Actions are recorded at the beginning of the decision

            state = self._get_state(grid_dist, surrounding_dist)
            reward = self._get_reward(grid_reward, surrounding_reward)

            all_grid_states.append(state)
            all_grid_rewards.append(reward)

            all_grid_states_cluster.append(grid_states_cluster)
            all_surrounding_states_cluster.append(surrounding_states_cluster)

            all_dissipation.append(self.system.get_dissipation())

            # The "System States" are recorded every 0.25 seconds. Excludes 0th second
            all_system_states.append(system_states)
            # Just to have a 1D array use extend
            all_system_rewards.append(system_rewards)
            all_system_states_cluster.append(system_states_cluster)

            if (i % 100 == 99):
                np.save(self.folder_name + "grid_rewards",
                        np.array(all_grid_rewards))
                np.save(self.folder_name + "grid_states",
                        np.array(all_grid_states, dtype=object))
                np.save(self.folder_name + "grid_states_cluster",
                        np.array(all_grid_states_cluster, dtype=object))

                np.save(self.folder_name + "surrounding_states_cluster",
                        np.array(all_surrounding_states_cluster))

                np.save(self.folder_name + "actions", np.array(all_actions))

                np.save(self.folder_name + "dissipation",
                        np.array(all_dissipation))
                np.save(self.folder_name + "system_states",
                        np.array(all_system_states))
                np.save(self.folder_name + "system_states_cluster",
                        np.array(all_system_states_cluster))
                np.save(self.folder_name + "system_rewards",
                        np.array(all_system_rewards))


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        centralize_code = int(sys.argv[-2])
        centralize_approach = sys.argv[-1]

        if (centralize_code == 0):
            centralize_states = False
            centralize_rewards = False
        elif (centralize_code == 1):
            centralize_states = True
            centralize_rewards = False
        elif (centralize_code == 2):
            centralize_states = False
            centralize_rewards = True
        elif (centralize_code == 3):
            centralize_states = True
            centralize_rewards = True
        else:
            raise ValueError("Wrong Centralize Code")

        if (centralize_approach == "none"):
            if (centralize_code > 0):
                print("Not running none with centralized states/rewards")
                exit()
            centralize_approach = None
        elif (centralize_code == 0):
            print("Not running not none without centralized states/rewards")
            exit()
        elif (centralize_approach == "all"):
            centralize_approach = "all"
        elif (centralize_approach == "plaquette"):
            centralize_approach = "plaquette"
        elif (centralize_approach == "grid_1"):
            centralize_approach = 1
        elif (centralize_approach == "grid_2"):
            centralize_approach = 2
        else:
            raise ValueError("Wrong Centralize Approach")

        c = CDQL(centralize_states=centralize_states,
                 centralize_rewards=centralize_rewards,
                 centralize_approach=centralize_approach)

    else:
        c = CDQL(centralize_states=True, centralize_rewards=True,
                 centralize_approach="plaquette")
    c.train()

    # c.model.load_networks()
    # c.test()
