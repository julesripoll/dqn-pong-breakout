from __future__ import annotations

import os
import random
import shutil
from datetime import datetime
from zipfile import ZIP_DEFLATED, ZipFile

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer
from utils.utils import image_to_state, state_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """DQN agent to be trained.

    Parameters
    ----------
    env: Environment to train the agent on (optimized for MinAtar/Breakout-v1)
    gamma: Discount factor for future rewards
    lr: Learning rate
    batch_size: Number of samples to use for each learning step
    epsilon_min: Minimum value of epsilon for the epsilon-greedy policy
    epsilon_decay: Decay rate of epsilon per timestep
    memory_size: Number of samples to store in the replay buffer
    update_rate: Number of timesteps between each learning step
    logging_directory: Directory to save the tensorboard logs to
    agent_name: Name of the agent
    log_images: Whether to log images of the last episode to tensorboard (won't work with another environment than MinAtar/Breakout-v1)
    checkpoint_directory: Directory to save the agent when it reaches a new best reward
    """
    def __init__(
            self,
            env: gym.Env,
            gamma: float = 0.99,
            lr: float = 0.001,
            batch_size: int = 32,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.999,
            memory_size: int = 100_000,
            update_rate:int = 4,
            logging_directory: str = './logs',
            agent_name: str = 'DQN',
            log_images: bool = True,
            checkpoint_directory: str | None ='./networks_weights',
    ) -> DQNAgent:
        self.agent_name = agent_name
        self.id = self.agent_name + '_' + datetime.now().strftime('%Y%m%d-%H%M%S')

        # Initialize environment
        self.env = env
        self.env.reset()
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        # Initialize hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.update_rate = update_rate

        # Initialize Q-network
        self.q_network = QNetwork(self.state_shape, self.action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-6)

        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size, batch_size)

        # Other variables
        self.checkpoint_path = None if checkpoint_directory is None else os.path.join(checkpoint_directory, f"Checkpoint_{self.id}.zip")
        self.best_reward = -np.inf
        self.timestep = 0
        self.epsilon = 1

        # Logging to tensorboard
        print(f"Run the following command to monitor training: tensorboard --logdir {logging_directory}")
        self.logging_directory = os.path.join(logging_directory, agent_name)
        self.writer = SummaryWriter(log_dir=self.logging_directory)
        self.log_images = log_images

    def train(self, max_steps: int) -> None:
        """Train the agent for a given number of steps.
        
        Parameters
        ----------
        max_steps: Number of steps to train the agent for
        """
        state = self.env.reset()
        current_reward = 0
        progress = tqdm(range(max_steps))
        for _ in progress:
            self.timestep += 1

            # Choose and take action
            action = self.act(state, train=True)
            next_state, reward, done, info = self.env.step(action)
            current_reward += reward

            # Save it
            self.memory.add(state, action, reward, next_state, done)

            # Update network when needed
            if (self.timestep % self.update_rate == 0) & (len(self.memory) > self.batch_size):
                loss = self.learn()
                progress.set_description(f"Timestep: {self.timestep}, Loss: {loss:02.4f}")
                self.writer.add_scalar('loss', loss, self.timestep)

            # Update epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Update state
            if done:
                self.writer.add_scalar('episode_reward', current_reward, self.timestep)
                if current_reward > self.best_reward:
                    self.writer.add_scalar('best_reward', current_reward, self.timestep)
                    if self.checkpoint_path is not None:
                        self.save(self.checkpoint_path, save_memory=True, override=True)
                    self.best_reward = current_reward
                if self.log_images:
                    img = np.transpose(state_to_image(next_state, show_ball=False), (2,0,1))
                    self.writer.add_image('last_episode_frame', img, self.timestep)
                state = self.env.reset()
                current_reward = 0
            else:
                state = next_state

            # Logging to tensorboard
            self.writer.add_scalar('epsilon', self.epsilon, self.timestep)

    def learn(self) -> float:
        """Update the Q-network.
        
        Returns
        -------
        loss: Loss of the network
        """
        # Sample from memory
        experiences_batch = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences_batch

        # Get the action with max Q-value
        action_values = self.q_network(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # If done just use reward, else update Q_target with discounted action values
        q_target = rewards + (self.gamma * max_action_values * (1 - dones))
        q_expected = self.q_network(states).gather(1, actions)

        # Update Q-network
        loss = F.mse_loss(q_target, q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state: np.ndarray, train: bool = False) -> int:
        """Choose an action based on the current state.

        Parameters
        ----------
        state: State of the environment
        train: Whether to use epsilon-greedy or directly the policy

        Returns
        -------
        action: Action chosen
        """
        # Epsilon-greedy action selection
        if train and (random.uniform(0, 1) < self.epsilon):
            return self.env.action_space.sample()
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.q_network(state)
            action = np.argmax(action_values.cpu().data.numpy())
            return action

    def save(self, path: str, save_memory: bool = True, override: bool = False) -> None:
        """Save the agent to a file.

        Parameters
        ----------
        path: Path to save the agent to
        save_memory: Whether to include the replay buffer (at the price of a larger file)
        override: Whether to override the file if it already exists
        """
        if os.path.exists(path) and not override:
            raise FileExistsError(f"Agent {path} already exists. Use override=True to overwrite it.")
        tmp_path = os.path.join(path + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'), "agent.pt")
        os.makedirs(os.path.dirname(tmp_path))
        writer_backup = self.writer
        memory_backup = self.memory
        try:
            self.writer = None
            if not save_memory:
                print("Memory not saved. To save it, specify save_memory=True (at the price of a higher disk usage).")
                self.memory = None
            torch.save(self, tmp_path)
            with ZipFile(path, 'w', ZIP_DEFLATED) as zip_file:
                zip_file.write(tmp_path, arcname=os.path.basename(tmp_path))
            shutil.rmtree(os.path.dirname(tmp_path))
        finally:
            self.writer = writer_backup
            if not save_memory:
                self.memory = memory_backup

    @staticmethod
    def load(path: str) ->  DQNAgent:
        """Load an agent from a file.

        Parameters
        ----------
        path: Path to load the agent from
        
        Returns
        -------
        agent: Loaded agent
        """
        tmp_path = path + '_' + datetime.now().strftime('%Y%m%d-%H%M%S')
        shutil.unpack_archive(path, tmp_path)
        try:
            agent = torch.load(tmp_path + "/agent.pt", map_location=device)
            agent.q_network.to(device)
            if not hasattr(agent, 'writer') or agent.writer is None:
                agent.writer = SummaryWriter(log_dir=agent.logging_directory)
            if agent.memory is None:
                print("No memory available to load. The experience replay will start empty if you continue training.")
                agent.memory = ReplayBuffer(agent.memory_size, agent.batch_size)
        finally:
            shutil.rmtree(tmp_path)
        return agent
