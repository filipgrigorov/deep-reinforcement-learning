import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import Memory, Experience
from model import Actor, Critic
from noise import OUNoise

class DDPGAgent:
    '''Class representing the DDPG algorithm'''
    def __init__(self, state_size, action_size, num_agents, device, config):
        '''Class constructor and parameters initialization'''
        self.device = device

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        seed = config['seed']
        self.gamma = config['gamma']

        # Learns argmax_a[Q(s, a); theta_mu] = mu(s, a; theta_mu)
        self.learnt_actor = Actor(seed, state_size, action_size).to(self.device) # learnt
        self.target_actor = Actor(seed, state_size, action_size).to(self.device) # soft-update tracking
        self.actor_optim = optim.Adam(self.learnt_actor.parameters(), lr=config['actor_lr'])

        # Learns to evaluate Q(s, mu(s, a); theta_q)
        self.learnt_critic = Critic(seed, state_size * num_agents, action_size * num_agents, 1).to(self.device) # learnt
        self.target_critic = Critic(seed, state_size * num_agents, action_size * num_agents, 1).to(self.device) # soft-update tracking
        self.critic_optim = optim.Adam(self.learnt_critic.parameters(), lr=config['critic_lr'])

        print(f'Summary:\nActor network:\n{self.learnt_actor}\nCritic network:\n{self.learnt_critic}')

        # Soft-update
        self.tau = config['tau']

        # Noise
        self.noise = OUNoise(action_size, seed)
        self.noise_decay = config['noise_decay']

    def reset_noise(self):
        '''Reset the noise state'''
        self.noise.reset()

    # Note: Decentralized actors (execution)
    def act(self, state):
        '''Sample an action from the policy'''
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        self.learnt_actor.eval()
        with torch.no_grad():
            actions = self.learnt_actor(state).cpu().data.numpy()
        self.learnt_actor.train()

        actions += self.noise_decay * self.noise.sample()

        return np.clip(actions, -1, 1)

    # Note: Centralized critic (training)
    def step(self, best_current_actions, best_next_actions, states, actions, rewards, next_states, dones):
        '''Optimizes the function apprximators and soft-updates'''

        self.__optimize_critic(best_next_actions, states, actions, rewards, next_states, dones)

        self.__optimize_actor(best_current_actions, states)

        self.__soft_update(self.learnt_actor, self.target_actor, self.tau)
        self.__soft_update(self.learnt_critic, self.target_critic, self.tau)

        self.noise_decay *= self.noise_decay
        self.reset_noise()

    def __optimize_critic(self, best_next_actions, states, actions, rewards, next_states, dones):
        '''Optimizes the critic approximator'''
        q_targets = rewards + self.gamma * self.target_critic(next_states, best_next_actions) * (1 - dones)

        q_predictions = self.learnt_critic(states, actions)

        self.critic_optim.zero_grad()
        critic_loss = F.mse_loss(q_predictions, q_targets)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learnt_critic.parameters(), 1)
        self.critic_optim.step()

    def __optimize_actor(self, best_current_actions, states):
        '''Optimizes the actor approximator'''
        advantage = -self.learnt_critic(states, best_current_actions).mean()
        
        self.actor_optim.zero_grad()
        advantage.backward()
        self.actor_optim.step()

    def __soft_update(self, learnt, target, tau):
        '''Soft-updates the target parameters'''
        for learnt_param, target_param in zip(learnt.parameters(), target.parameters()):
            target_param.data.copy_(tau * learnt_param.data + (1.0 - tau) * target_param.data)

# Reference: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" by Lowe, Wu, Tamar, Harb, Abbeel, Mordatch
class MADDPG:
    def __init__(self, state_size, action_size, num_agents, config):
        ''' Constructs the multi-agent eco-system '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device}')

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.ddpg_agents = [ DDPGAgent(state_size, action_size, num_agents, self.device, config) for _ in range(num_agents) ]

        self.update_every = config['update_every']
        self.update_iters = config['update_iterations']

        # Note: Could be replaced by parallel env batching
        seed = config['seed']
        self.batch_size = config['batch_size']
        self.memory = Memory(config['memory_size'], self.batch_size, seed)
        self.memory.to_device(self.device)

    def reset_noise(self):
        [ agent.reset_noise() for agent in self.ddpg_agents ]

    def act(self, states):
        ''' For each agent idx, select a_idx = policy_idx(o_idx) + noise '''
        actions = [ self.ddpg_agents[idx].act(states[np.newaxis, idx]).squeeze(0) for idx in range(self.num_agents) ]
        return actions

    # Note: We need to add all the observations, otherwise we break the stationarity of the environment
    def remember(self, states, actions, rewards, next_states, dones):
        '''Populates the replay memory with new batch of data; observations of all agents'''
        self.memory.add(Experience(states, actions, rewards, next_states, dones))

    def step(self, timestep):
        if len(self.memory) > self.batch_size and timestep % self.update_every == 0:
            for _ in range(self.update_iters):
                for idx in range(self.num_agents):
                    states, actions, rewards, next_states, dones = self.memory.sample()
                    
                    predicted_best_next_actions = torch.cat([self.ddpg_agents[idx].target_actor(next_states[:, idx, :]) for idx in range(self.num_agents)], dim=1)

                    predicted_best_current_actions = torch.cat([self.ddpg_agents[idx].learnt_actor(states[:, idx, :]) for idx in range(self.num_agents)], dim=1)

                    states = torch.cat([ states[:, idx, :] for idx in range(self.num_agents) ], dim=1)
                    actions = torch.cat([ actions[:, idx, :] for idx in range(self.num_agents) ], dim=1)
                    next_states = torch.cat([ next_states[:, idx, :] for idx in range(self.num_agents) ], dim=1)

                    self.ddpg_agents[idx].step(
                        predicted_best_current_actions,
                        predicted_best_next_actions,
                        states, actions, rewards, next_states, dones)

    def save(self, actor_weights_path, critic_weights_path):
        [ torch.save(self.ddpg_agents[idx].learnt_actor.state_dict(), actor_weights_path + str(idx + 1) + '.pth') for idx in range(self.num_agents) ]
        [ torch.save(self.ddpg_agents[idx].learnt_actor.state_dict(), critic_weights_path + str(idx + 1) + '.pth') for idx in range(self.num_agents) ]