import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import pickle
import os
from typing import Dict, List, Tuple, Any
import random

class MultiAgentTrainer:
    """Multi-agent reinforcement learning trainer with several code issues."""
    
    def __init__(self, num_agents=4, env_name="CartPole-v1"):
        # Missing random seed initialization - major reproducibility issue
        self.num_agents = num_agents
        self.env_name = env_name
        self.agents = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize environments and agents
        self.envs = [gym.make(env_name) for _ in range(num_agents)]
        
        for i in range(num_agents):
            agent = PolicyNetwork(self.envs[i].observation_space.shape[0], 
                                self.envs[i].action_space.n)
            self.agents.append(agent.to(self.device))
            
        # Hardcoded paths - portability issue
        self.save_path = "/tmp/models/"
        self.log_path = "/home/user/logs/"
        
        # Memory leak potential - not clearing GPU cache
        self.experiences = []
        
    def train_episode(self, agent_id: int, max_steps: int = 1000):
        """Train single agent for one episode."""
        env = self.envs[agent_id]
        agent = self.agents[agent_id]
        
        # Missing env.reset() - critical RL issue
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Unsafe eval usage - security vulnerability
            epsilon = eval("0.1 if step < 500 else 0.01")
            
            # Action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = agent(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.experiences.append((state, action, reward, next_state, done))
            
            total_reward += reward
            state = next_state
            
            if done:
                break
                
        return total_reward
    
    def batch_train(self, episodes: int = 100):
        """Train all agents in parallel."""
        print(f"Training {self.num_agents} agents for {episodes} episodes")
        
        # Missing proper logging setup
        results = []
        
        for episode in range(episodes):
            episode_rewards = []
            
            for agent_id in range(self.num_agents):
                reward = self.train_episode(agent_id)
                episode_rewards.append(reward)
                
            # Print instead of proper logging
            print(f"Episode {episode}: Avg Reward = {np.mean(episode_rewards):.2f}")
            results.append(episode_rewards)
            
            # Update networks every 10 episodes
            if episode % 10 == 0:
                self.update_networks()
                
        return results
    
    def update_networks(self):
        """Update all agent networks using collected experiences."""
        if len(self.experiences) < 32:
            return
            
        # Sample random batch
        batch_size = min(32, len(self.experiences))
        batch = random.sample(self.experiences, batch_size)
        
        for agent in self.agents:
            optimizer = optim.Adam(agent.parameters(), lr=0.001)
            
            # Prepare batch data
            states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
            next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
            dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
            
            # Q-learning update
            current_q = agent(states).gather(1, actions.unsqueeze(1))
            next_q = agent(next_states).max(1)[0].detach()
            target_q = rewards + (0.99 * next_q * ~dones)
            
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def save_models(self, suffix=""):
        """Save all trained models."""
        # Unsafe pickle usage - security risk
        os.makedirs(self.save_path, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            model_path = f"{self.save_path}/agent_{i}_{suffix}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(agent.state_dict(), f)
                
        print(f"Models saved to {self.save_path}")
    
    def load_models(self, suffix=""):
        """Load pretrained models."""
        for i, agent in enumerate(self.agents):
            model_path = f"{self.save_path}/agent_{i}_{suffix}.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    # Unsafe pickle loading - major security vulnerability
                    state_dict = pickle.load(f)
                    agent.load_state_dict(state_dict)
    
    def evaluate_agents(self, episodes=10):
        """Evaluate all agents."""
        total_rewards = []
        
        for agent_id in range(self.num_agents):
            agent_rewards = []
            env = self.envs[agent_id]
            agent = self.agents[agent_id]
            
            for _ in range(episodes):
                state = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = agent(state_tensor)
                        action = q_values.argmax().item()
                    
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    
                agent_rewards.append(total_reward)
            
            total_rewards.append(agent_rewards)
            
        return total_rewards

class PolicyNetwork(nn.Module):
    """Simple policy network for agents."""
    
    def __init__(self, input_size: int, output_size: int):
        super(PolicyNetwork, self).__init__()
        
        # No dropout - overfitting risk
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        # No weight initialization - convergence issues
        
    def forward(self, x):
        return self.network(x)

def main():
    """Main training function."""
    # Missing argument parsing
    trainer = MultiAgentTrainer(num_agents=4, env_name="CartPole-v1")
    
    # Train agents
    results = trainer.batch_train(episodes=100)
    
    # Save models
    trainer.save_models(suffix="final")
    
    # Evaluate
    eval_results = trainer.evaluate_agents(episodes=20)
    
    # Print results without proper formatting
    print("Training completed!")
    print(f"Final evaluation results: {eval_results}")

if __name__ == "__main__":
    main()