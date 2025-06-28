"""
Sample ML/RL code for testing the CodeGuard VS Code extension.
This file contains various issues that should be detected by CodeGuard.
"""

from torch import *  # Should trigger import issue
import pickle  # Should trigger security warning
import gym
import os

# Hardcoded credentials - security issue
API_KEY = "sk-1234567890abcdef"
password = "admin123"

def train_model():
    """Train a simple RL agent with various issues."""
    
    # Missing random seed - reproducibility issue
    env = gym.make("CartPole-v1")
    
    for episode in range(100):
        done = False
        total_reward = 0
        
        # Missing env.reset() - RL issue
        while not done:
            action = 1  # Hardcoded action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(f"Episode reward: {total_reward}")  # Should use logging
        
        # Unsafe pickle usage - security issue
        with open("/tmp/results.pkl", "wb") as f:
            pickle.dump({"reward": total_reward}, f)
    
    # Unsafe eval usage - security issue
    user_input = input("Enter code: ")
    eval(user_input)

def load_model():
    """Load model with unsafe practices."""
    
    # Hardcoded path - portability issue
    model_path = "/home/user/models/agent.pth"
    
    if os.path.exists(model_path):
        # Missing error handling
        model = torch.load(model_path)
        return model

def validate_model(model):
    """Validate model performance."""
    test_env = gym.make("CartPole-v1")
    
    # Missing reset
    obs = test_env.reset()
    
    # Infinite loop potential
    while True:
        action = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        
        if done:
            break

# Unused imports should be detected
import numpy as np
import tensorflow as tf

# Style issues
def poorly_formatted_function( x,y ):
    return x+y

# Type issues (if mypy is enabled)
def untyped_function(data):
    return data.process()

if __name__ == "__main__":
    train_model()