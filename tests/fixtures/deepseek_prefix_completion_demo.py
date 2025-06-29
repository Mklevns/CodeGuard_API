#!/usr/bin/env python3
"""
DeepSeek Chat Prefix Completion Demo for Code Improvements.
Demonstrates different ways to use prefix completion for structured code improvements.
"""

import json
import os
from typing import Dict, Any

def demo_json_structured_improvement():
    """Demo 1: JSON-structured code improvement response."""
    
    problematic_code = '''
import torch
import numpy as np

# Bad ML code with security issues
def train_model(data_path):
    # Security vulnerability: eval usage
    learning_rate = eval("0.001")
    
    # Missing random seeding
    model = torch.nn.Linear(10, 1)
    
    # Hardcoded path
    torch.save(model, "/tmp/model.pt")
    
    return model
'''
    
    print("=" * 60)
    print("DEMO 1: JSON-Structured Code Improvement")
    print("=" * 60)
    print("Original problematic code:")
    print(problematic_code)
    print("\nUsing Chat Prefix Completion to force JSON output...")
    
    # Simulate DeepSeek Chat Prefix Completion
    example_response = {
        "improved_code": '''
import torch
import numpy as np
import os

# Improved ML code with security fixes
def train_model(data_path, learning_rate=0.001, model_save_dir="./models"):
    # Fixed: Removed eval() security vulnerability
    # Added: Proper random seeding for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = torch.nn.Linear(10, 1)
    
    # Fixed: Use configurable path instead of hardcoded
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, "model.pt")
    torch.save(model, model_path)
    
    return model
''',
        "applied_fixes": [
            {
                "type": "security",
                "description": "Replaced eval() with direct assignment",
                "line": 7,
                "severity": "high"
            },
            {
                "type": "reproducibility", 
                "description": "Added torch.manual_seed() and np.random.seed()",
                "line": 11,
                "severity": "medium"
            },
            {
                "type": "portability",
                "description": "Replaced hardcoded path with configurable parameter",
                "line": 16,
                "severity": "medium"
            }
        ],
        "improvement_summary": "Fixed critical security vulnerability, added reproducibility seeding, and improved code portability",
        "confidence_score": 0.95,
        "warnings": ["Consider adding input validation for data_path parameter"]
    }
    
    print("DeepSeek JSON Response:")
    print(json.dumps(example_response, indent=2))
    
    return example_response

def demo_direct_code_completion():
    """Demo 2: Direct code completion with prefix."""
    
    incomplete_code = '''
import torch
import torch.nn as nn

class ImprovedPolicyNetwork(nn.Module):
    """Enhanced policy network with proper initialization and dropout."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super(ImprovedPolicyNetwork, self).__init__()
        
        # TODO: Add proper network architecture with dropout and initialization
'''
    
    print("\n" + "=" * 60)
    print("DEMO 2: Direct Code Completion")
    print("=" * 60)
    print("Incomplete code:")
    print(incomplete_code)
    print("\nUsing Chat Prefix Completion to complete the implementation...")
    
    # Simulate completion from "# TODO:" prefix
    completed_code = '''        
        # Enhanced network with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        
        # Proper weight initialization using Xavier/He initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU layers."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
'''
    
    print("Completed code:")
    print(incomplete_code + completed_code)
    
    return completed_code

def demo_security_focused_improvement():
    """Demo 3: Security-focused code improvement."""
    
    unsafe_code = '''
import pickle
import os

def load_model_config(config_path):
    # Dangerous: Using pickle.load on user input
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Dangerous: Using eval on config values
    if 'learning_rate' in config:
        lr = eval(config['learning_rate'])
    
    # Dangerous: Direct os.system call
    if 'cleanup_cmd' in config:
        os.system(config['cleanup_cmd'])
    
    return config
'''
    
    print("\n" + "=" * 60)
    print("DEMO 3: Security-Focused Improvement")
    print("=" * 60)
    print("Unsafe code with multiple security vulnerabilities:")
    print(unsafe_code)
    print("\nUsing prefix completion to generate secure alternatives...")
    
    # Simulate security-focused improvement
    secure_code = '''
import json
import subprocess
import os
from typing import Dict, Any

def load_model_config(config_path: str) -> Dict[str, Any]:
    """Safely load model configuration from JSON file."""
    
    # Security fix: Use JSON instead of pickle
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Invalid or missing config file: {e}")
    
    # Security fix: Validate learning rate instead of using eval
    if 'learning_rate' in config:
        try:
            lr = float(config['learning_rate'])
            if not (0 < lr < 1):
                raise ValueError("Learning rate must be between 0 and 1")
        except (ValueError, TypeError):
            raise ValueError("Invalid learning rate format")
    
    # Security fix: Use subprocess with shell=False for cleanup
    if 'cleanup_cmd' in config:
        allowed_commands = ['rm', 'mv', 'cp']  # Whitelist approach
        cmd_parts = config['cleanup_cmd'].split()
        
        if cmd_parts and cmd_parts[0] in allowed_commands:
            try:
                subprocess.run(cmd_parts, check=True, shell=False, timeout=30)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"Cleanup command failed safely: {e}")
        else:
            print(f"Cleanup command '{config['cleanup_cmd']}' not allowed")
    
    return config
'''
    
    print("Secure implementation:")
    print(secure_code)
    
    security_analysis = {
        "vulnerabilities_fixed": [
            "Replaced pickle.load() with json.load() to prevent code injection",
            "Eliminated eval() usage with proper type validation", 
            "Replaced os.system() with subprocess.run() with shell=False",
            "Added command whitelisting and timeout protection"
        ],
        "security_improvements": [
            "Input validation and error handling",
            "Type hints for better code safety",
            "Timeout protection for subprocess calls",
            "Graceful error handling without crashes"
        ]
    }
    
    print("\nSecurity Analysis:")
    print(json.dumps(security_analysis, indent=2))
    
    return secure_code

def demo_ml_best_practices():
    """Demo 4: ML/RL best practices improvement."""
    
    poor_ml_code = '''
import torch
import gym

def train_agent():
    env = gym.make("CartPole-v1")
    
    for episode in range(100):
        state = env.reset()
        
        while True:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            if done:
                break
                
        print(f"Episode {episode} completed")
'''
    
    print("\n" + "=" * 60)
    print("DEMO 4: ML/RL Best Practices Improvement")
    print("=" * 60)
    print("Poor ML code missing best practices:")
    print(poor_ml_code)
    print("\nImproving with ML/RL best practices...")
    
    improved_ml_code = '''
import torch
import gym
import numpy as np
import random
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_agent(seed: int = 42, num_episodes: int = 100) -> List[float]:
    """Train agent with proper ML/RL best practices."""
    
    # Reproducibility: Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Environment setup with proper seeding
    env = gym.make("CartPole-v1")
    env.seed(seed)
    
    episode_rewards = []
    
    try:
        for episode in range(num_episodes):
            # Proper state handling - env.reset() returns tuple in newer Gym
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Handle new Gym API
            
            total_reward = 0
            step_count = 0
            max_steps = 500  # Prevent infinite episodes
            
            while step_count < max_steps:
                # Random policy for demonstration
                action = env.action_space.sample()
                
                # Handle step return format
                step_result = env.step(action)
                if len(step_result) == 4:
                    next_state, reward, done, info = step_result
                else:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                total_reward += reward
                step_count += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            # Proper logging instead of print
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode}, Avg Reward (last 10): {avg_reward:.2f}")
    
    finally:
        # Cleanup resources
        env.close()
    
    return episode_rewards
'''
    
    print("Improved ML code with best practices:")
    print(improved_ml_code)
    
    best_practices_applied = {
        "reproducibility": [
            "Added seed parameter and seeded all random number generators",
            "Seeded environment for deterministic behavior"
        ],
        "robustness": [
            "Added proper error handling with try-finally",
            "Handle different Gym API versions (reset/step return formats)",
            "Added max_steps to prevent infinite episodes"
        ],
        "monitoring": [
            "Replaced print statements with proper logging",
            "Track and return episode rewards for analysis",
            "Periodic progress reporting"
        ],
        "resource_management": [
            "Proper environment cleanup with env.close()",
            "Type hints for better code maintainability"
        ]
    }
    
    print("\nML/RL Best Practices Applied:")
    print(json.dumps(best_practices_applied, indent=2))
    
    return improved_ml_code

def run_all_demos():
    """Run all DeepSeek Chat Prefix Completion demos."""
    
    print("DeepSeek Chat Prefix Completion for Code Improvements")
    print("=" * 60)
    print("This demonstrates different ways to use DeepSeek's prefix completion:")
    print("1. JSON-structured responses for comprehensive analysis")
    print("2. Direct code completion for implementation")
    print("3. Security-focused improvements")
    print("4. ML/RL best practices application")
    print()
    
    # Run all demos
    demo_json_structured_improvement()
    demo_direct_code_completion() 
    demo_security_focused_improvement()
    demo_ml_best_practices()
    
    print("\n" + "=" * 60)
    print("SUMMARY: Chat Prefix Completion Benefits")
    print("=" * 60)
    print("✓ Forces specific output formats (JSON, code blocks)")
    print("✓ Ensures structured responses for parsing")
    print("✓ Reduces need for complex response parsing logic")
    print("✓ Better control over model output format")
    print("✓ Ideal for CodeGuard's structured improvement workflow")
    print()
    print("Integration with CodeGuard API:")
    print("• JSON prefix: '{' for structured improvement responses")
    print("• Code prefix: '```python' for direct code completion")
    print("• Comment prefix: '# Fixed:' for explanation-focused output")

if __name__ == "__main__":
    run_all_demos()